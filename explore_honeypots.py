"""
Exploration script — find honeypot patterns in the full 100K dataset.
This is NOT part of the final ranker; just used to calibrate thresholds.
"""
import json
from datetime import date, datetime
from collections import Counter

COMPANY_SIZE_TO_MIN_AGE = {
    # rough heuristic: tiny companies are usually younger
    "1-10": 0,
    "11-50": 0,
    "51-200": 1,
    "201-500": 2,
    "501-1000": 3,
    "1001-5000": 5,
    "5001-10000": 8,
    "10001+": 10,
}

def load_candidates(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def check_expert_zero_duration(c):
    flags = []
    for s in c["skills"]:
        if s["proficiency"] == "expert" and s.get("duration_months", 1) == 0:
            flags.append(f"expert/{s['name']}/0mo")
    return flags

def check_too_many_experts(c, threshold=8):
    n = sum(1 for s in c["skills"] if s["proficiency"] == "expert")
    return n >= threshold, n

def check_overlapping_current_jobs(c):
    current = [ch for ch in c["career_history"] if ch["is_current"]]
    return len(current) > 1

def check_years_vs_career_history(c):
    """Total experience should roughly match sum of career history durations."""
    total_months_history = sum(ch["duration_months"] for ch in c["career_history"])
    yoe_months = c["profile"]["years_of_experience"] * 12
    diff = yoe_months - total_months_history
    return diff

def check_company_age_mismatch(c):
    """Heuristic: company_size 1-10 but duration_months > 60 (5 yrs) at that single company is suspicious
    for a 'just founded' company. We don't have founding dates, so use duration as proxy."""
    flags = []
    for ch in c["career_history"]:
        if ch["company_size"] == "1-10" and ch["duration_months"] > 96:  # 8+ years at a tiny co
            flags.append(f"{ch['company']}: {ch['duration_months']}mo at size 1-10")
    return flags

def check_completeness_vs_activity(c):
    sig = c["redrob_signals"]
    completeness = sig["profile_completeness_score"]
    last_active = date.fromisoformat(sig["last_active_date"])
    days_inactive = (date(2026, 6, 16) - last_active).days
    return completeness > 95 and days_inactive > 1000

# Run exploration
path = "data/candidates.jsonl"
total = 0
expert_zero_dur_count = 0
many_experts_count = 0
overlap_count = 0
yoe_mismatch_extreme = 0
tiny_co_long_tenure = 0
completeness_ghost = 0

sample_flagged = []

for c in load_candidates(path):
    total += 1
    f1 = check_expert_zero_duration(c)
    f2, n_expert = check_too_many_experts(c)
    f3 = check_overlapping_current_jobs(c)
    diff = check_years_vs_career_history(c)
    f5 = check_company_age_mismatch(c)
    f6 = check_completeness_vs_activity(c)

    if f1:
        expert_zero_dur_count += 1
    if f2:
        many_experts_count += 1
    if f3:
        overlap_count += 1
    if abs(diff) > 36:  # more than 3 years mismatch between stated YOE and career history sum
        yoe_mismatch_extreme += 1
    if f5:
        tiny_co_long_tenure += 1
    if f6:
        completeness_ghost += 1

    if (f1 or f2 or f3 or f5 or f6) and len(sample_flagged) < 20:
        sample_flagged.append({
            "id": c["candidate_id"],
            "title": c["profile"]["current_title"],
            "expert_zero_dur": f1,
            "many_experts": (f2, n_expert),
            "overlap": f3,
            "yoe_diff_months": diff,
            "tiny_co_long_tenure": f5,
            "completeness_ghost": f6,
        })

print(f"Total candidates: {total}")
print(f"expert_zero_dur_count: {expert_zero_dur_count}")
print(f"many_experts_count (>=8 expert skills): {many_experts_count}")
print(f"overlapping current jobs: {overlap_count}")
print(f"extreme YOE vs history mismatch (>36mo): {yoe_mismatch_extreme}")
print(f"tiny company + 8yr+ tenure: {tiny_co_long_tenure}")
print(f"completeness>95 + inactive>1000d: {completeness_ghost}")
print()
print("Sample flagged candidates:")
for s in sample_flagged:
    print(s)
