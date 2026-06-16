"""
audit_submission.py — sanity-check the produced submission against the
candidate pool: honeypot leakage, title-tier distribution, location mix,
and a look at the bottom of the ranking (since NDCG@50 still carries 30%
weight, the 11-50 band needs to be defensible too, not just top 10).
"""
import csv
import json

def load_candidates_by_id(path, ids):
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c["candidate_id"] in ids:
                result[c["candidate_id"]] = c
    return result

with open("outputs/submission.csv") as f:
    rows = list(csv.reader(f))[1:]

ids = {r[0] for r in rows}
candidates = load_candidates_by_id("data/candidates.jsonl", ids)

print(f"Total in submission: {len(rows)}")
print()

# Title tier check
TITLE_TIER_A = {
    "ml engineer", "machine learning engineer", "senior machine learning engineer",
    "staff machine learning engineer", "ai engineer", "senior ai engineer",
    "lead ai engineer", "ai research engineer", "data scientist",
    "senior data scientist", "senior software engineer (ml)", "nlp engineer",
    "senior nlp engineer", "computer vision engineer", "ai specialist",
    "junior ml engineer", "recommendation systems engineer",
    "applied ml engineer", "search engineer", "applied scientist",
    "senior applied scientist", "research engineer",
}
TITLE_TIER_D = {
    "business analyst", "hr manager", "mechanical engineer", "accountant",
    "project manager", "customer support", "operations manager",
    "content writer", "sales executive", "civil engineer",
    "graphic designer", "marketing manager",
}

tier_a_count = 0
tier_d_count = 0
for r in rows:
    c = candidates[r[0]]
    t = c["profile"]["current_title"].lower()
    if t in TITLE_TIER_A:
        tier_a_count += 1
    if t in TITLE_TIER_D:
        tier_d_count += 1

print(f"Tier A (ML/AI titled) in top 100: {tier_a_count}")
print(f"Tier D (clearly irrelevant title) in top 100: {tier_d_count}")
print()

# Honeypot leakage check
honeypot_count = 0
for r in rows:
    c = candidates[r[0]]
    for s in c["skills"]:
        if s["proficiency"] == "expert" and s.get("duration_months", 1) == 0:
            honeypot_count += 1
            print("HONEYPOT LEAK:", r[0], r[1])
            break
print(f"Honeypots leaked into top 100: {honeypot_count} (must be <=10)")
print()

# Location distribution
from collections import Counter
locs = Counter()
for r in rows:
    c = candidates[r[0]]
    locs[c["profile"]["location"]] += 1
print("Location distribution in top 100:")
for l, n in locs.most_common(15):
    print(f"  {n}  {l}")
print()

# Ghost-candidate check (inactive > 180 days) in top 20
print("Activity check, top 20:")
for r in rows[:20]:
    c = candidates[r[0]]
    sig = c["redrob_signals"]
    print(f"  rank {r[1]}: last_active={sig['last_active_date']}  "
          f"open_to_work={sig['open_to_work_flag']}  "
          f"response_rate={sig['recruiter_response_rate']}  "
          f"notice={sig['notice_period_days']}d")
print()

# Bottom 10 sanity check (ranks 91-100)
print("Bottom 10 (ranks 91-100):")
for r in rows[90:]:
    c = candidates[r[0]]
    print(f"  rank {r[1]}: {c['profile']['current_title']} | "
          f"{c['profile']['years_of_experience']} yrs | score={r[2]}")
    print(f"     reasoning: {r[3]}")
