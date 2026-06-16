#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon: Intelligent Candidate Discovery & Ranking

Produces the top-100 ranked candidates for the Senior AI Engineer (Founding
Team) role at Redrob AI, from a 100K candidate pool.

Usage:
    python rank.py --candidates ./data/candidates.jsonl --out ./outputs/submission.csv

Design summary (see README.md and presentation for full rationale):
    final_score = (
          0.30 * title_career_score
        + 0.25 * skills_score
        + 0.20 * experience_score
        + 0.10 * location_score
        + 0.15 * culture_fit_score
    ) * availability_multiplier

    Honeypots (subtly impossible profiles) are hard-filtered to score 0
    before any of the above is computed.

No GPU. No network calls. No hosted LLM calls. Pure Python + stdlib,
runs on 100K candidates in well under the 5-minute / 16GB budget.
"""

import argparse
import csv
import json
import sys
from datetime import date, datetime

# ---------------------------------------------------------------------------
# Reference date: the dataset's "today" — last_active_date values cluster
# right up to this point. Used for recency calculations.
# ---------------------------------------------------------------------------
TODAY = date(2026, 6, 16)

# ---------------------------------------------------------------------------
# Vocabulary tables — calibrated by inspecting the actual 100K-candidate
# pool's title and skill-frequency distributions (see explore_honeypots.py
# and the methodology notes in README.md).
# ---------------------------------------------------------------------------

# Tier A: titles that are themselves strong evidence of ML/AI engineering work.
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

# Tier B: adjacent engineering titles — plausible if career history / skills
# show real ML depth, but the title alone isn't sufficient evidence.
TITLE_TIER_B = {
    "senior software engineer", "software engineer", "backend engineer",
    "data engineer", "senior data engineer", "analytics engineer",
    "data analyst", "full stack developer",
}

# Tier C: generic tech roles — plausible only with very strong corroborating
# skill/career evidence (rare in this pool, mostly keyword-stuffer territory).
TITLE_TIER_C = {
    "devops engineer", "cloud engineer", "qa engineer", "mobile developer",
    ".net developer", "java developer", "frontend engineer",
}

# Tier D: explicitly not a fit regardless of skills listed (JD: "a candidate
# with all the AI keywords but title 'Marketing Manager' is not a fit").
TITLE_TIER_D = {
    "business analyst", "hr manager", "mechanical engineer", "accountant",
    "project manager", "customer support", "operations manager",
    "content writer", "sales executive", "civil engineer",
    "graphic designer", "marketing manager",
}

# Core "must-have" skills per JD: production embeddings/retrieval, vector
# search/hybrid infra, Python, and ranking-evaluation literacy. Calibrated
# as the rare (~1300/100K) skill tier vs. the much more common buzzword tier.
SKILLS_MUST_HAVE = {
    "python", "pytorch", "tensorflow", "nlp", "machine learning",
    "deep learning", "scikit-learn",
    "elasticsearch", "opensearch", "qdrant", "weaviate", "milvus",
    "pgvector", "faiss", "pinecone",
    "embeddings", "vector search", "semantic search", "information retrieval",
    "sentence transformers", "hugging face transformers",
    "learning to rank", "bm25", "haystack", "llamaindex",
}

# Good-to-have per JD (LLM fine-tuning, LTR, prior HR-tech exposure signals)
SKILLS_GOOD_TO_HAVE = {
    "lora", "qlora", "peft", "fine-tuning llms", "llms", "rag",
    "prompt engineering", "langchain", "recommendation systems",
}

# JD explicit negative: CV/speech/robotics-only profiles without NLP/IR.
SKILLS_OFF_DOMAIN = {
    "image classification", "object detection", "yolo", "computer vision",
    "opencv", "gans", "diffusion models", "speech recognition", "asr", "tts",
    "cnn",
}

# Services-only companies the JD explicitly flags as a weak signal (unless
# paired with prior product-company experience elsewhere in career history).
SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "mindtree",
}

SERVICES_INDUSTRIES = {"it services", "consulting"}
PRODUCT_INDUSTRIES = {
    "software", "saas", "fintech", "ai/ml", "e-commerce", "edtech",
    "adtech", "insurance tech", "healthtech", "healthtech ai",
    "conversational ai", "ai services", "voice ai", "internet", "gaming",
    "food delivery", "transportation",
}

# JD-preferred locations (Pune/Noida primary; other Tier-1 Indian hubs welcome)
PREFERRED_LOCATIONS = {"pune", "noida"}
WELCOME_LOCATIONS = {
    "hyderabad", "mumbai", "delhi", "gurgaon", "bangalore", "bengaluru",
}

INDIAN_LOCATION_HINTS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon", "bangalore",
    "bengaluru", "chennai", "kolkata", "ahmedabad", "jaipur", "indore",
    "chandigarh", "coimbatore", "kochi", "trivandrum", "bhubaneswar",
    "vizag",
}


def norm(s):
    return (s or "").strip().lower()


# ---------------------------------------------------------------------------
# Honeypot detection
# ---------------------------------------------------------------------------
def is_honeypot(c):
    """
    Hard-filter for subtly impossible profiles, per the spec's example:
    'expert proficiency in a skill with 0 months used.' Calibrated against
    the full 100K pool — this exact pattern isolates a clean, distinct
    population (21/100K) almost entirely on irrelevant titles with random
    unrelated 'expert' skills. We deliberately do NOT use a raw
    "many expert skills" rule, since that incorrectly flags genuine senior
    ML engineers who legitimately hold many expert-level skills.
    """
    for s in c.get("skills", []):
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0:
            return True
    return False


# ---------------------------------------------------------------------------
# Component scores
# ---------------------------------------------------------------------------
def title_career_score(c):
    """
    30% weight — the single most decisive signal. Directly encodes the JD's
    explicit guidance: title and demonstrated career trajectory matter more
    than a skills list. Looks at current title AND career history, so a
    candidate who has held an ML title before (even if their most recent
    title is more generic) still gets credit.
    """
    current_title = norm(c["profile"]["current_title"])
    history_titles = [norm(ch["title"]) for ch in c["career_history"]]
    all_titles = [current_title] + history_titles

    best = 0.0
    for i, t in enumerate(all_titles):
        if t in TITLE_TIER_A:
            tier_score = 1.0
        elif t in TITLE_TIER_B:
            tier_score = 0.55
        elif t in TITLE_TIER_C:
            tier_score = 0.25
        elif t in TITLE_TIER_D:
            tier_score = 0.0
        else:
            tier_score = 0.2  # unrecognized title, mild benefit of the doubt

        # Past titles count less than the current one — current role is
        # the strongest signal of what the person does today.
        recency_discount = 1.0 if i == 0 else 0.7
        best = max(best, tier_score * recency_discount)

    return best


def skills_score(c, signals):
    """
    25% weight. Applies a "trust filter": a skill only counts if it's listed
    as advanced/expert AND has duration_months > 6. This directly defeats
    the keyword-stuffer trap (skills listed as 'expert' with near-zero
    real-world use). Verified Redrob skill-assessment scores (when present)
    are an extra trust signal layered on top.
    """
    trusted_must = 0
    trusted_good = 0
    off_domain = 0

    for s in c.get("skills", []):
        name = norm(s["name"])
        prof = s.get("proficiency")
        dur = s.get("duration_months", 0)
        trusted = prof in ("advanced", "expert") and dur > 6

        if name in SKILLS_MUST_HAVE and trusted:
            trusted_must += 1
        if name in SKILLS_GOOD_TO_HAVE and trusted:
            trusted_good += 1
        if name in SKILLS_OFF_DOMAIN and trusted:
            off_domain += 1

    # Diminishing returns: first few must-have skills matter a lot, the
    # marginal 10th matters little.
    must_component = min(trusted_must, 6) / 6.0
    good_component = min(trusted_good, 3) / 3.0 * 0.3

    score = 0.75 * must_component + good_component

    # JD explicitly deprioritizes CV/speech/robotics-only profiles.
    if off_domain >= 3 and trusted_must == 0:
        score *= 0.4

    # Verified assessment bonus: an actual Redrob skill-assessment score of
    # 70+ on a must-have skill is a real signal, not a self-report.
    assess = signals.get("skill_assessment_scores", {}) or {}
    for skill_name, val in assess.items():
        if norm(skill_name) in SKILLS_MUST_HAVE and val >= 70:
            score = min(1.0, score + 0.05)

    return min(1.0, score)


def experience_score(c):
    """
    20% weight. JD sweet spot: 5-9 years (ideal 6-8), with meaningful time
    at product companies rather than pure IT-services shops.
    """
    yoe = c["profile"]["years_of_experience"]

    if 6 <= yoe <= 8:
        years_component = 1.0
    elif 5 <= yoe < 6 or 8 < yoe <= 9:
        years_component = 0.9
    elif 4 <= yoe < 5 or 9 < yoe <= 11:
        years_component = 0.7
    elif 3 <= yoe < 4 or 11 < yoe <= 13:
        years_component = 0.5
    elif yoe < 3:
        years_component = 0.3
    else:  # very senior, 13+
        years_component = 0.35

    # Product-company exposure across career history.
    product_months = 0
    services_only_months = 0
    total_months = 0
    for ch in c["career_history"]:
        dur = ch.get("duration_months", 0)
        total_months += dur
        ind = norm(ch.get("industry", ""))
        comp = norm(ch.get("company", ""))
        if ind in PRODUCT_INDUSTRIES:
            product_months += dur
        elif ind in SERVICES_INDUSTRIES or comp in SERVICES_COMPANIES:
            services_only_months += dur

    if total_months == 0:
        product_ratio = 0.0
    else:
        product_ratio = product_months / total_months

    # JD: "people who have only worked at consulting firms in their entire
    # career" are explicitly not wanted; prior product experience redeems it.
    if product_months == 0 and services_only_months > 0:
        product_penalty_mult = 0.5
    else:
        product_penalty_mult = 1.0

    company_component = min(1.0, product_ratio + 0.2) * product_penalty_mult

    return 0.6 * years_component + 0.4 * company_component


def location_score(c, signals):
    """
    10% weight. Pune/Noida preferred; other Indian metros explicitly
    welcomed; relocation willingness substitutes for current location.
    """
    loc = norm(c["profile"]["location"])
    country = norm(c["profile"]["country"])
    willing = signals.get("willing_to_relocate", False)

    loc_first_word = loc.split(",")[0].strip()

    if loc_first_word in PREFERRED_LOCATIONS:
        base = 1.0
    elif loc_first_word in WELCOME_LOCATIONS:
        base = 0.85
    elif country == "india" or loc_first_word in INDIAN_LOCATION_HINTS:
        base = 0.6
    else:
        base = 0.15

    if base < 1.0 and willing:
        base = min(1.0, base + 0.35)

    return base


def culture_fit_score(c):
    """
    15% weight. Encodes the JD's explicit "things we explicitly do NOT
    want" section: title-chasers (job-hopping every ~1.5 years while
    climbing a title ladder), framework-tutorial-only profiles, and
    pure-research-without-production backgrounds. This is a penalty-style
    score: starts at 1.0, deducted for red flags found in career history
    and summary text.
    """
    score = 1.0
    history = c["career_history"]

    # Title-chasing: many short (<=18mo) stints while title escalates
    # through seniority ladder words.
    short_stints = sum(1 for ch in history if ch.get("duration_months", 0) <= 18)
    if len(history) >= 3 and short_stints >= 3:
        score -= 0.25

    # Pure research / academic-only background (no industry production
    # signal at all) — JD explicit disqualifier.
    industries_seen = {norm(ch.get("industry", "")) for ch in history}
    if industries_seen and industries_seen.issubset({"academic", "research"}):
        score -= 0.5

    summary = norm(c["profile"].get("summary", ""))
    headline = norm(c["profile"].get("headline", ""))
    text = summary + " " + headline

    # JD: "if your AI experience consists primarily of recent LangChain
    # demos calling OpenAI" — a soft textual heuristic on the summary.
    if "langchain" in text and "tutorial" in text:
        score -= 0.2

    return max(0.0, score)


def availability_multiplier(signals):
    """
    Behavioral-signal multiplier layered on top of the fit score, per the
    JD's explicit instruction: "a perfect-on-paper candidate who hasn't
    logged in for 6 months and has a 5% recruiter response rate is, for
    hiring purposes, not actually available. Down-weight them appropriately."
    """
    mult = 1.0

    if not signals.get("open_to_work_flag", False):
        mult *= 0.75

    try:
        last_active = date.fromisoformat(signals["last_active_date"])
        days_inactive = (TODAY - last_active).days
    except Exception:
        days_inactive = 9999

    if days_inactive > 180:
        mult *= 0.4
    elif days_inactive > 90:
        mult *= 0.65
    elif days_inactive > 30:
        mult *= 0.9
    # else: active within 30 days, no penalty

    rr = signals.get("recruiter_response_rate", 0.5)
    if rr < 0.1:
        mult *= 0.55
    elif rr < 0.3:
        mult *= 0.8
    elif rr < 0.5:
        mult *= 0.95

    notice = signals.get("notice_period_days", 60)
    if notice <= 30:
        mult *= 1.0
    elif notice <= 60:
        mult *= 0.95
    elif notice <= 90:
        mult *= 0.85
    else:
        mult *= 0.7

    # Interview / offer behavior: a track record of not completing
    # interviews or rejecting offers is a real availability signal.
    icr = signals.get("interview_completion_rate", 1.0)
    if icr < 0.5:
        mult *= 0.85

    return max(0.05, mult)


# ---------------------------------------------------------------------------
# Final scoring
# ---------------------------------------------------------------------------
def score_candidate(c):
    if is_honeypot(c):
        return 0.0, {}

    signals = c.get("redrob_signals", {})

    t = title_career_score(c)
    sk = skills_score(c, signals)
    e = experience_score(c)
    l = location_score(c, signals)
    cu = culture_fit_score(c)

    base = (0.30 * t) + (0.25 * sk) + (0.20 * e) + (0.10 * l) + (0.15 * cu)
    mult = availability_multiplier(signals)
    final = base * mult

    breakdown = {
        "title_career": round(t, 3),
        "skills": round(sk, 3),
        "experience": round(e, 3),
        "location": round(l, 3),
        "culture_fit": round(cu, 3),
        "availability_mult": round(mult, 3),
    }
    return final, breakdown


# ---------------------------------------------------------------------------
# Reasoning generation — short, specific, non-templated per-candidate text.
# ---------------------------------------------------------------------------
def build_reasoning(c, breakdown):
    p = c["profile"]
    signals = c.get("redrob_signals", {})
    parts = []

    yoe = p["years_of_experience"]
    title = p["current_title"]
    company = p["current_company"]
    parts.append(f"{yoe:.1f} yrs experience, currently {title} at {company}")

    if breakdown.get("title_career", 0) >= 0.8:
        parts.append("title directly matches the ML/AI engineering profile the JD targets")
    elif breakdown.get("title_career", 0) >= 0.4:
        parts.append("adjacent engineering title with relevant signal in career history")
    else:
        parts.append("title is a weaker match for the role, included on other strengths")

    # mention 1-2 specific trusted must-have skills if present
    trusted_must_skills = [
        s["name"] for s in c.get("skills", [])
        if norm(s["name"]) in SKILLS_MUST_HAVE
        and s.get("proficiency") in ("advanced", "expert")
        and s.get("duration_months", 0) > 6
    ]
    if trusted_must_skills:
        sample = ", ".join(trusted_must_skills[:3])
        parts.append(f"hands-on with {sample}")

    loc = p["location"]
    willing = signals.get("willing_to_relocate", False)
    if breakdown.get("location", 0) >= 0.85:
        parts.append(f"based in {loc}, matches preferred location")
    elif willing:
        parts.append(f"based in {loc} but open to relocation")
    else:
        parts.append(f"based in {loc}, not an ideal location match")

    notice = signals.get("notice_period_days")
    if notice is not None:
        if notice <= 30:
            parts.append(f"{notice}-day notice period is ideal")
        elif notice > 90:
            parts.append(f"{notice}-day notice period is a real concern")

    last_active = signals.get("last_active_date")
    if last_active:
        try:
            days_inactive = (TODAY - date.fromisoformat(last_active)).days
            if days_inactive > 180:
                parts.append("inactive on the platform for 6+ months")
        except Exception:
            pass

    return "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_candidates(path):
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-n", type=int, default=100)
    args = ap.parse_args()

    scored = []
    for c in load_candidates(args.candidates):
        final, breakdown = score_candidate(c)
        if final > 0.0:
            scored.append((c, final, breakdown))

    # Sort by score desc, tie-break by candidate_id ascending (per spec
    # Section 3: "break score ties ... by candidate_id ascending").
    scored.sort(key=lambda x: (-x[1], x[0]["candidate_id"]))

    top = scored[: args.top_n]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (c, final, breakdown) in enumerate(top, start=1):
            reasoning = build_reasoning(c, breakdown)
            writer.writerow([c["candidate_id"], rank, round(final, 4), reasoning])

    print(f"Wrote {len(top)} ranked candidates to {args.out}")
    print(f"Total candidates scanned: scored {len(scored)} (non-honeypot), "
          f"out of pool")


if __name__ == "__main__":
    main()
