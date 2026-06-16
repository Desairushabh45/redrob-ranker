# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

Top-100 candidate ranking for the Senior AI Engineer (Founding Team) role
at Redrob AI, built for the *Data & AI Challenge* track of the Redrob
Hackathon (Hack2Skill / India Runs).

## Reproduce the submission

```bash
pip install -r requirements.txt   # stdlib only for rank.py; streamlit is for the sandbox demo
python rank.py --candidates ./data/candidates.jsonl --out ./outputs/submission.csv
```

Runs in ~12 seconds on a single CPU core, well under the 5-minute / 16GB /
CPU-only / no-network compute budget in `submission_spec.md` Section 3.

Validate before submitting:

```bash
python validate_submission.py outputs/submission.csv
```

## Run the sandbox demo locally

```bash
streamlit run app.py
```

Upload a small candidate sample (or use the bundled `sample_candidates.json`)
to see the ranker run end-to-end and download a ranked CSV in submission
format.

## Approach

We read the job description closely rather than treating this as a pure
embedding/keyword-matching problem. The JD is explicit that the intended
trap is "find candidates whose skills section contains the most AI
keywords" and that the right answer requires reasoning about title,
career trajectory, and behavioral availability — not just a skills list.

### Scoring

```
final_score = (
      0.30 * title_career_score    # current + past titles, tiered A-D
    + 0.25 * skills_score          # trust-filtered must-have/good-to-have skills
    + 0.20 * experience_score      # years-of-experience sweet spot + product-co history
    + 0.10 * location_score        # Pune/Noida preferred, India + relocation considered
    + 0.15 * culture_fit_score     # penalizes title-chasing, pure-research-only profiles
) * availability_multiplier        # behavioral signals: activity, response rate, notice period
```

Honeypot candidates are hard-filtered to a score of 0 *before* any of the
above runs, rather than relying on the weighted score to naturally push
them down.

### Honeypot detection — methodology, not guesswork

We didn't hand-pick thresholds. We wrote `explore_honeypots.py` and ran
several candidate impossible-profile heuristics across the full 100K pool
before deciding what to use in the real ranker:

- **"Expert proficiency with 0 months used"** — isolates a clean, distinct
  population of 21 candidates, almost entirely on irrelevant titles (HR
  Manager, Accountant, Civil Engineer, etc.) holding random unrelated
  "expert" skills. This is the one we use.
- **"8+ expert-level skills"** — initially looked like a honeypot signal,
  but checking it against the full pool showed it fires heavily on
  *genuinely* senior, legitimate ML/AI titles (Senior AI Engineer, NLP
  Engineer, Recommendation Systems Engineer). We explicitly rejected this
  rule — it would have demoted exactly the candidates we want at the top.
- **YOE vs. career-history-duration mismatch** — noisy across the pool
  (career history is capped at 10 entries per the schema, so long careers
  naturally show a gap). Not used as a hard honeypot filter; would have
  false-positived on strong candidates like a 15-year Recommendation
  Systems Engineer.

### Title tiering — calibrated against the real data

Inspecting title frequency across the 100K pool shows a clear intentional
structure: irrelevant titles (Business Analyst, HR Manager, Accountant,
etc.) at ~5,500-5,800 each, generic engineering titles at ~2,500-3,500
each, data-adjacent titles at ~700-770 each, and genuine ML/AI titles at
4-167 each — with the most senior ones (Senior AI Engineer, Lead AI
Engineer) appearing only 4-6 times in the entire pool. This directly
informed the four-tier title scoring table in `rank.py`.

### Skills — a rarity-calibrated vocabulary, not a flat keyword list

Skill-frequency analysis across the pool shows two distinct tiers: a
common "AI buzzword" tier (RAG, LLMs, Embeddings, Pinecone, FAISS — each
~5,000 occurrences, scattered across many unrelated titles) and a rarer
"core practitioner" tier (Python, PyTorch, Elasticsearch, Learning to
Rank, BM25, Qdrant, Weaviate, Milvus, PEFT, QLoRA — each ~1,300-1,400
occurrences). The JD's "things you absolutely need" section maps closely
onto this rarer tier, so `SKILLS_MUST_HAVE` weights it accordingly, with
a trust filter (`proficiency in {advanced, expert}` AND `duration_months
> 6`) to defeat keyword stuffing.

### Behavioral availability

Directly implements the JD's own instruction: *"a perfect-on-paper
candidate who hasn't logged in for 6 months and has a 5% recruiter
response rate is, for hiring purposes, not actually available."* This is
a multiplier, not an additive score component, so it scales the entire
fit score down for unavailable candidates without letting a single
strong signal compensate for it.

## Files

| File | Purpose |
|---|---|
| `rank.py` | The ranker. Single command produces the submission CSV. |
| `app.py` | Streamlit sandbox demo (Section 10.5 requirement). |
| `explore_honeypots.py` | Exploratory script used to calibrate honeypot detection against the real 100K pool. Not part of the ranking pipeline. |
| `audit_submission.py` | Post-hoc quality audit of the produced submission (title-tier distribution, honeypot leakage check, location mix, activity sanity check). |
| `validate_submission.py` | Official format validator, provided in the hackathon bundle. |
| `submission_metadata.yaml` | Portal metadata mirror, per spec Section 10.3. |
| `outputs/submission.csv` | The final ranked top-100. |

## Compute environment

See `submission_metadata.yaml` for the exact compute environment summary
used to produce this submission.
