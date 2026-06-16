"""
app.py — Streamlit sandbox demo for the Redrob Hackathon submission.

Lets a reviewer upload a small candidate sample (<=100 candidates, JSONL or
the provided sample_candidates.json) and see the ranker run end-to-end,
producing a ranked CSV — satisfying the spec's Section 10.5 sandbox
requirement.

Run locally:
    streamlit run app.py

Deploy: push this repo to GitHub and deploy on Streamlit Community Cloud
pointing at app.py.
"""
import json
import io
import csv as csv_module

import streamlit as st

from rank import score_candidate, build_reasoning

st.set_page_config(page_title="Redrob Ranker — Sandbox", layout="wide")

st.title("Redrob Hackathon — Intelligent Candidate Ranker")
st.caption(
    "Sandbox demo for the Data & AI Challenge submission. Upload a small "
    "candidate sample (JSONL, one candidate JSON object per line, or the "
    "sample_candidates.json provided in the hackathon bundle) to see the "
    "ranker run end-to-end."
)

with st.expander("How this ranker works", expanded=False):
    st.markdown(
        """
        **Architecture:** a five-component hybrid score, multiplied by a
        behavioral-availability modifier.

        ```
        final_score = (
              0.30 * title_career_score
            + 0.25 * skills_score
            + 0.20 * experience_score
            + 0.10 * location_score
            + 0.15 * culture_fit_score
        ) * availability_multiplier
        ```

        Honeypot candidates (e.g. "expert" proficiency in a skill listed
        with 0 months of use) are hard-filtered to a score of 0 before any
        of the above is computed.

        No GPU, no network calls, no hosted LLM calls — pure Python,
        runs the full 100K-candidate pool in well under the 5-minute /
        16GB compute budget specified in the rules.
        """
    )

uploaded = st.file_uploader(
    "Upload a candidate sample (.jsonl or .json, max 100 candidates)",
    type=["jsonl", "json"],
)

use_sample = st.button("Or use the bundled sample_candidates.json")

candidates = None

if uploaded is not None:
    raw = uploaded.read().decode("utf-8")
    if uploaded.name.endswith(".jsonl"):
        candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        data = json.loads(raw)
        candidates = data if isinstance(data, list) else [data]

elif use_sample:
    with open("data/sample_candidates.json") as f:
        candidates = json.load(f)

if candidates:
    if len(candidates) > 100:
        st.warning(
            f"Sample has {len(candidates)} candidates; sandbox is capped at "
            "100 for the quick demo. Using the first 100."
        )
        candidates = candidates[:100]

    st.write(f"Loaded **{len(candidates)}** candidates. Running ranker...")

    scored = []
    for c in candidates:
        final, breakdown = score_candidate(c)
        scored.append((c, final, breakdown))

    scored = [(c, round(final, 4), breakdown) for c, final, breakdown in scored]
    scored.sort(key=lambda x: (-x[1], x[0]["candidate_id"]))

    rows = []
    for rank, (c, final, breakdown) in enumerate(scored, start=1):
        reasoning = build_reasoning(c, breakdown)
        rows.append(
            {
                "rank": rank,
                "candidate_id": c["candidate_id"],
                "score": final,
                "title": c["profile"]["current_title"],
                "company": c["profile"]["current_company"],
                "location": c["profile"]["location"],
                "reasoning": reasoning,
                "is_honeypot": final == 0.0,
            }
        )

    st.success(f"Ranked {len(rows)} candidates.")
    st.dataframe(
        [{k: v for k, v in r.items() if k != "is_honeypot"} for r in rows],
        use_container_width=True,
        height=500,
    )

    n_honeypot = sum(1 for r in rows if r["is_honeypot"])
    st.caption(f"Honeypots detected and zeroed out in this sample: {n_honeypot}")

    # Downloadable CSV in the exact submission format
    output = io.StringIO()
    writer = csv_module.writer(output)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in rows:
        writer.writerow([r["candidate_id"], r["rank"], r["score"], r["reasoning"]])

    st.download_button(
        "Download ranked CSV (submission format)",
        data=output.getvalue(),
        file_name="sandbox_ranking.csv",
        mime="text/csv",
    )
else:
    st.info("Upload a file or click the sample-data button above to see the ranker run.")
