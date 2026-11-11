"""Filter the frame to the eligible pool, logging the funnel and exclusion reasons.

Rules:
  - desk-rejected, or no official reviews                      -> out
  - paper missing from Pangram or fraction_ai is null          -> out (unknown AI share; conservative)
  - paper fraction_ai >= 0.5 (majority or fully AI text)       -> out; fraction_ai kept as covariate
  - keep only reviews Pangram labels "Fully human-written" and whose scores were not changed
    after the Nov 28 2025 reversion; fewer than 3 such reviews  -> out
Withdrawn papers stay in: their reviews are pre-rebuttal judgments like any other.

Inputs: data/frame/all_ids.txt, data/derived/{papers,reviews}.parquet, data/raw/pangram/*.json,
        data/derived/post_freeze_reviews.csv (review_id, reason)
Outputs: data/frame/{eligible_ids.txt, eligible_ids.sha256, funnel.csv},
         data/derived/{paper_exclusions.csv, review_exclusions.csv}
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FRAME = ROOT / "data" / "frame"
DERIVED = ROOT / "data" / "derived"
PANGRAM = ROOT / "data" / "raw" / "pangram"

MIN_HUMAN_REVIEWS = 3
MAX_FRACTION_AI = 0.5  # exclude papers with fraction_ai >= this
HUMAN_LABEL = 1  # Pangram prediction_sort: 1 = Fully human-written


def main():
    frame = (FRAME / "all_ids.txt").read_text().split()
    papers = pd.read_parquet(DERIVED / "papers.parquet").set_index("forum_id").loc[frame]
    reviews = pd.read_parquet(DERIVED / "reviews.parquet")
    ps = pd.DataFrame(json.load(open(PANGRAM / "submissions.json"))).set_index("submission_id")
    pr = pd.DataFrame(json.load(open(PANGRAM / "reviews.json"))).set_index("review_id")
    post_freeze = pd.read_csv(DERIVED / "post_freeze_reviews.csv")

    # Review-level exclusions, one reason per review (first match wins).
    label = reviews.review_id.map(pr.prediction_sort)
    rev_reason = pd.Series(None, index=reviews.index, dtype="object")
    rev_reason[label.isna()] = "no_pangram_label"
    rev_reason[rev_reason.isna() & (label != HUMAN_LABEL)] = "pangram_not_fully_human"
    freeze_reason = reviews.review_id.map(post_freeze.set_index("review_id").reason)
    rev_reason[rev_reason.isna()] = freeze_reason[rev_reason.isna()]
    reviews["exclusion"] = rev_reason
    n_kept = reviews[rev_reason.isna()].groupby("forum_id").size()

    # Paper-level funnel, applied in order.
    fa = papers.index.to_series().map(ps.fraction_ai)
    steps = [
        ("desk_rejected", papers.status == "desk_rejected"),
        ("no_official_reviews", papers.n_reviews == 0),
        ("no_pangram_fraction_ai", fa.isna()),
        ("pangram_fraction_ai_ge_0.5", fa >= MAX_FRACTION_AI),
        (f"fewer_than_{MIN_HUMAN_REVIEWS}_human_reviews",
         papers.index.to_series().map(n_kept).fillna(0) < MIN_HUMAN_REVIEWS),
    ]
    reason = pd.Series(None, index=papers.index, dtype="object")
    funnel = [{"step": "frame", "removed": 0, "remaining": len(papers)}]
    for name, mask in steps:
        hit = reason.isna() & mask
        reason[hit] = name
        funnel.append({"step": name, "removed": int(hit.sum()), "remaining": int(reason.isna().sum())})

    eligible = sorted(reason[reason.isna()].index)
    body = "".join(f"{i}\n" for i in eligible)
    (FRAME / "eligible_ids.txt").write_text(body)
    digest = hashlib.sha256(body.encode()).hexdigest()
    (FRAME / "eligible_ids.sha256").write_text(f"{digest}  eligible_ids.txt\n")
    pd.DataFrame(funnel).to_csv(FRAME / "funnel.csv", index=False)
    (reason.dropna().rename("reason").rename_axis("forum_id").reset_index()
     .to_csv(DERIVED / "paper_exclusions.csv", index=False))
    (reviews.loc[rev_reason.notna(), ["review_id", "forum_id", "exclusion"]]
     .rename(columns={"exclusion": "reason"}).to_csv(DERIVED / "review_exclusions.csv", index=False))

    print(pd.DataFrame(funnel).to_string(index=False))
    print("review exclusions:", reviews.exclusion.value_counts(dropna=False).to_dict())
    print(f"eligible: {len(eligible)} papers, sha256 {digest}")


if __name__ == "__main__":
    main()
