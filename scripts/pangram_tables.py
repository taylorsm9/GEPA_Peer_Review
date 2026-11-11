"""Tidy Pangram tables keyed on OpenReview IDs, for filtering and covariates.

Writes, for every paper in the frame and every official review:
  data/derived/pangram_papers.parquet   forum_id, fraction_ai, in_pangram, eligible, exclusion
  data/derived/pangram_reviews.parquet  review_id, forum_id, prediction_sort, prediction_display,
                                        in_pangram, exclusion (null = counts as a human review)
Exclusion reasons come from build_eligible.py's outputs, so the two never disagree.
Inputs: data/raw/pangram/*.json, data/derived/{papers,reviews}.parquet, data/frame/eligible_ids.txt,
data/derived/{paper,review}_exclusions.csv.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "data" / "derived"
PANGRAM = ROOT / "data" / "raw" / "pangram"


def main():
    ps = pd.DataFrame(json.load(open(PANGRAM / "submissions.json"))).set_index("submission_id")
    pr = pd.DataFrame(json.load(open(PANGRAM / "reviews.json"))).set_index("review_id")
    eligible = set((ROOT / "data" / "frame" / "eligible_ids.txt").read_text().split())

    papers = pd.read_parquet(DERIVED / "papers.parquet")[["forum_id"]]
    papers["fraction_ai"] = papers.forum_id.map(ps.fraction_ai)
    papers["in_pangram"] = papers.forum_id.isin(ps.index)
    papers["eligible"] = papers.forum_id.isin(eligible)
    papers["exclusion"] = papers.forum_id.map(
        pd.read_csv(DERIVED / "paper_exclusions.csv").set_index("forum_id").reason)
    papers.to_parquet(DERIVED / "pangram_papers.parquet", index=False)

    reviews = pd.read_parquet(DERIVED / "reviews.parquet")[["review_id", "forum_id"]]
    reviews["prediction_sort"] = reviews.review_id.map(pr.prediction_sort).astype("Int64")
    reviews["prediction_display"] = reviews.review_id.map(pr.prediction_display)
    reviews["in_pangram"] = reviews.review_id.isin(pr.index)
    reviews["exclusion"] = reviews.review_id.map(
        pd.read_csv(DERIVED / "review_exclusions.csv").set_index("review_id").reason)
    reviews.to_parquet(DERIVED / "pangram_reviews.parquet", index=False)

    print(len(papers), "papers;", papers.eligible.sum(), "eligible")
    print(reviews.groupby(["prediction_sort", "prediction_display"], dropna=False).size().to_string())


if __name__ == "__main__":
    main()
