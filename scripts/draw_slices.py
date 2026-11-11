"""Shuffle the eligible pool once (seed 42) and cut consecutive slices.

Writes data/frame/slices.json (full shuffled order + slice of each position) and prints each
slice's balance (decision, primary area, mean human rating). Refuses to overwrite an existing
draw: seed 42 is used once. Replacements come from the slice's reserve, in shuffled order.
"""
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FRAME = ROOT / "data" / "frame"
DERIVED = ROOT / "data" / "derived"

SEED = 42
SLICES = [  # (name, size), assigned as consecutive blocks of the shuffled order
    ("gepa_train", 80),
    ("gepa_val", 40),
    ("locked_test", 150),
    ("reserve_a", 50),
    ("reserve_b", 155),
    ("reserve_c", 50),
]


def main():
    out = FRAME / "slices.json"
    if out.exists():
        raise SystemExit(f"{out} exists; the seed-{SEED} draw is done once and never redrawn.")

    text = (FRAME / "eligible_ids.txt").read_text()
    ids = sorted(text.split())
    random.Random(SEED).shuffle(ids)

    assignment, pos = {}, 0
    for name, size in SLICES:
        assignment[name] = ids[pos:pos + size]
        pos += size
    json.dump({
        "seed": SEED,
        "eligible_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "slice_sizes": dict(SLICES),
        "shuffled_order": ids,
        "slices": assignment,
    }, out.open("w"), indent=1)

    # Balance report.
    papers = pd.read_parquet(DERIVED / "papers.parquet").set_index("forum_id")
    reviews = pd.read_parquet(DERIVED / "reviews.parquet")
    excluded = set(pd.read_csv(DERIVED / "review_exclusions.csv").review_id)
    human = reviews[~reviews.review_id.isin(excluded)]
    papers["mean_rating"] = human.groupby("forum_id").rating.mean()
    papers["accepted"] = papers.decision.fillna("").str.startswith("Accept")
    groups = {"eligible_pool": ids, **assignment}
    rows = []
    for name, members in groups.items():
        p = papers.loc[members]
        rows.append({"slice": name, "n": len(p), "accepted": round(p.accepted.mean(), 3),
                     "withdrawn": round((p.status == "withdrawn").mean(), 3),
                     "mean_rating": round(p.mean_rating.mean(), 2),
                     "n_areas": p.primary_area.nunique()})
    balance = pd.DataFrame(rows)
    balance.to_csv(FRAME / "slice_balance.csv", index=False)
    print(balance.to_string(index=False))
    print(f"used {pos} of {len(ids)} eligible papers")


if __name__ == "__main__":
    main()
