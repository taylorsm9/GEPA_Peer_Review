"""Re-slice over the eligible papers that have an as-submitted PDF.

Keeps the seed-42 shuffled order from data/frame/slices.json, drops papers without a PDF
(data/derived/pdfs.csv, pdf_status == "ok"), and cuts consecutive slices from what remains.
Papers over MAX_PAPER_TOKENS are then dropped from their slice and replaced by the next papers in the order
after the last slice, train first, then val, then test, skipping any that are also too long. Lengths need
both MinerU tiers, so the script stops if a paper it has to measure isn't converted yet.
Everything after the replacements is left unassigned ("pool"). Writes data/frame/slices_pdf.json
and data/frame/slice_balance_pdf.csv, and prints each slice's balance.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from intake.mineru import paper_text
from reviewer.review import MAX_PAPER_TOKENS

FRAME = ROOT / "data" / "frame"
DERIVED = ROOT / "data" / "derived"

SLICES = [  # (name, size), assigned as consecutive blocks of the filtered shuffled order
    ("gepa_train", 80),
    ("gepa_val", 40),
    ("locked_test", 150),
]


def paper_tokens(fid, tok):
    std, adv = ROOT / f"data/mineru/zips/{fid}.zip", ROOT / f"data/mineru_adv/zips/{fid}.zip"
    if not (std.exists() and adv.exists()):
        sys.exit(f"{fid} isn't converted by both MinerU tiers yet; rerun when it is")
    return len(tok.encode(paper_text(std, adv)[0]).ids)


def main():
    draw = json.load((FRAME / "slices.json").open())
    pdfs = pd.read_csv(DERIVED / "pdfs.csv")
    has_pdf = set(pdfs.loc[pdfs.pdf_status == "ok", "forum_id"])
    ids = [i for i in draw["shuffled_order"] if i in has_pdf]

    assignment, pos = {}, 0
    for name, size in SLICES:
        assignment[name] = ids[pos:pos + size]
        pos += size

    tok = Tokenizer.from_file(str(ROOT / "data/tokenizers/deepseek-v3_tokenizer.json"))
    dropped = {}
    for name, _ in SLICES:
        keep = []
        for fid in assignment[name]:
            if paper_tokens(fid, tok) <= MAX_PAPER_TOKENS:
                keep.append(fid)
                continue
            dropped[fid] = name
            while paper_tokens(ids[pos], tok) > MAX_PAPER_TOKENS:
                dropped[ids[pos]] = "pool"
                pos += 1
            keep.append(ids[pos])
            print(f"{name}: {fid} is over {MAX_PAPER_TOKENS} tokens, replaced by {ids[pos]}")
            pos += 1
        assignment[name] = keep
    json.dump({
        "seed": draw["seed"],
        "source": "slices.json shuffled_order, filtered to pdf_status == ok",
        "slice_sizes": dict(SLICES),
        "shuffled_order": ids,
        "slices": assignment,
        "max_paper_tokens": MAX_PAPER_TOKENS,
        "dropped_too_long": dropped,
    }, (FRAME / "slices_pdf.json").open("w"), indent=1)

    # Balance report, same columns as draw_slices.py.
    papers = pd.read_parquet(DERIVED / "papers.parquet").set_index("forum_id")
    reviews = pd.read_parquet(DERIVED / "reviews.parquet")
    excluded = set(pd.read_csv(DERIVED / "review_exclusions.csv").review_id)
    human = reviews[~reviews.review_id.isin(excluded)]
    papers["mean_rating"] = human.groupby("forum_id").rating.mean()
    papers["accepted"] = papers.decision.fillna("").str.startswith("Accept")
    groups = {"eligible_all": draw["shuffled_order"], "eligible_with_pdf": ids,
              **assignment, "pool": [i for i in ids[pos:] if i not in dropped]}
    rows = []
    for name, members in groups.items():
        p = papers.loc[members]
        rows.append({"slice": name, "n": len(p), "accepted": round(p.accepted.mean(), 3),
                     "withdrawn": round((p.status == "withdrawn").mean(), 3),
                     "mean_rating": round(p.mean_rating.mean(), 2),
                     "n_areas": p.primary_area.nunique()})
    balance = pd.DataFrame(rows)
    balance.to_csv(FRAME / "slice_balance_pdf.csv", index=False)
    print(balance.to_string(index=False))


if __name__ == "__main__":
    main()
