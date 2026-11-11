"""Parse the raw ICLR 2026 pull into papers.parquet and reviews.parquet, and snapshot the frame.

Reads data/raw/iclr2026/submissions_*.jsonl (from pull_iclr2026.py).
Writes data/derived/{papers,reviews,other_replies}.parquet and, only if absent,
data/frame/all_ids.txt + all_ids.sha256 (the frame is never regenerated).

Scores are kept both raw (as served) and as parsed integers; scales are verified
by printing the observed value sets, not assumed.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "iclr2026"
DERIVED = ROOT / "data" / "derived"
FRAME = ROOT / "data" / "frame"

SCORE_FIELDS = ["rating", "soundness", "presentation", "contribution", "confidence"]


def val(content, key):
    v = content.get(key)
    return v.get("value") if isinstance(v, dict) else v


def leading_int(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return int(x)
    m = re.match(r"\s*(-?\d+)", str(x))
    return int(m.group(1)) if m else None


def reply_kind(reply):
    # Invitation ids look like ICLR.cc/2026/Conference/Submission123/-/Official_Review
    return reply["invitations"][0].rsplit("/-/", 1)[-1]


def main():
    DERIVED.mkdir(parents=True, exist_ok=True)
    FRAME.mkdir(parents=True, exist_ok=True)

    notes = [json.loads(line) for p in sorted(RAW.glob("submissions_*.jsonl")) for line in p.open()]
    print(f"{len(notes)} submissions")

    papers, reviews, others = [], [], []
    kinds = Counter()
    for n in notes:
        c = n["content"]
        replies = n.get("details", {}).get("replies", [])
        decision = None
        n_reviews = 0
        for r in replies:
            kind = reply_kind(r)
            kinds[kind] += 1
            rc = r["content"]
            if kind == "Official_Review":
                n_reviews += 1
                row = {"review_id": r["id"], "forum_id": n["id"], "signature": r["signatures"][0],
                       "cdate": r.get("cdate"), "mdate": r.get("mdate"),
                       "tcdate": r.get("tcdate"), "tmdate": r.get("tmdate")}
                for f in SCORE_FIELDS:
                    row[f"{f}_raw"] = None if val(rc, f) is None else str(val(rc, f))
                    row[f] = leading_int(val(rc, f))
                for k in rc:
                    if k not in SCORE_FIELDS:
                        v = val(rc, k)
                        row[f"text_{k}"] = v if isinstance(v, str) else json.dumps(v)
                reviews.append(row)
            else:
                if kind == "Decision":
                    decision = val(rc, "decision")
                others.append({"reply_id": r["id"], "forum_id": n["id"], "kind": kind,
                               "signature": r["signatures"][0], "cdate": r.get("cdate"),
                               "mdate": r.get("mdate"), "content": json.dumps(rc)})

        venueid = val(c, "venueid")
        if venueid and venueid.endswith("Desk_Rejected_Submission"):
            status = "desk_rejected"
        elif venueid and venueid.endswith("Withdrawn_Submission"):
            status = "withdrawn"
        else:
            status = "decided" if decision else "no_decision"
        papers.append({
            "forum_id": n["id"], "number": n.get("number"),
            "title": val(c, "title"), "abstract": val(c, "abstract"),
            "primary_area": val(c, "primary_area"),
            "keywords": json.dumps(val(c, "keywords")),
            "venueid": venueid, "venue": val(c, "venue"),
            "status": status, "decision": decision, "n_reviews": n_reviews,
            "pdf": val(c, "pdf"), "cdate": n.get("cdate"), "mdate": n.get("mdate"),
            "pdate": n.get("pdate"), "odate": n.get("odate"),
        })

    papers_df = pd.DataFrame(papers).sort_values("forum_id")
    reviews_df = pd.DataFrame(reviews).sort_values(["forum_id", "review_id"])
    others_df = pd.DataFrame(others)
    papers_df.to_parquet(DERIVED / "papers.parquet", index=False)
    reviews_df.to_parquet(DERIVED / "reviews.parquet", index=False)
    others_df.to_parquet(DERIVED / "other_replies.parquet", index=False)

    # Verification printout: observed scales and reply types, not assumptions.
    print("reply kinds:", dict(kinds))
    print("paper status:", papers_df["status"].value_counts().to_dict())
    print("decisions:", papers_df["decision"].value_counts(dropna=False).to_dict())
    for f in SCORE_FIELDS:
        print(f"{f}: {reviews_df[f].value_counts(dropna=False).sort_index().to_dict()}")
        print(f"  raw examples: {reviews_df[f + '_raw'].dropna().unique()[:6].tolist()}")
    print("review text fields:", [k for k in reviews_df.columns if k.startswith("text_")])
    for s in ["withdrawn", "desk_rejected"]:
        sub = papers_df[papers_df.status == s]
        print(f"{s}: {len(sub)} papers, {int((sub.n_reviews > 0).sum())} expose reviews")

    ids_path = FRAME / "all_ids.txt"
    if ids_path.exists():
        print(f"{ids_path} exists; frame not regenerated")
    else:
        body = "".join(f"{i}\n" for i in sorted(papers_df["forum_id"]))
        ids_path.write_text(body)
        digest = hashlib.sha256(body.encode()).hexdigest()
        (FRAME / "all_ids.sha256").write_text(f"{digest}  all_ids.txt\n")
        print(f"frame: {len(papers_df)} ids, sha256 {digest}")


if __name__ == "__main__":
    main()
