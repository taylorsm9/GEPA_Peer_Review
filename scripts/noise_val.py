"""Noise check: review the 40 val papers again with the starting prompt and a different seed (43; the baseline
used 42, which would likely reproduce the same output). The val-score difference between this run and
runs/baseline sets the threshold for extending GEPA. Writes runs/noise_val/<forum_id>.json.

Rerunning skips papers already reviewed, so a rerun after it finishes just prints the threshold:
twice the standard error of the mean per-paper score difference between the two runs (paired, n = 40).
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from reviewer import review as rv

OUT = ROOT / "runs" / "noise_val"
SEED, WORKERS = 43, 16


def run(fid):
    path = OUT / f"{fid}.json"
    if path.exists():
        return
    rec = {"forum_id": fid, "seed": SEED, "settings": rv.SETTINGS, "model": rv.MODEL,
           **rv.review((ROOT / f"data/mineru/text/{fid}.md").read_text(), seed=SEED)}
    path.write_text(json.dumps(rec, indent=1))
    print(fid, "overall", (rec["review"] or {}).get("Overall"), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    val = json.load((ROOT / "data/frame/slices_pdf.json").open())["slices"]["gepa_val"]
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(run, val))

    import numpy as np
    import pandas as pd
    reviews = pd.read_parquet(ROOT / "data/derived/reviews.parquet")
    excluded = set(pd.read_csv(ROOT / "data/derived/review_exclusions.csv").review_id)
    human = reviews[~reviews.review_id.isin(excluded)].groupby("forum_id").rating.mean()

    def score(path, fid):
        r = json.load(path.open())
        o = (r.get("review") or {}).get("Overall")
        return 1 - abs(o - human[fid]) / 10 if not r.get("missing_fields") and isinstance(o, (int, float)) else 0.0
    a = np.array([score(ROOT / f"runs/baseline/{f}.json", f) for f in val])
    b = np.array([score(OUT / f"{f}.json", f) for f in val])
    d = b - a
    print(f"val score: baseline {a.mean():.4f}, seed {SEED} {b.mean():.4f}, difference {d.mean():+.4f}; "
          f"papers whose overall changed: {(d != 0).sum()}/{len(val)}; "
          f"NOISE_THRESHOLD = 2 x SE = {2 * d.std(ddof=1) / np.sqrt(len(d)):.4f}")


if __name__ == "__main__":
    main()
