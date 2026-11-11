"""Locked test: review the 150 locked-test papers once with the seed prompt and with GEPA's best candidate.

The best candidate is the one with the highest val aggregate score in runs/gepa/main (GEPA's own choice,
candidate 28 at 0.9095). Both prompts are frozen to runs/locked_test/freeze.json, with their hashes, before
any test paper is reviewed; a rerun refuses to go on if the frozen prompts no longer match.

Each paper's text is built the same way as for train/val (standard-tier text + advanced-tier figure
descriptions, intake/mineru.py) and saved to data/mineru/text/<forum_id>.md. Reviews use seed 42, like the
baseline and the GEPA run, up to 16 at once (the server's limit), paper by paper so partial results stay
paired. Writes runs/locked_test/<seed|best>/<forum_id>.json. Rerunning skips finished reviews, so after a
stop it resumes, and once everything is reviewed it only prints the report (also saved to report.json).

    .venv/bin/python -u scripts/locked_test.py 2>&1 | tee -a logs/locked_test.log
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from intake.mineru import paper_text
from reviewer import review as rv

OUT = ROOT / "runs" / "locked_test"
TEXT = ROOT / "data" / "mineru" / "text"
GEPA_DIR = ROOT / "runs" / "gepa" / "main"
SEED, WORKERS, BOOT = 42, 16, 10000
DIMS = [("Overall", "rating"), ("Soundness", "soundness"), ("Presentation", "presentation"),
        ("Contribution", "contribution"), ("Confidence", "confidence")]


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def freeze():
    """The two prompts under test, written once; later runs must find the same ones."""
    from gepa.core.state import GEPAState
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_gepa import cand_hash
    state = GEPAState.load(str(GEPA_DIR))
    scores = state.program_full_scores_val_set
    best = int(np.argmax(scores))
    candidates = json.load((GEPA_DIR / "candidates.json").open())
    seed = json.load((ROOT / "reviewer/seed_candidate.json").open())["components"]
    assert candidates[0] == seed
    frozen = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "gepa_reviews_used": state.total_num_evals,
              "prompts": {"seed": {"candidate_idx": 0, "val_score": scores[0], "hash": cand_hash(seed),
                                   "components": seed},
                          "best": {"candidate_idx": best, "val_score": scores[best],
                                   "hash": cand_hash(candidates[best]), "components": candidates[best]}}}
    path = OUT / "freeze.json"
    if path.exists():
        old = json.load(path.open())
        assert all(old["prompts"][k]["hash"] == frozen["prompts"][k]["hash"] for k in ("seed", "best")), \
            "the GEPA run changed after the test prompts were frozen"
        return old["prompts"]
    path.write_text(json.dumps(frozen, indent=1))
    return frozen["prompts"]


def run(fid, name, components):
    path = OUT / name / f"{fid}.json"
    if path.exists():
        return
    text_path = TEXT / f"{fid}.md"
    if not text_path.exists():
        text, _ = paper_text(ROOT / f"data/mineru/zips/{fid}.zip", ROOT / f"data/mineru_adv/zips/{fid}.zip")
        text_path.write_text(text)
    text = text_path.read_text()
    for attempt in range(6):  # server hiccups: wait and retry, then leave the paper for a rerun
        try:
            rec = rv.review(text, system_prompt=components["system_prompt"],
                            review_form=components["review_form"], seed=SEED)
            break
        except requests.RequestException as e:
            if attempt == 5:
                log(fid, name, "FAILED:", repr(e)[:300])
                return
            time.sleep(120)
    rec = {"forum_id": fid, "prompt": name, "seed": SEED, "settings": rv.SETTINGS, "model": rv.MODEL, **rec}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(path)
    log(fid, name, "overall", (rec["review"] or {}).get("Overall"),
        f"{sum(c['seconds'] for c in rec['calls']):.0f}s")


def report(test):
    reviews = pd.read_parquet(ROOT / "data/derived/reviews.parquet")
    excluded = set(pd.read_csv(ROOT / "data/derived/review_exclusions.csv").review_id)
    human = reviews[~reviews.review_id.isin(excluded)].groupby("forum_id")[[c for _, c in DIMS]].mean()
    done = [f for f in test if all((OUT / n / f"{f}.json").exists() for n in ("seed", "best"))]
    res = {"papers_reviewed_by_both": len(done), "papers_in_test": len(test)}
    per = {}
    for name in ("seed", "best"):
        rows = []
        for f in done:
            r = json.load((OUT / name / f"{f}.json").open())
            final = r.get("review") or {}
            ok = not r.get("missing_fields") and isinstance(final.get("Overall"), (int, float))
            rows.append({"forum_id": f, "valid": ok,
                         "score": 1 - abs(final["Overall"] - human.rating[f]) / 10 if ok else 0.0,
                         **{d: (final.get(d) if ok else None) for d, _ in DIMS}})
        per[name] = pd.DataFrame(rows).set_index("forum_id")
        df = per[name]
        v = df[df.valid]
        m = {"score": df.score.mean(), "valid": int(df.valid.sum()),
             "pearson": float(np.corrcoef(v.Overall.astype(float), human.rating[v.index])[0, 1]),
             "spearman": float(v.Overall.astype(float).corr(human.rating[v.index], method="spearman")),
             "mean_signed_error": float((v.Overall - human.rating[v.index]).mean())}
        for d, col in DIMS:
            p = pd.to_numeric(v[d], errors="coerce")
            m[f"mae_{d.lower()}"] = float((p - human[col][v.index]).abs().mean())
        res[name] = m
    diff = (per["best"].score - per["seed"].score).to_numpy()
    rng = np.random.default_rng(SEED)
    boot = rng.choice(diff, (BOOT, len(diff))).mean(1)
    from scipy import stats
    res["paired"] = {"score_difference": float(diff.mean()),
                     "bootstrap_95ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                     "paired_t_p": float(stats.ttest_rel(per["best"].score, per["seed"].score).pvalue),
                     "wilcoxon_p": float(stats.wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0,
                     "best_better_same_worse": [int((diff > 0).sum()), int((diff == 0).sum()), int((diff < 0).sum())]}
    (OUT / "report.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    for name in ("seed", "best"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    prompts = freeze()
    test = json.load((ROOT / "data/frame/slices_pdf.json").open())["slices"]["locked_test"]
    log(f"locked test: {len(test)} papers; seed = candidate 0 ({prompts['seed']['hash']}), "
        f"best = candidate {prompts['best']['candidate_idx']} ({prompts['best']['hash']}, val {prompts['best']['val_score']:.4f})")
    jobs = [(f, n) for f in test for n in ("seed", "best") if not (OUT / n / f"{f}.json").exists()]
    log(f"{len(jobs)} reviews to do")
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(lambda j: run(j[0], j[1], prompts[j[1]]["components"]), jobs))
    report(test)


if __name__ == "__main__":
    main()
