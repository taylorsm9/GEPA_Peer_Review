"""GEPA run: evolve the reviewer's system prompt and review form so its overall score matches the human mean.

Setup:
- seed: reviewer/seed_candidate.json; minibatch 6; val 40; up to 16 reviews at once (the server's limit);
- score per paper: 1 - |pred overall - human mean overall| / 10, and 0 for a malformed review;
- reflection: DeepSeek, GEPA's default reflection prompt plus the AAAI background section; feedback has
  every dimension's predicted vs. human scores, the human reviewers' weaknesses, the final review and the
  self-critique THOUGHT;
- 1,000-review chunks on one run folder; the next chunk starts at once if the best val score rose by more
  than NOISE_THRESHOLD during the chunk.

Reviews are cached on disk per (candidate, paper), and the seed's cache is filled from runs/baseline, so
the seed is never re-reviewed. A candidate is reviewed with seed 42, like the baseline.

    .venv/bin/python scripts/run_gepa.py          # start, or resume the run in runs/gepa/main
    .venv/bin/python scripts/run_gepa.py --check  # build everything and render one reflection prompt; no GEPA, no model calls
    .venv/bin/python scripts/run_gepa.py --smoke  # one real GEPA proposal in runs/gepa/smoke, to catch integration bugs
    touch runs/gepa/main/gepa.stop                # stop cleanly after the current step
"""
import argparse
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.strategies.instruction_proposal import InstructionProposalSignature
from reviewer import review as rv

RUN_DIR = ROOT / "runs" / "gepa" / "main"
CACHE = ROOT / "runs" / "gepa" / "review_cache"
SEED_FILE = ROOT / "reviewer" / "seed_candidate.json"
WORKERS, MINIBATCH, CHUNK = 16, 6, 1000  # the server runs at most 16 requests at once
NOISE_THRESHOLD = 0.0341  # 2 x SE of the paired val-score difference, seed 43 vs baseline (scripts/noise_val.py, 2026-10-31)
WEAKNESS_CHARS = 1500  # per human review, so a minibatch's feedback fits one reflection call
THOUGHT_CHARS = 3000
DIMS = [("Overall", "rating"), ("Soundness", "soundness"), ("Presentation", "presentation"),
        ("Contribution", "contribution"), ("Confidence", "confidence")]

BACKGROUND = """Background. The assistant is an LLM paper reviewer; the goal is for its overall score to match the mean score given by the paper's human ICLR reviewers. A related system, the AAAI-26 AI Review Pilot (an LLM reviewer used at conference scale; arXiv 2604.13940), was surveyed by authors and program committee members. They reported that its reviews, compared with human reviews:
- often failed to judge novelty, significance of contributions, and overall impact;
- overemphasized minor issues and had difficulty prioritizing the significance of issues;
- made factual errors and misread some equations and tables;
- showed shallow understanding of particular research domains.
These are reports about a different reviewer and may or may not apply to this one. Use the examples and feedback above to judge whether they do.

"""
_DEFAULT = InstructionProposalSignature.default_prompt_template
REFLECTION_TEMPLATE = _DEFAULT.replace("Your task is to write a new instruction", BACKGROUND + "Your task is to write a new instruction", 1)
assert REFLECTION_TEMPLATE != _DEFAULT


def cand_hash(candidate):
    return hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest()[:16]


def load_data():
    """Train and val instances: paper id, title, abstract, human scores, human weaknesses."""
    slices = json.load((ROOT / "data/frame/slices_pdf.json").open())["slices"]
    papers = pd.read_parquet(ROOT / "data/derived/papers.parquet").set_index("forum_id")
    reviews = pd.read_parquet(ROOT / "data/derived/reviews.parquet")
    excluded = set(pd.read_csv(ROOT / "data/derived/review_exclusions.csv").review_id)
    human = reviews[~reviews.review_id.isin(excluded)]

    def inst(fid):
        h = human[human.forum_id == fid]
        return {"forum_id": fid, "title": papers.title[fid], "abstract": papers.abstract[fid],
                "human": {name: [int(v) for v in h[col].dropna()] for name, col in DIMS},
                "weaknesses": [w.strip()[:WEAKNESS_CHARS] for w in h.text_weaknesses.dropna() if w.strip()]}
    return [inst(f) for f in slices["gepa_train"]], [inst(f) for f in slices["gepa_val"]]


def thought_of(content):
    m = re.search(r"THOUGHT:\s*(.*?)\s*REVIEW JSON:", content or "", re.DOTALL)
    return (m.group(1) if m else "").strip()[:THOUGHT_CHARS]


class ReviewerAdapter:
    propose_new_texts = None  # GEPA reads this attribute directly; None means its default proposer

    def __init__(self, seed_candidate, log_path):
        self.seed_hash = cand_hash(seed_candidate)
        self.log_path, self.lock = log_path, threading.Lock()

    def _review(self, inst, candidate, h):
        """One review, from the cache if possible. Returns (record, newly_reviewed)."""
        path = CACHE / h / f"{inst['forum_id']}.json"
        if path.exists():
            return json.load(path.open()), False
        baseline = ROOT / f"runs/baseline/{inst['forum_id']}.json"
        if h == self.seed_hash and baseline.exists():
            rec, new = json.load(baseline.open()), False
        else:
            text = (ROOT / f"data/mineru/text/{inst['forum_id']}.md").read_text()
            for attempt in range(6):  # server hiccups: wait and retry, then give up on the whole run
                try:
                    rec = rv.review(text, system_prompt=candidate["system_prompt"],
                                    review_form=candidate["review_form"])
                    break
                except requests.RequestException:
                    if attempt == 5:
                        raise
                    time.sleep(120)
            new = True
        path.parent.mkdir(parents=True, exist_ok=True)
        # write then rename, so a thread reviewing the same paper never reads a half-written file
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(path)
        return rec, new

    def evaluate(self, batch, candidate, capture_traces=False):
        h = cand_hash(candidate)
        with ThreadPoolExecutor(WORKERS) as ex:
            results = list(ex.map(lambda i: self._review(i, candidate, h), batch))
        outputs, scores, trajectories = [], [], []
        for inst, (rec, new) in zip(batch, results):
            final = rec.get("review") or {}
            pred = {name: final.get(name) for name, _ in DIMS}
            human = {name: (float(np.mean(v)) if v else None) for name, v in inst["human"].items()}
            ok = not rec.get("missing_fields") and isinstance(pred["Overall"], (int, float))
            score = 1 - abs(pred["Overall"] - human["Overall"]) / 10 if ok else 0.0
            calls = rec.get("calls", [])
            out = {"forum_id": inst["forum_id"], "candidate": h, "pred": pred, "human": human, "score": score,
                   "valid": ok, "new_review": new, "seconds": sum(c.get("seconds", 0) for c in calls)}
            outputs.append(out)
            scores.append(score)
            trajectories.append({"inst": inst, "review": final, "valid": ok, "missing": rec.get("missing_fields"),
                                 "thought": thought_of(calls[1]["content"]) if len(calls) > 1 else "",
                                 "pred": pred, "human": human})
        with self.lock, self.log_path.open("a") as f:
            for out in outputs:
                f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **out}) + "\n")
        return EvaluationBatch(outputs=outputs, scores=scores,
                               trajectories=trajectories if capture_traces else None,
                               num_metric_calls=sum(new for _, new in results))

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        items = []
        for t in eval_batch.trajectories:
            inst, human = t["inst"], t["human"]
            if t["valid"]:
                lines = [f"Predicted overall {t['pred']['Overall']}; human mean {human['Overall']:.2f} "
                         f"(individual ratings {inst['human']['Overall']}); "
                         f"error {t['pred']['Overall'] - human['Overall']:+.2f}."]
                lines += [f"{name}: predicted {t['pred'][name]}, human mean "
                          f"{human[name]:.2f}" if human[name] is not None else f"{name}: predicted {t['pred'][name]}"
                          for name, _ in DIMS[1:]]
            else:
                lines = [f"The review was malformed (missing or invalid fields: {t['missing']}), so it scored 0."]
            lines.append("Weaknesses written by the human reviewers:")
            lines += [f"[Reviewer {i}] {w}" for i, w in enumerate(inst["weaknesses"], 1)]
            items.append({
                "Inputs": f"Paper title: {inst['title']}\nAbstract: {inst['abstract']}\n"
                          "(The assistant was given the full paper text; it is omitted here for length.)",
                "Generated Outputs": f"Final review: {json.dumps(t['review'], ensure_ascii=False)}\n\n"
                                     f"The assistant's self-critique of its first review: {t['thought']}",
                "Feedback": "\n".join(lines),
            })
        return {c: items for c in components_to_update}


class ReflectionLM:
    """DeepSeek with the reviewer's settings; a new seed per call so retried prompts don't repeat."""
    def __init__(self):
        self.n = 0

    def __call__(self, prompt):
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        self.n += 1
        return rv.chat(messages, 1000 + self.n)["content"]


class ValMetrics:
    """After every val evaluation: correlation with the human mean and per-dimension error, to MLflow and a file."""
    def __init__(self, valset, path):
        self.valset, self.path = valset, path

    def on_valset_evaluated(self, event):
        outs = [o for o in (event.get("outputs_by_val_id") or {}).values() if o and o["valid"]]
        if len(outs) < 3:
            return
        pred = np.array([o["pred"]["Overall"] for o in outs], float)
        hum = np.array([o["human"]["Overall"] for o in outs], float)
        m = {"val_pearson": float(np.corrcoef(pred, hum)[0, 1]),
             "val_spearman": float(pd.Series(pred).corr(pd.Series(hum), method="spearman")),
             "val_mean_signed_error": float((pred - hum).mean()), "val_valid": len(outs)}
        for name, _ in DIMS:
            pairs = [(o["pred"][name], o["human"][name]) for o in outs
                     if isinstance(o["pred"][name], (int, float)) and o["human"][name] is not None]
            if pairs:
                m[f"val_mae_{name.lower()}"] = float(np.mean([abs(a - b) for a, b in pairs]))
        with self.path.open("a") as f:
            f.write(json.dumps({"iteration": event["iteration"], "candidate_idx": event["candidate_idx"],
                                "average_score": event["average_score"], **m}) + "\n")
        try:
            import mlflow
            mlflow.log_metrics(m, step=event["iteration"])
        except Exception:
            pass


def best_val(run_dir, upto):
    """Best val score among candidates found within the first `upto` reviews, so a resumed chunk is
    judged from the chunk's true start, not from wherever it crashed."""
    state = run_dir / "gepa_state.bin"
    if not state.exists():
        return None
    from gepa.core.state import GEPAState
    s = GEPAState.load(str(run_dir))
    return max(v for v, n in zip(s.program_full_scores_val_set, s.num_metric_calls_by_discovery) if n <= upto)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global RUN_DIR
    if args.smoke:
        RUN_DIR = ROOT / "runs" / "gepa" / "smoke"
    seed = json.load(SEED_FILE.open())["components"]
    train, val = load_data()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    adapter = ReviewerAdapter(seed, RUN_DIR / "reviews.jsonl")

    if args.check:
        batch = adapter.evaluate(train[:MINIBATCH], seed, capture_traces=True)  # all from the baseline cache
        assert batch.num_metric_calls == 0, "check must not call the model"
        ds = adapter.make_reflective_dataset(seed, batch, ["system_prompt"])["system_prompt"]
        side = "\n\n".join(f"# Example {i}\n" + "\n".join(f"## {k}\n{v}" for k, v in d.items())
                           for i, d in enumerate(ds, 1))
        prompt = REFLECTION_TEMPLATE.replace("<curr_param>", seed["review_form"]).replace("<side_info>", side)
        (RUN_DIR / "check_reflection_prompt.md").write_text(prompt)
        tokens = len(rv._TOK.encode(prompt).ids)
        print(f"train {len(train)}, val {len(val)}; minibatch scores {[round(s, 3) for s in batch.scores]}")
        print(f"reflection prompt for the review form: {tokens} tokens -> {RUN_DIR / 'check_reflection_prompt.md'}")
        return

    common = dict(
        seed_candidate=seed, trainset=train, valset=val, adapter=adapter,
        reflection_lm=ReflectionLM(), reflection_minibatch_size=MINIBATCH,
        reflection_prompt_template=REFLECTION_TEMPLATE, candidate_selection_strategy="pareto",
        module_selector="round_robin", run_dir=str(RUN_DIR), write_agent_state=True,
        callbacks=[ValMetrics(val, RUN_DIR / "val_metrics.jsonl")],
        use_mlflow=True, mlflow_tracking_uri=f"sqlite:///{RUN_DIR.parent / ('mlflow_smoke.db' if args.smoke else 'mlflow.db')}",
        mlflow_experiment_name="gepa-iclr-smoke" if args.smoke else "gepa-iclr", seed=42, display_progress_bar=True)

    if args.smoke:
        from gepa.utils.stop_condition import MaxCandidateProposalsStopper
        result = gepa.optimize(**common, stop_callbacks=[MaxCandidateProposalsStopper(1)])
        print(f"smoke done: {len(result.candidates)} candidates, val scores {result.val_aggregate_scores}", flush=True)
        return

    assert NOISE_THRESHOLD is not None, "set NOISE_THRESHOLD from the val noise check first"
    chunk = 1
    while True:
        used = 0
        if (RUN_DIR / "gepa_state.bin").exists():
            from gepa.core.state import GEPAState
            used = GEPAState.load(str(RUN_DIR)).total_num_evals
        chunk = max(chunk, used // CHUNK + 1)
        before = best_val(RUN_DIR, CHUNK * (chunk - 1))
        print(f"chunk {chunk}: budget {CHUNK * chunk} reviews (used {used}), best val before {before}", flush=True)
        result = gepa.optimize(**common, max_metric_calls=CHUNK * chunk)
        after = max(result.val_aggregate_scores)
        gained = after - (before if before is not None else result.val_aggregate_scores[0])
        print(f"chunk {chunk} done: best val {after:.4f}, gain {gained:+.4f} (threshold {NOISE_THRESHOLD})", flush=True)
        if (RUN_DIR / "gepa.stop").exists() or gained <= NOISE_THRESHOLD:
            break
        chunk += 1
    print("best candidate:", RUN_DIR / "pareto", flush=True)


if __name__ == "__main__":
    main()
