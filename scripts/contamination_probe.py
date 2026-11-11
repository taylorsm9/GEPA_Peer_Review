"""Memorization probe: does the reviewer model already know ICLR 2026 papers?

Gives DeepSeek only a paper's title and asks for its abstract and ICLR decision. 30 papers are drawn
(seed 42) from the GEPA slices in data/frame/slices_pdf.json; 3 famous older ICLR papers are positive
controls, to show the probe can detect a paper the model clearly knows.

Scoring, per paper:
  - sim:        word-level similarity (difflib ratio) of the model's abstract to the true abstract
  - sim_null:   mean similarity of the model's abstract to the *other* probe papers' true abstracts
  - longest_run: longest run of consecutive words shared with the true abstract
  - decision accuracy, against the base rate of always guessing the majority outcome

Writes data/derived/contamination/probe.jsonl (raw answers) and prints a summary.
"""
import difflib
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "derived" / "contamination"
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/v1")
MODEL = "deepseek-v4-flash-0731"
N = 30
SEED = 42

CONTROLS = [  # (title, year, abstract opening used as reference)
    ("Adam: A Method for Stochastic Optimization", 2015,
     "We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective "
     "functions, based on adaptive estimates of lower-order moments. The method is straightforward to "
     "implement, is computationally efficient, has little memory requirements, is invariant to diagonal "
     "rescaling of the gradients, and is well suited for problems that are large in terms of data and/or "
     "parameters."),
    ("Very Deep Convolutional Networks for Large-Scale Image Recognition", 2015,
     "In this work we investigate the effect of the convolutional network depth on its accuracy in the "
     "large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of "
     "increasing depth using an architecture with very small (3x3) convolution filters, which shows that a "
     "significant improvement on the prior-art configurations can be achieved by pushing the depth to 16-19 "
     "weight layers."),
    ("An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", 2021,
     "While the Transformer architecture has become the de-facto standard for natural language processing "
     "tasks, its applications to computer vision remain limited. In vision, attention is either applied in "
     "conjunction with convnets, or used to replace certain components of convolutional networks while "
     "keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a "
     "pure transformer applied directly to sequences of image patches can perform very well on image "
     "classification tasks."),
]

PROMPT = """Here is the title of a paper submitted to ICLR {year}:

"{title}"

Do you know this specific paper? Answer in JSON with these keys:
- "known": true if you recognize this exact paper, false otherwise
- "abstract": the paper's abstract, reproduced as closely as you can. If you don't know the paper, give your best guess anyway.
- "decision": its ICLR {year} outcome, one of "Accept (Oral)", "Accept (Spotlight)", "Accept (Poster)", "Reject", "Withdrawn". Give your best guess if unsure.

Return only the JSON object."""


def ask(title, year):
    r = requests.post(f"{BASE_URL}/chat/completions", timeout=3600, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(title=title, year=year)}],
        "temperature": 1.0, "top_p": 0.95, "seed": SEED, "max_tokens": 32768,
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"},  # "max" = real "high" on 0731
    })
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"] or ""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {"raw": text}
    except json.JSONDecodeError:
        return {"raw": text}


def words(s):
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def sim(a, b):
    return difflib.SequenceMatcher(None, words(a), words(b), autojunk=False).ratio()


def longest_run(a, b):
    m = difflib.SequenceMatcher(None, words(a), words(b), autojunk=False)
    return m.find_longest_match().size


def outcome(decision, status):
    if status == "withdrawn":
        return "Withdrawn"
    return "Accept" if str(decision).startswith("Accept") else "Reject"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    slices = json.load((ROOT / "data/frame/slices_pdf.json").open())["slices"]
    ids = sorted(i for s in slices.values() for i in s)
    picked = random.Random(SEED).sample(ids, N)
    papers = pd.read_parquet(ROOT / "data/derived/papers.parquet").set_index("forum_id")

    items = [{"forum_id": i, "title": papers.at[i, "title"], "year": 2026, "control": False,
              "true_abstract": papers.at[i, "abstract"],
              "true_outcome": outcome(papers.at[i, "decision"], papers.at[i, "status"])} for i in picked]
    items += [{"forum_id": None, "title": t, "year": y, "control": True, "true_abstract": a,
               "true_outcome": "Accept"} for t, y, a in CONTROLS]

    with ThreadPoolExecutor(6) as ex:
        answers = list(ex.map(lambda it: ask(it["title"], it["year"]), items))

    probe = [it for it in items if not it["control"]]
    for it, ans in zip(items, answers):
        gen = ans.get("abstract", "")
        it["answer"] = ans
        it["sim"] = round(sim(gen, it["true_abstract"]), 3)
        it["longest_run"] = longest_run(gen, it["true_abstract"])
        others = [p["true_abstract"] for p in probe if p is not it]
        it["sim_null"] = round(sum(sim(gen, o) for o in others) / len(others), 3)
        pred = str(ans.get("decision", ""))
        it["pred_outcome"] = "Withdrawn" if pred == "Withdrawn" else ("Accept" if pred.startswith("Accept") else "Reject")
        it["decision_correct"] = it["pred_outcome"] == it["true_outcome"]
    with (OUT / "probe.jsonl").open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    df = pd.DataFrame(items)
    cols = ["control", "title", "sim", "sim_null", "longest_run", "true_outcome", "pred_outcome"]
    print(df[cols].assign(title=df.title.str[:50]).to_string(index=False))
    p = df[~df.control]
    print(f"\nICLR 2026 probe (n={len(p)}): claims known {p.answer.map(lambda a: a.get('known') is True).mean():.2f}, "
          f"sim {p.sim.mean():.3f} vs null {p.sim_null.mean():.3f}, max longest_run {p.longest_run.max()}, "
          f"decision acc {p.decision_correct.mean():.2f} (majority base rate {p.true_outcome.value_counts(normalize=True).max():.2f})")
    c = df[df.control]
    print(f"Controls (n={len(c)}): sim {c.sim.mean():.3f}, longest_run {c.longest_run.tolist()}")


if __name__ == "__main__":
    main()
