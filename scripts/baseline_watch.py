"""Overnight baseline: review each train/val paper as soon as MinerU has converted it (both tiers).

Builds the paper's text (standard-tier text + advanced-tier figure descriptions, intake/mineru.py), saves it
to data/mineru/text/<forum_id>.md, and reviews it -> runs/baseline/<forum_id>.json. Keeps 6 reviews in flight. A request the server rejects (HTTP 400, e.g. the
paper doesn't fit the context) is recorded and not retried; other failures (server down, timeouts) are
retried up to 3 times. Exits when every train/val paper has a record, or when MinerU has stopped and
nothing is left to review. Locked-test papers are never reviewed here.
"""
import json
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from reviewer import review as rv
from intake.mineru import paper_text
from tokenizers import Tokenizer

OUT = ROOT / "runs" / "baseline"
TEXT = ROOT / "data" / "mineru" / "text"
WORKERS, POLL, MAX_TRIES = 6, 120, 3


def running(pattern):
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TEXT.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer.from_file(str(ROOT / "data/tokenizers/deepseek-v3_tokenizer.json"))
    slices = json.load((ROOT / "data/frame/slices_pdf.json").open())["slices"]
    todo = slices["gepa_train"] + slices["gepa_val"]
    tries, inflight = {}, {}

    def run(fid):
        text, meta = paper_text(ROOT / f"data/mineru/zips/{fid}.zip", ROOT / f"data/mineru_adv/zips/{fid}.zip")
        (TEXT / f"{fid}.md").write_text(text)
        rec = {"forum_id": fid, "paper_tokens": len(tok.encode(text).ids), **meta, "settings": rv.SETTINGS,
               "model": rv.MODEL}
        try:
            rec.update(rv.review(text))
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code != 400:
                raise
            rec["error"] = f"HTTP 400: {e.response.text[:500]}"
        (OUT / f"{fid}.json").write_text(json.dumps(rec, indent=1))
        return rec

    with ThreadPoolExecutor(WORKERS) as ex:
        while True:
            pending = [f for f in todo if not (OUT / f"{f}.json").exists() and f not in inflight
                       and tries.get(f, 0) < MAX_TRIES]
            ready = [f for f in pending if (ROOT / f"data/mineru/zips/{f}.zip").exists()
                     and (ROOT / f"data/mineru_adv/zips/{f}.zip").exists()]
            for fid in ready[:WORKERS - len(inflight)]:
                tries[fid] = tries.get(fid, 0) + 1
                inflight[fid] = ex.submit(run, fid)
            if not inflight:
                if not pending or not (running("mineru-kit") or running("run_mineru.sh")):
                    break
                time.sleep(POLL)
                continue
            finished, _ = wait(list(inflight.values()), timeout=POLL, return_when=FIRST_COMPLETED)
            for fid, fut in list(inflight.items()):
                if fut not in finished:
                    continue
                del inflight[fid]
                try:
                    rec = fut.result()
                    calls = rec.get("calls", [])
                    log(fid, "paper", rec["paper_tokens"], f"figs {rec['figures_described']}/{rec['figures']}", "|", " ; ".join(
                        f"out {c['usage'].get('completion_tokens')} answer {c['answer_tokens']} {c['seconds']}s {c['finish_reason']}"
                        for c in calls), "| overall", (rec.get("review") or {}).get("Overall"), rec.get("error", ""))
                except Exception as e:
                    log(fid, f"FAILED try {tries[fid]}/{MAX_TRIES}:", repr(e)[:300])
    done = sum((OUT / f"{f}.json").exists() for f in todo)
    log(f"finished: {done}/{len(todo)} train+val papers have a record")


if __name__ == "__main__":
    main()
