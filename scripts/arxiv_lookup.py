"""Find a pre-discussion arXiv version for each sampled paper.

OpenReview does not expose revision history (edits return empty), so the as-submitted PDF
comes from arXiv where possible. For every eligible paper, in the seed-42 shuffled order
(data/frame/slices.json, so the sliced papers come first and the rest are labeled 'pool'):
  1. search the arXiv API by exact title phrase;
  2. accept a hit if title similarity >= 0.9 and at least one author surname is shared;
  3. fetch its version history (the arXiv abs page) and choose the latest version dated before
     reviews were released (2025-11-12), recording its distance from the submission deadline.

Raw responses are cached unmodified in data/raw/arxiv/{search,abs}/ (resumable).
Polite: one request at a time, >= 3 s apart (arXiv API terms), no parallelism.
Writes data/derived/arxiv_matches.csv.
"""
import difflib
import json
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "arxiv"
DERIVED = ROOT / "data" / "derived"

API = "https://export.arxiv.org/api/query"
ABS = "https://arxiv.org/abs/"
HEADERS = {"User-Agent": "GEPA_ICLR research data script (python-requests)"}
MIN_INTERVAL_S = 3.1
DEADLINE = datetime(2025, 9, 25, 11, 59, tzinfo=timezone.utc)
REVIEWS_RELEASED = datetime(2025, 11, 12, tzinfo=timezone.utc)
MIN_TITLE_SIM = 0.9

ATOM = {"a": "http://www.w3.org/2005/Atom"}
VERSION_RE = re.compile(r"\[(v\d+)\].*?(\w{3}, \d{1,2} \w{3} \d{4} \d\d:\d\d:\d\d) UTC", re.S)

_last_request = 0.0


def polite_get(url, params):
    global _last_request
    for attempt in range(4):
        wait = MIN_INTERVAL_S - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.time()
        r = requests.get(url, params=params, headers=HEADERS, timeout=60)
        if r.status_code in (429, 503):
            time.sleep(int(r.headers.get("Retry-After", 30)) * (attempt + 1))
            continue
        r.raise_for_status()
        return r.text
    raise RuntimeError(f"arXiv kept refusing {url} {params}")


def cached(path, url, params):
    if path.exists():
        return path.read_text()
    text = polite_get(url, params)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".tmp").write_text(text)
    path.with_suffix(".tmp").rename(path)
    return text


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"\$[^$]*\$", " ", s)  # inline LaTeX
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def surname(name):
    parts = norm(name).split()
    return parts[-1] if parts else ""


def load_sampled():
    draw = json.load(open(ROOT / "data" / "frame" / "slices.json"))
    slice_of = {pid: name for name, ids in draw["slices"].items() for pid in ids}
    order = {pid: i for i, pid in enumerate(draw["shuffled_order"])}
    rows = []
    for p in sorted((ROOT / "data" / "raw" / "iclr2026").glob("submissions_*.jsonl")):
        for line in p.open():
            n = json.loads(line)
            if n["id"] in order:
                c = n["content"]
                rows.append({"forum_id": n["id"], "slice": slice_of.get(n["id"], "pool"),
                             "title": c["title"]["value"], "authors": c.get("authors", {}).get("value", [])})
    return sorted(rows, key=lambda r: order[r["forum_id"]])


def best_hit(paper, feed):
    root = ET.fromstring(feed)
    want_title = norm(paper["title"])
    want_surnames = {surname(a) for a in paper["authors"]} - {""}
    best = None
    for e in root.findall("a:entry", ATOM):
        title = " ".join(e.findtext("a:title", "", ATOM).split())
        sim = difflib.SequenceMatcher(None, want_title, norm(title)).ratio()
        names = [a.findtext("a:name", "", ATOM) for a in e.findall("a:author", ATOM)]
        overlap = len(want_surnames & {surname(n) for n in names})
        arxiv_id = re.sub(r"v\d+$", "", e.findtext("a:id", "", ATOM).rsplit("/abs/", 1)[-1])
        if sim >= MIN_TITLE_SIM and overlap >= 1 and (best is None or sim > best["title_sim"]):
            best = {"arxiv_id": arxiv_id, "arxiv_title": title, "title_sim": round(sim, 3), "author_overlap": overlap}
    return best


def versions(abs_html):
    history = abs_html.split("Submission history", 1)[-1]
    return [(v, datetime.strptime(d, "%a, %d %b %Y %H:%M:%S").replace(tzinfo=timezone.utc))
            for v, d in VERSION_RE.findall(history)]


def main():
    rows = []
    papers = load_sampled()
    if len(sys.argv) > 2 and sys.argv[1] == "--limit":  # first N in shuffled order
        papers = papers[:int(sys.argv[2])]
    for i, paper in enumerate(papers):
        query = f'ti:"{norm(paper["title"])}"'
        row = {"forum_id": paper["forum_id"], "slice": paper["slice"], "title": paper["title"]}
        try:
            feed = cached(RAW / "search" / f"{paper['forum_id']}.xml", API,
                          {"search_query": query, "max_results": 5})
        except RuntimeError as e:  # not cached, so the next run retries it
            print(e, flush=True)
            rows.append({**row, "provenance": "lookup_failed"})
            continue
        hit = best_hit(paper, feed)
        if hit:
            row.update(hit)
            abs_html = cached(RAW / "abs" / f"{hit['arxiv_id'].replace('/', '_')}.html",
                              ABS + hit["arxiv_id"], None)
            vs = versions(abs_html)
            row["versions"] = json.dumps([(v, d.isoformat()) for v, d in vs])
            pre = [(v, d) for v, d in vs if d < REVIEWS_RELEASED]
            if pre:
                v, d = pre[-1]
                row.update(chosen_version=v, chosen_date=d.isoformat(),
                           days_from_deadline=round((d - DEADLINE).total_seconds() / 86400, 1),
                           provenance="arxiv_pre_discussion")
            else:
                row["provenance"] = "arxiv_only_after_release"
        else:
            row["provenance"] = "no_arxiv_match"
        rows.append(row)
        if (i + 1) % 100 == 0:
            pd.DataFrame(rows).to_csv(DERIVED / "arxiv_matches.csv", index=False)
            print(f"{i + 1}/{len(papers)}", pd.Series([r["provenance"] for r in rows]).value_counts().to_dict(), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(DERIVED / "arxiv_matches.csv", index=False)
    print(df.provenance.value_counts().to_dict())


if __name__ == "__main__":
    main()
