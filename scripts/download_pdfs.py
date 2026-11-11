"""Download the chosen pre-discussion arXiv version of each matched paper in three formats.

Reads data/derived/arxiv_matches.csv (rows with provenance arxiv_pre_discussion), in its
shuffled order, so sliced papers come first. Saves to data/pdf/iclr2026/<forum_id>/:
  arxiv_<id><v>.pdf            the PDF
  arxiv_<id><v>_src.<kind>     the e-print exactly as served: gz (gzipped tar or single .tex),
                               tar, or pdf (no source uploaded). Stored, never unpacked here.
  arxiv_<id><v>.html           arXiv's LaTeXML HTML page only (no images or styles);
                               <name>.unavailable marks papers arXiv has no HTML for
Records path, kind, sha256, size and status per format in data/derived/pdfs.csv.
Resumable: files (and .unavailable markers) on disk are skipped.
Polite: one request at a time, >= 3 s apart. Files are not redistributed (per-paper licenses).
"""
import hashlib
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "data" / "derived"
PDF_DIR = ROOT / "data" / "pdf" / "iclr2026"
HEADERS = {"User-Agent": "GEPA_ICLR research data script (python-requests)"}
MIN_INTERVAL_S = 3.1

SRC_KINDS = [(b"\x1f\x8b", "gz"), (b"%PDF", "pdf")]


class Unavailable(Exception):
    pass


def src_kind(data):
    for magic, kind in SRC_KINDS:
        if data.startswith(magic):
            return kind
    return "tar" if data[257:262] == b"ustar" else "bin"


def check_pdf(data):
    if not data.startswith(b"%PDF"):
        raise ValueError("response is not a PDF")
    return "pdf"


def check_html(data):
    if b"ltx_document" not in data:  # LaTeXML output; arXiv serves a notice page when there is none
        raise Unavailable("no LaTeXML HTML for this version")
    return "html"


# (format, url template, validate-and-name function, file name for a given kind)
FORMATS = [
    ("pdf", "https://arxiv.org/pdf/{v}", check_pdf, lambda stem, kind: f"{stem}.pdf"),
    ("src", "https://arxiv.org/e-print/{v}", src_kind, lambda stem, kind: f"{stem}_src.{kind}"),
    ("html", "https://arxiv.org/html/{v}", check_html, lambda stem, kind: f"{stem}.html"),
]

_last_request = 0.0


def fetch(url):
    global _last_request
    for attempt in range(4):
        time.sleep(max(0, MIN_INTERVAL_S - (time.time() - _last_request)))
        _last_request = time.time()
        r = requests.get(url, headers=HEADERS, timeout=120)
        if r.status_code in (429, 503):
            time.sleep(int(r.headers.get("Retry-After", 30)) * (attempt + 1))
            continue
        if r.status_code == 404:
            raise Unavailable("404")
        r.raise_for_status()
        return r.content
    raise RuntimeError(f"arXiv kept refusing {url}")


def existing(folder, fmt, stem):
    pattern = {"pdf": f"{stem}.pdf", "src": f"{stem}_src.*", "html": f"{stem}.html"}[fmt]
    hits = [f for f in folder.glob(pattern) if f.suffix != ".tmp"] if folder.exists() else []
    return hits[0] if hits else None


def get_format(folder, stem, version_id, fmt, url, validate, filename):
    """Download one format unless already on disk. Returns the row fields for that format."""
    marker = folder / f"{stem}.{fmt}.unavailable"
    path = existing(folder, fmt, stem)
    if path is None and not marker.exists():
        try:
            data = fetch(url.format(v=version_id))
            kind = validate(data)
            path = folder / filename(stem, kind)
            folder.mkdir(parents=True, exist_ok=True)
            path.with_suffix(".tmp").write_bytes(data)
            path.with_suffix(".tmp").rename(path)
        except Unavailable as e:
            folder.mkdir(parents=True, exist_ok=True)
            marker.write_text(str(e) + "\n")
        except Exception as e:  # keep going; the failure is logged per paper and retried next run
            return {f"{fmt}_status": f"failed: {e}"}
    if marker.exists():
        return {f"{fmt}_status": "unavailable"}
    b = path.read_bytes()
    kind = path.name.rsplit("_src.", 1)[-1] if fmt == "src" else fmt
    return {f"{fmt}_status": "ok", f"{fmt}_path": str(path.relative_to(ROOT)), f"{fmt}_kind": kind,
            f"{fmt}_sha256": hashlib.sha256(b).hexdigest(), f"{fmt}_bytes": len(b)}


def main():
    matches = pd.read_csv(DERIVED / "arxiv_matches.csv", dtype={"arxiv_id": str})  # keep trailing zeros
    todo = matches[matches.provenance == "arxiv_pre_discussion"]
    rows = []
    for i, m in enumerate(todo.itertuples()):
        version_id = f"{m.arxiv_id}{m.chosen_version}"
        stem = f"arxiv_{version_id.replace('/', '_')}"
        folder = PDF_DIR / m.forum_id
        row = {"forum_id": m.forum_id, "slice": m.slice, "source": "arxiv_pre_discussion",
               "arxiv_id": m.arxiv_id, "version": m.chosen_version}
        for fmt, url, validate, filename in FORMATS:
            row.update(get_format(folder, stem, version_id, fmt, url, validate, filename))
        rows.append(row)
        if (i + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(DERIVED / "pdfs.csv", index=False)
            df = pd.DataFrame(rows)
            print(f"{i + 1}/{len(todo)}", {f: df[f"{f}_status"].str.split(":").str[0].value_counts().to_dict()
                                            for f, *_ in FORMATS}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(DERIVED / "pdfs.csv", index=False)
    for fmt, *_ in FORMATS:
        col = f"{fmt}_bytes"
        size = df[col].sum() / 1e9 if col in df else 0
        print(fmt, df[f"{fmt}_status"].str.split(":").str[0].value_counts().to_dict(), f"{size:.2f} GB")


if __name__ == "__main__":
    main()
