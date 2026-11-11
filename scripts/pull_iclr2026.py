"""Pull all ICLR 2026 submissions with their replies from OpenReview API v2.

Writes raw API responses, unmodified, one JSONL file per page, to data/raw/iclr2026/.
Resumable: pages already on disk are skipped. Throttled to stay well under rate limits.

Credentials come from OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD in the project .env
(the API requires an account; anonymous note queries are challenge-gated).
"""
import importlib.metadata
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import openreview
from dotenv import load_dotenv

BASEURL = "https://api2.openreview.net"
VENUE = "ICLR.cc/2026/Conference"
INVITATION = f"{VENUE}/-/Submission"
PAGE_SIZE = 250
SLEEP_S = 2.0

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "iclr2026"


def fetch_page(client, offset):
    # Raw request (not client.get_notes) so the response is saved exactly as served.
    params = {"invitation": INVITATION, "details": "replies", "offset": offset,
              "limit": PAGE_SIZE, "sort": "number:asc"}
    for attempt in range(5):
        r = client.session.get(client.notes_url, params=params, headers=client.headers)
        if r.status_code == 429:
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()["notes"]
    raise RuntimeError(f"rate-limited repeatedly at offset {offset}")


def main():
    load_dotenv(ROOT / ".env")  # supplies OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD
    if not (os.environ.get("OPENREVIEW_USERNAME") and os.environ.get("OPENREVIEW_PASSWORD")):
        raise SystemExit("OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD missing from .env. "
                         "Anonymous note queries are challenge-gated.")
    OUT.mkdir(parents=True, exist_ok=True)
    client = openreview.api.OpenReviewClient(baseurl=BASEURL)

    offset = 0
    while True:
        path = OUT / f"submissions_{offset:06d}.jsonl"
        if path.exists():
            n = sum(1 for _ in path.open())
        else:
            notes = fetch_page(client, offset)
            tmp = path.with_suffix(".tmp")
            with tmp.open("w") as f:
                for note in notes:
                    f.write(json.dumps(note) + "\n")
            tmp.rename(path)
            n = len(notes)
            print(f"offset {offset}: {n} notes", flush=True)
            time.sleep(SLEEP_S)
        if n < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    manifest = {
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseurl": BASEURL,
        "invitation": INVITATION,
        "details": "replies",
        "page_size": PAGE_SIZE,
        "openreview_py": importlib.metadata.version("openreview-py"),
        "n_submissions": sum(1 for p in OUT.glob("submissions_*.jsonl") for _ in p.open()),
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest)


if __name__ == "__main__":
    main()
