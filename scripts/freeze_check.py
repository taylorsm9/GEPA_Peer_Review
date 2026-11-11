"""Flag reviews whose scores may not be the pre-rebuttal (frozen) state.

ICLR reverted review text and scores to "the beginning of the discussion period"; OpenReview ran
the reversion on 2025-11-28 from 10:00 am to 9:30 pm EST (15:00 Nov 28 to 02:30 Nov 29 UTC)
(https://blog.iclr.cc/?p=1063). Reviews were released on 2025-11-12.

A review's last modification (mdate) is trusted when it falls
  - before 2025-11-13 UTC (release-day processing, before any discussion), or
  - inside the reversion window (restored by the reversion).
Otherwise it is flagged:
  - discussion_edit_not_reverted: modified 2025-11-13 up to the reversion, and not touched by it;
    what changed can't be verified from timestamps, so excluded conservatively
  - post_freeze_edit: modified after the reversion window

Writes data/derived/post_freeze_reviews.csv (review_id, forum_id, mdate_utc, reason).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "data" / "derived"

DISCUSSION_START = pd.Timestamp("2025-11-13 00:00", tz="UTC")
REVERSION_START = pd.Timestamp("2025-11-28 15:00", tz="UTC")
REVERSION_END = pd.Timestamp("2025-11-29 02:30", tz="UTC")


def main():
    r = pd.read_parquet(DERIVED / "reviews.parquet")
    r["mdate_utc"] = pd.to_datetime(r.mdate, unit="ms", utc=True)
    reason = pd.Series(None, index=r.index, dtype="object")
    reason[(r.mdate_utc >= DISCUSSION_START) & (r.mdate_utc < REVERSION_START)] = "discussion_edit_not_reverted"
    reason[r.mdate_utc > REVERSION_END] = "post_freeze_edit"
    r["reason"] = reason

    flagged = r[reason.notna()][["review_id", "forum_id", "mdate_utc", "reason"]]
    flagged.to_csv(DERIVED / "post_freeze_reviews.csv", index=False)
    in_window = ((r.mdate_utc >= REVERSION_START) & (r.mdate_utc <= REVERSION_END)).sum()
    print(f"reviews: {len(r)} | before discussion: {(r.mdate_utc < DISCUSSION_START).sum()} "
          f"| reverted in window: {in_window}")
    print("flagged:", flagged.reason.value_counts().to_dict())


if __name__ == "__main__":
    main()
