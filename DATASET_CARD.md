# Dataset card: ICLR 2026 reviews and papers

## Sources

| Source | What | How obtained | Where |
|---|---|---|---|
| OpenReview API v2 | All ICLR 2026 submissions with replies (reviews, meta-reviews, decisions, withdrawals, desk rejections) | `scripts/pull_iclr2026.py`, logged in (guest access is refused), `openreview-py` 2.9.1 | `data/raw/iclr2026/` (raw JSONL, unmodified, `MANIFEST.json`) |
| Pangram (iclr.pangram.com) | Per-paper `fraction_ai`, per-review AI label | Site's JSON files, downloaded in a browser (Cloudflare-gated) | `data/raw/pangram/` (see `PROVENANCE.md` for hashes) |
| arXiv | As-submitted paper versions: PDF, e-print source, LaTeXML HTML | `scripts/arxiv_lookup.py`, `scripts/download_pdfs.py`, one request at a time, ≥ 3.1 s apart | `data/raw/arxiv/` (search and abs-page cache), `data/pdf/iclr2026/<forum_id>/` |

No leaked or deanonymized data from the Nov 2025 security incident is used.

## Frame

- 19,814 submissions, `data/frame/all_ids.txt`, SHA-256 `55a96bf8fb96c6fd8ec4c256fd145d9ce592efa56e16a6c71c2a708b70177482`.
- Status: 13,702 decided (8,816 Reject, 5,134 Poster, 225 Oral), 5,204 withdrawn (5,087 of them with reviews), 908 desk-rejected.
- 75,859 official reviews. Scales: rating {0,2,4,6,8,10}; soundness, presentation and contribution 1–4; confidence 1–5.

## Review freeze

Reviews were released 2025-11-12. After the security incident, scores were reverted in a window from 2025-11-28 15:00 to 2025-11-29 02:30 UTC.

- 69,493 reviews were last modified before the discussion phase, and 5,821 inside the reversion window. Both kinds are kept.
- Excluded: 543 reviews modified between Nov 13 and the reversion window (`discussion_edit_not_reverted`), and 2 modified after it (`post_freeze_edit`).
- Cross-check: Pangram's scores match OpenReview on 99.8% of reviews. All 133 mismatches were modified on Nov 28–29, which is consistent with reversion.
- List: `data/derived/post_freeze_reviews.csv`.

## Filters (`scripts/build_eligible.py`)

A review counts as human only if Pangram labels it fully human and it passes the freeze check.

| Step | Removed | Remaining |
|---|---|---|
| Frame | | 19,814 |
| Desk-rejected | 908 | 18,906 |
| No official reviews | 117 | 18,789 |
| No Pangram `fraction_ai` | 167 | 18,622 |
| `fraction_ai` ≥ 0.5 | 1,512 | 17,110 |
| Fewer than 3 human reviews | 13,111 | **3,999** |

Eligible list: `data/frame/eligible_ids.txt`, SHA-256 `4ea0635de4c1d6aabdd4fc3c01a4800d2f19a1b6dd0c911a01346e0f2c8ef9ec`. Per-item reasons: `data/derived/paper_exclusions.csv`, `review_exclusions.csv`.

## Sampling

`random.Random(42).shuffle(sorted(eligible_ids))`, drawn once (`scripts/draw_slices.py` refuses to redraw). Consecutive blocks: gepa_train 80, gepa_val 40, locked_test 150, then reserves. Order and slices: `data/frame/slices.json`; balance: `data/frame/slice_balance.csv`.

Only 1,598 eligible papers have an as-submitted PDF, so the GEPA slices were re-cut from the same seed-42 order filtered to those papers: train 80, val 40, locked test 150 (`scripts/draw_slices_pdf.py` → `data/frame/slices_pdf.json`, balance in `data/frame/slice_balance_pdf.csv`).

## As-submitted papers

OpenReview revision history is not public, so the as-submitted version comes from arXiv: the **latest version dated before 2025-11-12 00:00 UTC**. A search hit is accepted when title similarity is ≥ 0.9 and at least one author surname is shared.

| Provenance | All 3,999 | First 525 (sliced) |
|---|---|---|
| `arxiv_pre_discussion` | 1,602 | 210 |
| `arxiv_only_after_release` | 602 | 75 |
| `no_arxiv_match` | 1,795 | 240 |

- Downloaded: 1,598 PDFs and 1,598 sources (4 versions return 404 from arXiv); 1,587 HTML pages (arXiv has none for the rest). The source is gzip for 1,597 papers and a PDF for 1 with no source uploaded. Total ~19 GB. Hashes and sizes are in `data/derived/pdfs.csv`.
- Archives are stored as served and never unpacked in place.
- **Known bias:** accepted papers are likelier to be on arXiv, so restricting to papers with a PDF shifts the acceptance balance: 41% of the PDF slices are accepted, against 31% of all eligible papers (mean human rating 4.58 against 4.26). This is reported, not corrected.
- **Reading the CSVs:** load `arxiv_id` as text (`pd.read_csv(..., dtype={"arxiv_id": str})`); parsed as a number, IDs ending in 0 lose the zero (2503.19990 → 2503.1999).

## Paper text

Each arXiv PDF is converted with MinerU 4.0.12 (`scripts/run_mineru.sh`): text from the standard tier, figure descriptions from the advanced tier, joined by `intake/mineru.py`. Papers over 75,000 tokens are dropped from their slice and replaced by the next paper in the shuffled order (2 train, 2 test; `scripts/draw_slices_pdf.py`).

## Licenses and use

- OpenReview review content: CC BY 4.0.
- arXiv papers carry per-paper licenses. Files are used locally for research and **not redistributed**.
- Pangram labels: used as published on their site.

## Environment

Python 3.14 venv at `.venv`, packages in `requirements.txt`. MinerU runs in its own environment.
