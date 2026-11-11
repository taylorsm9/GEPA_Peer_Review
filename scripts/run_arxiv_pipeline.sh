#!/usr/bin/env bash
# arXiv lookup + PDF download for the eligible pool: sliced papers (first 525 in shuffled order)
# first, then the rest. Every step is resumable, so re-running picks up where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "== lookup 525";  $PY -u scripts/arxiv_lookup.py --limit 525
echo "== pdfs 525";    $PY -u scripts/download_pdfs.py
echo "== lookup all";  $PY -u scripts/arxiv_lookup.py
echo "== pdfs all";    $PY -u scripts/download_pdfs.py
echo "== done"
