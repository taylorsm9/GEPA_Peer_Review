#!/usr/bin/env bash
# MinerU intake for every downloaded ICLR 2026 PDF, same settings as SPIQA_Deepseek
# (https://github.com/taylorsm9/SPIQA_Deepseek):
#   standard tier, image analysis off  -> data/mineru/zips      (paper text)
#   advanced tier, image analysis on   -> data/mineru_adv/zips  (figure descriptions)
# Order: train+val papers through both tiers first (the baseline needs them), then everything else.
# Each pass only takes papers that don't have a zip yet, so the script can be rerun after a stop.
# Input is data/mineru/in/<forum_id>.pdf (symlinks to the arXiv PDFs in data/derived/pdfs.csv).
# MinerU 4.0.12 runs in its own environment: MINERU_KIT is its mineru-kit binary (default: mineru-kit on
# PATH), and MINERU_HOME, if set, is its model cache.
set -euo pipefail
cd "$(dirname "$0")/.."
KIT=${MINERU_KIT:-mineru-kit}
$KIT version > logs/mineru_version.txt 2>&1 || true

# pass <name> <tier: standard|advanced> <ids: trainval|rest>
pass() {
    local name=$1 tier=$2 which=$3 out in
    out=data/mineru/zips; [ "$tier" = advanced ] && out=data/mineru_adv/zips
    in=data/mineru/pass_$name; rm -rf "$in"; mkdir -p "$in" "$out"
    .venv/bin/python - "$which" "$out" "$in" <<'EOF'
import json, os, sys
which, out, inp = sys.argv[1:]
s = json.load(open("data/frame/slices_pdf.json"))["slices"]
tv = set(s["gepa_train"] + s["gepa_val"])
for f in sorted(os.listdir("data/mineru/in")):
    fid = f[:-4]
    if (fid in tv) == (which == "trainval") and not os.path.exists(f"{out}/{fid}.zip"):
        os.symlink(os.path.realpath(f"data/mineru/in/{f}"), f"{inp}/{f}")
EOF
    local n; n=$(ls "$in" | wc -l)
    echo "$(date +%T) pass $name: $n papers"
    if [ "$n" -gt 0 ]; then
        local args=(--tier "$tier" -f zip); [ "$tier" = standard ] && args+=(--disable-image-analysis)
        $KIT parse "$in" -o "$out" "${args[@]}" > "logs/mineru_$name.log" 2>&1
    fi
    echo "$(date +%T) pass $name done: $(ls "$out" | wc -l) zips in $out"
}

pass std_trainval standard trainval
pass adv_trainval advanced trainval
pass std_rest standard rest
pass adv_rest advanced rest
