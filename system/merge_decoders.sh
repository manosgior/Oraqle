#!/bin/sh
# Merge the three decoder sweeps into decoders_summary.{txt,json} exactly as published (§7.7, decoder_choice.pdf).
#
# The published summary = analyze_decoders.py over THREE files, in this order (first file wins for a duplicated
# (d, T, f, pattern) job; the json is written next to the first file):
#   ler_decoders.jsonl         main run, 126 of 136 jobs (stopped once only d=11/13 soft points were missing)
#   ler_decoders_soft13.jsonl  d=13 soft points (600 ns/5% and 400 ns/10%), union-find off
#   ler_decoders_soft11.jsonl  the 2 missing d=11 400 ns/10% patterns, union-find off (--done-from the main run)
# Source: Claude transcript 94019386, line 2151 (2026-10-29T17:59Z), run on steve in oraqle-latsim:dec:
#   python analyze_decoders.py /work/data/ler_decoders.jsonl /work/data/ler_decoders_soft13.jsonl \
#          /work/data/ler_decoders_soft11.jsonl > data/decoders_summary.txt
# helpers/original/watch_decoders.sh passed only the first two files; its output was superseded by this merge.
#
# Usage: system/merge_decoders.sh [DATA_DIR]      (default $ORAQLE_WORK/system, i.e. work/system; writes there)
#        PYTHON=python3 (any python with numpy; no stim needed)
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")
DATA=${1:-${ORAQLE_WORK:-$REPO/work}/system}
for f in ler_decoders.jsonl ler_decoders_soft13.jsonl ler_decoders_soft11.jsonl; do
  [ -f "$DATA/$f" ] || { echo "missing $DATA/$f" >&2; exit 1; }
done
"${PYTHON:-python3}" "$HERE/analyze_decoders.py" "$DATA/ler_decoders.jsonl" "$DATA/ler_decoders_soft13.jsonl" \
  "$DATA/ler_decoders_soft11.jsonl" > "$DATA/decoders_summary.txt"
echo "wrote $DATA/decoders_summary.txt and decoders_summary.json"
