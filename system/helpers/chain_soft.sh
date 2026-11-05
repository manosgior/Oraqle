#!/bin/sh
# Path-plumbed copy of the cluster-only helper ~/oraqle-ics-sims/chain_soft.sh (steve, 2026-10-26; verbatim in
# helpers/original/chain_soft.sh). Waits until the five soft-CNN containers (oraqle-soft-*, soft_readout_cnn.py)
# have finished, then starts the soft-readout QEC sweep (sweep_soft.py) in oraqle-latsim.
# Original mounts: -v ~/oraqle-ics-sims:/work, soft outputs in /work/data/soft, results in /work/data.
# Here: OUTDIR (default $ORAQLE_WORK/system, i.e. work/system) plays the role of the old data/ directory.
set -eu
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUTDIR=${OUTDIR:-${ORAQLE_WORK:-$REPO/work}/system}
mkdir -p "$OUTDIR"
while [ -n "$(docker ps -q --filter name=oraqle-soft-)" ]; do sleep 60; done
docker run -d --name oraqle-softqec --user "$(id -u):$(id -g)" -e HOME=/tmp --cpus 150 \
  -v "$REPO":/repo:ro -v "$OUTDIR":/out -w /repo oraqle-latsim \
  sh -c "python system/sweep_soft.py --soft /out/soft --out /out/ler_soft.jsonl --workers 150 > /out/softqec.log 2>&1"
