#!/usr/bin/env bash
# Path-plumbed copy of the cluster-only helper ~/oraqle-ics-sims/watch_decoders.sh (steve, 2026-10-29; verbatim in
# helpers/original/watch_decoders.sh). Stops the full decoder run (container oraqle-decoders) once every job except
# the d=13 soft points is in (those run without union-find in oraqle-decoders-13), waits for the d=13 run, then
# merges. CORRECTION: the original merged only ler_decoders.jsonl + ler_decoders_soft13.jsonl. In the paper run the
# main run was stopped by hand at 126 jobs, the 2 missing d=11 soft patterns were run separately
# (ler_decoders_soft11.jsonl), and the published summary merges all three (system/merge_decoders.sh). This copy
# calls merge_decoders.sh and therefore needs ler_decoders_soft11.jsonl (system/run_paper_sweeps.sh, step dec11).
set -eu
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUTDIR=${OUTDIR:-${ORAQLE_WORK:-$REPO/work}/system}
count() {
  python3 -c '
import json, sys
print(sum(1 for l in open(sys.argv[1]) if not (json.loads(l)["d"] == 13 and json.loads(l)["f"] > 0)))' "$OUTDIR/ler_decoders.jsonl"
}
until [ "$(count)" -ge 128 ]; do sleep 120; done
echo "$(date) main run has all 128 jobs below d=13 soft; stopping it"
docker stop oraqle-decoders
while [ -n "$(docker ps -q --filter name='^oraqle-decoders-13$')" ]; do sleep 60; done
echo "$(date) d=13 soft run finished"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$REPO":/repo:ro -v "$OUTDIR":/out -w /repo \
  oraqle-latsim:dec sh system/merge_decoders.sh /out
echo "$(date) done"
