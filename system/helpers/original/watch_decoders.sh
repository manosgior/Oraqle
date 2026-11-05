#!/usr/bin/env bash
# Stop the full decoder run once every job except the d=13 soft points is in (those run without union-find in
# oraqle-decoders-13), then analyze both files together.
cd ~/oraqle-ics-sims
count() {
  python3 -c '
import json, sys
print(sum(1 for l in open(sys.argv[1]) if not (json.loads(l)["d"] == 13 and json.loads(l)["f"] > 0)))' data/ler_decoders.jsonl
}
until [ "$(count)" -ge 128 ]; do sleep 120; done
echo "$(date) main run has all 128 jobs below d=13 soft; stopping it"
docker stop oraqle-decoders
while [ -n "$(docker ps -q --filter name='^oraqle-decoders-13$')" ]; do sleep 60; done
echo "$(date) d=13 soft run finished"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v ~/oraqle-ics-sims:/work -w /work/sims oraqle-latsim:dec \
  python analyze_decoders.py /work/data/ler_decoders.jsonl /work/data/ler_decoders_soft13.jsonl \
  > data/decoders_summary.txt 2>&1
echo "$(date) done"
