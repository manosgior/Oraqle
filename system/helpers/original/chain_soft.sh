#!/bin/sh
cd ~/oraqle-ics-sims
while [ -n "$(docker ps -q --filter name=oraqle-soft-)" ]; do sleep 60; done
docker run -d --name oraqle-softqec --user $(id -u):$(id -g) -e HOME=/tmp --cpus 150 -v ~/oraqle-ics-sims:/work -w /work/sims oraqle-latsim sh -c "python sweep_soft.py --soft /work/data/soft --out /work/data/ler_soft.jsonl --workers 150 > /work/logs/softqec.log 2>&1"
