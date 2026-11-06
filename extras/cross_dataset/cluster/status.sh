#!/usr/bin/env bash
# Short status of the cross-dataset runs. Prints failures first.
cd ~/oraqle-work/cross_dataset
for c in small transformer cao transformer_cao; do docker logs oraqle-xd-$c 2>&1 | grep -E "rc=[^0]" | cut -c1-90; done
for d in runs/*; do [ -d "$d" ] && printf '%s:%s ' "$(basename $d)" "$(ls $d/*.json 2>/dev/null | wc -l)"; done; echo
docker ps --format "{{.Names}}:{{.Status}}" | grep oraqle-xd | tr '\n' ' '; echo
tail -n1 logs/chain.log | cut -c1-60; cut -d' ' -f1 /proc/loadavg
