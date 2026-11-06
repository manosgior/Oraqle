#!/usr/bin/env bash
# Re-run the one-off inspection scripts, so their output (only ever printed in the session) is kept.
cd ~/oraqle-work/cross_dataset; S=/w/cross_dataset/scripts; mkdir -p logs/exploration
run() { n=$1; shift; ./xrun.sh 4 $S/$n.py "$@" > logs/exploration/$n.log 2>&1; echo "$n rc=$? $(wc -l < logs/exploration/$n.log) lines"; }
run explore_qick /w/external/qick_14427490 /w/cross_dataset/out/qick_means.npz &
run explore_small2 /w/external &
run explore_cao /w/external/cao_DPMID3 &
run check_cao_labels /w/external/cao_DPMID3 &
run cao_t0 &
run explore_zheng &
run onset &
run onset2 &
run risetime &
wait
