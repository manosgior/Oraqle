#!/usr/bin/env bash
# Starts each stage when its input is ready. Runs detached on jamie.
cd ~/oraqle-work/cross_dataset; R=/w/cross_dataset/repo/extras/cross_dataset
until [ -f cache/lienhard_q5/meta.json ]; do sleep 20; done
echo "$(date +%T) control ready"; ./qstart.sh small 12 6
until grep -q "done cao" ../external/logs/cao_all.log; do sleep 30; done
echo "$(date +%T) cao downloaded"; ./xrun.sh 16 $R/prepare.py cao > logs/prepare_cao.log 2>&1
echo "$(date +%T) cao prepared"; ./xrun.sh 8 $R/sweep_linear.py cao > logs/linear_cao.log 2>&1
./qstart.sh cao 8 6
echo "$(date +%T) chain done"
