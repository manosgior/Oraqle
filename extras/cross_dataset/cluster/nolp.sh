#!/usr/bin/env bash
# Unfiltered linear baselines and sweeps; keeps the earlier filtered files.
cd ~/oraqle-work/cross_dataset; R=/w/cross_dataset/repo/extras/cross_dataset
for n in qick cao; do [ -f linear_$n.json ] && mv -n linear_$n.json linear_lp_$n.json; done
./xrun.sh 8 $R/sweep_linear.py qick cao > logs/linear_nolp.log 2>&1
./xrun.sh 8 $R/check_linear_filter.py > logs/check_filter.log 2>&1
for t in 200 400 600 800 1000 1250 1500 1800 2000 2400; do OMP=4 ./xrun.sh 4 $R/run_one.py --dataset qick --model linear --T $t --tag nolp > logs/lin_q_$t.log 2>&1 & done
for t in 200 400 600 800 1000 1200 1360; do OMP=4 ./xrun.sh 4 $R/run_one.py --dataset cao --model linear --T $t --tag nolp > logs/lin_c_$t.log 2>&1 & done
wait
./xrun.sh 8 $R/analyze.py > logs/analyze.log 2>&1
tail -n 12 logs/check_filter.log | cut -c1-110
grep -A9 "== qick" logs/analyze.log | head -3 | cut -c1-110; grep -A3 "== cao" logs/analyze.log | cut -c1-90
echo DONE
