#!/usr/bin/env bash
# One short run of every model on QICK at 400 ns, each with its own thread budget.
cd ~/oraqle-work/cross_dataset; R=/w/cross_dataset/repo/extras/cross_dataset; mkdir -p logs/smoke
for m in linear fnn qubic herqules klinq transformer cnn; do
  ( OMP=8 ./xrun.sh 8 $R/run_one.py --dataset qick --model $m --T 400 --threads 8 --epochs 1 --trials 1 --tag smoke > logs/smoke/$m.log 2>&1 ) &
done
wait
