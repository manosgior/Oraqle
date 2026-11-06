#!/usr/bin/env bash
# usage on jamie: qstart.sh <queue> <workers> <threads>   One detached container per queue.
q=$1; w=$2; t=$3
cd ~/oraqle-work/cross_dataset; mkdir -p logs
docker rm -f "oraqle-xd-$q" >/dev/null 2>&1
docker run -d --name "oraqle-xd-$q" --cpus "$((w * t))" --memory 500g -u "$(id -u):$(id -g)" \
  -v "$HOME/oraqle-work:/w:rw" -v "$HOME/qubit_readout_klinq/data/five_qubit_data:/data:ro" \
  -e ORAQLE_WORK=/w -e ORAQLE_DATA=/data -e ORAQLE_OUT=/w/cross_dataset/out -e HOME=/tmp \
  -w /w/cross_dataset --entrypoint python oraqle-torch-gpu:latest -I /w/cross_dataset/repo/extras/cross_dataset/run_queue.py "$q" --workers "$w" --threads "$t"
