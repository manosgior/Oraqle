#!/usr/bin/env bash
# MCMit-T runs for the cross-dataset table, on a GPU only. Waits until this host's GPU has room, then
# runs each missing run once. Safe to start on several hosts: a run is claimed by an atomic mkdir.
#   usage: gpu_wait.sh <min free MiB>
need=${1:-12000}; image=${IMAGE:-oraqle-torch-gpu:latest}
cd ~/oraqle-work/cross_dataset; R=/w/cross_dataset/repo/extras/cross_dataset; mkdir -p runs/locks logs/gpu
RUNS="qick:600 qick:800 qick:1000 qick:1800 cao:400 cao:600 cao:800 cao:1000 cao:1360"
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | tail -1; }
for r in $RUNS; do
  ds=${r%%:*}; t=${r##*:}
  [ -f runs/$ds/transformer_T${t}_s1.json ] && continue
  mkdir runs/locks/transformer_${ds}_${t} 2>/dev/null || continue
  until [ "$(free_mib)" -ge "$need" ]; do sleep 300; done
  echo "$(date +%F_%T) $(hostname) start $ds $t (free $(free_mib) MiB)"
  docker run --rm --device nvidia.com/gpu=all -u "$(id -u):$(id -g)" -v "$HOME/oraqle-work:/w:rw" -e ORAQLE_WORK=/w -e HOME=/tmp \
    -w /w/cross_dataset --entrypoint python $image -I $R/run_one.py --dataset $ds --model transformer \
    --T $t --threads 8 > logs/gpu/${ds}_T${t}.log 2>&1
  if [ -f runs/$ds/transformer_T${t}_s1.json ]; then echo "$(date +%F_%T) done $ds $t"; else echo "$(date +%F_%T) FAILED $ds $t"; rmdir runs/locks/transformer_${ds}_${t}; fi
done
echo "$(date +%F_%T) $(hostname) queue finished"
