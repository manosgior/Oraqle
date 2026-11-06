#!/usr/bin/env bash
# Copy the code the cross-check needs to the cluster (shared NFS home). Data never goes the other way.
cd "$(dirname "$0")/../../.." && tar cf - --exclude='__pycache__' oraqle_env.py extras/cross_dataset discriminators/networks discriminators/disc_paths.py readout/baseline_no_ML.py results/discriminators results/system/ler_disc.jsonl system/sweep_ler.py system/discriminators.py system/analyze.py qec/lattice-sim | ssh -o BatchMode=yes steve 'mkdir -p ~/oraqle-work/cross_dataset/repo && tar xf - -C ~/oraqle-work/cross_dataset/repo'
