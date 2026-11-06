#!/usr/bin/env bash
# usage on a cluster host: xrun.sh <cpus> <python args...>   Runs python -I in the Oraqle torch image, CPU only.
cpus=$1; shift
exec docker run --rm --cpus "$cpus" -u "$(id -u):$(id -g)" \
  -v "$HOME/oraqle-work:/w:rw" -v "$HOME/qubit_readout_klinq/data/five_qubit_data:/data:ro" \
  -e ORAQLE_WORK=/w -e ORAQLE_DATA=/data -e ORAQLE_OUT=/w/cross_dataset/out -e HOME=/tmp -e OMP_NUM_THREADS="${OMP:-$cpus}" \
  -w /w/cross_dataset --entrypoint python oraqle-torch-gpu:latest -I "$@"
