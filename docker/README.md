# docker/ — the images the paper used, pinned

| Image (tag) | Build | Base / key packages | Runs |
|---|---|---|---|
| `oraqle-latsim` | `docker build -t oraqle-latsim docker/latsim` | python 3.11.17-slim; stim 1.16.0, PyMatching 2.4.0, numpy 2.3.1, qiskit 2.1.0, qiskit-aer 0.17.1, sinter 1.16.0, matplotlib 3.11.2 (`latsim/requirements.lock`) | §7 lattice-sim sweeps (`system/sweep_ler.py`, `sweep_soft.py`, `decode_timing.py`, `helpers/probe_*.py`), `system/derive_uniform_fraction.py`, `verify/qec_system.py --stim` |
| `oraqle-latsim:dec` | `docker build -f docker/latsim/Dockerfile.dec -t oraqle-latsim:dec docker/latsim` (after `oraqle-latsim`) | + ldpc 2.4.1 (`latsim/requirements-dec.lock`) | §7.7 decoder sweep (`system/sweep_decoders.py`), `crosscheck_stim.py`, `sweep_large.py`, `replay_points.py`, `helpers/test_uf.py` |
| `oraqle-torch-gpu` | `docker build -t oraqle-torch-gpu docker/torch-gpu` | python 3.11.17-slim; torch 2.5.1+cu124, numpy 2.1.3, h5py 3.16.0, scikit-learn 1.9.1 (`torch-gpu/requirements.lock`) | §7.6 soft-readout CNN retraining (`system/soft_readout_cnn.py`, needs the dataset and a GPU) |
| `oraqle-torch` | `docker build -t oraqle-torch docker/torch` | python 3.11.11-slim; torch 2.5.1+cpu, same versions (`torch/requirements.lock`) | leakage / error-mode analyses (docs/reproduce/leakage.md); CPU fallback for the discriminator scripts (docs/reproduce/discriminators.md) |
| `eccentric` | `qec/eccentric/fetch_eccentric.sh && docker build -f work/eccentric/Dockerfile.oraqle -t eccentric work/eccentric` | python 3.11-bookworm (digest-pinned); ECCentric @da7164c, 235 pinned packages (stim 1.14.0, PyMatching 2.3.1, qiskit 1.4.2, numpy 1.26.4, tqec 0.2.0) | §6 and appendix ECCentric simulations (`docker/eccentric/README.md`) |
| `oraqle` (CUDA) | `docker build -f docker/oraqle-cuda/Dockerfile -t oraqle .` (context = repo root) | nvidia/cuda 12.2.2-cudnn8 ubuntu22.04; `discriminators/requirements.txt` **unpinned** | the May-June 2026 discriminator training/HPO runs (docs/reproduce/discriminators.md). The original image no longer exists, so its versions are unknown |

Figures, tables and `verify/qec_system.py` (without `--stim`) need only Python with numpy and matplotlib
(+ poppler-utils for `--figures`); no image is required.

## Conventions for every run

The repository is mounted at `/repo` (read-only where possible) and is the working directory; outputs go to a mounted
directory, never into `results/`. The raw Lienhard et al. dataset (not distributed) is mounted read-only at `/data`:

```bash
export ORAQLE_DATA=/path/to/five_qubit_data        # holds DRaw_C_Tr_v0-001, DRaw_C_Te_v0-002 (only for dataset steps)
mkdir -p work/system out
COMMON="--rm --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/repo:ro -w /repo \
        -e ORAQLE_RESULTS=/repo/results -v $PWD/work:/work -e ORAQLE_WORK=/work -v $PWD/out:/out -e ORAQLE_OUT=/out"
DATA="-v $ORAQLE_DATA:/data:ro -e ORAQLE_DATA=/data"
```

## Run recipes (§6, §7, QEC appendix)

```bash
# exact replay of published §7 sweep jobs (bit-identical shots/failures), ~1 min
docker run $COMMON oraqle-latsim:dec python system/replay_points.py
# tiny fresh lattice-sim LER sweep (d = 5, every discriminator point, --quick), ~10 s
docker run $COMMON oraqle-latsim python system/sweep_ler.py --set disc --distances 5 --quick --workers 16 --out /work/system/disc_quick.jsonl
# tiny decoder sweep (d = 5, 7, 6 readout points, all 7 decoders, --quick), ~30 s
docker run $COMMON oraqle-latsim:dec python system/sweep_decoders.py --quick --workers 32 --out /work/system/dec_quick.jsonl
# verification incl. the stim re-derivation of UNIFORM_FRACTION, ~10 s
docker run $COMMON oraqle-latsim python verify/qec_system.py --stim
# soft-readout CNN: dataset + GPU (one readout length; CNN_LEN = raw samples of 2 ns), ~30 min on an H200
docker run $COMMON $DATA --device=nvidia.com/gpu=all -e CNN_LEN=300 oraqle-torch-gpu \
    python system/soft_readout_cnn.py --out /work/system/soft
# the full paper sweeps, step by step (as run on steve/graham; hours): system/run_paper_sweeps.sh <step>
# ECCentric (§6 / appendix): docker/eccentric/README.md
```
`--device=nvidia.com/gpu=all` is the CDI syntax used on steve; with the classic NVIDIA runtime use `--gpus all`.

## Which image produced which committed result

| Result | Image | Host, date |
|---|---|---|
| `results/system/ler_{grid,disc,disc_hi,soft}.jsonl`, `soft_params.json` | oraqle-latsim (cc757391b5b1) | steve, 2026-10-26 |
| `results/system/decode_timing_d*.jsonl`, `decode_large/` | oraqle-latsim (0456515d488a, Python 3.11.11; packages identical) | graham, 2026-10-26 |
| `results/system/soft/cnn_soft_*ns.npz` | oraqle-torch-gpu (8e0bef81a5fc) | steve (H200), 2026-10-26 |
| `results/system/ler_decoders*.jsonl`, `ler_crosscheck.jsonl` | oraqle-latsim:dec (ff7469718be1) | steve, 2026-10-29 |
| `results/system/{summary,summary_soft,decoders_summary,system_cost,uncertainty}.json` | none (laptop Python, numpy) | 2026-10-26..29 |
| `results/qec/experiment_results/**` | eccentric (2026-06-09 build) | graham, 2026-06..07 |
| `results/discriminators/**` | oraqle (CUDA) | graham, 2026-05..06 |
| `results/leakage/**` | oraqle-torch (f40dbb670f94) | graham, 2026-11 |

Exact package sets of the original images: `freeze/*.pip-freeze.txt` (`pip freeze` inside each image on 2026-11-05)
and `freeze/*.image-inspect.txt` (image id, creation time, Python version). Provenance: `PROVENANCE.md`.
