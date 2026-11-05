# ECCentric runs of the paper (§6 and the QEC appendix)

ECCentric (TUM-DSE/eccentric, public) is the circuit-level QEC simulator behind §6 (`qec_impact.tex`) and the
appendix sections "ECCentric vs. lattice-sim", "Decoherence-only noise model" and "Fixed-fidelity readout".
The library is **not** vendored: `fetch_eccentric.sh` clones the exact revision over HTTPS. Only the Oraqle-specific
pieces that are not in that revision, or that needed a path fix, live here.

| Path | What |
|---|---|
| `fetch_eccentric.sh [DEST] [--with-7858c2c]` | clone eccentric @da7164c + submodules qiskit-qec @7136727, more-bacon-less-threshold @df98313 (HTTPS), verify the SHAs, copy the overlay, the pinned Dockerfile/lock and the smoke wrapper into DEST (default `$ORAQLE_WORK/eccentric` = `work/eccentric`). `--with-7858c2c` adds a worktree at 7858c2c for the appendix comparison drivers. |
| `overlay/run_ler_discriminators.py` | da7164c's driver with one change: `master_fidelity.csv` is found via `$ORAQLE_MASTER_FIDELITY`, else `$ORAQLE_RESULTS/discriminators/master_fidelity.csv`, else the original `../oraqle_reports/master_fidelity.csv`. |
| `plots/` | the plotting scripts that made the paper PDFs, path-plumbed (`_paths.py`: inputs `$ECC_RESULTS` = `results/qec/experiment_results`, outputs `$ECC_FIG_OUT` = `$ORAQLE_OUT/paper_figures/qec`). Need only numpy + matplotlib. `plots/original/` holds the unmodified sources; `plots/recovered/` the recovered `plot_paper_pairs.py`. |
| `paper_numbers.py` | every §6 prose number and the appendix checks from the CSVs; `--check` compares with `results/qec/paper_values.json` (exit 1 on mismatch). |
| `smoke/smoke_point.py` | one-configuration smoke test that calls a driver's own `run_one()` (copied into the checkout). |
| `cluster_only/` | three more scripts that existed only in graham `~/eccentric` (MCMit thesis plots; **not** used by the Oraqle paper), kept verbatim for completeness. |
| (doc rows) | merged into `docs/reproduce/qec_system.md` (section 4). |

## Regenerate the figures and numbers (no simulator needed, seconds)

```bash
make -C paper_figures/qec all compare numbers PY=python3    # into $ORAQLE_OUT/paper_figures/qec (default out/)
```
All seven figures come out pixel-identical to the paper PDFs (300 dpi raster, 0 differing pixels; matplotlib 3.10.9
vs. the paper's 3.9.3). `make extras` draws the CNN-readout version of the simulator comparison (see below).

## Re-run the simulations (ECCentric image, CPU; hours per sweep)

```bash
qec/eccentric/fetch_eccentric.sh                                   # -> work/eccentric
docker build -f work/eccentric/Dockerfile.oraqle -t eccentric work/eccentric
mkdir -p work/ecc_out
docker run --rm -v "$PWD/work/ecc_out:/qec/experiment_results" -v "$PWD/results:/results:ro" \
  -e ORAQLE_RESULTS=/results -e LER_MODE=current eccentric python -u run_ler_noise_models.py
```
Every driver writes `experiment_results/<group>/<file>.csv` relative to its working directory (`/qec` in the image),
so mount an empty host directory there. Drivers use all cores (`min(cpu_count-2, 60)` processes).
The samplers are **not seeded** (`compile_detector_sampler()` without a seed), so a re-run reproduces the CSVs
statistically (binomial error), not bit for bit.

### Which invocation produced which paper CSV

All drivers are in eccentric @da7164c, unchanged except `run_ler_discriminators.py` (overlay). "Documented" = the
command is written down in the eccentric repository (`run_cnn_all.sh`, RESULTS.md, docstrings); "reconstructed" = the
environment variables are inferred from the CSV contents (codes, shot counts) and the run-log headers on graham
(`~/eccentric/experiment_results/discriminators/run*.log`: config counts 180/180/120/105/140).

| CSV (results/qec/experiment_results/...) | Paper item | Command (cwd = checkout, or `/qec` in the image) | Source |
|---|---|---|---|
| `noise_models/ler_noise_models.csv` | fig:six_noise (left), fig:six_decoh, CNN line of fig:ler_discriminators (d=5) | `LER_MODE=current python run_ler_noise_models.py` | documented (run_cnn_all.sh) |
| `noise_models/ler_noise_models_boston_lowt_backlog.csv` | fig:six_noise (right), "~27%" | `LER_MODE=boston_lowt_backlog python run_ler_noise_models.py` | reconstructed (mode documented in the driver and EXPERIMENTS_INDEX.md) |
| `noise_models/ler_noise_models_highfid.csv` | fig:six_future (left) | `LER_MODE=highfid python run_ler_noise_models.py` | documented |
| `noise_models/ler_noise_models_futuristic.csv` | fig:six_future (right) | `LER_MODE=futuristic python run_ler_noise_models.py` | documented |
| `noise_models/ler_noise_models_futuristic_fixedreadout.csv` | fig:six_fut_fixed | `LER_MODE=futuristic_fixedreadout python run_ler_noise_models.py` | reconstructed (mode documented) |
| `discriminators/ler_discriminators_d5.csv` | fig:ler_discriminators (left), §6.1 numbers | `LER_DIST=5 python run_ler_discriminators.py` | documented (RESULTS.md) |
| `discriminators/ler_discriminators_d7.csv` | (not in the paper) | `LER_DIST=7 python run_ler_discriminators.py` | documented |
| `discriminators/ler_discriminators_d9.csv` | (not in the paper) | `LER_DIST=9 LER_CODES=surface,bacon,color,hh python run_ler_discriminators.py` | reconstructed |
| `discriminators/ler_discriminators_d12.csv` | fig:ler_discriminators (right), d=12 numbers | `LER_DIST=12 LER_CODES=surface,bacon,hh LER_INCLUDE_CNN=1 python run_ler_discriminators.py` | reconstructed |
| `discriminators/ler_discriminators_d12_futuristic.csv` | (not in the paper) | `LER_DIST=12 LER_CODES=surface,bacon,hh,gross LER_INCLUDE_CNN=1 LER_FUTURISTIC=1 python run_ler_discriminators.py` | reconstructed |
| `readout_compare/eccentric_readout_ler_herqules.csv` | fig:ler_current (a), fig:ler_futuristic (a) | `compare_readout_eccentric.py` @7858c2c (`READOUT_MODE="herqules"`) with d+1 memory rounds | **partial**: the committed 7858c2c file sets `MEMORY_ROUNDS = 1` (it then writes `..._nmr1.csv`); the d+1 state that wrote this CSV on 2026-06-10 10:29 UTC was never committed. Its log header (graham `readout_compare/eccentric_run_herqules.log`) matches the 7858c2c shots/distances. |
| `readout_compare/eccentric_readout_ler_herqules_nmr1.csv` | (control, not plotted) | `python compare_readout_eccentric.py` in the `--with-7858c2c` worktree | exact code |
| `readout_compare/artifact_readout_ler_herqules.csv` (= `synchronization-artifact/artifact_readout_ler_herqules.csv`) | fig:ler_current (b), fig:ler_futuristic (b) | `cd synchronization-artifact && python compare_readout_artifact.py` @7858c2c (`READOUT_MODE="herqules"`; lattice-sim, idle x3, 500k shots; runs in the eccentric image), then copy into `experiment_results/readout_compare/` | exact code |
| `readout_compare/eccentric_readout_ler_cnn{,_nmr1}.csv` | (CNN rerun; extras) | `MEMORY_ROUNDS=dplus1` / `MEMORY_ROUNDS=1 python compare_readout_eccentric.py` @da7164c | documented (run_cnn_all.sh) |
| `synchronization-artifact/artifact_readout_ler_cnn.csv` | (CNN rerun; extras) | `cd synchronization-artifact && python compare_readout_artifact.py` @da7164c (`READOUT_MODE="cnn"`) | exact code; 49 of 50 points (d=13, 200 ns, current missing) |
| `noise_models/{readout_length_tradeoff,surface_readout_mechanisms}.*`, `RESULTS.md` | (not in the paper) | `make_readout_tradeoff.py`, `explore_readout.py`, `gen_noise_results_md.py` | documented |
| `*/buggy_herqules/*` | (superseded; kept because the appendix comparison figures still use this readout) | earlier runs with the mixed-shot HERQULES table | moved there by commit 5658bb8 |

The other Oraqle drivers in the pinned revision, used unchanged: `run_ler_noise_models.py`,
`compare_readout_eccentric.py`, `explore_readout.py`, `make_readout_tradeoff.py`, `gen_noise_results_md.py`,
`run_cnn_all.sh`, `synchronization-artifact/{compare_readout_artifact,run_herqules_both,run_herqules_futuristic,run_tradeoff_final}.py`
(the last three are MCMit runs). The §6 noise constants (qec_impact:20: CZ 1.191e-3, SX 1.637e-4,
T1/T2 284.95/322.68 µs) are checked against these drivers by `paper_numbers.py`.

`run_ler_discriminators.py` reads the per-qubit fidelities of six discriminators from `master_fidelity.csv`
(produced by the discriminator part of this repository). The paper's CSVs used exactly the current file
(md5 35bbb4b4c0693094e71c5924c5058074): `measure_error` in every row of the d=5/7/9/12 CSVs equals
1 − gmean(Q1, Q3, Q4, Q5) of that file to the last digit. The MCMit-CNN line uses the hard-coded `CNN_ACC` table.

## Smoke test (tier 2)

```bash
docker run --rm -e LER_DIST=5 -e SMOKE_SHOTS=2000 -e ORAQLE_RESULTS=/results \
  -e ORAQLE_MASTER_FIDELITY=/results/discriminators/master_fidelity.csv \
  -v "$PWD/results:/results:ro" eccentric python smoke_point.py discriminators surface MCMit-T 600
```
prints the LER of one configuration and its z-score against the paper's CSV. Runs on graham, 2026-11-05 (image
`unify-qec-eccentric` built from this recipe), every one within |z| < 1.3 of the paper's CSV:

| Driver and point | Shots | LER (rerun) | LER (paper CSV) | z | Wall |
|---|---|---|---|---|---|
| run_ler_discriminators, surface d=5, MCMit-T, 600 ns | 20k | 0.0717 | 0.07115 | 0.21 | 4 s |
| run_ler_discriminators, color d=5 (BP-OSD), HERQULES, 400 ns | 5k | 0.1876 | 0.1842 | 0.44 | 25 s |
| run_ler_noise_models current, surface d=5, heron_cnn, 600 ns | 20k | 0.07275 | 0.0703 | 0.96 | 4 s |
| run_ler_noise_models boston_lowt_backlog, surface d=5, 200 ns | 50k | 0.02242 | 0.0222 | 0.24 | 4 s |
| compare_readout_eccentric @7858c2c (nmr1 state), d=5, 600 ns, current | 100k | 0.3631 | 0.36084 | ≈1.0 | 23 s |
| compare_readout_artifact @7858c2c (lattice-sim), d=5, 600 ns, current | 500k | 0.006478 | 0.006688 | ≈−1.3 | 17 s |

## The extra figure

`extras/qec/plot_readout_compare_cnn.py` is eccentric's own CNN variant of the comparison plot, pointed at the CNN
reruns of both simulators. The paper's fig:ler_current / fig:ler_futuristic use the older HERQULES readout
(contrary to qec_impact:20); the CNN version shows the same qualitative picture (ECCentric saturated, lattice-sim
U-shaped with its minimum at 600-800 ns).
