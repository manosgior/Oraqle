# Oraqle

Oraqle is a benchmarking framework and artifact for studying how qubit-state **readout** shapes
quantum error correction (QEC) as a systems problem. It covers the signal level (readout duration),
the discriminator level (learned and classical state discriminators, including their FPGA cost),
the readout-error physics (state changes, overlap, leakage) and the QEC/system level (logical error
rate, machine size, decoder load).

This repository contains **all code, small results and checkpoints** behind the Oraqle paper. Every
figure, table and quantitative claim maps to a command in [REPRODUCE.md](REPRODUCE.md).

## The one thing that is not included: the raw dataset

The experiments use the five-qubit multiplexed readout traces of Lienhard et al., *Deep-Neural-Network
Discrimination of Multiplexed Superconducting-Qubit States*, Phys. Rev. Applied 17, 014024 (2022).
The dataset is **not public and is not redistributed here**. Obtain it from its authors, put the two
files in a directory and point `ORAQLE_DATA` at it. Everything else then works unchanged. See
[dataset/README.md](dataset/README.md) for the expected files, their layout and their checksums.

Without the dataset you can still verify every number and figure of the paper from the shipped
results (next section).

## Quick start: verify the paper without the dataset

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-verify.txt
make verify          # every paper number from results/ (seconds)
make verify-figures  # also re-render figures and compare them with the paper's PDFs
make figures         # regenerate all data figures/tables into $ORAQLE_OUT (default: out/)
```

`make verify` exits non-zero on any mismatch. The paper's few known typos (for example a value printed
as 0.936 where the data says 0.963) are recorded as documented discrepancies in
`results/*/paper_values.json` rather than hidden.

## Full reproduction with the dataset

Training the discriminators, the QEC sweeps and the leakage analysis run in Docker images defined in
[docker/](docker/README.md). They read the dataset read-only from `$ORAQLE_DATA`. Per-part commands,
runtimes and hardware are in `docs/reproduce/`. Full retraining and the large sweeps take GPU- and
CPU-days. Every step was smoke-tested from this layout, and several were replayed bit-identically
(see the "verified" column of each table).

## Layout

| Path | Contents | Paper |
|---|---|---|
| `oraqle_env.py` | central path configuration (`ORAQLE_DATA`, `ORAQLE_WORK`, `ORAQLE_RESULTS`, `ORAQLE_CHECKPOINTS`, `ORAQLE_OUT`) | all |
| `dataset/` | how to plug in the raw dataset (README only) | all |
| `readout/` | signal-level analysis (readout-duration sweeps, IQ clouds, baselines) | §4 |
| `discriminators/` | discriminator models, trainers and runners (MCMit-CNN, MCMit-T, HERQULES, KLiNQ, QubiCML, FNN, linear baselines), Table 2 builder | §5, App. |
| `fpga/` | hls4ml synthesis scripts and Vitis HLS reports for the FPGA cost table (contributed by Felix Gust) | §5 |
| `leakage/` | readout error-mode / leakage analysis (state change, overlap, leakage attribution) | §5.3, App. |
| `qec/lattice-sim/` | QEC simulator used for the system study (vendored, MIT) | §7 |
| `qec/eccentric/` | ECCentric drivers and plots; ECCentric itself is fetched at a pinned commit | §6, App. |
| `system/` | system-level study: spacetime, decoder load, provisioning, soft readout, decoder choice | §7 |
| `results/` | small result files the paper is computed from, plus `paper_values.json` per part | all |
| `checkpoints/` | trained model checkpoints (`checkpoints/INDEX.csv` maps them to paper rows) | §5 |
| `paper_figures/` | one script per paper figure/table, from `results/` | all |
| `verify/` | automatic checks of every paper number | all |
| `docker/` | the Docker images used for the paper, with pinned package versions | all |
| `docs/reproduce/` | per-part reproduction tables: paper item, command, output, runtime, verification status | all |
| `extras/` | real work that is not in the paper (three-level/leakage-aware trainers, crosstalk, seed sweeps, synthetic data, superseded scripts) | none |

Every folder has a `PROVENANCE.md` that records where each file came from (repository and commit, or
cluster path) and what was changed during consolidation. Changes were limited to path handling and imports.

## License

MIT (see [LICENSE](LICENSE)). Parts adapted from HERQULES and KLiNQ (Apache-2.0), the vendored
lattice-sim (MIT) and the FPGA scripts contributed by Felix Gust (MIT) are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The raw dataset is not included and is not covered.
