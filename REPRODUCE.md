# Reproducing the paper

Each paper item (figure, table, number) is listed with its command, output, runtime, whether it needs
the raw dataset, and how it was verified, in one table per part:

| Paper part | Reproduction table | Verify |
|---|---|---|
| §2–§5: readout duration, discriminators, Table 2 (`tab:fidelity`), model comparison, FPGA cost (`tab:fpga_resources`), HERQULES appendix tables | [docs/reproduce/discriminators.md](docs/reproduce/discriminators.md) | `make verify-discriminators` |
| §5.3 failure analysis: `tab:error_modes`, `fig:iq_taxonomy`, Takeaway #5, error-mode appendix | [docs/reproduce/leakage.md](docs/reproduce/leakage.md), [leakage/README.md](leakage/README.md) | `make verify-leakage` |
| §6 QEC impact (ECCentric), §7 system bottleneck (lattice-sim), discussion, QEC appendix figures | [docs/reproduce/qec_system.md](docs/reproduce/qec_system.md) | `make verify-qec` |

## Three levels of verification

1. **From shipped results (no dataset, seconds):** `make verify` recomputes every number from
   `results/` and compares it with the printed value (`results/*/paper_values.json`, with
   chapter:line references). `make verify-figures` re-renders the figures and compares them with the
   paper's PDFs, most of them pixel-identical.
2. **Re-running analysis steps (dataset, minutes to hours):** feature extraction, error-mode
   attribution, the §7 analysis chain and exact replays of published simulation points. The commands
   are in the tables above, and the outputs were checked byte-identical where marked.
3. **Full re-training and sweeps (dataset, GPU/CPU-days):** the exact commands, seeds, images and
   protocols are documented. These were not re-run during consolidation. Note that the Table 2 rows
   come from different training protocols, as documented in the discriminators table.

## Environment

- Level 1: Python ≥ 3.10 with `requirements-verify.txt`.
- Levels 2–3: the Docker images in [docker/](docker/README.md). Mount the dataset read-only and set
  `ORAQLE_DATA`, `ORAQLE_WORK`, `ORAQLE_OUT` (see `oraqle_env.py`).

## Not reproducible from this repository

- The raw dataset (see [dataset/README.md](dataset/README.md)).
- Decoder hardware latencies and resource counts taken from other work (Helios, Micro Blossom, qShell,
  CXL-Quantum). They are recorded with their sources in [results/external/](results/external/).
