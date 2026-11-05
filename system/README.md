# system/ — §7 "Readout as a System Bottleneck"

lattice-sim sweeps (logical error vs readout length, readout error, distance, soft readout, decoder type), decoder
timing, and the cost model that turns them into qubits, runtime, bandwidth and decoder hardware.

| Step | Script | Image | Writes |
|---|---|---|---|
| LER sweeps | `sweep_ler.py --set grid / disc` | oraqle-latsim | `ler_grid.jsonl`, `ler_disc(_hi).jsonl` |
| decoder timing | `decode_timing.py` | oraqle-latsim | `decode_timing_d*.jsonl`, `decode_large/` |
| soft CNN outputs | `soft_readout_cnn.py` (needs `$ORAQLE_DATA`, GPU) | oraqle-torch-gpu | `soft/cnn_soft_*ns.npz` |
| soft-readout sweep | `sweep_soft.py` | oraqle-latsim | `ler_soft.jsonl`, `soft_params.json` |
| decoder sweep | `sweep_decoders.py` | oraqle-latsim:dec | `ler_decoders*.jsonl` |
| summaries | `analyze.py`, `summarize_soft.py`, `merge_decoders.sh` | any python + numpy | `summary.json`, `summary_soft.json`, `decoders_summary.{txt,json}` |
| cost model | `system_cost.py` (+ `workloads.py`, `discriminators.py`) | any python + numpy | `system_cost.json` |
| figures | `plot_readout_bottleneck.py` (or `make -C paper_figures/system`) | python + matplotlib | 5 PDFs in `$ORAQLE_OUT/paper_figures/system` |
| paper numbers | `paper_numbers.py --check` | python + numpy | stdout |
| exact replay (smoke) | `replay_points.py` | oraqle-latsim:dec | stdout |
| UNIFORM_FRACTION | `derive_uniform_fraction.py` | oraqle-latsim (stim 1.16.0) | `results/external/uniform_fraction_d9.json` |
| not in the paper | `uncertainty.py` (bootstrap error bars), `crosscheck_stim.py` (simulator cross-check), `sweep_large.py` (d up to 49; never run to completion) | | |

The committed outputs of every step are in `results/system/`. All paper commands, in order: `run_paper_sweeps.sh`.
Readers (`plot_readout_bottleneck.py`, `paper_numbers.py`, `replay_points.py`) default to `results/system`; writers
(`analyze.py`, `summarize_soft.py`, `merge_decoders.sh`, `system_cost.py`, `uncertainty.py`) work in `--data DIR`,
default `$ORAQLE_WORK/system`, so they never overwrite `results/`. `verify/qec_system.py` re-runs the cheap chain
from `results/system` in `$ORAQLE_OUT/verify/system` and checks it, and every §7 number, against the committed files.
Provenance and path changes: `PROVENANCE.md`. Full reproduction table: `docs/reproduce/qec_system.md`.
