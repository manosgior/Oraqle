# results/system — provenance

Every result file of §7 (system_bottleneck.tex). 46 MB, largest file 9.5 MB (`soft/cnn_soft_200ns.npz`).

| Files | Origin | Producer (repo path) | Check |
|---|---|---|---|
| all `*.jsonl`, `*.json`, `decoders_summary.txt`, `decode_large/`, `soft/` (55 files) | `oraqle-ics/data/sims/` (paper repository, git, HEAD d7cd4e9) | see below | md5-identical (55/55) to `oraqle-ics/data/sims`; the 51 files that also exist in steve/graham `~/oraqle-ics-sims/data` are md5-identical there (the cluster lacks `summary_soft.json`, `system_cost.json`, `uncertainty.json`, written on the laptop) |
| `logs/` (52 files, 228 KB) | steve/graham `~/oraqle-ics-sims/logs/` (2026-10-26..29) | stdout of the sweep containers | copied verbatim; `dec_d19.log`, `dec_d21.log` are the failed d = 19/21 timing runs (decoder-latency lookup), superseded by `decode_large/` |
| `reference_figures/*.pdf` (5) | `oraqle-ics/plots/{readout_spacetime,decoder_load,system_cost,soft_readout,decoder_choice}.pdf` (git, last change 0cf05da; Matplotlib 3.10.3, 2026-10-29) | `system/plot_readout_bottleneck.py` | regenerated PDFs have identical drawing streams (only Creator/CreationDate differ) |
| `paper_values.json` | written for this artifact from system_bottleneck.tex (oraqle-ics working tree 2026-11-05) | — | `system/paper_numbers.py --check`: 165 values, 0 mismatches; 11 known content issues listed |

| Result | Producer | Image / host / date |
|---|---|---|
| `ler_disc.jsonl` (target 200), `ler_disc_hi.jsonl` (target 1000, cap ×5), `ler_grid.jsonl` | `system/sweep_ler.py` | oraqle-latsim, steve, 2026-10-26 |
| `decode_timing_d{5..17}.jsonl`, `decode_large/d{21..57}_T*.jsonl` | `system/decode_timing.py` (one pinned core per job) | oraqle-latsim, graham, 2026-10-26 |
| `soft/cnn_soft_{200..1000}ns.npz` (per-shot CNN probability of \|1>, 1.12 M test shots; expensive: GPU retrain on the raw dataset) | `system/soft_readout_cnn.py` | oraqle-torch-gpu, steve H200, 2026-10-26 |
| `ler_soft.jsonl`, `soft_params.json` | `system/sweep_soft.py` | oraqle-latsim, steve, 2026-10-26 |
| `ler_decoders.jsonl` (126/136 jobs), `ler_decoders_soft13.jsonl`, `ler_decoders_soft11.jsonl` | `system/sweep_decoders.py` | oraqle-latsim:dec, steve, 2026-10-29 |
| `ler_crosscheck.jsonl` (not cited) | `system/crosscheck_stim.py` | oraqle-latsim:dec, steve, 2026-10-29 |
| `summary.json` | `system/analyze.py` | any |
| `summary_soft.json` | `system/summarize_soft.py` (was an inline call) | any |
| `decoders_summary.{txt,json}` | `system/merge_decoders.sh` (3 files) | any |
| `system_cost.json` | `system/system_cost.py` | any |
| `uncertainty.json` (not cited) | `system/uncertainty.py` | any |

Exact commands: `system/run_paper_sweeps.sh`; reproduction table: `docs/reproduce/qec_system.md`.
