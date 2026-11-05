# readout/ provenance

Section 4 signal-level code (integration-and-threshold baseline, IQ-cloud figure, old trace plots).

Sources:
* **graham `~/Oraqle`** (NFS home shared with steve; not a git repo; last change 2026-06-22). Copied from its byte-identical laptop mirror `/home/manosgior/Documents/Oraqle` (all source files md5-identical to graham, checked 2026-11-05).
* **graham `~/oraqle_reports`** and **`~/oraqle_models`**: not git; copied from the byte-identical laptop mirrors `/home/manosgior/Documents/oraqle_reports` and `/home/manosgior/Documents/oraqle_models`.
* **Oraqle git** (`github.com/manosgior/Oraqle`, this repository): files from the old tree `Discriminators/`, `scratch/`, `plots/` and the top-level scripts, at branch `unify` HEAD `7be6083` (the "git commit" column is the last commit that touched the file). The old copies were deleted from the working tree; git history keeps them.

18 files, 6 modified after copying (all modifications are path/import plumbing; edited lines carry `[unify]`).

| File | Origin | Host | Git commit | Change |
|---|---|---|---|---|
| `PROVENANCE.md` | new (2026-11-05) | - | - | this file |
| `README.md` | new (2026-11-05) | - | - | folder guide |
| `baseline_no_ML.py` | Oraqle git baseline_no_ML.py | laptop (git manosgior/Oraqle) | 186c1fa | repo copy (git HEAD + Oct 15 axis-label edit; newer than the cluster copy); raw paths -> `disc_paths`; `./optimization_reports` -> `REPORTS_OUT`; CSV-replot defaults -> results/discriminators. `__main__` (hand-edited with exit() calls) unchanged: use readout/run_baseline_sweeps.py |
| `export_traces.py` | Oraqle git export_traces.py | laptop (git manosgior/Oraqle) | d960adc | path plumbing only (container mounts / hard-coded paths -> oraqle_env via discriminators/disc_paths.py; every edited line tagged `[unify]`) |
| `iq_clouds_vs_length.py` | graham:~/Oraqle/Discriminators/runners/iq_clouds_vs_length.py (laptop mirror /home/manosgior/Documents/Oraqle/Discriminators/runners/iq_clouds_vs_length.py) | graham (NFS ~/Oraqle; not git) | - | `REPORTS` -> `IQ_REPORTS` env or `REPORTS_OUT`; plot() falls back to the published npz in results/discriminators |
| `legacy_plots/readout_clusters_1000_shots.png` | Oraqle git plots/readout_clusters_1000_shots.png | laptop (git manosgior/Oraqle) | b41eeb7 | verbatim |
| `legacy_plots/readout_clusters_200_shots.png` | Oraqle git plots/readout_clusters_200_shots.png | laptop (git manosgior/Oraqle) | b41eeb7 | verbatim |
| `legacy_plots/readout_clusters_30_shots.png` | Oraqle git plots/readout_clusters_30_shots.png | laptop (git manosgior/Oraqle) | b41eeb7 | verbatim |
| `legacy_plots/readout_clusters_600_shots.png` | Oraqle git plots/readout_clusters_600_shots.png | laptop (git manosgior/Oraqle) | b41eeb7 | verbatim |
| `legacy_plots/readout_traces_figure_1.png` | Oraqle git plots/readout_traces_figure_1.png | laptop (git manosgior/Oraqle) | b41eeb7 | verbatim |
| `legacy_plots/readout_traces_figure_2.png` | Oraqle git plots/readout_traces_figure_2.png | laptop (git manosgior/Oraqle) | b41eeb7 | verbatim |
| `legacy_plots/readout_traces_figure_3.png` | Oraqle git plots/readout_traces_figure_3.png | laptop (git manosgior/Oraqle) | b41eeb7 | verbatim |
| `legacy_plots/readout_traces_figure_4.png` | Oraqle git plots/readout_traces_figure_4.png | laptop (git manosgior/Oraqle) | b41eeb7 | verbatim |
| `legacy_plots/readout_traces_figure_5.png` | Oraqle git plots/readout_traces_figure_5.png | laptop (git manosgior/Oraqle) | b41eeb7 | verbatim |
| `plot_clusters.py` | Oraqle git plot_clusters.py | laptop (git manosgior/Oraqle) | d960adc | path plumbing only (container mounts / hard-coded paths -> oraqle_env via discriminators/disc_paths.py; every edited line tagged `[unify]`) |
| `plot_readout_sweeps.py` | Oraqle git plot_readout_sweeps.py | laptop (git manosgior/Oraqle) | 186c1fa | `RES = Discriminators/training_results` -> results/discriminators (override `ORAQLE_READOUT_RES`); default output -> `REPORTS_OUT` |
| `replot_baseline.py` | graham:~/Oraqle/replot_baseline.py (laptop mirror /home/manosgior/Documents/Oraqle/replot_baseline.py) | graham (NFS ~/Oraqle; not git) | - | `R = "/r"` mount -> results/discriminators; figures -> `REPORTS_OUT` |
| `run_baseline_sweeps.py` | new (2026-11-05) | - | - | driver for baseline_no_ML.py's sweep / dF/dT / sliding-window steps |
