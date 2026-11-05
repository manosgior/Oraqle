# readout/ — signal-level readout analysis (paper §4)

| File | Paper item | Needs dataset? |
|---|---|---|
| `baseline_no_ML.py` | integration-and-threshold baseline (demodulate, integrate to T, logistic-regression boundary refit at every T): truncation sweep, dF/dT, sliding windows; also the Linear Threshold rows of tab:fidelity_linear | yes (functions) |
| `run_baseline_sweeps.py` | driver that runs the steps of `baseline_no_ML.py` in order and writes `Baseline_Threshold_*.csv` (the original `__main__` was hand-edited between runs) | yes |
| `plot_readout_sweeps.py` | fig:readout_sweeps (qubit_readout.tex:8) from the pinned published CSVs | no |
| `iq_clouds_vs_length.py` | fig:iq_clouds (qubit_readout.tex:35): `extract()` (HERQULES front-end demodulation, needs the dataset) + `plot()` (`IQ_REPLOT=1`, uses `results/discriminators/iq_clouds_vs_length_q5.npz`) | extract: yes; plot: no |
| `replot_baseline.py` | older two-panel versions of the §4 figures (not in the paper) | no |
| `export_traces.py`, `plot_clusters.py`, `legacy_plots/` | early averaged-trace and IQ-cluster figures (May 2026; not in the paper) | yes |

    python readout/plot_readout_sweeps.py out.pdf                       # no dataset
    IQ_REPLOT=1 python readout/iq_clouds_vs_length.py                   # no dataset
    python readout/run_baseline_sweeps.py                               # needs ORAQLE_DATA; all 1.6 M shots need ~130 GB RAM
    python readout/run_baseline_sweeps.py --max-samples 20000 --steps sweep    # quick

Outputs go to `$ORAQLE_OUT/discriminators/reports`. `paper_figures/discriminators/fig_*.py`
wrap the figure scripts. Paper-vs-data differences in the §4 prose are listed in
docs/reproduce/discriminators.md and checked by verify/discriminators.py.
