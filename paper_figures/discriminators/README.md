# paper_figures/discriminators/ — regenerate the discriminator/readout figures and tables

No dataset needed; everything is rebuilt from `results/discriminators/` and `fpga/hls4ml/reports/`
into `$ORAQLE_OUT/paper_figures/discriminators` (override with `ORAQLE_FIG_OUT`).

    python paper_figures/discriminators/make_all.py [--paper-root ../oraqle-ics]   # ~20 s

| Script | Paper item | Check |
|---|---|---|
| `fig_readout_sweeps.py` | fig:readout_sweeps (qubit_readout.tex:8) | pixel-identical to plots/readout_sweeps.pdf (100 dpi raster, mean diff 0) |
| `fig_iq_clouds.py` | fig:iq_clouds (qubit_readout.tex:34) | same data/layout; only the font differs when Arial is installed (paper: DejaVu Sans in the cluster container) |
| `fig_pareto.py` | fig:pareto (ml_discriminators.tex:263) | same; font only |
| `tab_fidelity.py` | tab:fidelity (ml_discriminators.tex:145) + tab:fidelity_linear (appendix_herqules.tex:224) | master_fidelity.csv byte-identical; all 180 Table 2 values identical; `--paper-root` prints the cell diff |
| `tab_model_comparison_params.py` | tab:model_comparison "Est. Parameters" (overview.tex:31) | |
| `tab_herqules_pertrace.py` | tab:herqules-pertrace (appendix_herqules.tex:147) | |
| `tab_herqules_noretrain.py` | tab:herqules-noretrain, "Retraining across durations" (appendix_herqules.tex:196) | 1000 ns row: see the known discrepancy |
| `tab_fpga_resources.py` | tab:fpga_resources (ml_discriminators.tex:275) | rows character-identical to the paper |

Hand-made items (no script; sources are in the paper repository): fig:qec_pipeline
(figures/qec_pipeline.tex), fig:axes (figures/taxonomy.tex), fig:background
(figures/oraqle-Background.pdf; exported from a web drawing tool, editable source missing),
the block diagrams of tab:model_comparison (figures/oraqle-models.tex), fig:oraqle_overview
(figures/oraqle_overview.tex), fig:footprint (TikZ in appendix_herqules.tex), and the
text tables tab:scope, tab:research_questions. tab:error_modes and fig:iq_taxonomy belong to
the leakage part (`leakage/`).
