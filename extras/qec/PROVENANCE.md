# extras/qec — provenance (not paper figures)

| File | Origin | Changes |
|---|---|---|
| `plot_readout_compare_cnn.py` | `plot_readout_compare.py` of TUM-DSE/eccentric @da7164c (the CNN-readout variant) | path plumbing; points the lattice-sim panel at `synchronization-artifact/artifact_readout_ler_cnn.csv` (2026-06-13, 49 of 50 points), which eccentric never copied next to the ECCentric CSV. Output: `$ORAQLE_OUT/extras/qec/readout_ler_cnn_{current,futuristic}.pdf` (`make -C paper_figures/qec cnn49`). |

| `run_latsim_cnn_missing_point.py` | new 2026-11-05 (audit) | computes the missing d=13 / 200 ns / current point of `artifact_readout_ler_cnn.csv` with the unchanged driver `qec/lattice-sim/compare_readout_artifact.py` (500k shots in 50 processes); ran on steve in the `oraqle-latsim` image (47 s) → `results/qec/latsim_cnn_d13_200ns_current.{csv,log}`, merged into `results/qec/artifact_readout_ler_cnn_complete.csv` by `paper_figures/qec/make_readout_ler_cnn.py --merge` |

Purpose: `plot_readout_compare_cnn.py` was the first draft of the simulator comparison with the MCMit-CNN readout
that qec_impact.tex:21 describes (49 lattice-sim points, its own styling); it is superseded by the paper figure
since the 2026-11-05 [audit fix] (fig:ler_current / fig:ler_futuristic = plots/readout_ler_cnn_*.pdf, made by
`paper_figures/qec/make_readout_ler_cnn.py` with all 50 points). Kept as `make -C paper_figures/qec cnn49`.
