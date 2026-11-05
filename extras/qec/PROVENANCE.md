# extras/qec — provenance (not paper figures)

| File | Origin | Changes |
|---|---|---|
| `plot_readout_compare_cnn.py` | `plot_readout_compare.py` of TUM-DSE/eccentric @da7164c (the CNN-readout variant) | path plumbing; points the lattice-sim panel at `synchronization-artifact/artifact_readout_ler_cnn.csv` (2026-06-13, 49 of 50 points), which eccentric never copied next to the ECCentric CSV. Output: `$ORAQLE_OUT/extras/qec/readout_ler_cnn_{current,futuristic}.pdf` (`make -C paper_figures/qec extras`). |

Purpose: the simulator comparison with the MCMit-CNN readout that qec_impact.tex:20 describes; the paper's
fig:ler_current / fig:ler_futuristic use the older HERQULES readout (docs/reproduce/qec_system.md, known issue 12).
