# Provenance: paper_figures/qec

| File | What |
|---|---|
| `Makefile` | one target per §6 / QEC-appendix figure (`ler_discriminators six_noise six_future ler_current ler_futuristic six_decoh six_fut_fixed`), plus `compare`, `numbers`, `extras` (the superseded HERQULES-readout `readout_ler_herqules_*`), `compare-extras`, `cnn49` |
| `make_figure.py` | calls the vendored plotting code (`qec/eccentric/plots`) exactly as its `__main__` block did, one figure per label; `fig:ler_current` / `fig:ler_futuristic` go through `make_readout_ler_cnn.py`, `extra:ler_*_herqules` draw the old HERQULES-readout version |
| `make_readout_ler_cnn.py` | new 2026-11-05 (audit): fig:ler_current / fig:ler_futuristic with the MCMit-CNN readout (`plots/readout_ler_cnn_{current,futuristic}.pdf`, in the paper since the [audit fix]); the unchanged `plot_readout_compare.make_figure` @7858c2c on `readout_compare/eccentric_readout_ler_cnn.csv` + `results/qec/artifact_readout_ler_cnn_complete.csv`; `--merge` writes the latter |
| `compare_figures.py` | regenerated vs `results/qec/reference_figures` (top level = the paper's 7 PDFs; `--ref .../extras` for the superseded ones): page size, pdftotext text, 100 dpi raster difference, and whether the decompressed PDF drawing streams are identical |

New files (2026-11-05); the plotting code they call and its origin are listed in `qec/eccentric/PROVENANCE.md`.

## Verification (tier 1 and tier 2)

| Run | Environment | Result for all 7 figures |
|---|---|---|
| laptop, `make all compare` | Oraqle/Discriminators/.venv: Python 3.13.2, numpy 2.4.4, matplotlib 3.10.9 | page size and text identical; 0 differing pixels at 100 and 300 dpi; every decompressed drawing stream identical (only creator/date metadata differ) |
| graham, `make_figure.py all` inside the pinned `eccentric` image (`unify-qec-eccentric`) | Python 3.11.15, numpy 1.26.4, matplotlib 3.9.3 (the paper's version) | same; for the four figures made on the cluster in June/July even the non-stream PDF objects are identical once the creation date is removed |

2026-11-05, after the [audit fix]: `make verify-figures` on the laptop (same environment) regenerates all 7 paper
figures, incl. readout_ler_cnn_{current,futuristic}.pdf: 0 differing pixels, drawing streams identical; the two
HERQULES-readout extras (`make extras compare-extras`) likewise. The graham run above predates the CNN figures.

Runtime: ~10 s for all figures.
