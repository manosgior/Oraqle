# Provenance: paper_figures/qec

| File | What |
|---|---|
| `Makefile` | one target per §6 / QEC-appendix figure (`ler_discriminators six_noise six_future ler_current ler_futuristic six_decoh six_fut_fixed`), plus `compare`, `numbers`, `extras` |
| `make_figure.py` | calls the vendored plotting code (`qec/eccentric/plots`) exactly as its `__main__` block did, one figure per label |
| `compare_figures.py` | regenerated vs `results/qec/reference_figures`: page size, pdftotext text, 100 dpi raster difference, and whether the decompressed PDF drawing streams are identical |

New files (2026-11-05); the plotting code they call and its origin are listed in `qec/eccentric/PROVENANCE.md`.

## Verification (tier 1 and tier 2)

| Run | Environment | Result for all 7 figures |
|---|---|---|
| laptop, `make all compare` | Oraqle/Discriminators/.venv: Python 3.13.2, numpy 2.4.4, matplotlib 3.10.9 | page size and text identical; 0 differing pixels at 100 and 300 dpi; every decompressed drawing stream identical (only creator/date metadata differ) |
| graham, `make_figure.py all` inside the pinned `eccentric` image (`unify-qec-eccentric`) | Python 3.11.15, numpy 1.26.4, matplotlib 3.9.3 (the paper's version) | same; for the four figures made on the cluster in June/July even the non-stream PDF objects are identical once the creation date is removed |

Runtime: ~10 s for all figures.
