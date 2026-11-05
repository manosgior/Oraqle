# paper_figures/leakage

Regenerates the error-mode items of the paper from `results/leakage` (no dataset needed):

| paper item | command | output | check |
|---|---|---|---|
| Table `tab:error_modes` (ml_discriminators.tex l.397-418) | `python paper_figures/leakage/make_error_modes_table.py` | `$ORAQLE_OUT/leakage/tab_error_modes.tex` | identical to `reference/tab_error_modes.tex` (the paper block) |
| same, from the stored 2-decimal shares | `... make_error_modes_table.py --stored` | `tab_error_modes_stored.tex` | differs only in Q5 overlap: 8.8 (a double rounding of 8.75; the paper printed it until the 2026-11-05 [audit fix], now 8.7) |
| Fig. `fig:iq_taxonomy` (l.342-353) | `python paper_figures/leakage/make_iq_taxonomy.py` | `$ORAQLE_OUT/leakage/iq_error_gallery_q5_v2.{pdf,png}` | pixel-identical to `reference/iq_error_gallery_q5_v2.pdf` at 300 dpi |

The figure needs matplotlib and seaborn, `results/discriminators/iq_error_gallery_q5.npz` and the generator
`discriminators/runners/iq_error_gallery.py` (override with `--npz`, `--gen`). Its extraction step (raw data ->
npz) is described in the script docstring. App. `app:error-modes` has no figure or table of its own (it refers to
these two). Both checks run in `verify/leakage.py` (`--figure` for the figure).
