# paper_figures/system — provenance

New for this artifact. `Makefile` targets call `system/plot_readout_bottleneck.py --only <figure>` (the script that
made the paper's PDFs, `oraqle-ics/sims/plot_readout_bottleneck.py` @d7cd4e9, path plumbing only) and
`tab_sys_reaction.py`, which prints the cells of Table tab:sys_reaction from `system_cost.json` (the paper's table was
hand-transcribed from `system_cost.py` output; all 20 cells match). `compare_figures.py` reuses
`paper_figures/qec/compare_figures.py` with `results/system/reference_figures`.

| Target | Paper item | Output |
|---|---|---|
| sys_spacetime | fig:sys_spacetime (system_bottleneck.tex:29) | readout_spacetime.pdf |
| sys_decoder | fig:sys_decoder (:54) | decoder_load.pdf |
| sys_cost | fig:sys_cost (:98) | system_cost.pdf |
| sys_soft | fig:sys_soft (:143) | soft_readout.pdf |
| sys_decoders | fig:sys_decoders (:170) | decoder_choice.pdf |
| sys_reaction | tab:sys_reaction (:61-78) | tab_sys_reaction.tex |

Verified 2026-11-05 with Matplotlib 3.10.9 (paper: 3.10.3): page size, text and raster identical, drawing streams
identical; the PDF bytes differ only in the Creator/Producer version and CreationDate.
