# §6 (qec_impact.tex) and QEC-appendix figures, regenerated from results/qec into $(ORAQLE_OUT)/paper_figures/qec.
# Needs python3 with numpy + matplotlib (any recent version; the paper PDFs were made with matplotlib 3.9.3 and the
# regenerated ones are pixel-identical with 3.10.9) and, for `compare`, poppler-utils.
#   make all        every figure            make compare   regenerated vs results/qec/reference_figures
#   make numbers    §6 prose numbers vs results/qec/paper_values.json
#   make extras     CNN-readout simulator comparison (not in the paper)
PY ?= python3
HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
FIG := $(PY) $(HERE)make_figure.py

.PHONY: all ler_discriminators six_noise six_future ler_current ler_futuristic six_decoh six_fut_fixed compare numbers extras
all: ler_discriminators six_noise six_future ler_current ler_futuristic six_decoh six_fut_fixed
ler_discriminators: ; $(FIG) fig:ler_discriminators   # qec_impact.tex:25  plots/ler_pair_discriminators.pdf
six_noise:          ; $(FIG) fig:six_noise            # qec_impact.tex:99  plots/ler_pair_noise.pdf
six_future:         ; $(FIG) fig:six_future           # qec_impact.tex:113 plots/ler_pair_future.pdf
ler_current:        ; $(FIG) fig:ler_current          # appendix_herqules.tex:415 plots/readout_ler_herqules_current.pdf
ler_futuristic:     ; $(FIG) fig:ler_futuristic       # appendix_herqules.tex:424 plots/readout_ler_herqules_futuristic.pdf
six_decoh:          ; $(FIG) fig:six_decoh            # appendix_herqules.tex:437 plots/noise_models_ler_decoherence_only.pdf
six_fut_fixed:      ; $(FIG) fig:six_fut_fixed        # appendix_herqules.tex:460 plots/noise_models_ler_futuristic_fixedreadout.pdf
compare: ; $(PY) $(HERE)compare_figures.py
numbers: ; $(PY) $(HERE)../../qec/eccentric/paper_numbers.py --check
extras:  ; $(PY) $(HERE)../../extras/qec/plot_readout_compare_cnn.py
