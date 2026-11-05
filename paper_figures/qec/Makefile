# §6 (qec_impact.tex) and QEC-appendix figures, regenerated from results/qec into $(ORAQLE_OUT)/paper_figures/qec.
# Needs python3 with numpy + matplotlib (any recent version; the paper PDFs were made with matplotlib 3.9.3 and the
# regenerated ones are pixel-identical with 3.10.9) and, for `compare`, poppler-utils.
#   make all        every paper figure      make compare   regenerated vs results/qec/reference_figures
#   make numbers    §6 prose numbers vs results/qec/paper_values.json
#   make extras     not in the paper: the HERQULES-readout simulator comparison (fig:ler_current / fig:ler_futuristic
#                   until the 2026-11-05 [audit fix]); `make compare-extras` checks it against reference_figures/extras
#   make cnn49      not in the paper: the earlier 49-point CNN-readout draft (extras/qec/plot_readout_compare_cnn.py)
PY ?= python3
HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
FIG := $(PY) $(HERE)make_figure.py

.PHONY: all ler_discriminators six_noise six_future ler_current ler_futuristic six_decoh six_fut_fixed compare numbers \
	extras ler_current_herqules ler_futuristic_herqules compare-extras cnn49
all: ler_discriminators six_noise six_future ler_current ler_futuristic six_decoh six_fut_fixed
ler_discriminators: ; $(FIG) fig:ler_discriminators   # qec_impact.tex:26         plots/ler_pair_discriminators.pdf
six_noise:          ; $(FIG) fig:six_noise            # qec_impact.tex:102        plots/ler_pair_noise.pdf
six_future:         ; $(FIG) fig:six_future           # qec_impact.tex:116        plots/ler_pair_future.pdf
ler_current:        ; $(FIG) fig:ler_current          # appendix_herqules.tex:427 plots/readout_ler_cnn_current.pdf
ler_futuristic:     ; $(FIG) fig:ler_futuristic       # appendix_herqules.tex:438 plots/readout_ler_cnn_futuristic.pdf
six_decoh:          ; $(FIG) fig:six_decoh            # appendix_herqules.tex:451 plots/noise_models_ler_decoherence_only.pdf
six_fut_fixed:      ; $(FIG) fig:six_fut_fixed        # appendix_herqules.tex:474 plots/noise_models_ler_futuristic_fixedreadout.pdf
compare: ; $(PY) $(HERE)compare_figures.py
numbers: ; $(PY) $(HERE)../../qec/eccentric/paper_numbers.py --check
extras: ler_current_herqules ler_futuristic_herqules
ler_current_herqules:    ; $(FIG) extra:ler_current_herqules     # readout_ler_herqules_current.pdf (not in the paper)
ler_futuristic_herqules: ; $(FIG) extra:ler_futuristic_herqules  # readout_ler_herqules_futuristic.pdf (not in the paper)
compare-extras: ; $(PY) $(HERE)compare_figures.py --ref $(HERE)../../results/qec/reference_figures/extras
cnn49:  ; $(PY) $(HERE)../../extras/qec/plot_readout_compare_cnn.py
