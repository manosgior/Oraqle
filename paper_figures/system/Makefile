# §7 (system_bottleneck.tex) figures and table, regenerated from results/system into $(ORAQLE_OUT)/paper_figures/system.
# Needs python3 with numpy + matplotlib (paper PDFs: matplotlib 3.10.3; with 3.10.9 the regenerated PDFs differ only
# in the Creator/Producer version string and CreationDate) and, for `compare`, poppler-utils.
#   make all       every figure + the table        make compare   regenerated vs results/system/reference_figures
#   make analysis  re-derive the summaries from the raw sweeps into $(ORAQLE_OUT)/system_rerun (see verify/qec_system.py)
PY ?= python3
HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
REPO := $(abspath $(HERE)../..)
PLOT := $(PY) $(REPO)/system/plot_readout_bottleneck.py
DATA ?= $(REPO)/results/system

.PHONY: all sys_spacetime sys_decoder sys_cost sys_soft sys_decoders sys_reaction compare analysis
all: sys_spacetime sys_decoder sys_cost sys_soft sys_decoders sys_reaction
sys_spacetime: ; $(PLOT) --data $(DATA) --only spacetime        # fig:sys_spacetime system_bottleneck.tex:30  plots/readout_spacetime.pdf
sys_decoder:   ; $(PLOT) --data $(DATA) --only decoder          # fig:sys_decoder   system_bottleneck.tex:58  plots/decoder_load.pdf
sys_cost:      ; $(PLOT) --data $(DATA) --only system           # fig:sys_cost      system_bottleneck.tex:103  plots/system_cost.pdf
sys_soft:      ; $(PLOT) --data $(DATA) --only soft             # fig:sys_soft      system_bottleneck.tex:149 plots/soft_readout.pdf
sys_decoders:  ; $(PLOT) --data $(DATA) --only decoder_choice   # fig:sys_decoders  system_bottleneck.tex:177 plots/decoder_choice.pdf
sys_reaction:  ; $(PY) $(HERE)tab_sys_reaction.py --data $(DATA) # tab:sys_reaction  system_bottleneck.tex:65-82
compare: ; $(PY) $(HERE)compare_figures.py
analysis: ; $(PY) $(REPO)/verify/qec_system.py --only system
