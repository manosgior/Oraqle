# Top-level entry points. Override the interpreter with `make PY=/path/to/python`.
PY ?= python3

.PHONY: verify verify-discriminators verify-leakage verify-qec verify-figures figures help

help:
	@echo "make verify          check every paper number from results/ (no dataset needed)"
	@echo "make verify-figures  also re-render figures and compare with the paper PDFs"
	@echo "make figures         regenerate all data figures/tables into \$$ORAQLE_OUT (default out/)"

verify: verify-discriminators verify-leakage verify-qec

verify-discriminators:
	$(PY) verify/discriminators.py

verify-leakage:
	$(PY) verify/leakage.py

verify-qec:
	$(PY) verify/qec_system.py

verify-figures:
	$(PY) verify/leakage.py --figure
	$(PY) verify/qec_system.py --figures

figures:
	$(PY) paper_figures/discriminators/make_all.py
	$(PY) paper_figures/leakage/make_error_modes_table.py
	$(PY) paper_figures/leakage/make_iq_taxonomy.py
	$(MAKE) -C paper_figures/system all PY=$(PY)
	$(MAKE) -C paper_figures/qec all PY=$(PY)
