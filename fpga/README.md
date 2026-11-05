# fpga/ — FPGA cost of the deployable discriminators (tab:fpga_resources)

`tab:fpga_resources` (chapters/ml_discriminators.tex:275-293) and the FPGA claims in the
prose (ml_discriminators.tex:297: "≤74 ns", "~70% of LUTs/DSPs") come from Vitis HLS
synthesis reports of hls4ml models. The hls4ml scripts and reports were written by
**Felix Gust** (github.com/gustifix/mcmit-model, commit `56cfcc0`, 2026-06-15) and are
vendored here unchanged in `hls4ml/`.

## Verify (no Vitis, no dataset)

    python fpga/parse_reports.py                                  # per-report numbers + the paper rows
    python paper_figures/discriminators/tab_fpga_resources.py     # LaTeX rows (identical to the paper)
    python verify/discriminators.py                               # includes all 16 cells

All 16 cells reproduce exactly:

| Paper row | LUT / FF / DSP / latency | Composition (reports in `hls4ml/reports/`) | Model synthesized | Same as the Table 2 model? |
|---|---|---|---|---|
| QubiCML† | 3% / 3% / 5% / 38 ns | 5 × `qubic.txt` (2.68 / 3.16 / 5.27 %; 19 cycles × 2 ns) | `Arxiv240618807FNN`, 65 params; `checkpoints/QubiCML_q0_1000ns.pth` = `checkpoints/Arxiv_q0_len500_trial3.pth` (Table 2 QubiCML, Q1, 1000 ns) | yes (Q1 instance; one per qubit) |
| HERQULES | 6% / 9% / 17% / 38 ns | 1 × `herqules.txt` (5.98 / 8.63 / 16.78 %; 19 cycles) | `Net_rmf`, 1002 params; `HERQULES_1000ns.pth` = `checkpoints/HERQULES_len500.pth` | same architecture; this checkpoint is the June-2 per-trace model (`_train_and_analyze_errors.py`); the Table 2 sweep saved no checkpoint |
| MCMit CNN | 29% / 16% / 20% / 74 ns | 1 × `cnn-m8-parallelizationfactors.txt` (28.81 / 16.37 / 19.85 %; 37 cycles) | `CNN(m_param=8)`, **957 params**, ds40 input (10, 13), single head; `CNN_small_m8_ds40_1000ns_singlehead.pth` (made by `hls_input_prep/_cnn_small_hls.py`) | **no**: Table 2's MCMit-CNN is m_param=16, 3061 params. The m16 model is `cnn-m16.txt`: 65% / 54% / 47%, 95-99 cycles (198 ns) |
| KLiNQ‡ | 70% / 49% / 69% / 42 ns | 3 × `klinq.txt` + 2 × `klinq-larger-reuse4.txt` (70.24 / 48.61 / 69.01 %; latency = slowest instance, 21 cycles) | 705-param student (= `checkpoints/KLiNQ_student_paper_q0_1000ns.pth`) on Q1/Q4/Q5 and 3425-param student (= `KLiNQ_student_paper_q2_1000ns.pth`) on Q2/Q3, ReuseFactor 4 for the larger one | yes (8965 params in total, Table 2 KLiNQ) |

Notes and discrepancies (also in docs/reproduce/discriminators.md):

* The numbers are **Vitis HLS 2023.2 C-synthesis estimates** (part `xczu49dr-ffvf1760-2-e`,
  2.00 ns target clock = 500 MHz; estimated clock 1.38-1.60 ns), not post-place-and-route
  results. ml_discriminators.tex:259 calls them "measured".
* Only the neural network is synthesized. The demodulation / matched-filter / FIR
  front-ends of every model are not part of the hls4ml graph (`hls4ml/inputs/README.txt`).
* The MCMit CNN row is the smaller m_param=8 variant, not the 3061-parameter model whose
  fidelity is in tab:fidelity (its own fidelity: F5Q ≈ 0.908 per `_cnn_shallow_hls.py`'s
  docstring, vs 0.910 for m16; not stored as a result file). The user asked to ignore the
  discrepancy between this table's CNN LUT share and the 0.33% LUT quoted in the MICRO paper.
* "The Baseline FNN and MCMit-T do not fit on the ZCU216" has no report behind it
  (no synthesis of either model exists): recorded as unverifiable in paper_values.json.
* Variants whose exact run script was not committed in mcmit-model (edit the vendored
  script, then run it):
  * `klinq-larger-reuse4.txt` (used in the paper): `run-klinq-larger.py` plus
    `config['Model']['ReuseFactor'] = 4` after the `ClockPeriod` line (the report shows II = 4).
  * `cnn-m8-reuse2.txt` / `cnn-m8-reuse4.txt`: `run-cnn.py` with the commented
    `config['Model']['ReuseFactor']` line set to 2 / 4 and without the ParallelizationFactor lines.
  * `cnn-m8.txt`: `run-cnn.py` as of mcmit-model commit `fc054b6` (no ParallelizationFactor).
  * `cnn-m16.txt`: `run-cnn.py` as of commit `0d016af` (`CNN(m_param=16)`, `CNN_1000ns_singlehead.pth`, ds20 (10, 25) input; `networks/CNN.py` at HEAD hard-codes the m=8 head).
  `cnn-m8-parallelizationfactors.txt`, `qubic.txt`, `herqules.txt`, `klinq.txt` and
  `klinq-larger.txt` come from the vendored scripts as they are.

## Regenerate the reports (needs Vitis HLS 2023.2)

    cd fpga/hls4ml
    pip install -r requirements.txt        # hls4ml 1.3.0, torch 2.12 (Felix's environment; pinned)
    python run-qubic.py                    # -> hls-project-qubic/  (gitignored)
    python run-herqules.py
    python run-cnn.py                      # m8 + ParallelizationFactor = the paper's CNN row
    python run-klinq.py
    python run-klinq-larger.py             # add ReuseFactor 4 for the paper's variant (see above)

The scripts use paths relative to `fpga/hls4ml/` (checkpoints/, inputs/) and were not
modified. `hls4ml.report.read_vivado_report()` prints the report that is stored in
`reports/`.

## Layout

| Path | What |
|---|---|
| `parse_reports.py` | parses the reports and composes the paper rows (new) |
| `hls4ml/run-*.py`, `hls4ml/networks/` | Felix Gust's hls4ml conversion scripts and the PyTorch model classes they load (copies of discriminators/networks with a single-head CNN) |
| `hls4ml/checkpoints/` | the 7 checkpoints that were synthesized (104 KB) |
| `hls4ml/inputs/` | README.txt only; the 512-shot example inputs are not shipped (see "Data note" below) |
| `hls4ml/reports/` | the 10 Vitis HLS reports (the verifiable artifact) |
| `hls4ml/micro-submission/` | Felix's earlier Keras/HGQ CNN scripts for the MICRO submission; not used for this table (data path plumbed to `$ORAQLE_WORK/cnn`) |
| `hls4ml/requirements.txt` | Felix's pinned environment |
| `hls_input_prep/` | Oraqle runners that produced the checkpoints/inputs handed to Felix: `_make_example_inputs.py` (QubiCML / KLiNQ / ds20 CNN / HERQULES inputs + reference outputs), `_make_klinq_large_input.py` (larger KLiNQ student), `_cnn_small_hls.py` (trains the m8/ds40 CNN and writes its single-head checkpoint + inputs), `_cnn_shallow_hls.py` (shallower m8 variants; not synthesized in `reports/`). They need the raw dataset; outputs go to `$ORAQLE_OUT/fpga/` |

Data note: the 512-shot example inputs/outputs/labels (`hls4ml/inputs/*.npy`) are NOT shipped,
because they are preprocessed slices of the non-public Lienhard et al. test set. The reports and the
table do not depend on them. Regenerate them with `hls_input_prep/_make_example_inputs.py` (and
`_make_klinq_large_input.py`, `_cnn_small_hls.py`) once ORAQLE_DATA is set, then copy them into
`hls4ml/inputs/` before running the `run-*.py` scripts.
