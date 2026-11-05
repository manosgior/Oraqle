# Reproducing the readout and discriminator results (paper §3 Stage 1-2, §4, §5.1-5.2, App. HERQULES / linear baseline)

Scope: abstract, intro, background, motivation, overview, qubit_readout, ml_discriminators
(except tab:error_modes, fig:iq_taxonomy and the §5.3 error-mode numbers, see the leakage
part) and the HERQULES/linear appendix tables. Paper source: oraqle-ics working tree of
2026-11-05 (HEAD e2ba1ac + the uncommitted `[audit fix]` chapter edits of the same day); line numbers below refer
to it (the `% [audit fix] original:` comment lines are counted).

## 1. Quick start

Without the dataset (everything below runs in ~30 s with the repository's Python
environment; `Discriminators/.venv` on the author's laptop: Python 3.13, numpy 2.4, torch 2.11,
matplotlib 3.10.9; also tested in the `oraqle-torch` container, Python 3.11, numpy 2.1.3):

    python verify/discriminators.py [--paper-root ../oraqle-ics]          # 368 values; exit 1 on mismatch
    python paper_figures/discriminators/make_all.py [--paper-root ../oraqle-ics]
    python discriminators/tables/build_table2.py --check                  # master_fidelity.csv byte-identical
    python fpga/parse_reports.py                                          # tab:fpga_resources

With the dataset (`export ORAQLE_DATA=...`, see dataset/README.md):

    python discriminators/smoke/smoke_table2.py all       # every producer, tiny config, minutes
    # full runs: section 4 below

Outputs go to `$ORAQLE_OUT` (default `out/`) and `$ORAQLE_WORK` (default `work/`), never into
`results/` or `checkpoints/`.

## 2. Status

* **Table 2 (tab:fidelity) is reproducible from the per-model result CSVs.**
  `discriminators/tables/build_table2.py` rebuilds `results/discriminators/master_fidelity.csv`
  **byte for byte** (all 350 cells, incl. the Linear Threshold rows; all 180 cells of Table 2)
  and emits the LaTeX table. Diff against ml_discriminators.tex:154-188: **0 value
  differences, 0 highlighting differences**. The highlighting rule (`build_table2.flags()`):
  bold = best printed (3-decimal) value per model, the shortest duration on ties; shading =
  best printed value per column over all rows, a tie at printed precision goes to MCMit-CNN
  (Table 2 Q5: MCMit-CNN 600 ns 0.9700 is shaded, tied at 0.970 with MCMit-T 800 ns 0.97039
  and MCMit-CNN 1000 ns). The same bold rule gives tab:fidelity_linear's F4Q bold at 800 ns
  (0.941). `verify/discriminators.py --paper-root` and `build_table2.py --check --paper-tex
  ... --paper-appendix ...` fail on any bold/shading difference.
  How the published file came about: build_master_and_table.py (2026-06-01) wrote it; then
  the HERQULES and MCMit-CNN rows were replaced by hand (2026-06-13/22) with
  `herqules_length_faithful_vs_demux.csv` (variant `deployment_pertrace`) and
  `cnn_length_deployment.csv`, copying the F5Q column from those files (for MCMit-CNN at 600
  and 1000 ns it differs by 1e-4 from the gmean of the rounded Q values). No script for the
  merge or for the `cnn_length_deployment.csv` aggregation survived; build_table2.py now
  encodes both (the aggregate = the per-length `cnn_length_<ns>ns.csv` files from
  `_cnn_length_sweep.py` + `gmean4_no_q2` from the rounded Q1, Q3, Q4, Q5) and reproduces both
  files exactly. The original builder (`tables/build_master_and_table.py`) still reproduces its
  stale `fidelity_table.tex` byte for byte; its HERQULES rows come from
  `HERQULES_len*_pertrace_*` and its CNN rows from the `CNN_m16` Optuna runs (= the CNN rows of
  `master_fidelity.csv.bak`).
* **tab:fpga_resources is reproducible from the Vitis HLS reports** (all 16 cells), see §6.
* **verify/discriminators.py**: 368 paper values checked, all agree (the 16 known discrepancies of
  the first audit were fixed in the paper on 2026-11-05, §5), 1 claim without an artifact
  ("Baseline FNN and MCMit-T do not fit on the ZCU216").
* **Smoke tests (tier 2)**: all 9 cases passed on graham (§7).
* **No model was retrained at full scale** (not requested). Re-running a producer gives
  statistically similar, not bit-identical numbers (unseeded Optuna samplers, GPU
  non-determinism; see the seed column below).

Verification tiers used below: **tier1** = recomputed from the committed results without
the dataset (verify/discriminators.py, figure/table scripts); **tier2** = the producer ran
end-to-end on the raw data from the new layout with a tiny config (smoke test); **not run**.

## 3. Table 2 producers

All rows were trained in May-June 2026 inside the `oraqle` Docker image (`docker/oraqle-cuda/Dockerfile`:
`nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04` + the **unpinned**
`discriminators/requirements.txt`; the image no longer exists on graham or steve, so the
library versions of those runs are unknown) on a CUDA GPU of the TUM cluster (graham/steve,
shared NFS home; `device=cuda` in the CSVs; `hyper_optimize_transformer.py` sizes its runs
for a 48 GB A40; today graham has no GPU driver and steve has an H200 NVL, so the exact GPU
is not recorded). Mounts: raw data at `/data/five_qubit_data` (now `ORAQLE_DATA`), reports
at `/app/optimization_reports` (now `$ORAQLE_OUT/discriminators/reports`), models at
`/app/saved_models` (now `$ORAQLE_WORK/checkpoints`). Runtimes are inferred from output
timestamps (UTC).

| Row | Producer and exact command (new layout) | Protocol | Seeds | Ran | Result files |
|---|---|---|---|---|---|
| Baseline FNN | `python discriminators/runners/hyper_optimize_paper.py --models FNN` | Optuna, **10 trials × 25 epochs**, lengths 50..500 samples, **random subsample 320k train-file / 80k test-file shots** (`MAX_TOTAL_SAMPLES=400_000`), per-qubit accuracy on the 80k test subset | subsample `SAMPLE_SEED=42`, `train_test_split(random_state=42)`; Optuna and torch unseeded | 2026-05-06 08:07 → 05-12 14:14 (~6.3 days, 10 lengths) | `FNN_len*.csv`, `checkpoints/FNN_len*_trial*.pth` |
| QubiCML | `python discriminators/runners/hyper_optimize_paper.py --models Arxiv240618807FNN` | same protocol; one 65-param model per qubit on the demodulated, time-averaged (I, Q) point | as above | 2026-05-13 14:06 → 05-14 12:41 (~23 h) | `Arxiv240618807FNN_q*_len*.csv`, `checkpoints/Arxiv_q*_len*_trial*.pth` |
| HERQULES | `cd $ORAQLE_WORK && python $REPO/discriminators/runners/_herqules_length_both.py` (`HERQULES_TRACE_LENGTH` default 500) | per-trace (deployment-faithful) MF/RMF features, BOX=[1,1,9,2,9]; train+test files merged, per state 20,000 train+val / 30,000 test, val ratio 0.35; Net_rmf (1002 params), 100 epochs, retrained per length; no checkpoint | `train_eval(seed=1)`; HERQULES_original `SAMPLE_SEED=42` | output 2026-06-12 22:11; runtime not recorded | `herqules_length_faithful_vs_demux.csv` (rows `deployment_pertrace`) |
| KLiNQ | `python discriminators/runners/reproduce_klinq_paper.py` | paper recipe, no Optuna: per-qubit teacher [1000→500→250→1] on the 1 µs trace (300 epochs), student FNN-A (705 params; Q1/Q4/Q5) / FNN-B (3425; Q2/Q3), KD "T4-ST2-CE8", 50 epochs, full train file (10% val), full test file; assignment fidelity | `SEED=42` (torch, numpy, splits) | run `20260601_141220` (≤1.5 h; the script's mtime is 15:23, after the run, so it may have been edited afterwards: PROBABLE) | `KLiNQ_paper_repro[_F5Q]_20260601_141220.csv`, `checkpoints/KLiNQ_student_paper_q*_*ns.pth` |
| MCMit-T | `python discriminators/runners/hyper_optimize_transformer.py` | Optuna (TPE), **50 trials × 30 epochs**, 4 parallel trials, lengths 50..500, **full train file** (z-scored, MF-hard-trace removal searched), full test file, per-qubit accuracy | `TPESampler(seed=42)`, `SAMPLE_SEED=42`; cudnn.benchmark (non-deterministic) | 2026-05-21 12:15 → 05-29 10:58 (~8 days) | `Transformer_len*.csv`, `checkpoints/Transformer_len*_trial*.pth` |
| MCMit-CNN | `CNN_LEN=<100,200,300,400,500> python discriminators/runners/_cnn_length_sweep.py` (one length per process, can run in parallel) | `networks/CNN.py` m_param=16 (3061 params), colleague preprocessing (`_colleague_prep.py`: FFT frequency/phase calibration on 500 samples, windowed-sinc FIR, ↓40), full train file (80/20 split) and full test file, 30 epochs, lr 5e-4; assignment fidelity; retrained per length; no checkpoint | `torch.manual_seed(1)`, `np.random.seed(1)`, split `random_state=42` | 2026-06-05 (files 17:38-18:29 UTC) | `cnn_length_<ns>ns.csv` → `cnn_length_deployment.csv` (aggregated by build_table2.py) |
| (Linear Threshold, App. tab:fidelity_linear) | `python readout/run_baseline_sweeps.py` | demodulate, trapezoid-integrate to T, standardize, logistic regression refit at each T; train+test files merged and shuffled | `SAMPLE_SEED=42`; `MAX_SAMPLES` of the May-14 runs not recorded (file default None) | 2026-05-14 12:58-20:44 (six partial runs) | `Baseline_Threshold_*.csv` |

**Mixed protocols (paper caveat).** The six rows were not produced under one protocol: the
Baseline FNN and QubiCML rows come from a 10-trial Optuna search on a 400k-shot subsample
(evaluated on 80k test shots), MCMit-T from a 50-trial search on the full training file, KLiNQ
from the fixed published recipe, and MCMit-CNN and HERQULES from fixed recipes on the full
data (HERQULES with its own per-state 40/60 split of the merged files). FNN/QubiCML/MCMit-T
report per-qubit accuracy, the others balanced assignment fidelity (nearly equal here because
the classes are balanced).

**hyper_optimize.py versions.** `runners/hyper_optimize.py` is the last cluster version
(2026-06-01; constants set for a KLiNQ-student run: `TRACE_LENGTHS=[500]`, 15 trials, 30
epochs, full data). The FNN/QubiCML objectives are byte-identical to Oraqle git `b41eeb7`;
the CSVs record `n_optuna_trials=10`, `epochs=25`, which is the configuration of `7d3b10a`
(b41eeb7 itself says 30 trials). `runners/hyper_optimize_paper.py` sets exactly those
constants; it changes no code.

## 4. Paper item → command

| Paper item (label, chapter:line) | Command | Output | Dataset? | Runtime | Verified |
|---|---|---|---|---|---|
| Readout latency/error claims (intro.tex:25, background.tex:34) | cited (Yang2022Fpga, Chen2023Transmon, ...); tab:readout-errors is in the appendix part | - | - | - | n/a (citations) |
| fig:qec_pipeline, fig:axes, fig:background, fig:oraqle_overview, tab:model_comparison diagrams, tab:scope, tab:research_questions, fig:footprint | hand-made (TikZ / drawing; sources in the paper repo; fig:background has no editable source) | - | - | - | n/a |
| tab:model_comparison "Est. Parameters" (overview.tex:31-35), overview.tex:68 | `python paper_figures/discriminators/tab_model_comparison_params.py` | `tab_model_comparison_params.csv` | no | <1 s | tier1 (10 values) |
| Dataset facts: 1.6 M shots, 500 MS/s, 2 µs, 32 × 50,000 (overview.tex:140) | `results/discriminators/dataset_metadata.json` (probed on graham) | - | to re-probe: yes | - | tier1 (5 values) |
| fig:readout_sweeps (qubit_readout.tex:8) | `python paper_figures/discriminators/fig_readout_sweeps.py`; data: `python readout/run_baseline_sweeps.py` | `readout_sweeps.pdf` | figure: no; data: yes | <5 s; data ~hours (full set needs ~130 GB RAM) | tier1 (pixel-identical), producer tier2 |
| §4 prose numbers (qubit_readout.tex:11, 56, 60, 134, 138) | `python verify/discriminators.py -v` | - | no | <5 s | tier1: 34 values, all agree (11 fixed in the paper, §5) |
| fig:iq_clouds (qubit_readout.tex:35) | `python paper_figures/discriminators/fig_iq_clouds.py`; extract: `python readout/iq_clouds_vs_length.py` | `iq_clouds_vs_length_q5.pdf` | plot: no; extract: yes | <5 s; extract: minutes, ~30 GB RAM | tier1 (same content; font), extract tier2 |
| tab:fidelity (ml_discriminators.tex:145-191) | `python paper_figures/discriminators/tab_fidelity.py --paper-root ../oraqle-ics`; producers: §3 | `tables/master_fidelity.csv`, `tab_fidelity.tex` | table: no; producers: yes | <2 s; producers hours-days on a GPU | tier1 (180/180 values), producers tier2 |
| §5.1 prose (ml_discriminators.tex:85, 91, 95) | `python verify/discriminators.py -v` | - | no | <5 s | tier1: 13 values, all agree |
| fig:pareto (ml_discriminators.tex:264) | `python paper_figures/discriminators/fig_pareto.py` | `pareto_params_vs_fidelity_600ns_1000ns.pdf` | no | <5 s | tier1 (same content; font) |
| §5.2 prose (ml_discriminators.tex:273, 301, 303) | `python verify/discriminators.py -v` | - | no | <5 s | tier1: 12 values (incl. 2 FPGA claims), all agree, 1 without artifact |
| tab:fpga_resources (ml_discriminators.tex:277-296) | `python fpga/parse_reports.py`; `python paper_figures/discriminators/tab_fpga_resources.py`; synthesis: `fpga/hls4ml/run-*.py` | `tab_fpga_resources.tex` | no | <1 s; synthesis needs Vitis HLS 2023.2 | tier1 (16/16 cells, from the HLS reports); synthesis not run |
| §5.3, tab:error_modes, fig:iq_taxonomy | leakage part; inputs here: `results/discriminators/{preds_*.npy, rmf_len*.pkl, iq_error_gallery_q5.npz}`, `checkpoints/HERQULES_len{200,500}.pth`, producers `runners/_xtalk_cnn.py`, `_xtalk_herqules_deploy.py`, `_train_and_analyze_errors.py`, `iq_error_gallery.py` | - | yes | - | not run here |
| App. seed-to-seed variance ≈0.0002 (appendix_herqules.tex:79) | data: `runners/_herqules_full.py` | `herqules_full_dataset.csv` | data: yes | - | tier1 (value), producer tier2 |
| App. 21% / 31-35% / 0.794 vs 0.702 (appendix_herqules.tex:137-138) | `python verify/discriminators.py -v` | - | no | <5 s | tier1 (4 values) |
| App. 0.003% (appendix_herqules.tex:130), Q3 0.924 / Q5 0.970 / 0.917 / 0.967 (:141-142) | leakage part (`em_e3herq_summary_v3.json`) | - | - | - | not covered here |
| tab:herqules-pertrace (appendix_herqules.tex:147-174) | `python paper_figures/discriminators/tab_herqules_pertrace.py`; data: `_herqules_length_both.py` | `tab_herqules_pertrace.tex` | table: no | <1 s | tier1 (20/20), producer tier2 |
| tab:herqules-noretrain + prose (appendix_herqules.tex:190-217) | `python paper_figures/discriminators/tab_herqules_noretrain.py`; data: `runners/_herqules_truncation_pertrace.py` | `tab_herqules_noretrain.tex` | table: no | <1 s | tier1 (39 values, incl. "within 0.002"), producer tier2 |
| tab:fidelity_linear (appendix_herqules.tex:228-246) | `python paper_figures/discriminators/tab_fidelity.py`; data: `readout/run_baseline_sweeps.py` | `tab_fidelity_linear.tex` | table: no | <1 s | tier1 (35 cells + bold pattern), producer tier2 |

## 5. Known discrepancies: none open (all fixed in the paper, 2026-11-05)

The 16 paper-vs-data discrepancies of the first audit were corrected in the paper (`[audit fix]` edits); the
entries in paper_values.json now carry the new text and status `match`:

| Where (current line) | Paper before | Paper now (= data) |
|---|---|---|
| qubit_readout.tex:11, :60 | dF/dT "turns negative beyond ≈1 µs", "crosses zero between 0.8 and 1.0 µs" | "at ≈0.7-1.7 µs", "between 0.7 (Q3) and 1.7 µs (Q2)"; Q3/Q5 "beyond 0.7-0.9 µs" |
| qubit_readout.tex:56 | "T ∈ [0.1, 2.0] µs", gmean "~0.86 at 0.3 µs", Q3/Q5 "peak around 0.7-1.0 µs (~0.90, ~0.96)", "up to ~40%" | "[0.1, 1.9]", "~0.82", "0.5-0.6 µs (~0.91, ~0.97)", "up to ~50% by T=1.9 µs" |
| qubit_readout.tex:134, :138 | "Q1 and Q5 strongest", "~21% below" | "Q1 and Q5 mostly strongest" (72% / 80% of positions), "~25%" |
| ml_discriminators.tex:273 | "five orders of magnitude" | "six" (10^5.98) |
| appendix_herqules.tex:240 | Linear 800 ns Q5 0.936, F4Q 0.934 (bold F4Q at 600 ns) | 0.963, **0.941** (bold moved to 800 ns) |
| appendix_herqules.tex:191, :210 | no-retrain 1000 ns row Q3 0.924, Q4 0.944, "reproduces ... exactly" | 0.925, 0.942 (the freeze run), "within 0.002" (max 0.0017) |

Notes on values that agree but need context: "seed-to-seed variance ≈0.0002"
(appendix_herqules.tex:79) is the **standard deviation** of F5Q over 5 seeds, measured with the
released (demux) assembly (F5Q 0.925), not with the per-trace model; "31-35%" uses the table
rows 600/400/200 ns (the CSV's 250 ns row gives 40.8%); "~10% more error" (ml:91, :95) is the
mean over the six models (+3.9% QubiCML to +12.1% MCMit-CNN); "Q2 ~0.69-0.75 for every model"
(ml:95) holds for 400-1000 ns; "100-1000× weights" (ml:301) = 208× (FNN) and 1559× (MCMit-T)
over MCMit-CNN. ml_discriminators.tex:260 and the tab:fpga_resources caption now call the FPGA
numbers HLS estimates (they said "measured" before the 2026-11-05 fix). Two half-way cells are rounded differently in the paper
(tab:herqules-pertrace 200 ns per-trace F4Q 0.7595 → printed 0.759; tab:herqules-noretrain
600 ns F5Q 0.7745 → printed 0.775); the verifier accepts both, the table scripts round half-up.

## 6. tab:fpga_resources

Source: Felix Gust's hls4ml/Vitis HLS flow, vendored in `fpga/hls4ml/` from
github.com/gustifix/mcmit-model @ `56cfcc0` (see fpga/README.md, fpga/PROVENANCE.md).
Vitis HLS 2023.2, part xczu49dr-ffvf1760-2-e (425,280 LUT / 850,560 FF / 4,272 DSP), 2 ns clock.

| Row | Paper | From reports | Composition |
|---|---|---|---|
| QubiCML† | 3 / 3 / 5 %, 38 ns | 2.68 / 3.16 / 5.27 %, 19 cycles × 2 ns | 5 × `qubic.txt` |
| HERQULES | 6 / 9 / 17 %, 38 ns | 5.98 / 8.63 / 16.78 %, 19 cycles | `herqules.txt` |
| MCMit CNN | 29 / 16 / 20 %, 74 ns | 28.81 / 16.37 / 19.85 %, 37 cycles | `cnn-m8-parallelizationfactors.txt` |
| KLiNQ‡ | 70 / 49 / 69 %, 42 ns | 70.24 / 48.61 / 69.01 %, 21 cycles (slowest instance) | 3 × `klinq.txt` + 2 × `klinq-larger-reuse4.txt` |

The plain single reports do not match QubiCML and KLiNQ because the table counts all five
per-qubit instances (footnotes † and ‡). **The synthesized CNN is not the Table 2 model**: it
is the m_param=8 variant with 957 parameters on ds40 input (`_cnn_small_hls.py`), while
Table 2's MCMit-CNN has m_param=16 and 3061 parameters; the m16 synthesis (`cnn-m16.txt`,
from the Optuna checkpoint `CNN_m16_len500_trial36.pth`) needs 65% LUT / 54% FF / 47% DSP and
198 ns. The user asked to ignore the gap to the 0.33% CNN LUT figure of the MICRO paper.
QubiCML, KLiNQ (both students) and HERQULES used the published checkpoints
(`Arxiv_q0_len500_trial3`, `KLiNQ_student_paper_q{0,2}_1000ns`, `HERQULES_len500`).
The run script of the `klinq-larger-reuse4` variant was never committed (fpga/README.md
lists the one-line edit).

## 7. Smoke tests (tier 2), 2026-11-05 on graham

Setup: my part of the tree rsynced to `graham:~/oraqle-unify-smoke/discriminators/repo`;
image `oraqle-torch` (python 3.11-slim, torch 2.5.1 CPU, numpy 2.1.3, h5py, scipy, sklearn)
plus `pip install --user loguru optuna pandas matplotlib seaborn` into a mounted user site
(optuna 5.0.0, pandas 3.1.1, matplotlib 3.11.2, seaborn 0.13.2, loguru 0.7.3); raw data
mounted read-only. Command per case (`run_case.sh`):

    docker run -d --name unify-disc-<case> --user $(id -u):$(id -g) --cpus 16 \
      -v ~/qubit_readout_klinq/data/five_qubit_data:/data:ro -v ~/oraqle-unify-smoke/discriminators:/smoke \
      -e PYTHONUSERBASE=/smoke/pyuser -e OMP_NUM_THREADS=16 -e ORAQLE_DATA=/data \
      -e ORAQLE_OUT=/smoke/out/<case> -e ORAQLE_WORK=/smoke/work/<case> oraqle-torch \
      python /smoke/repo/discriminators/smoke/smoke_table2.py <case>

`smoke_table2.py` leaves the repository code unchanged and shrinks each run in memory (module
constants, or exact-string substitutions that must match exactly once):

| Case | Entry point | Shrinking | Outcome (exit 0 for all) |
|---|---|---|---|
| fnn_qubicml | `runners/hyper_optimize_paper.py` | length 100, 1 trial, 1 epoch, 8000-shot subsample | 5 s; `FNN_len100_*.csv` + 5 `Arxiv240618807FNN_q*_len100_*.csv` |
| transformer | `runners/hyper_optimize_transformer.py` | length 100, 1 trial, 1 epoch, 6000/3000 shots | 6 s; `Transformer_len100_*.csv` |
| klinq | `runners/reproduce_klinq_paper.py` | 200 ns only, 1+1 epochs, every 40th/80th shot | 9 s; `KLiNQ_paper_repro[_F5Q]_*.csv` |
| herqules | `runners/_herqules_length_both.py` | 10% subsample at import, 1500/3000 per state, 200 ns, 1 epoch | 11 s; `herqules_length_faithful_vs_demux.csv` (first attempt failed: a `[unify]` comment had swallowed `BOX = ...`; fixed in 4 runners, AST check of all edited files added) |
| cnn | `runners/_cnn_length_sweep.py` | CNN_LEN=100, 1 epoch, **full data** | 60 s; `cnn_length_200ns.csv` (F5Q 0.7625 vs 0.7638 published after 30 epochs) |
| baseline | `readout/run_baseline_sweeps.py` | 4000 shots per file | 8 s; 19 `Baseline_Threshold_len*`, dF/dT and both sliding-window CSVs |
| herqules_noretrain | `runners/_herqules_truncation_pertrace.py` | subsample, 400/1000 ns, 1 epoch | 38 s; `herqules_truncation_pertrace_deployment.csv` |
| herqules_full | `runners/_herqules_full.py` | subsample, 2 seeds, 1 epoch | 18 s; `herqules_full_dataset.csv` |
| iq_clouds | `readout/iq_clouds_vs_length.py` | subsample, 500 shots per state | 21 s; npz + pdf + png |

`verify/discriminators.py` and `paper_figures/discriminators/make_all.py` also pass inside
the same container. Logs: `graham:~/oraqle-unify-smoke/discriminators/logs/`.
Not smoke-tested: the leakage-input producers (`_xtalk_cnn.py`, `_xtalk_herqules_deploy.py`,
`_train_and_analyze_errors.py`, `iq_error_gallery.py` extract), the FPGA input preparation
(`fpga/hls_input_prep/`), `helpers/extract_data*.py` and everything in `extras/`.

## 8. Remaining gaps

* Full-scale retraining was not repeated; library versions of the original `oraqle` image are
  unknown (unpinned requirements), and Optuna/torch seeding makes FNN/QubiCML/MCMit-T reruns
  non-deterministic.
* `reproduce_klinq_paper.py` was modified (mtime) about an hour after the run that produced
  the KLiNQ CSVs; the code may differ slightly from the producing version.
* `MAX_SAMPLES` of the May-14 baseline runs is not recorded.
* The exact GPU of the May-June runs is not recorded.
* "Baseline FNN and MCMit-T do not fit on the ZCU216" has no synthesis report.
* Open paper issues in this part: the "do not fit on the ZCU216" claim (no report) and the CNN
  synthesized for tab:fpga_resources being the m8 variant, not the Table 2 m16 model (§6).
