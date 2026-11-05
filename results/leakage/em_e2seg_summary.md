# em_e2seg: HMM-free segment-based error-mode attribution of the canonical MCMit-CNN errors

Tag `e2seg`, 2026-11-05.
- Scripts: `sims/leakage/em_e2seg_attr_v3.py` (detector and attribution), `em_e2seg_assemble.py` (tables), `em_e2seg_oldmethod.py` and `em_e2seg_oldmethod_l200scan.py` (old procedure).
- Results: `data/leakage/em_e2seg_results.json`.
- Raw outputs: `em_e2seg_v3_raw_main.json` (and `em_e2seg_v2_raw_main.json`, which contains the bug described below), `em_e2seg_oldmethod*.json`.

Predictions and windows:
- **1 us:** canonical `preds_MCMitCNN_pb.npy`, with true bits equal to the prepared bits (verified).
- **600 / 400 ns:** soft models `cnn_soft_{600,400}ns.npz`, hard decision P > 0.5. These reproduce the paper's MCMit-CNN Tab. fidelity errors exactly:
  - 600 ns: 4.71 / 27.48 / 6.16 / 6.31 / 2.99 %.
  - 400 ns: 9.32 / 30.28 / 7.36 / 9.88 / 4.87 %.

All CIs are 95%:
- error rates: Wilson;
- mirror- or FP-subtracted counts: ±1.96 sqrt(N+ + N−)/N, or ±1.96 sqrt(flags + MC var of the FP sum)/N.

## Method (MEASURED unless marked INFERRED)

**Mode 1: leakage.** I built my own line frame per qubit and per neighbour configuration (16):
- c0 and c1 are mean-shift modes of the window means on TRAIN.
- z is the perpendicular distance from the c0–c1 line, in units of the prepared-|0> MAD width.
- The flag is z > 3, minus the mirror count z < −3.
- It cross-checks against `f_cnn_leak_xtab`:
  - E3 within 0.04 pp on every qubit and window;
  - leaked share of canonical errors: Q3 3.144 % vs 3.136 %, Q5 1.034 % vs 1.043 %, pooled Q1Q2Q3Q5 0.814 % vs 0.81 %.
- Q4 is n/m (multimodal |1>).

**Mode 2: mid-readout transition.** This is a two-segment GLRT step test.

How the detector works:
- Per configuration, I build per-bin core trajectories mu0(k) and mu1(k) from 100 ns boxcar bins.
- The core shots are selected on the *independent* 1–2 us window. In v3 this selection is seeded from the 1 us core.
- Each trace becomes a normalised coordinate x_k, which is 0 at the |0> core and 1 at the |1> core in every bin.
- The noise covariance is a robust GK covariance from the core shots.
- The step template includes the fitted resonator ring-up.
- The test statistic is LLR = max over the step time t_s of the step log-likelihood, minus the best stationary level.
- Relaxation is tested on prepared-|1> errors, excitation on prepared-|0> errors.

Null and thresholds:
- The null is "no transition": stationary Gaussian bin noise with the measured covariance, simulated exactly *conditional on the shot's matched-filter level m*.
- Thresholds come from one bank; FP rates come from an independent bank. I take the max over the two stationary branches, which is conservative.
- FP-corrected count = flags − Σ FP_i over the pool, at α = 0.01.

Real-data checks of the null:
- TEST shots in the |b> core of the 1–2 us window (observed/expected flags).
- Real prepared-|0> core carriers.

Sensitivity:
- Steps are injected into real carriers at t_s = 0…1000 ns.

Gating: a transition number is reported only if two conditions hold. Otherwise it is n/m and the errors stay in the residual.
1. The quasi-null observed/expected ratio is in [0.6, 1.3].
2. The efficiency given a misread by the CNN proxy is at least 5 %.

**Mode 3: residual (no detected event).** I characterise it independently against two boundaries:
- the distance to a CNN-like boundary (a per-config linear fit of the soft-CNN logit, in noise widths);
- the near-Bayes stationary-Gaussian error count of the per-config GLS matched filter. This bounds what pure overlap can explain for *any* classifier.

**Attribution order.** A = 1 → 2 → 3 (primary); B = 2 → 1 → 3. Overlaps are reported. No mode is true by construction.

## 1 us breakdown (order A, % of that qubit's errors)

| Qubit | Error rate | Leakage | Mid-readout transition | Residual (no detected event) | Overlap explains at most |
|---|---|---|---|---|---|
| Q1 | 2.98 [2.95, 3.01] | 0.04 [−0.02, 0.11] | **4.33 [4.08, 4.57]** (relax. 5.05 of p1-err; excit. 0.95 of p0-err) | 95.63 [95.37, 95.88] | 11 % of p1-err, 41 % of p0-err |
| Q2 | 25.43 [25.35, 25.51] | 0.34 [0.32, 0.37] | n/m (efficiency ≈1 %) | 99.66 | all (Gaussian predicts ≥ the observed count) |
| Q3 | 5.84 [5.80, 5.89] | **3.14 [3.00, 3.29]** | 1.75 [1.65, 1.86] partial: excit. **9.45 [8.88, 10.03] of p0-err**; relax. n/m (null fails, obs/exp 2.48) | 95.10 [94.93, 95.28] | 2 % |
| Q4 | 5.31 [5.27, 5.36] | n/m | n/m (two-level null invalid) | 100 (unattributed) | n/m |
| Q5 | 2.96 [2.93, 2.99] | **1.03 [0.90, 1.17]** | **5.02 [4.76, 5.29]** (relax. 4.44 of p1-err; excit. 8.47 of p0-err) | 93.94 [93.65, 94.24] | <0.1 % (5 of 33,162) |
| Pooled, all 5 | 8.51 | 0.71 [0.68, 0.74] | 0.89 [0.86, 0.92] (partial) | 98.39 | |
| Pooled Q1Q3Q4Q5 | 4.27 | 1.26 [1.21, 1.32] | 2.22 [2.15, 2.29] (partial) | 96.52 | |

As % of shots:

| Qubit | Leakage | Transition |
|---|---|---|
| Q1 | 0.001 | 0.129 [0.122, 0.136] |
| Q3 | 0.184 [0.175, 0.192] | 0.102 |
| Q5 | 0.031 [0.027, 0.035] | 0.149 [0.141, 0.157] |
| Q2 | 0.087 | – |

**Hidden leakage (leaked shots as % of all shots; fraction of leaked shots read correctly):**

| Qubit | Leaked shots | Read correctly |
|---|---|---|
| Q1 | 0.41 % [0.39, 0.43] | 99.7 % |
| Q2 | 0.90 % [0.88, 0.92] | 90.3 % |
| Q3 | 6.14 % [6.10, 6.19] | 97.0 % (prep \|1>: 99.3 %; prep \|0>: 64.7 %) |
| Q5 | 2.40 % [2.37, 2.43] | 98.7 % |

**Efficiency-corrected transition shares (INFERRED; FP-corrected share ÷ injection efficiency, uniform t_s).** Each range is from the matched-filter misread efficiency to the CNN-proxy misread efficiency:

| Qubit | Errors | Corrected share |
|---|---|---|
| Q1 | prepared-\|1> | 22–27 % |
| Q5 | prepared-\|1> | 10–16 % |
| Q5 | prepared-\|0> | 19–48 % |
| Q3 | prepared-\|0> | 19–51 % |

**600 ns:**
- Q1: L 0.00, T 1.46 [1.32, 1.59].
- Q3: L 1.64 [1.54, 1.74], excitation 6.00 of p0-err; relaxation n/m (obs/exp 1.75).
- Q5: L 0.18 [0.08, 0.27], T 2.53 [2.33, 2.73].
- Q2 and Q4: n/m.
- Pooled Q1Q3Q4Q5: L 0.53, T 1.12.

**400 ns:**
- Q3: L 0.70 [0.63, 0.76], T 0.71 [0.62, 0.80].
- Q1 and Q5 transitions: n/m (efficiency 3–4 %).
- Q1 and Q5 leakage: ≈0.

**Residual character.**
- On Q1, Q3 and Q5 at 1 us the residual is **not** boundary overlap:
  - only 4–20 % of prepared-|1> residual errors lie within 1 σ of the CNN-like boundary (the Gaussian expectation is 95–97 %);
  - 56–84 % lie at least 2 σ beyond it.
- These errors are in the other state for most of the window. Possible causes are a state-preparation error, thermal population, or a transition before the window or in its first ~150 ns. The detector cannot resolve transitions that early: at t_s = 100 ns, p_flag is 0.05–0.18 (INFERRED identity).
- Q2 alone is overlap-dominated: 71–82 % of its residual is within 1 σ, and the Gaussian overlap count is at least the observed error count.
- Overlap matters more as the window shortens. At 400 ns it explains 91 % (Q1 p0), 66 % (Q1 p1) and 32 % (Q5 p1) of errors.

## Null and sensitivity (1 us, α = 0.01)

**Quasi-null observed/expected flags:**

| Qubit | Relaxation test | Excitation test |
|---|---|---|
| Q1 | 0.97 | 0.79 |
| Q2 | 1.03 | 0.80 |
| Q3 | **2.48** | 0.81 |
| Q4 | 1.81 | 0.00 |
| Q5 | 1.13 | 0.77 |

- Real |0>-core carriers flag 0.4–0.9 % of shots, against 1 % nominal.
- On calibrated qubits the max-branch FP is therefore conservative by about 0–25 %, so FP-corrected shares are lower bounds.
- Robustness of the FP correction:
  - selection-aware null (CNN-proxy-misread null traces) and prepared-branch FP: Q1/Q5 shares change by ≤ 0.5 pp;
  - Q5 relaxation stays at 4.4–4.6 %.
- Injection, P(flag) vs t_s (relaxation; excitation is similar):

  | Qubit | t_s = 100 ns | 300 ns | 500 ns | 700 ns | 900 ns |
  |---|---|---|---|---|---|
  | Q1 | 0.05 | 0.32 | 0.44 | 0.33 | 0.05 |
  | Q3 | 0.05 | 0.59 | 0.76 | 0.69 | 0.14 |
  | Q5 | 0.18 | 0.67 | 0.79 | 0.52 | 0.04 |
  | Q2 | ≤ 0.03 | | | | |

  Transitions in the first ~150 ns or the last ~150 ns are invisible.
- Efficiency given a CNN-proxy misread: Q1 0.19, Q3 0.18, Q5 0.28 (relaxation). At 600 ns it is 0.06–0.14; at 400 ns it is 0.03–0.08.

## Robustness

**Order swap.** The residual moves by ≤ 0.03 pp on every qubit.
- The L/T split moves only on Q3: L 3.14 → 1.70 and T 1.75 → 3.18. Mid-window departures from |0> into the off-axis state are both leaked and step-flagged; 7.7 % of Q3 prepared-|0> errors carry both labels.
- Q5 moves L 1.03 → 0.89 and T 5.02 → 5.16.

**α dependence (FP-corrected shares).** These shrink with α, at α = 0.05 / 0.01 / 0.002:
- Q1 relaxation: 11.8 / 5.1 / 1.7 %.
- Q5 relaxation: 11.6 / 4.4 / 1.3 %.

So the detected share is detector-dependent; only the FP-corrected lower bound at a stated α is MEASURED.

**Pipeline variants.** Q1/Q3/Q5 shares agree within ≤ 1.1 pp across:
- the FIR feature pipeline (`fir1us`);
- a step template without the resonator lag (`tau0`);
- soft1000 instead of the canonical predictions.

**v2 bug (fixed in v3).** The late-window |1> core of Q3 for Q1 = Q5 = 1 (configs 9/11/13/15) sat on the wrong cloud: σ_m1 was 0.83–1.0 instead of ~0.18, and those configs gave 0 flags. Only Q3 and Q4 numbers changed. The v2 outputs are kept.

## What the old numbers measured (`_train_and_analyze_hybridcnn.py` + `herqules_error_report.py`)

**Model.** HybridCNN, not MCMit-CNN, with sparse-FIR demodulation (DS = 40 at 1 us; the 400 ns checkpoint reproduces only with DS = 20).

**Rules**, applied in exclusive priority t1 > amp > near:
- **"Near-boundary":** P(correct) < 0.6 from the model's own sigmoid, which is true for every error.
- **"Relaxation (T1)":**
  - The Wiener envelope mean(x0 − x1)/var(x0 − x1) is projected separately onto the raw (un-centred) first and second halves of the window.
  - A shot is flagged if both |mf| > 0.5·median and the two signs differ.
  - Because nothing is centred, the sign is not a state indicator. On Q1 and Q3 both states project to the same sign, so the test can never fire. On Q5 it flags 9.5 % of *all* prepared-|0> shots.
- **"Leakage":** |z| > 3 on the time-mean |IQ|, with TRAIN mean/std pooled over states.

**Reproduction.**
- Re-running HybridCNN_len500.pth reproduces every CSV count to within 0.06 %.
- The old procedure applied to the **canonical MCMit-CNN** gives near / T1 / amp pooled = 97.41 / 2.44 / 0.16 % (paper: 97.4 / 2.4 / 0.2). Per qubit, T1 = 0.0 / 2.3 / 0.0 / 0.55 / 14.24 %, the same as the paper.

**Decomposition of the old-vs-new difference.**
- (a) The wrong model changes only the error-rate column, by 0.04–0.37 pp (3.2 → 2.98, 25.8 → 25.43, 6.1 → 5.84, 5.6 → 5.31, 3.0 → 2.96 %).
- (b) The wrong method accounts for the entire difference in mode shares:
  - "T1" 14.3 % on Q5 becomes 5.0 % detected mid-window transitions;
  - "T1" 0 % on Q1 becomes 4.3 %;
  - "Leakage" 0.1 % on Q3 becomes 3.14 % (Q5: 0.0 → 1.03 %);
  - "Near-boundary 97.4 %" is a residual by construction. The measured residual on Q1/Q3/Q5 is dominated by errors far from the boundary.

## Caveats

- Mirror subtraction for decayed prepared-|1> errors is biased by about 0.1 pp (curved decay paths), e.g. Q5 p1 L = −0.09 [−0.17, −0.02].
- The excitation flags on Q3/Q5 include transitions out of |0> into the off-axis state, not only |0> → |1>.
- The 600 ns old-procedure masks assume DS = 40 (no HybridCNN 600 ns checkpoint to validate against).
- Efficiency corrections assume uniform transition times; real transitions are front-loaded.
