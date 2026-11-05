# em_e4recon: final error-mode numbers for Table tab:error_modes and RQ#5

Tag `e4recon`, 2026-11-05. This reconciles e1hmm (HMM) and e2seg (segment GLRT) and applies one final attribution to MCMit-CNN and HERQULES.

**Files**
- Machine-readable results, every number with CI and source: `data/leakage/em_e4recon_final.json`. Raw per-bit counts and variances: `data/leakage/em_e4recon_analysis.json`.
- Scripts:
  - `sims/leakage/em_e4recon_glrt.py` re-runs the e2seg v3 detector on all TEST shots, importing it read-only.
  - `sims/leakage/em_e4recon_analyze.py` does the attribution for every model.
  - `sims/leakage/em_e4recon_assemble.py` builds the final JSON.
- Cluster: `graham:~/oraqle-ics-sims/leakage/em/e4recon/{out,logs}`.

**Reproduction.** The re-run reproduces `em_e2seg_v3_raw_main.json` byte for byte for the canonical, soft1000, soft600 and soft400 attributions on all 5 qubits.

**Error rates (MEASURED, they match the paper's tables):**

| Model | Window | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|---|
| Canonical MCMit-CNN | 1 us | 2.98 | 25.43 | 5.84 | 5.31 | 2.96 % |
| Soft MCMit-CNN | 600 ns | 4.71 | 27.48 | 6.16 | 6.31 | 2.99 % |
| Soft MCMit-CNN | 400 ns | 9.32 | 30.28 | 7.36 | 9.88 | 4.87 % |
| HERQULES per-trace | 1 us | 3.15 | 26.07 | 7.60 | 5.86 | 3.05 % |
| HERQULES per-trace | 400 ns | 9.43 | 30.44 | 8.38 | 10.14 | 5.50 % |

## 1. Final definitions (primary) and attribution order

Each error is accounted for once, per qubit and prepared bit; then the counts are summed:

**N = L + T + O + S**

| Mode | Definition | Status | Why this definition |
|---|---|---|---|
| **L: leakage** (non-computational) | Window-mean z perpendicular to the per-configuration core-to-core line (frame fit on TRAIN, applied to TEST). z > 3 minus the mirror z < −3. Windows: 1 us = `feat1us` bins 0–9; 600/400 ns = `featbox` 0–5/0–3. | MEASURED, k = 3 lower bound. Q4 n/m (multimodal \|1>). | Identical in e1hmm, e2seg and `f_cnn_leak_xtab` (within 0.01 pp). |
| **T: resolved mid-readout transition** | e2seg v3 two-segment GLRT at α = 0.01: relaxation test on prepared-\|1> errors, excitation test on prepared-\|0> errors. FP-corrected by the sum of per-shot FP from the exact stationary conditional null. Gated: real quasi-null obs/exp must lie in [0.6, 1.3] and efficiency given a misread must be ≥ 5 %, otherwise n/m. | MEASURED lower bound; efficiency ≈ 16–28 % at 1 us. | Its null is validated on real quasi-null shots (obs/exp 0.77–1.13), with one free choice (α). e1hmm's in-window split varies 0.4–28 % with the start prior, and its null fails goodness of fit (χ²/dof 25–330). |
| **O: near-boundary overlap** | min(R, Σ_cfg N(b,cfg)·p_ov(b,cfg)), with R = N − L − T. p_ov is the model's **own** error rate on prepared-\|b> shots whose *late* window (1–2 us, never seen by any model) sits in the \|b> core, i.e. on shots that stayed in their state. | MEASURED upper bound. | Model-native: no surrogate, no Gaussian assumption, no HMM. It works for any model, including HERQULES. |
| **S: no resolved event, in excess of overlap** | R − O | MEASURED lower bound, by elimination. Identity INFERRED: the qubit was in the other computational state for most of the window. | |

**State change = T + S = N − L − O.** This is relaxation or excitation before or during the readout. It does not depend on the step detector, and it is the robust quantity.

**Order and swap.** The order is L → T → O → S. In order B (T before L) the residual, O and S move by ≤ 0.03 pp. Only the L/T split moves: on Q3, L goes 3.14 → 1.70 % and T 1.75 → 3.18 %, because 7.7 % of Q3 prepared-\|0> errors are both leaked and step-flagged.

No mode is true by construction:
- S ≈ 0 on Q2.
- O is capped only for Q2 prepared \|1>.
- T is a small minority.

**95 % CIs:**
- error rates: Wilson;
- L: ±1.96 √(N₊ + N₋);
- T: ±1.96 √(flags + MC var of the FP sum);
- O: binomial per configuration;
- S: var(L) + var(T) + var(O);
- T + S: var(L) + var(O).

All CIs are statistical only. The method spread is given in §5.

## 2. Proposed Table tab:error_modes (canonical MCMit-CNN, 1 us; % of that qubit's errors)

| Qubit | Error rate | Leakage | State change, T+S (of which resolved step T) | Near-boundary overlap O | Leaked shots, % of all shots (read correctly) |
|---|---|---|---|---|---|
| Q1 | 2.98 [2.95, 3.01] | 0.04 [−0.02, 0.11] | **81.2** [80.6, 81.8] (4.3 [4.1, 4.6]) | ≤18.8 [18.2, 19.4] | 0.41 [0.39, 0.42] (99.7) |
| Q2 | 25.43 [25.35, 25.51] | 0.34 [0.32, 0.37] | 1.2 [0.9, 1.4] (n/m) | ≤98.5 [98.2, 98.7] | 0.90 [0.88, 0.92] (90.3) |
| Q3 | 5.84 [5.80, 5.89] | 3.14 [3.00, 3.29] | **76.8** [76.1, 77.4] (1.8 [1.6, 1.9], excitation only) | ≤20.1 [19.5, 20.7] | 6.14 [6.10, 6.19] (97.0) |
| Q4 | 5.31 [5.27, 5.36] | n/m | 86.8 [86.3, 87.4] (n/m) | ≤13.2 [12.6, 13.7] | n/m |
| Q5 | 2.96 [2.93, 2.99] | 1.03 [0.90, 1.17] | **90.2** [89.8, 90.7] (5.0 [4.8, 5.3]) | ≤8.8 [8.3, 9.2] | 2.40 [2.37, 2.43] (98.7) |
| Q1, Q3–Q5 | 4.27 | 1.26 [1.21, 1.32] | **83.0** [82.7, 83.3] (2.2) | ≤15.7 [15.4, 16.0] | – |
| All 5 | 8.51 | 0.71 [0.68, 0.74] | 34.1 [33.9, 34.3] (0.9) | ≤65.2 [65.0, 65.4] | – |

Notes on the table:
- **Q3, step test:** the relaxation test fails its real quasi-null check (obs/exp 2.48). That is probably L→\|1> returns, which look like a down-step along the axis (INFERRED). Q3's T therefore counts excitation only. The ungated value would be 2.1 %.
- **Q4:** the multimodal \|1> response makes leakage and steps n/m. Errors on its reduced-shift \|1> sub-cloud fall in S.
- **Q2:** O is a loose upper bound, because its late \|0>/\|1> clouds overlap. Every estimate says overlap dominates: late core 98.5 %, e2seg near-Bayes Gaussian ≥ the observed count, e1hmm Gaussian at the CNN surrogate 58 %.

A LaTeX version of the table is in §8.

## 3. HERQULES with the same attribution

**Which HERQULES.** The model used is the **per-trace HERQULES** checkpoint:
- It is the only single-shot HERQULES.
- It is the model behind tab:fidelity.
- It is panel (b) of the cut figure.

The "demux-pipeline / scrambled" HERQULES builds each decision from the MF and RMF of *different* shots (same shot in 0.0029 % of rows), so a per-shot attribution of its errors is ill-defined. Its numbers below are INDICATIVE only.

Shares in % of errors. CNN − HERQ differences use an independent-sample CI, which is conservative because the error sets overlap.

| | CNN 1 us | HERQ-pt 1 us | CNN − HERQ (pp) | CNN 400 ns | HERQ-pt 400 ns | CNN − HERQ (pp) |
|---|---|---|---|---|---|---|
| **Q3 error rate** | 5.84 | 7.60 (ratio 0.77) | | 7.36 | 8.38 (ratio 0.88) | |
| leakage | 3.1 | 3.5 | −0.3 [−0.5, −0.2] | 0.7 | 0.7 | +0.1 [−0.0, 0.1] |
| resolved step | 1.8 | 0.9 | +0.8 [0.7, 1.0] | 0.7 | 1.5 | −0.8 [−0.9, −0.6] |
| state change T+S | 76.8 | 69.4 | **+7.4 [6.5, 8.3]** | 59.2 | 59.4 | −0.2 [−1.3, 0.9] (same) |
| overlap | 20.1 | 27.1 | −7.1 [−8.0, −6.1] | 40.1 | 40.0 | +0.2 [−0.9, 1.3] (same) |
| **Q5 error rate** | 2.96 | 3.05 (ratio 0.97) | | 4.87 | 5.50 (ratio 0.88) | |
| leakage | 1.0 | 1.1 | −0.1 [−0.3, 0.1] | 0.0 | 0.0 | 0.0 |
| resolved step | 5.0 | 6.2 | −1.2 [−1.5, −0.8] | n/m | n/m | |
| state change T+S | 90.2 | 87.9 | **+2.4 [1.7, 3.0]** | 44.1 | 38.3 | +5.8 [4.6, 7.0] |
| overlap | 8.8 | 11.0 | −2.3 [−2.9, −1.6] | 55.9 | 61.7 | −5.8 [−7.0, −4.6] |

**Per shot at 1 us (state change / overlap, % of shots):**

| Qubit | CNN | HERQ-pt |
|---|---|---|
| Q3 | 4.48 / 1.17 | 5.27 / 2.06 |
| Q5 | 2.67 / 0.26 | 2.68 / 0.34 |

So the CNN's lower Q3 error comes from both modes.

**The other qubits at 1 us** (HERQ per-trace: error rate, then L / T / O / T+S):
- Q1: 3.15 %; 0.1 / 6.2 / 21.9 / 78.1.
- Q2: 26.07 %; 0.3 / n/m / 98.2 / 1.4.
- Q4: 5.86 %; n/m / n/m / 13.5 / 86.5.
- Pooled Q1, Q3–Q5: L 1.5, T 2.3, O 19.7, T+S 78.7 (CNN: 1.3 / 2.2 / 15.7 / 83.0).

**Co-failure (e3herq, MEASURED).** 68.2 % of HERQ-pt Q3 errors and 92.1 % of Q5 errors are also CNN errors at 1 us.

**Is there a 2× gap? No, between single-shot models.**
- CNN/HERQ-pt error ratio: 0.77–0.98 at 1 us and 0.88–0.99 at 400 ns, on every qubit.
- CNN/HERQ-scrambled: 1.58 (Q3) and 2.21 (Q5) at 1 us; 1.61 and 1.76 at 400 ns. Here HERQULES is the *better* one, and only because each decision combines two shots.

**Are the proportions the same?**
- Qualitatively yes: same dominant mode, same ordering.
- Statistically no at 1 us: Q3 differs by 7 pp in state change vs overlap, Q5 by 2.4 pp. At 400 ns Q3 is identical within CIs; Q5 differs by 6 pp.
- The scrambled HERQULES gives very different proportions: Q3 T+S 51 %, Q5 62 %, with O 35–47 % (INDICATIVE).

## 4. Numbers for Results / Analysis / Takeaway #5

All MCMit-CNN. 1 us = canonical, 600/400 ns = soft. "% err" = share of that qubit's errors.

**Q3 at 1 us vs 400 ns**

| | 1 us | 400 ns |
|---|---|---|
| state change | ≥76.8 [76.1, 77.4] % err | ≥59.2 [58.4, 60.0] |
| overlap | ≤20.1 [19.5, 20.7] | ≤40.1 [39.4, 40.9] |
| leakage | 3.1 [3.0, 3.3] | 0.7 [0.6, 0.8] |
| resolved step | 1.8 [1.6, 1.9] | 0.7 [0.6, 0.8] |

**Q5 at 1 us vs 400 ns**

| | 1 us | 400 ns |
|---|---|---|
| state change | ≥90.2 [89.8, 90.7] | ≥44.1 [43.3, 45.0] |
| overlap | ≤8.8 [8.3, 9.2] | ≤55.9 [55.0, 56.8] |
| leakage | 1.0 [0.9, 1.2] | ≈0 (−0.0 [−0.1, 0.0]) |
| resolved step | 5.0 [4.8, 5.3] | n/m (efficiency 3 %) |

At 600 ns: Q3 state change 72.4, overlap 26.0; Q5 86.5 / 13.3.

**Per shot, from 400 ns → 1 us (% of shots)**

| Qubit | State change | Overlap |
|---|---|---|
| Q3 | 4.36 [4.30, 4.41] → 4.48 [4.45, 4.52] | 2.95 → 1.17 |
| Q5 | 2.15 [2.10, 2.19] → 2.67 [2.66, 2.68] | 2.72 → 0.26 |
| Q1 | 1.55 → 2.42 | 7.78 → 0.56 |

**Pooled Q1, Q3–Q5 (state change / overlap / leakage / resolved step, % err)**
- 1 us: 83.0 [82.7, 83.3] / 15.7 [15.4, 16.0] / 1.3 [1.2, 1.3] / 2.2 [2.1, 2.3].
- Window trend in the state-change share: 38.5 → 66.9 → 83.0 % (400 / 600 / 1000 ns), with overlap 61.4 → 32.5 → 15.7 %.
- All 5 at 1 us: 34.1 / 65.2 / 0.7 (Q2's 285k errors dominate).

**Relaxation is not confined to the clean qubit.** State change as a share of prepared-\|1> errors at 1 us:
- Q1 89.0 [88.5, 89.6]
- Q3 87.6 [87.0, 88.3]
- Q4 89.2 [88.5, 90.0]
- Q5 96.5 [96.1, 96.8]
- Q2 ≈0

**Resolved mid-readout steps (lower bound).**
- Q1 4.3 [4.1, 4.6] % err; Q5 5.0 [4.8, 5.3].
- Relaxation share of prepared-\|1> errors: Q1 5.0, Q5 4.4.
- Q3 excitation: 9.4 % of prepared-\|0> errors.
- Sensitivity: P(flag) ≥ 0.5 only between about 225 and 800 ns. Q1 peaks at 0.44. The test is blind before about 150 ns and in the last about 150 ns.
- Efficiency-corrected shares (INFERRED, assume uniform transition times):
  - Q1 prepared \|1>: 22–27 %;
  - Q5 prepared \|1>: 10–16 %;
  - Q5 prepared \|0>: 19–48 %;
  - Q3 prepared \|0>: 19–51 %.

**Late-window confirmation (model-free, MEASURED).** Prepared-\|1> errors sitting in the \|0> core during the unused 1–2 us window, compared with correctly read prepared-\|1> shots:
- Q5: 65.0 % [64.5, 65.6] vs 7.5 %
- Q3: 63.9 vs 19.0 %
- Q1: 60.6 vs 3.1 %
- Q4: 57.7 vs 2.7 %

Prepared-\|0> errors sitting in the \|1> core, against the same comparison:
- Q1: 27.4 vs 0.1 %
- Q3: 10.9 vs 1.3 %
- Q5: 24.1 vs 1.0 %

**Hidden leakage at 1 us** (leaked shots; share of them read "correctly"; leaked errors as % of shots):
- Q3: 6.14 [6.10, 6.19] % of shots, i.e. 11.46 [11.38, 11.55] % of prepared \|1> and 0.82 % of prepared \|0>; read correctly 97.0 [96.9, 97.1] %; errors 0.184 [0.175, 0.192] %.
- Q5: 2.40 % (4.72 % of prepared \|1>); 98.7 %; 0.031 %.
- Q2: 0.90 %; 90.3 %; 0.087 %.
- Q1: 0.41 %; 99.7 %; 0.001 %.

**Q3 leakage grows with the window** (prepared \|1>, one consistent line frame):
- 3.51 [3.46, 3.56] / 6.45 [6.39, 6.52] / 11.46 [11.38, 11.55] % at 400 / 600 / 1000 ns.
- The report's 11.3 % is the multi-frame central value (frames 10.7–11.8).
- Q5: 0.45 / 1.68 / 4.72 %.

**Replacement for "excitation errors 10⁻³–10⁻² % of shots."** Excitation-side state changes (prepared \|0> read as \|1>) are:
- Q1: 0.47 [0.45, 0.49] % of prepared-\|0> shots;
- Q3: 0.63 [0.59, 0.67] %;
- Q5: 0.45 [0.43, 0.47] %.

Of these, resolved mid-readout excitations are 0.010 / 0.205 / 0.072 %. Separately, 0.82 % of Q3 prepared-\|0> shots leak; that is 13.4 % of Q3's prepared-\|0> errors.

## 5. Robustness / cross-method (1 us, % of errors)

| Qubit | e4recon T+S (primary) | e1hmm HMM state change (variant range) | e4recon O (primary, upper) | e1hmm Gauss @ CNN surrogate | e2seg Gauss @ LS proxy | e2seg near-Bayes MF |
|---|---|---|---|---|---|---|
| Q1 | 81.2 | 79.9 (79.8–83.3) | 18.8 | 20.7 | 30.6 | 16.1 |
| Q2 | 1.2 | 50.8 (50.8–56.2), not credible | 98.5 | 57.8 | 109.9 | 109.5 |
| Q3 | 76.8 | 82.4 (82.0–82.7) | 20.1 | 14.5 | 8.3 | 1.6 |
| Q4 | 86.8 | 71.0 (71.0–73.3), χ²/dof 331 | 13.2 | 23.8 | 18.6 | 32.7 |
| Q5 | 90.2 | 89.9 (89.5–90.4) | 8.8 | 6.7 | 2.9 | 0.02 |

What the comparison shows:
- On Q1/Q3/Q5 the state-change share agrees across methods to within 0.3–5.6 pp.
- The near-Bayes MF bound is not the CNN's boundary. The CNN weights early bins heavily to survive decays, so on Q5 its overlap rate on stationary \|1> shots is about 300× the MF's: 0.17 % vs 6×10⁻⁴ %.

**At 600 / 400 ns** (state change; e4recon vs e1hmm):

| Qubit | 600 ns | 400 ns |
|---|---|---|
| Q1 | 42.7 vs 36.3 | 16.6 vs 10.6 |
| Q3 | 72.4 vs 81.1 | 59.2 vs 67.5 |
| Q5 | 86.5 vs 87.6 | 44.1 vs 40.3 |
| pooled Q1, Q3–Q5 | 66.9 vs 62.7 | 38.5 vs 32.5 |

**The in-window-transition share is not robust.**
- e2seg α = 0.05 / 0.01 / 0.002: Q1 9.9 / 4.3 / 1.5 %; Q5 11.9 / 5.0 / 1.6 %.
- The selection-aware null and the prepared-branch FP change it by ≤ 0.5 pp.
- e1hmm gives 0.4–28 % depending on the prior.
- **Only report it as a lower bound.**

**Soft-output depth split, rejected.** I tried splitting each error by the model's own P(correct). Errors were called near-boundary if P(correct) was above the 5th percentile of the late-core overlap errors.
- That threshold is p* = 0.004–0.06, and it puts 85–93 % of residual errors in "near".
- The count-level late-core test allows only 4–12 % of prepared-\|1> errors to be overlap on Q1/Q3/Q5.
- The CNN is calibrated against decays and is *not* confident on early-decay traces, so model confidence is no near-boundary criterion. This is also why the old P(correct) < 0.6 rule meant nothing.

**Soft1000 vs canonical.** All shares agree within 0.6 pp.

## 6. What the old numbers measured, and old vs new

Source: `Oraqle/Discriminators/runners/_train_and_analyze_hybridcnn.py`, aggregated by `herqules_error_report.py`. I checked the lines myself.

**Model.** HybridCNN, on a sparse-FIR demodulation with DS = 40.

**Near-boundary** (l.28, l.146-148). P(correct) < 0.60 from the model's own sigmoid. Every error has P(correct) < 0.5, so this is true for every error.

**"Relaxation (T1)"** (l.48-56, l.156-164).
- Envelope env = mean(z−o)/var(z−o), per feature, on TRAIN pairs drawn with RandomState(0).
- mf1 and mf2 are the projections of the *un-centred* first and second halves of the flattened [I, Q] window.
- A shot is flagged if |mf| > 0.5·median|mf| in both halves and sign(mf1) ≠ sign(mf2).
- Without centring the sign tracks the IQ origin, not the state:
  - Q1/Q3 can never fire;
  - on Q5 it fires on 9.5 % of all prepared-\|0> shots.

**"Leakage"** (l.150-154). |z| > 3 on the time-mean |IQ|, with z against the TRAIN mean/std pooled over states.

**Priority.** t1 > amp > near.

**Old vs new.** Old procedure on canonical MCMit-CNN (e1hmm and e2seg agree), pooled near / T1 / amp: 97.41 / 2.44 / 0.16. Paper (HybridCNN): 97.4 / 2.4 / 0.2. So:
- **(a) Wrong model.** It moves only the error-rate column, by 0.04–0.37 pp: 3.2 / 25.8 / 6.1 / 5.6 / 3.0 → 2.98 / 25.43 / 5.84 / 5.31 / 2.96. Shares move by ≤ 0.1 pp.
- **(b) Wrong method.** It accounts for everything else:
  - "near-boundary 97.4 %" → overlap ≤65.2 % (all 5), ≤15.7 % (Q1, Q3–Q5);
  - "T1 2.4 %" → state change ≥34.1 % (all 5), ≥83.0 % (Q1, Q3–Q5), of which ≥0.9 / 2.2 % are resolved steps;
  - "leakage 0.2 %" → 0.71 % (Q1/Q2/Q3/Q5: 0.81 %).
  - Per qubit: Q5 T1 14.3 % → state change 90.2 % (5.0 % resolved); Q3 near 99.9 % → overlap ≤20.1 %; Q3 leakage 0.1 → 3.1 %.

## 7. Narrative: what survives

**"Near-boundary ambiguity dominates" dies** for Q1, Q3, Q4 and Q5 at 1 us:
- An independent, model-native test bounds overlap at ≤8.8–20.1 % of errors.
- The rest are qubits that were in the other computational state for most of the window. The late-window check confirms it: 58–65 % of prepared-\|1> errors are already in \|0> in the next microsecond, vs 3–19 % of correctly read shots.

It survives in two places:
- for Q2, the low-SNR qubit (overlap ≥58–98 %);
- at short windows, where overlap reaches 40–84 % at 400 ns on Q1/Q3/Q4/Q5.

**"T1 only surfaces on the clean qubit" dies.**
- Relaxation-side state changes are 88–97 % of prepared-\|1> errors on Q1, Q3, Q4 and Q5.
- The old test could not fire on Q1/Q3 by construction.

**"We have hit physics, not algorithms" survives qualitatively; the number dies.**
- ≥83 % of errors on Q1, Q3–Q5 come from relaxation or excitation before or during the readout, ≤16 % from overlap, 1.3 % from leakage. Q2 is SNR-bound.
- Two different single-shot models fail on largely the same shots (68 % / 92 %), and with similar per-shot state-change rates on Q5 (2.67 vs 2.68 %).

Three qualifiers:
1. On Q3 the CNN has 23 % fewer errors than per-trace HERQULES, in both modes (state change 4.48 vs 5.27 % of shots).
2. The overlap/state-change split depends on the model's boundary. The CNN accepts about 300× the near-Bayes overlap on Q5 to tolerate decays.
3. Much of the "physics" is readout-induced and therefore depends on readout design, not only on the qubit. Q3's decay rate doubles to triples when Q1/Q5 are excited, and leakage grows with the window: 6.1 % of Q3 shots leave the computational subspace but are counted as correct 97 % of the time.

**The hypothesis "mid-readout relaxation/excitation are hardest, more so for longer readouts"** is supported in a refined form:
- State changes are the dominant error source at 1 us.
- Their share grows with the window: 38.5 → 66.9 → 83.0 % (pooled Q1, Q3–Q5).
- Their per-shot rate grows only mildly (Q5 2.15 → 2.67 %, Q3 4.36 → 4.48 %). The share rises mainly because overlap errors fall 2.5–14×.
- The costly transitions are early ones, before about 250 ns or before the window, not mid-window ones. Resolvable mid-window steps are only ≥2–5 % of errors (10–27 % efficiency-corrected, INFERRED).

## 8. LaTeX for the table (drop-in replacement)

```latex
\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{MCMit-CNN error modes at $1\,\mu$s, as a share of that qubit's errors (95\% CIs $\le\pm0.7$\,pp). \emph{Leak.}: off-axis (non-computational) population. \emph{State change}: relaxation or excitation before or during the readout, i.e.\ errors not explained by leakage or by the model's own misread rate on shots that stay in their state (lower bound); in parentheses the part resolved as a mid-readout step. \emph{Overlap}: near-boundary SNR errors (upper bound). Last column: leaked shots in \% of all shots (\% of them read correctly).}
\label{tab:error_modes}
\begin{tabular}{l r r r r r}
\toprule
Qubit & Err.\ (\%) & Leak.\ (\%) & State change (step) (\%) & Overlap (\%) & Leaked shots (\%) \\
\midrule
Q1 & 2.98 & 0.0 & 81.2 (4.3) & 18.8 & 0.4 (99.7) \\
Q2 (hardware) & 25.43 & 0.3 & 1.2 (n/m) & 98.5 & 0.9 (90.3) \\
Q3 & 5.84 & 3.1 & 76.8 (1.8$^{a}$) & 20.1 & 6.1 (97.0) \\
Q4$^{b}$ & 5.31 & n/m & 86.8 (n/m) & 13.2 & n/m \\
Q5 & 2.96 & 1.0 & 90.2 (5.0) & 8.8 & 2.4 (98.7) \\
\midrule
Q1, Q3--Q5 & 4.27 & 1.3 & \textbf{83.0} (2.2) & 15.7 & --- \\
All qubits & 8.51 & 0.7 & 34.1 (0.9) & 65.2 & --- \\
\bottomrule
\end{tabular}\\[1pt]
{\raggedright\scriptsize $^{a}$Excitation only; the relaxation step test fails its null check on Q3. $^{b}$Multimodal $|1\rangle$ response: leakage and steps not measurable.\par}
\end{table}
```

## 9. Caveats

- **Bounds, not point estimates.** O is an upper bound: the late-core rate is multiplied by all prepared-b shots, and its reference contains a few shots that changed state and returned. S and T+S are therefore lower bounds. L is a k = 3 lower bound, never \|2>.
- **What S contains (INFERRED).** S is established by elimination plus the late-window check. Its physical split is INFERRED: preparation or π-pulse error, thermal population, decay before or during ring-up (< 150–250 ns), and sub-threshold leakage (Q3 prepared \|0>). On Q4 it also contains the reduced-shift \|1> sub-cloud.
- **Q2.** The late core is impure. Q2 is overlap-dominated in every estimate, but its O / S split is loose.
- **HERQULES.** Only the per-trace HERQULES gives a valid per-shot attribution. The scrambled numbers are indicative. HERQULES 600 ns was not validated (e3herq), so it is not attributed.
- **CIs.** All CIs are statistical; the method spread in §5 is the systematic band.
