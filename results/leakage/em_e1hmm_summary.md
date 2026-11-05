# em_e1hmm: HMM-based error-mode attribution of the canonical MCMit-CNN errors

Tag `e1hmm`, 2026-11-05.
- Script: `sims/leakage/em_e1hmm_attrib_v6.py`. It imports the validated helpers from `_v3` and `_v4`.
- Cluster output: `~/oraqle-ics-sims/leakage/em/e1hmm/full_v6/`.
- Results: `data/leakage/em_e1hmm_results.json`, assembled by `sims/leakage/em_e1hmm_summarize.py`.
- Old-procedure replica: `sims/leakage/em_e1hmm_oldproc.py`, output in `.../e1hmm/oldproc/`.

Earlier versions are superseded (see "History" below).

## Bottom line (1 us, canonical `preds_MCMitCNN_pb`)

Most MCMit-CNN errors are **state changes**, not boundary ambiguity: the qubit is in the other computational state for part of the window, or already at the window start.
- Pooled over Q1/Q3/Q4/Q5:
  - **79.7%** [79.5, 79.9] of errors are detected state changes;
  - **1.26%** [1.21, 1.32] are leakage;
  - **19.0%** [18.9, 19.2] have no detected event.
- That residual behaves like pure Gaussian overlap:
  - its count matches the overlap expectation at the CNN's own boundary (Q1 6,702 vs 6,901 expected; Q3 9,492 vs 9,463);
  - it sits within 1 sigma of that boundary (Q1 95%).
- Q2 (low SNR) is the exception: 49% of its errors are residual.
- The paper's "near-boundary 97.4%" was true for every error by construction. Nothing in it was a measurement.

## Tables (primary: featbox HMM, fitted label prior, per-config null Ac; order leakage -> state change -> residual)

Column definitions:
- Shares are % of that qubit's CNN errors, with 95% CIs.
- **Leakage**: line-frame E3 flag, mirror-corrected. CI is the analytic ±1.96·sqrt(N+ + N-)/N. Q4 is not measurable (n/m).
- **Detected state change** = (transition + wrong-from-start HMM calls on non-leaked errors) − the calls expected from stationary overlap errors (null Ac). CI is a bootstrap over errors and null matrices; it is statistical only.
- **In-window transition**: the part of the state changes that the HMM places inside the window (after about 150-200 ns). This split depends on the start prior (see Robustness).
- **No detected event** = errors − leakage − state change.
- **Within 1/2 sigma**: distance of those residual errors to the per-config linear CNN-surrogate boundary, in units of the prepared state's noise width.
- **Gauss-overlap expected**: the expected error count for stationary Gaussian states read with that same boundary (independent of the HMM).

### 1 us (canonical `preds_MCMitCNN_pb`)

| Qubit | Error rate % [CI] | Leakage | Detected state change | of which in-window transition | No detected event | resid. within 1 / 2 sigma of CNN boundary | Gauss-overlap expected / resid. count |
|---|---|---|---|---|---|---|---|
| Q1 | 2.98 [2.95, 3.01] | 0.05 [-0.02, 0.11] | 79.9 [79.4, 80.3] | 20.2 [19.5, 20.7] | 20.1 [19.6, 20.6] | 95 / 98 | 6901 / 6702 |
| Q2 | 25.43 [25.35, 25.51] | 0.34 [0.32, 0.37] | 50.8 [50.5, 51.1] | 18.2 [17.8, 18.6] | 48.9 [48.6, 49.2] | 86 / 99 | 164686 / 139242 |
| Q3 | 5.84 [5.80, 5.89] | 3.14 [3.00, 3.28] | 82.4 [82.1, 82.6] | 12.2 [11.9, 12.5] | 14.5 [14.3, 14.8] | 55 / 80 | 9463 / 9492 |
| Q4 | 5.31 [5.27, 5.36] | n/m | 71.0 [70.6, 71.4] | 15.9 [15.5, 16.4] | 29.0 [28.6, 29.4] | 38 / 52 | 14187 / 17268 |
| Q5 | 2.96 [2.93, 2.99] | 1.04 [0.91, 1.18] | 89.9 [89.6, 90.2] | 27.9 [27.2, 28.7] | 9.1 [8.7, 9.3] | 70 / 84 | 2234 / 3005 |
| pooled all 5 | 8.51 [8.48, 8.53] | 0.71 [0.68, 0.74] | 62.4 [62.2, 62.6] | 17.9 [17.6, 18.2] | 36.9 [36.7, 37.1] | | |
| pooled Q1/Q3/Q4/Q5 | 4.27 [4.26, 4.29] | 1.26 [1.21, 1.32] | 79.7 [79.5, 79.9] | 17.5 [17.2, 17.7] | 19.0 [18.9, 19.2] | | |
| pooled Q1/Q2/Q3/Q5 | 9.30 [9.28, 9.33] | 0.81 [0.78, 0.85] | 61.2 [61.0, 61.4] | 18.2 [17.9, 18.5] | 38.0 [37.8, 38.2] | | |

Pooled leakage counts Q4 as 0.

**Mode rates as % of all shots at 1 us**, written leakage / state change / residual:
- Q1: 0.001 / 2.38 / 0.60
- Q2: 0.087 / 12.9 / 12.4
- Q3: 0.183 [0.175, 0.191] / 4.81 [4.77, 4.85] / 0.85
- Q4: n/m / 3.77 / 1.54
- Q5: 0.031 [0.027, 0.035] / 2.66 / 0.27
- Pooled all 5: 0.061 / 5.31 / 3.14

**By prepared bit at 1 us** (shares of that bit's errors):

| | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| prep-\|1> error rate | 4.91% | 30.57% | 9.52% | 7.47% | 5.07% |
| state change (relaxation side) | 88.2 [87.7, 88.6] | 82.6 [82.2, 83.0] | 94.8 [94.6, 95.0] | 89.7 [89.4, 90.0] | 97.1 [96.9, 97.3] |
| in-window 1->0 (detected) | 24.2 [23.4, 24.7] | 30.1 [29.4, 30.7] | 11.3 [11.0, 11.6] | 20.7 [20.2, 21.3] | 28.3 [27.4, 29.2] |
| prep-\|0> error rate | 1.05% | 20.29% | 2.17% | 3.16% | 0.86% |
| leakage (prep-\|0> errors) | 0.3 [0.1, 0.6] | 0.8 [0.7, 0.8] | 13.4 [12.7, 14.0] | n/m | 7.7 [6.9, 8.5] |
| state change (excitation side) | 40.9 [39.5, 42.0] | 2.8 [2.7, 2.9] | 27.8 [27.0, 28.6] | 26.9 [26.1, 27.6] | 47.3 [46.0, 48.8] |
| in-window 0->1 (detected) | 1.5 [1.2, 2.0] | 0.3 [0.3, 0.4] | 16.1 [15.3, 16.7] | 4.5 [4.1, 5.0] | 25.8 [24.2, 27.2] |

### 600 ns (soft model `cnn_soft_600ns`, threshold 0.5; error rates reproduce the paper sweep within 0.02 pp)

| Qubit | Error rate % [CI] | Leakage | Detected state change | of which in-window transition | No detected event | resid. within 1 / 2 sigma | Gauss expected / resid. count |
|---|---|---|---|---|---|---|---|
| Q1 | 4.71 [4.67, 4.75] | 0.00 [-0.04, 0.05] | 36.3 [35.8, 36.8] | 4.2 [4.0, 4.4] | 63.7 [63.1, 64.2] | 96 / 100 | 32059 / 33615 |
| Q2 | 27.48 [27.40, 27.57] | 0.16 [0.13, 0.18] | 27.7 [27.4, 28.1] | 2.6 [2.4, 2.8] | 72.1 [71.8, 72.4] | 89 / 99 | 223613 / 221954 |
| Q3 | 6.16 [6.12, 6.21] | 1.66 [1.55, 1.76] | 81.1 [80.8, 81.4] | 10.2 [9.9, 10.5] | 17.3 [17.0, 17.5] | 63 / 85 | 13555 / 11927 |
| Q4 | 6.31 [6.27, 6.36] | n/m | 52.6 [52.2, 53.0] | 6.9 [6.6, 7.3] | 47.4 [47.0, 47.8] | 72 / 78 | 34422 / 33525 |
| Q5 | 2.99 [2.96, 3.03] | 0.17 [0.08, 0.27] | 87.6 [87.3, 88.0] | 24.9 [24.3, 25.5] | 12.2 [11.9, 12.6] | 87 / 94 | 4367 / 4101 |
| pooled all 5 | 9.53 | 0.32 [0.30, 0.34] | 42.5 [42.3, 42.8] | 5.7 [5.6, 5.8] | 57.2 [56.9, 57.4] | | |
| pooled Q1/Q3/Q4/Q5 | 5.05 | 0.53 [0.50, 0.57] | 62.7 [62.5, 62.9] | 9.9 [9.8, 10.1] | 36.8 [36.6, 37.0] | | |

### 400 ns (soft model `cnn_soft_400ns`)

| Qubit | Error rate % [CI] | Leakage | Detected state change | of which in-window transition | No detected event | resid. within 1 / 2 sigma | Gauss expected / resid. count |
|---|---|---|---|---|---|---|---|
| Q1 | 9.32 [9.26, 9.37] | -0.03 [-0.06, 0.00] | 10.6 [10.4, 10.9] | 0.1 [0.1, 0.2] | 89.4 [89.2, 89.6] | 92 / 100 | 83854 / 93295 |
| Q2 | 30.28 [30.19, 30.36] | 0.10 [0.08, 0.12] | 13.0 [12.6, 13.3] | 0.1 [0.0, 0.1] | 86.9 [86.6, 87.3] | 86 / 99 | 281720 / 294563 |
| Q3 | 7.36 [7.31, 7.41] | 0.71 [0.64, 0.77] | 67.5 [67.1, 68.0] | 5.7 [5.4, 5.9] | 31.8 [31.3, 32.2] | 79 / 93 | 22386 / 26176 |
| Q4 | 9.88 [9.83, 9.94] | n/m | 23.2 [22.9, 23.5] | 0.6 [0.5, 0.7] | 76.8 [76.5, 77.1] | 88 / 94 | 82740 / 84964 |
| Q5 | 4.87 [4.83, 4.90] | -0.02 [-0.07, 0.03] | 40.3 [39.7, 40.7] | 5.5 [5.1, 5.9] | 59.8 [59.3, 60.3] | 96 / 99 | 25254 / 32562 |
| pooled all 5 | 12.34 | 0.13 [0.11, 0.14] | 23.0 [22.7, 23.1] | 1.3 [1.2, 1.3] | 76.9 [76.8, 77.1] | | |
| pooled Q1/Q3/Q4/Q5 | 7.86 | 0.15 [0.13, 0.17] | 32.5 [32.3, 32.7] | 2.4 [2.3, 2.5] | 67.3 [67.2, 67.5] | | |

As the window shortens, the extra errors are overlap errors. The residual grows, while the state-change rate per shot stays roughly constant (Q3: 4.8 / 5.0 / 5.0% of shots at 1 us / 600 ns / 400 ns).

**Hidden leakage**: leaked shots as % of shots, and the share of leaked shots the CNN reads correctly. Mirror-corrected, 1 us.

| | Q1 | Q2 | Q3 | Q5 |
|---|---|---|---|---|
| leaked, % of all shots | 0.43 [0.41, 0.44] | 0.91 [0.89, 0.93] | 6.14 [6.09, 6.19] | 2.43 [2.39, 2.46] |
| leaked, % of prepared \|1> | 0.86 | 1.62 | 11.45 [11.37, 11.54] | 4.76 |
| read correctly, % | 99.7 [99.2, 100.1] | 90.4 [89.5, 91.2] | 97.0 [96.9, 97.2] | 98.7 [98.6, 98.9] |

Q4 is n/m. For Q1 and Q5 the prepared-|0> leaked counts are about 0, so their prepared-|0> read-correctly ratios are not meaningful.

At 600 / 400 ns, Q3's leaked share of prepared |1> is 6.46 / 3.52% (read correctly 97.0 / 97.2%).

## Method (MEASURED unless marked INFERRED)

### Leakage (mode 1)
- Replica of `f_cnn_leak_xtab.py`, reused from `full_v3/shared_inputs.npz`.
- Per neighbour configuration, z is the perpendicular distance from the line through the TRAIN density cores of |0> and |1>. Its unit is the prepared-|0> width; frames are fit on TRAIN and applied to TEST.
- Flag: z > 3, minus the mirror count z < -3. Window means come from `feat1us` bins 0-9 (1 us) and `featbox` bins 0-5 / 0-3 (600 / 400 ns).
- It reproduces LEAKAGE_REPORT 3.5 exactly (Q3 3.14%, Q5 1.04%, Q2 0.34%, Q1 0.05%).

### Transition detector (mode 2)
- Uses the m3time 3-state HMM (0, 1, L), fitted on TRAIN (`featbox_xt1`), with the canonical relabelling validated against `out_re/`: posterior occupancies agree to 1e-16.
- A, pi and the tied white Sig[k] are frozen. Per-config emission means are refit on TRAIN by EM on the means only. The shift was measured only for Q3 and Q5 (v4 smoke run): at most 0.28 and 0.45 window-sigma. The log-likelihood gain is 0.008-0.08 nats per shot for featbox and 0.10-0.11 for feat Q4/Q5.
- On TEST, a forward pass over **in-window bins only** gives J(a,e) = P(s_1 = a, s_{K-1} = e | x_0..x_{K-1}):
  - K = 10 / 6 / 4 for 1 us / 600 ns / 400 ns;
  - featbox bin k is samples 50k..50k+49, so there is no look-ahead;
  - the start state is read at bin 1 (100-200 ns), because bin 0 is ring-up.
- Calls: transition if J(b, 1-b) > 0.5. Otherwise the call is the largest of wrong-throughout J(1-b, 1-b), stay J(b, b) and other (L-involving).

### Null (no transitions) and FP correction
- Null Ac, per config c and state s: synthetic stationary sequences mu_c[s,k] + N(0, Sig[k]), white. The noise check (`em_e1hmm_noisecheck.json`) supports this: lag-1/2 correlation below 0.02, |1>/|0> noise ratio 1.00-1.05 (Q4 1.07-1.13).
- Variant Aci inflates the prepared-|1> noise by that ratio.
- Synthetic errors come in three types, all selected as **errors of a per-config linear surrogate of the CNN**. The surrogate is a logistic regression on the CNN's hard output; held-out agreement is 99.6 / 93.8 / 96.9 / 98.7 / 99.7% for Q1-Q5 at 1 us. The types are:
  - overlap: stationary in b, exact truncated sampling;
  - wrong-throughout: stationary in 1-b;
  - transition: b -> 1-b at a uniform time in the window.
- Per config and prepared bit, the 4 call categories of non-leaked errors (mirror weights) are deconvolved by NNLS into N_ov, N_wr and N_tr.
- Detected state change = (trans + wrong calls) − [P(trans|ov) + P(wrong|ov)]·N_ov.
- In-window transition = trans calls − P(trans|ov)·N_ov − P(trans|wr)·N_wr.
- CIs: 300 Poisson bootstrap replicates over errors, with multinomial resampling of the null matrices (6,000 sequences per config, bit and type).

### Attribution order and overlaps
- Primary order: leakage -> state change -> residual.
- In the swap, state change is computed on all errors first and leakage counted only among the remainder:
  - 1 us state change: Q3 82.7 vs 82.4%, Q5 89.7 vs 89.9%;
  - leakage after the swap: Q3 2.78 vs 3.14%, Q5 1.19 vs 1.04%;
  - pooled all 5: state change 62.5 vs 62.4%, leakage 0.62 vs 0.71%.
- Overlap between leakage and a transition call is stored in the JSON.

### Residual (mode 3, "no detected event")
- Characterised independently of the HMM, in two ways:
  1. Distance to the CNN-surrogate boundary and to the optimal stationary LDA boundary, in noise units.
  2. The pure-Gaussian-overlap prediction: the expected error count, and the expected fraction within 1 / 2 sigma (94-97% / 99.9% at 1 us).
- **INFERRED:** the deconvolved composition (overlap / wrong-throughout / transition) is in the JSON. The deconvolved overlap count agrees with the Gaussian expectation (Q1 5,752 vs 6,901, Q3 7,543 vs 9,463, Q5 2,063 vs 2,234).

## Null and sensitivity

**FP per stationary shot** (null Ac, primary detector, 1 us), prepared |0> / |1>:

| Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|
| 0.005 / 0.30% | 0.01 / 15.4% | 0.19 / 1.21% | 0.12 / 1.22% | 0.05 / 0.27% |

Q2's 15% reflects its low SNR. The FP rate that matters is the one on overlap errors; it is the subtracted term in every share above.

**Closure test** (synthetic error sets of known composition, 16 configs, three compositions, null Ac and Aci):
- Recovered vs true detected transitions agree within 1-5% (e.g. Q3 prep 1: 3,161 vs 3,162; Q5 prep 0: 7,914 vs 7,866).
- In compositions dominated by wrong-throughout errors the error grows to 5-20% (Q5 400 ns prep 0: 1,989 vs 1,635).

**Injection** (transition b -> 1-b at time tau; P(transition call) ≥ 0.5 between tau_first and tau_last; primary detector, 1 us):
- Q3: 150-825 ns (prep 0), 175-900 ns (prep 1).
- Q5: 150-850 ns and 175-900 ns.
- Q4: 200-700 ns and 200-800 ns.
- Q1 prep 1: 200-700 ns. Q1 prep 0 never reaches 0.5 (maximum 0.45). Q2 prep 1: 175-750 ns (maximum 0.69); Q2 prep 0 never reaches it.
- Transitions in the last ~100-150 ns are invisible. Before ~150 ns a transition looks like a wrong initial state (P = 0.02-0.33 at tau = 0).
- The CNN surrogate errs only on early transitions. Q5 prep 1: P(error) = 1.0 for tau ≤ 200 ns, 0.71 at 300 ns, 0.18 at 400 ns. Q3: 0.60 at 200 ns, 0.17 at 300 ns.
- So the transitions that cause CNN errors sit mostly in the 100-300 ns band where the start state is weakly identified. This is why the transition / wrong-from-start split is prior-dependent while their sum is not.

**600 / 400 ns**: tau_first is the same (about 150-200 ns), and tau_last moves to 425-500 ns at 600 ns and to about 200-275 ns at 400 ns (Q3).

## Robustness

**Detected-state-change share at 1 us, by variant.** The variants are:
- primary;
- agnostic start prior (p50);
- Viterbi;
- future-informed (all 20 bins, reference only);
- p > 0.8;
- inflated null Aci;
- FIR `feat` features;
- order swap.

| Qubit | Range of the detected-state-change share across these variants |
|---|---|
| Q1 | 79.8-83.3% |
| Q2 | 50.8-56.2% |
| Q3 | 82.0-82.7% |
| Q4 | 70.7-73.3% |
| Q5 | 89.5-90.3% |

- Residual ranges: Q1 16.6-20.1%, Q3 14.4-14.9%, Q5 8.6-9.5%.
- Pooled Q1/Q3/Q4/Q5: state change 79.6-81.1%.

**The in-window share depends on the prior and is reported as a range.** With the agnostic prior, early transitions move into wrong-from-start:
- Q1: 20.2 (primary) vs 2.8%;
- Q3: 12.2 vs 3.8%;
- Q5: 27.9 vs 2.9%;
- FIR features give 25.6 / 12.0 / 25.8%.

**600 / 400 ns.** The agnostic-prior variants fail their goodness of fit (chi2/dof up to 250). The fitted-prior primary fits well, e.g. Q1 36/32 and 16/32, Q5 290/32 and 65/32. The quoted range is therefore primary vs future-informed:
- pooled Q1/Q3/Q4/Q5: state change 62.7-66.8% (600 ns) and 32.5-41.0% (400 ns);
- Q3: 80.8-81.1% and 67.5-71.8%;
- Q5: 87.6-88.1% and 40.3-47.7%.

**Systematics.** The deconvolution's goodness of fit at full statistics is chi2/dof = 3 (Q1), 16 (Q2), 55 (Q3), 25 (Q5) and 330 (Q4) at 1 us. The null does not reproduce the real call pattern in detail, so the bootstrap CIs are statistical only. Use the variant ranges above as the systematic band.

## Old paper numbers: what they measured, and old-vs-new decomposition

**Old procedure.** Source: `Oraqle/Discriminators/runners/_train_and_analyze_hybridcnn.py`, aggregated by `herqules_error_report.py`. Verified line by line; the replica is `em_e1hmm_oldproc.py`.
- **Front end:** sparse-FIR windowed-sinc demod (101-tap 5 MHz LPF, every 40th sample), physical I/Q.
- **Near-boundary:** P(correct) < 0.60. Every error has P(correct) < 0.5, so this is true for every error (`any_n_near_boundary == n_error` in every row).
- **"Relaxation (T1)":**
  - Wiener envelope env = mean(z−o)/var(z−o) over the flattened [I0,Q0,I1,Q1,...] TRAIN traces;
  - mf1 = first half · env, mf2 = second half · env, **uncentred** (no threshold subtracted);
  - T1 = (|mf_i| > 0.5·median|mf_i| for both halves) AND sign(mf1) ≠ sign(mf2), applied to all errors whatever the prepared bit.
  - On Q3, mf1 and mf2 are negative for 100% of shots, so T1 = 0.0% by construction. On Q1 it fires once.
- **"Excitation" / paper "Leakage":** |z| > 3, where z is the time-mean |IQ| amplitude standardised by a TRAIN mean/std pooled over all states.
- **Priority:** T1 > amp > near.
- **Replica check:** at 1 us it reproduces the HybridCNN CSV to within one shot (Q2 T1 6,637 vs 6,636; amp 673 vs 673). At 400 ns the saved len200 checkpoint does not reproduce its CSV (Q1 427k vs 120k errors), so the 400 ns HybridCNN row cannot be re-derived. The old flags on MCMit errors at 400 ns are still valid.

**(a) Wrong model.** The old procedure applied to canonical MCMit-CNN at 1 us gives near / T1 / amp per qubit:
- Q1 100.0 / 0.0 / 0.0
- Q2 97.5 / 2.30 / 0.22
- Q3 99.9 / 0.0 / 0.14
- Q4 99.4 / 0.55 / 0.03
- Q5 85.8 / 14.24 / 0.0
- pooled 97.41 / 2.44 / 0.156

The paper (HybridCNN) has 97.42 / 2.42 / 0.161. So the model swap changes only the error-rate column (3.21 / 25.76 / 6.14 / 5.63 / 3.04 -> 2.98 / 25.43 / 5.84 / 5.31 / 2.96) and the shares by ≤ 0.1 pp.

**(b) Wrong method.** Everything else comes from the method. Pooled over all five qubits at 1 us:

| | Old | New |
|---|---|---|
| "Near-boundary" vs residual | 97.4% | 36.9% (19.0% excluding Q2) |
| "T1" vs detected state change | 2.4% | 62.4%, of which 17.9% in-window transition |
| "Leakage" | 0.16% | 0.71% (Q1/Q2/Q3/Q5: 0.81%) |

Per qubit, old -> new:
- Q3 leakage 0.1 -> 3.14%, Q5 leakage 0.0 -> 1.04%.
- Q5 T1 14.3% -> state change 89.9% (in-window 27.9%).

The old method on MCMit soft models at 600 / 400 ns gives pooled near-boundary 95.3 / 92.3%.

## Caveats

1. The null's goodness of fit fails at full statistics for Q2-Q5 (chi2/dof 16-330). The CIs are statistical; the systematic band is the variant spread.
2. "Detected state change" mixes in-window transitions with a qubit already in the wrong state at about 100-200 ns. The physical cause of the latter (π-pulse error, thermal population, or decay before or during ring-up) is INFERRED, not measured. HMM pi_prep1(0) = 2.3 / 9.2 / 7.7 / 3.0 / 1.5% for Q1-Q5.
3. Q4: leakage n/m. Its multimodal |1> response is outside the 3-state HMM (chi2/dof 330), so the Q4 split is unreliable beyond its residual.
4. Q2: low SNR. The detector false-alarms on 15% of stationary prepared-|1> shots, and Q2's state-change share is 51-56% depending on the variant.
5. The residuals of Q3 and Q5 at 1 us are broader than pure overlap (55 / 70% within 1 sigma vs 95-97% predicted; Q5 count 1.35x expected). Part of them are undetected state changes or non-Gaussian errors.
6. Leakage E3 is a k = 3 lower bound. The state is non-computational, identity unresolved; it is never labelled |2>.
7. The surrogate is linear. CNN errors that the surrogate does not reproduce (Q3: 11%, Q2: 11%) are attributed with a slightly mismatched null.

## History (all earlier outputs kept, superseded)

- **v1** never ran (wrong path).
- **v2 / v3** (`full_v3`): config-independent HMM means, label prior, null A, LDA-selected errors. This produced a negative transition share (Q3 prep 0: −7%) and chi2 up to 9,180. Do not use `full_v3/em_e1hmm_results_full.json`.
- **v4**: per-config means. Showed that v3's 3.5-sigma centroid mismatch was an artefact of the contaminated null B.
- **v5**: CNN-surrogate selection.
- **v6**: robust state-change quantity (primary).
