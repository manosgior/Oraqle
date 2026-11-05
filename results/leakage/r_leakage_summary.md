# Leakage in the Lienhard 5-qubit readout dataset: reconciled answer (tag r)

Date: 2026-11-04. This combines four independent analyses on the shared features (m1tail tail statistics, m2mix mixture fits, m3time 3-state HMM, m4conf confounder screens), three surveys (s1local, s2cluster, s3lit), and my own spot checks on graham (`r_spotcheck.py`, `r_spot2.py`).

## 1. Answer

**Yes, the dataset contains real, unlabelled third-state populations. Their size depends strongly on the qubit.**

- **Q3 / q2: leakage is present. This is the strongest case.**
  - At the paper's 1 us window, **11.2 % [11.1, 11.3]** of prepared-|1> shots and **0.73 % [0.62, 0.84]** of prepared-|0> shots sit in a third state. Overall that is **6.0 %** of all Q3 shots.
  - The state lies beyond |1>, at about 1.7-1.9 separations along the axis and +1.0 to +1.5 separations perpendicular to it.
  - It decays to |1> (96-99.9 % of exits) with a 1.8 us lifetime. That is shorter than the |1>->|0> lifetime under readout (2.5 us, or 4.8 us without exchange partners), which is the |2> signature.
  - Most of it is created during the readout, at a |1>->L rate of about 2.6 % per 100 ns. Only 0.6-2.0 % of prepared |1> is already in it in the first 100-300 ns.
- **Q5 / q4: a |2>-like population of 4.5 % [4.3, 4.6]** of prepared |1> (0.10 % of prepared |0>, 2.3 % overall).
  - Q3 feeds it: 6.8-8.9 % when Q3 is prepared |1> (depending on Q4), against 0.33-0.49 % when Q3 and Q4 are both prepared |0>.
  - Shot by shot, Q5 enters the state at the same time as Q3 drops |1>->|0> (10x the chance rate). It leaves the state when Q3 jumps back |0>->|1> (20x the chance rate).
  - INFERRED: a two-qubit excitation exchange |1_Q3 1_Q5> -> |0_Q3 2_Q5> during readout. That is still leakage of Q5. Its own baseline is at most 0.3-0.5 %.
- **Q1 / q0:** 0.69 % [0.59, 0.79] of prepared |1>, at most 0.02 % of prepared |0> (0.35 % overall). It follows the same Q3-paired mechanism: 0.13 % when Q3 is prepared |0>, 1.1-1.4 % when Q3 is prepared |1>.
- **Q2 / q1:** 1.53 % [1.48, 1.58] of prepared |1>, 0.17 % [0.15, 0.19] of prepared |0> (0.85 % overall). It is |2>-like: sourced from |1> and decays to |1>. Confidence is only moderate, because this qubit's SNR is 1.2-2.1 and part of the signal is correlated with Q3's third state.
- **Q4 / q3: inconclusive. Leakage is not measurable here.**
  - The prepared-|1> response is itself multimodal: a stationary sub-cloud between |0> and |1> holds about 21 % of prepared |1>, and the main |1> cloud is broad and elongated.
  - The sub-cloud is fed in the same proportion from thermally excited prepared-|0> shots (0.30 % of shots, against 1.55 % main-cloud shots in prepared |0>). That argues for a sub-response of |1> (a fluctuator) rather than |2>.
  - No method isolates a |2> population on Q4.

**Size relative to the error budget.** Take the canonical MCMit-CNN on the test file at 1 us. Its error rates are Q1 2.98 %, Q2 25.43 %, Q3 5.84 %, Q4 5.31 %, Q5 2.96 %; the paper's table prints the HybridCNN values 3.2/25.8/6.1/5.6/3.0 %.

- Q3's leaked population (6.0 % of shots) is as large as its whole assignment error. Q5's (2.3 %) is 0.8x its error.
- Leaked shots are nevertheless only a small share of the CNN's *errors*: Q3 4.1 % [3.9, 4.2], Q5 1.2 % [1.0, 1.3], Q2 0.32 %, Q1 0.14 %.
- The reason is that the binary CNN reads 98-99 % of leaked prepared-|1> shots as "1", which counts as correct.
- So leakage is a hidden channel *beside* the two-level error budget, not a slice of it. The paper's "Leakage, share of errors" column (0.0-0.2 %) is not a leakage measurement. For Q3 it undercounts even the leaked share of errors by 30-40x (4.06 % vs 0.13 %, the unrounded 0.1 %).

## 2. Windows, data, definitions

- **Sampling and windows.** 500 MS/s, 2 ns per sample; the stored trace is 1000 samples = 2 us. The paper's "1 us" = raw samples 0-499 (the `feat1us` bins 0-9, strictly causal FIR, identical to the CNN input). The paper's "400 ns" = samples 0-199 (feat bins 0-3).
- **Qubit naming.** Code q = paper Q(q+1); the prepared bit is (y>>q)&1.
- **Data.** All headline numbers use the TEST split: 560,000 shots per prepared bit per qubit, 1,120,000 per qubit. Every method replicates on TRAIN within its CI.
- **"Overall"** = (prep0 + prep1)/2. Both prepared bits have equal counts, so this equals the share of all shots.
- **Primary statistic.** The one-sided mirror excess E3 = [N(z > 3) - N(z < -3)] / N. Here z = (perp - centre of the prepared 5-bit state) / (perpendicular noise width of the prepared-|0> state with the same 16-configuration neighbours).
  - It is MEASURED, and it needs only a symmetric core.
  - It counts shots whose 1 us window mean is displaced off the |0>-|1> line by more than the Gaussian core allows.
  - It is a lower bound on "visited the third state during the window", because late entries are only partly displaced.
  - Two independent implementations agree: m1tail and my `r_spotcheck.py`, which uses a different frame (truncated medians instead of truncated-Gaussian fits).
- **Best estimate and CI.** The best estimate is the mean of the two E3 implementations. The CI is the envelope of their two statistical 95 % CIs, so it includes the frame systematic. The "method range" in the table spans the other methods' definitions (mixture weight, HMM occupancy).
- **MEASURED vs INFERRED.** The existence, location, size, timing and correlations of these populations are MEASURED. Calling them |2> is INFERRED from their dynamics; no |2> was ever prepared or labelled.

## 3. Per-qubit table (paper 1 us window, TEST)

| Qubit | Prepared \|0> | Prepared \|1> | Overall | Supporting methods (prep1 value) | Verdict |
|---|---|---|---|---|---|
| Q1 / q0 | 0.00 % [-0.01, 0.02] (at most 0.02 %) | 0.69 % [0.59, 0.79]; method range 0.43-0.77 % | 0.35 % [0.29, 0.40] | m1 E3 0.61 %, r E3 0.77 %, m2 0.43 % [0.19, 0.75]; m4 5-sigma floor 0.017 %; m3 HMM unreliable here (L only 1.5 sigma/bin from \|1>, null false-positive 1.2 %) | Third-state-like population, mostly paired with Q3 (exchange). Own part: 0.13 % [0.11, 0.16] of prepared \|1> when Q3 is prepared \|0> |
| Q2 / q1 | 0.17 % [0.15, 0.19]; range 0.05-0.34 % | 1.53 % [1.48, 1.58]; range 0.46-2.8 % | 0.85 % [0.82, 0.88] | m1 1.54 %, r 1.52 %, m2 1.50 % [1.40, 1.63], m3 visited-L 2.80 % (null 0.08 %), m4 5-sigma floor 0.66 % | Probable \|2>-like (sourced from \|1>, decays to \|1>); moderate confidence |
| Q3 / q2 | 0.73 % [0.62, 0.84]; range 0.33-1.25 % | 11.2 % [11.1, 11.3]; range 8.9-14.5 % | 5.96 % [5.85, 6.07] | m1 11.22 %, r 11.16 %, m2 8.94 % [8.80, 9.04], m3 visited-L 14.5 % [14.4, 14.6] / in L at 1 us 12.6 %, m4 5-sigma floor 4.66 % (95-96 % of candidates survive all confounder screens) | **Leakage present (\|2>-like), strong** |
| Q4 / q3 | not measurable | not measurable | - | m1 inconclusive (two-sided halo); m2 sub-cloud 20.8 % (not attributed to leakage); m3 "L" = the far half of the broad \|1> cloud; m4: no off-axis tail (2/800,000 beyond 5 sigma) | **Inconclusive**: multimodal \|1> response, origin unknown |
| Q5 / q4 | 0.10 % [0.085, 0.12]; range 0.04-0.79 % | 4.47 % [4.34, 4.60]; range 3.1-13.3 % | 2.29 % [2.22, 2.36] | m1 4.40 %, r 4.54 %, m2 3.09 % [2.73, 3.37], m3 visited-L 13.3 % (null 1.2 %), m4 5-sigma floor 0.15 % | \|2>-like population created by exchange with Q3 (and weakly with Q4). Own baseline 0.33-0.49 % (Q3 = Q4 = prepared \|0>) |

### Conditioning on Q3 (prepared |1>, 1 us; r E3, with m1 per-configuration values in brackets)

| | Q3 prepared \|0> | Q3 prepared \|1> |
|---|---|---|
| Q1 | 0.13 % [0.11, 0.16] (configs 0.08-0.16 %) | 1.40 % [1.35, 1.45] (configs 0.98-1.26 %) |
| Q2 | 1.27 % [1.23, 1.32] | 1.76 % [1.71, 1.82] |
| Q5 | 0.91 % [0.87, 0.95]: Q4 = 0 gives 0.33-0.49 %, Q4 = 1 gives 1.35-1.66 % | 8.17 % [8.06, 8.28]: Q4 = 0 gives 8.6-8.9 %, Q4 = 1 gives 6.8-7.1 % |

Q3's own rate depends only weakly on its neighbours: 9.5-13.6 % across the 16 configurations (m1).

## 4. Window dependence and what is present at readout start

E3 from r_spotcheck (TEST, 560,000 per prepared bit; 95 % CI about +/-0.02-0.09 pp):

| Qubit | 0-200 ns, p1 / p0 | 0-400 ns (paper 400 ns), p1 / p0 | 0-1 us (paper), p1 / p0 | 0-2 us, p1 / p0 |
|---|---|---|---|---|
| Q1 | 0.00 / 0.00 % | 0.06 / 0.01 % | 0.77 / 0.00 % | 2.98 / 0.01 % |
| Q2 | 0.08 / 0.02 % | 0.40 / 0.05 % | 1.52 / 0.17 % | 2.65 / 0.26 % |
| Q3 | 0.86 / 0.03 % | **3.57 / 0.12 %** | 11.16 / 0.65 % | 23.2 / 1.1 % |
| Q5 | 0.00 / 0.01 % | 0.48 / 0.00 % | 4.54 / 0.10 % | 10.1 / 0.60 % |

- The population grows with window length for every qubit, so most of it is created during the measurement. A short (400 ns) readout cuts Q3's leaked share to about 1/3 of its 1 us value.
- **Present at readout start (upper bounds; 100 ns bins cannot separate "at t = 0" from "entered during ring-up"):**
  - Q3: the HMM puts 0.61 % [0.59, 0.62] of prepared |1> in L at bin 0, and filtered Viterbi puts 1.97 % [1.93, 2.01] in L at 100-300 ns (noise-only null 0.005 %). For prepared |0> the figures are 0.0097 % and 0.054 %.
  - Q5: with Q3 and Q4 both prepared |0>, Q5's start-in-L is 0.39 % [0.36, 0.43], equal to the 0.36 % false-positive rate. Q5 therefore has no preparation leakage of its own.
  - Q2: 0.28 % (null 0.02 %).
  - Q1: not resolved.
- **Cross-check with Lienhard et al. Tab. V.** Their fits use a 0.5 us matched-filter boxcar for Q3 and give P(2|pi) = 0.052. m1's 0-500 ns sliding window gives Q3 prepared-|1> E3 = 5.05 %, which is consistent.
  - Their Q5 value (<<0.001) is blind to Q5's state, which sits perpendicular to the axis above |1>. A 1-D matched-filter projection maps that state onto |1>.

## 5. Why the four methods give different numbers (reconciliation)

The methods measure different things; once that is accounted for they agree.

1. **m1 tail (E3, 3 sigma0) and r E3** measure the same quantity. They agree to within 0.06 pp for Q3 prepared |1> and to within 0.02-0.16 pp elsewhere, and that difference is the frame systematic.
2. **m4 set A (|z| > 5)** uses the prepared bit's own pooled MAD as its scale, which is 1.29x the |0> noise width for Q3 and 1.16x for Q5. Its cut is therefore at about 6.5 sigma0 for Q3 and 5.8 sigma0 for Q5, so it keeps only the far tail.
   - I reproduced its candidate rate exactly: Q3 prepared |1> at 1 us gives 4.869 % (r) against 4.872 % (m4, train+test).
   - Threshold curve for Q3 prepared |1>: E2 15.5 %, E3 11.2 %, E4 9.1 %, E5 7.5 %, E6 6.1 %, m4-style 4.9 %.
   - For Q5: 6.1 / 4.5 / 2.3 / 0.72 / 0.13 / 0.21 %. Q5's state is only about 4 sigma0 off-axis at 1 us, so m4's 0.15 % for Q5 is a far-tail floor, not a disagreement.
   - m4's value lies in its screens rather than its rates. 70-96 % of its candidates survive the static-crosstalk, simultaneity, raw-glitch, DC, burst and drift screens, and in raw traces the candidates deviate only on their own tone. The off-axis events are therefore single-channel physical events, not artefacts.
3. **m2 mixture weight** splits Q3's continuous streak (|1> -> L transitions caught mid-window) between the far Gaussian and the cores, giving 8.9 % (model range 8.9-14.3 %). For Q5, Q1 and Q2 it agrees with E3 within its model range. Its Q4 "20.8 %" is the sub-cloud between |0> and |1>, which m2 itself declines to call leakage.
4. **m3 HMM** reports occupancies:
   - "start in L" (0.6-2.0 % for Q3 prepared |1>);
   - "visited L within 1 us" (14.5 %);
   - "in L at 1 us" (12.6 %).
   These bracket E3. Problems:
   - For Q1 and Q5, the HMM's L lies only 1.5-2.0 sigma per bin from |1>, so noise-only false positives are 1.2 % and 0.36 % for start-in-L. Its higher numbers for those qubits are inflated, and only its timing/correlation results are used.
   - For Q4, my 4-component GMM cross-tabulation of segment means shows that the HMM's |1> <-> L telegraph is an exchange between two halves of one broad |1> cloud: the halves trade 24-34 % of shots between 0.5-1 us and 1.2-2 us. That is not leakage.

## 6. Evidence for |2> (Q3) and for exchange (Q5, Q1)

- **Q3, destination.** L decays to |1>, not |0>:
  - HMM branching 99.91 % [99.90, 99.93];
  - shots that start in L: 96.4 % [95.8, 97.0] exit to |1>;
  - after the exit the shot dwells in |1> for a median of 400 ns (87-89 % stay at least 300 ns) before the normal |1>->|0> decay.
- **The same holds for L episodes that began from |0>** in prepared-|0> shots: 96.8 % [96.2, 97.4] exit to |1> (r_spot2, HMM paths).
  - A model-light check (nearest-reference labels of segment means, no transition model) gives the same 3:1 |1>:|0> destination ratio for prepared-|0> and prepared-|1> L shots.
  - A resonator artefact would return to the state it came from. A qubit level above |1> cascades through |1>.
- **Q3, source.** The probability of entering L depends on the qubit's actual state at readout start, not on its prepared label. Between 0.3 and 1 us it is 13.0 % (prepared |1>) and 15.8 % (prepared |0>) for shots that start in |1>, and 1.3 % / 1.05 % for shots that start in |0>. The state is therefore a physical level, mostly reached from |1>.
- **Q3, lifetime.** tau_L = 1.79 us [1.77, 1.80] (no-FIR 1.63, survival 1.81). Against Q3's |1>->|0> lifetime under readout that is 0.72x (2.47 us), and 0.37x (4.78 us) when Q1 and Q5 are prepared |0> (no exchange partner). That is close to the T1/2 expected for |2> -> |1>.
- **Q3, robustness.** Location is fixed across both prepared bits and all 16 neighbour configurations (per-configuration SD at most 0.05 separations). The result is the same without the FIR and without crosstalk correction. Candidates survive every confounder screen, and train and test agree.
- **Q5/Q1 exchange.**
  - Q5's |1>->L transitions coincide with Q3's |1>->|0> at 10.3x the chance rate at lag 0, and Q5's L->|1> with Q3's |0>->|1> at 20.4x. For Q1 the factors are 9.4x and 18.5x.
  - Q3's |1>->|0> rate per 100 ns is 2.07 % with Q1 = Q5 = |0>, 3.92 % with Q5 = |1>, 4.33 % with Q1 = |1> and 6.09 % with both: the contributions add.
  - Q1 and Q5 are in L together less often than chance (lift 0.6-0.7).
  - Q3's flip precedes Q5's perpendicular rise by about 200 ns.
  - A static readout cross-shift cannot produce Q5's L. The "Q3 = 0, Q5 = 1" pointer is the ordinary |1> location (static shifts are at most 0.94 sigma).
  - Lienhard's frequency table puts Q3's f01 = 5.000 GHz only 35 MHz from Q5's f12 = 4.965 GHz, so |1_Q3 1_Q5> and |0_Q3 2_Q5> are nearly degenerate.
  - INFERRED: readout-induced exchange into Q5's |2>. Q3's apparent "T1 decay" during readout is partly this exchange.

## 7. Error budget: canonical MCMit-CNN (1 us, test, n = 1,120,000 per qubit)

Source: `~/oraqle_reports/preds_MCMitCNN_{pb,tb}.npy`, with true bits verified equal to the prepared bits. Leaked shots among errors are mirror-corrected: errors with z > 3 minus errors with z < -3. The HMM "visited L within 1 us" flag is given in brackets.

| Qubit | CNN error | Leaked share of shots (1 us) | Leaked share of CNN errors | Same, prepared-\|0> errors only | CNN error on leaked prepared-\|1> shots | Paper column |
|---|---|---|---|---|---|---|
| Q1 | 2.98 % | 0.35 % | 0.14 % [0.08, 0.21] (1.1 %) | 0.15 % | 1.2 % (z > 3), 0 % (z > 5) | 0.0 % |
| Q2 | 25.43 % | 0.85 % | 0.32 % [0.29, 0.35] (0.67 %) | 0.59 % | 4.0 % / 0.4 % | 0.2 % |
| Q3 | 5.84 % | 5.96 % | **4.06 % [3.90, 4.22]** (4.05 %) | 10.7 % [10.0, 11.3] (HMM 16.2 %) | 2.2 % / 0.6 % | 0.1 % |
| Q4 | 5.31 % | n/a | n/a | n/a | n/a | 0.04 % |
| Q5 | 2.96 % | 2.29 % | **1.16 % [1.02, 1.30]** (3.9 %) | 7.5 % (HMM 22 %) | 0.3 % / 0.05 % | 0.0 % |

- Pooled over Q1, Q2, Q3 and Q5: 4,002 of 416,773 CNN errors fall on leaked shots, i.e. 0.96 % [0.93, 0.99]. The paper's pooled figure is 0.2 %, but its pool also includes Q4. Without Q2: 2.34 %.
- **For the ternary-head question:**
  - About 98 % of leaked prepared-|1> shots are read as "1". For QEC they are undetected leakage events: about 11-15 % of Q3 measurements of |1> end in L after 1 us, and about 3.6-5 % after 400-500 ns (E3 at 0-400 ns; m1 0-500 ns window).
  - For prepared |0>, leaked shots are 28-44 % misread, and they make up 11-16 % of Q3's P(1|0) errors.

## 8. Existing numbers: assessment

| Existing number | Value | Assessment | Why |
|---|---|---|---|
| Paper tab:error_modes "Leakage" (share of errors) | Q1 0.0, Q2 0.2, Q3 0.1, Q4 0.04, Q5 0.0 %; pooled 0.2 % | **Inconsistent** for Q3 and Q5; consistent in order only for Q1 and Q2; **not a leakage measurement** | It comes from HybridCNN (not MCMit-CNN) and counts only *misclassified* shots with \|z(time-mean \|IQ\|)\| > 3. That radial test misses an off-axis state, and leaked shots read as correct are never counted. Measured leaked share of canonical MCMit-CNN errors: Q3 4.06 % (30-40x), Q5 1.2-3.9 %, Q2 0.32 %, Q1 0.14 %. Its "near-boundary 97.4 %" is a residual by construction. |
| Same column converted to share of shots (s1local) | Q2 0.060 %, Q3 0.0081 %, Q4 0.0023 %, Q1 = Q5 = 0; pooled 0.014 % of 5.6 M qubit-shots | **Inconsistent**, by 700x for Q3 and >10^3 for Q5 | Measured leaked shares of shots: Q3 5.96 %, Q5 2.29 %, Q2 0.85 %, Q1 0.35 %. |
| Paper text "Excitation errors stay at 10^-3-10^-2 % of shots" | same amplitude-outlier metric | **Mislabelled and inconsistent** | Same metric, called "excitation". |
| three_level_labels.hdf5 (perp > 3 sigma_pooled, k = 3 verified) | overall 0.58 / 1.07 / 5.08 / 1.00 / 1.92 %; prep1 0.93 / 1.77 / 9.43 / 1.94 / 3.65 %; prep0 0.23 / 0.38 / 0.74 / 0.06 / 0.19 % | **Circular calibration** ("tuned to 0.03-1.1 %", and that range is not even met). Prepared-\|1> magnitudes are roughly consistent for Q2, Q3 and Q5 (1.77 vs 1.53, 9.43 vs 11.2, 3.65 vs 4.47 %) by coincidence of geometry. **Inconsistent** for Q1 and Q4, and for prepared \|0> | Q1 prepared-\|0> labels are pure Gaussian tail: sign-symmetric, 0 % survive. Q4's 1.94 % is its broad \|1> core: 0.013 % survive. Per-shot agreement with confounder-screened candidates is poor (Jaccard 0.008-0.52). The perpendicular-only rule also misses along-axis displacement. |
| Mude et al. 2025 (arXiv 2405.08982): 487 (Q1) to 17,642 (Q4) of 1.6 M traces | 0.030-1.10 % of all traces | **Not comparable** (heuristic cluster size); **inconsistent** if their numbering matches ours | Spectral clustering gives the size of the "smallest cluster", which depends on the method (a GMM gives 6-22 %). The largest measured population here is Q3 (5.96 % of shots, about 95,000 per 1.6 M), not Q4. Q4 is not measurable, and their Q4 cluster most likely reflects its multimodal \|1> response. Q1 overall is 0.35 % here vs 0.030 %. |
| Error-mode CSVs (herqules_error_modes_*, herqules_error_breakdown_{hybridcnn,full,...}) | amplitude outliers among errors at 1 us: HybridCNN Q2 673, Q3 91, Q4 26; HERQULES Q2 641, Q3 162, Q4 18; full-data Q2 537, Q3 137 | **Not a leakage measurement; inconsistent** | Same radial \|z\| > 3 rule. The measured count of leaked shots among canonical CNN errors for Q3 is about 2,660 of 65,425, against 91. Outliers concentrate on Q2, the low-SNR qubit. |
| Synthetic leakage sweeps (synth_transfer_sweep_leakage_F0.98_*) | gmean drops 0.978 -> 0.953 (HERQULES), 0.971 -> 0.942 (CNN) for p = 0-0.2 | **Not comparable** | Synthetic data with \|2> placed on the matched-filter boundary and labelled 1. In the real data the third states of Q1, Q3 and Q5 lie on or beyond the \|1> side, so they cost little two-level accuracy. The injected p range (5-20 %) brackets Q3's real prepared-\|1> rate, but the geometry differs. Run 102850 is broken. |
| HERQULES preclassifier "ket2" bucket | 41-88 % of \|1>-prepared | **Not comparable (geometric artefact)** | Tangent purity circles collect the Gaussian tail. |
| iq_error_gallery Q5 "Leakage \|2>" cloud (paper Fig. iq_taxonomy) | at least 2,500 shots; 1.20 % of a 4,000-shot \|1> subsample | **Consistent** with the measured Q5 population (4.5 %; the gallery rule is a stricter subset) | It contradicts the paper's own table (Q5 0.0 %). Its origin is mainly exchange with Q3. |
| s2cluster Gaussian-null excess (prep1, k = 3, HERQULES frame) | Q1 0.60, Q2 1.44, Q3 8.78, Q4 -0.72, Q5 3.14 % | **Consistent** | Same one-sided idea in a different frame. |
| s2cluster CNN errors on old labels | pooled 0.85 %; Q3 2.73 % | **Roughly consistent** (labels include Gaussian tails and miss along-axis shots) | Measured, mirror-corrected: Q3 4.06 %. |
| Lienhard et al. Tab. V P(2\|pi) | Q1 0.001, Q2 0.019, Q3 0.052, Q4 0.734, Q5 <<0.001 | Q3 **consistent** (5.05 % in 0-500 ns); Q2 roughly consistent; Q1 and Q5 **not comparable** (a 1-D matched filter cannot see off-axis states); Q4 not a \|2> population | Q4's 73 % most likely fits the split \|1> response. |
| s3lit thermal prior | at most about 2e-3 per \|1>-prepared shot | **Inconsistent for Q3** (11 %): its leakage is readout-induced, not thermal | The readout-induced range in the literature (0.1-8 % per readout, Hazra et al.) covers it. |
| build_three_level_labels docstring | "far from both": Q3 4 %; 3-way GMM 6-22 % | Q3 4 % is roughly consistent with 5.96 % overall; the GMM range is method-dependent | Not reproducible (no stored output). |
| inflate_leakage / three_level_common | "about 1 %/qubit, q0 low, q2 high"; "<5 %" | Qualitatively right (Q1 low, Q3 high), but circular in origin; Q3 prepared \|1> (11 %) exceeds "<5 %" | |

## 9. Uncertainties and limits

1. **Identity of the state (INFERRED).** IQ data cannot tell |2> from |3> or from a "chaotic" high level. For Q3, decay to |1> with a lifetime below T1 and a cascade via |1> favour |2>. The Q5/Q1 exchange mechanism is inferred from timing, additive decay rates, mutual exclusion and the frequency plan. A correlated readout-chain effect is not strictly excluded.
2. **Definition dependence.** E3 is a lower bound on "visited the third state within the window" and misses late entries. HMM "visited L" is higher (Q3 14.5 %, Q5 13.3 %, though Q5's figure is inflated by weak separation). The tables give method ranges. Rates grow with window length, so always quote the window.
3. **Frame systematic.** Two E3 implementations differ by up to 0.16 pp (Q1 prepared |1>, Q3 prepared |0>). The CIs above are envelopes of both. At 2 us the prepared-|0> frame is less stable (m1 2.3 % vs r 1.1 % for Q3), so the 2 us prepared-|0> values are indicative only.
4. **Q4.** Its leakage is not measurable. The multimodal |1> response (a stationary sub-cloud with shot-to-shot memory, plus a broad, internally exchanging main cloud) could hide a |2> population of any size up to the sub-structure itself. The prepared-|0> / prepared-|1> ratio test favours a fluctuator-type sub-response of |1>.
5. **Q2.** Low SNR (separation 1.2-2.1 noise sigma), and part of the signal is correlated with Q3's third state (mirror IFs +25.4 / -24.8 MHz; 30 % of the prepared-|1> excess and 65 % of the prepared-|0> excess sit in shots where Q3 is off-axis). The L lifetime (0.7 us) is not separable from Q2's own short T1 under readout (0.64-1.04 us).
6. **Start versus readout-induced.** 100 ns bins and the resonator ring-up do not resolve leakage present at t = 0 from leakage created in the first 200 ns. For Q3 the bracket is 0.61-1.97 % of prepared |1>.
7. **CNN cross-tab.** The leaked share of errors uses the mirror-corrected z flag or HMM paths, not ground truth. The two agree for Q3 (4.06 / 4.05 %) and differ for Q5 (1.16 / 3.9 %) because Q5's state is close to |1>.
8. **CIs** are statistical (Wilson, Clopper-Pearson or mirror Poisson) unless stated. Method ranges are not CIs.

## 10. Files

- Spot checks:
  - `/home/manosgior/Documents/GitHub/oraqle-ics/sims/leakage/r_spotcheck.py` and `r_spot2.py` (run on graham in docker, outputs in `~/oraqle-ics-sims/leakage/r/`).
  - Results: `/home/manosgior/Documents/GitHub/oraqle-ics/data/leakage/r_spotcheck.json`, `r_spot2.json`.
- Summary: `/home/manosgior/Documents/GitHub/oraqle-ics/data/leakage/r_leakage_summary.json` (machine-readable version of this file, built by `sims/leakage/r_build_summary.py`).
- Inputs: the m1tail / m2mix / m3time / m4conf / s1local / s2cluster / x_* files in the same directory.
