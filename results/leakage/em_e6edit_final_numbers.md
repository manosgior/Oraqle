# em_e6edit: final error-mode numbers used in the paper (2026-11-05)

These are the only numbers the leakage fix puts into the paper. The machine-readable version is
`data/leakage/em_e6edit_final_numbers.json`, which gives each number's value, 95% CI, status and source key,
plus a `paper_text_map` from each paper location to its numbers.
Generator: `sims/leakage/em_e6edit_finalize.py`. It reads existing results only:

- `em_e4recon_final.json`
- `em_e5vstat_summary.json` and `em_e5vstat_check.json`
- `em_e5vrepro_results.json`
- `em_e5vrepro_diag_partners.json`
- `LEAKAGE_REPORT.md`

**Setup.** TEST split, 1.12M shots per qubit. 1 µs = samples 0-499. The 1 µs model is the canonical MCMit-CNN
(`preds_MCMitCNN_pb.npy`); the 600 / 400 ns models are the soft MCMit-CNN trained with the same recipe. HERQULES means the
per-trace checkpoint used in tab:fidelity.

**CIs.** Shares are of that qubit's errors unless marked. All CIs are statistical 95% intervals:
- error rates: Wilson;
- L: ±1.96 sqrt(N+ + N-);
- O: binomial per configuration;
- var(T+S) = var(L) + var(O).

Systematics (reference definition, detector) are given as ranges.

**Status.**
- MEASURED: L (a k = 3 lower bound), O (an upper bound), T+S (a lower bound), and the late-window fractions.
- INFERRED: that S really is a state change, the efficiency-corrected T, the neighbour-exchange mechanism, and that leakage is "not |2>".

## 1. Table tab:error_modes (MCMit-CNN, 1 µs)

| Qubit | Error rate (% shots) | Leakage (% err) | State change T+S, ≥ (% err) | Overlap O, ≤ (% err) | Leaked shots, % of shots (% read correctly) |
|---|---|---|---|---|---|
| Q1 | 2.98 [2.95, 3.01] | 0.04 [−0.02, 0.11] → "0.0" | 81.2 [80.6, 81.8] | 18.8 [18.2, 19.4] | 0.41 [0.39, 0.42] (99.7 [99.5, 99.8]) |
| Q2 | 25.43 [25.35, 25.51] | 0.34 [0.32, 0.37] | **n/m** (late-core value of 1.2 is uninformative) | **n/m** (98.5 uninformative) | 0.90 [0.88, 0.92] (90.3 [89.7, 90.8]) |
| Q3 | 5.84 [5.80, 5.89] | 3.14 [3.00, 3.29] | 76.8 [76.1, 77.4] | 20.1 [19.5, 20.7] | 6.14 [6.10, 6.19] (97.0 [96.9, 97.1]) |
| Q4 | 5.31 [5.27, 5.36] | n/m (multimodal \|1>) | 86.8 [86.3, 87.4], indicative (82.1-88.7 by reference) | 13.2 [12.6, 13.7], indicative | n/m |
| Q5 | 2.96 [2.93, 2.99] | 1.03 [0.90, 1.17] | 90.2 [89.8, 90.7] | 8.8 [8.3, 9.2] | 2.40 [2.37, 2.43] (98.7 [98.6, 98.9]) |
| **Q1, Q3, Q5** | 3.93 [3.91, 3.95] | 1.83 [1.75, 1.91] | **81.3 [80.9, 81.6]**; reference band 78.6-81.4 | 16.9 [16.5, 17.3]; band 16.8-19.6 | – |

**Why Q2 is n/m.** In the unused 1-2 µs window, Q2's |0> and |1> cores are only 0.52 σ apart (Q1 5.5, Q3 7.7, Q4 13.0, Q5 9.7 σ).
That makes O ≈ N, and S ≈ 0, by construction.

**Why Q2 is still called SNR-bound.** A linear filter (TRAIN LDA) has Δ = 1.151, so its Gaussian overlap alone predicts
Φ(−Δ/2) = 28.3% error, more than the CNN's measured 25.4%.

**Pools that are not headlined:**
- Q1, Q3-Q5: 83.0 / 15.7 / 1.26. Q4's leakage enters as 0, and its O is only indicative.
- All 5: 34.1 / 65.2 / 0.71. Q2's circular split dominates it.

## 2. Results paragraph

| Claim in the paper | Value [95% CI] |
|---|---|
| Q3 state change "≥76%", 1 µs | 76.76 [76.13, 77.40] (the CI lower edge, rounded down) |
| Q5 state change "≥90%" | 90.22 [89.76, 90.67] |
| Q3 overlap "≤21%" | 20.09 [19.47, 20.72] (the CI upper edge, rounded up) |
| Q5 overlap "≤9%" | 8.75 [8.31, 9.19] |
| Q3 prepared-\|1> errors on the \|0> side of the 1-2 µs midline (half-plane) vs correctly read shots | 90.97 [90.72, 91.21] vs 32.96 [32.83, 33.09] |
| Same for Q5 | 94.66 [94.40, 94.92] vs 15.33 [15.23, 15.42] |
| "only 2-8% show a step inside the window" | Two detectors (CIs in the json): Q1 4.33 [4.08, 4.57] (e2seg) to 7.76 [7.39, 8.13] (e5vrepro, empirical null). Q5 5.02 [4.76, 5.29] to 7.36 [7.01, 7.70]. Q3 (excitation only) 1.68-1.75 |
| Q3 overlap at 400 ns, "40%" | 40.14 [39.35, 40.93] |
| Q5 overlap at 400 ns, "56%" | 55.90 [55.04, 56.76] |
| Q3 prepared \|1> leaked, 1 µs, "11.5%" | 11.46 [11.38, 11.55]; by frame and IF, 11.3-11.6 |
| Q5 prepared \|1> leaked, "4.7%" | 4.72 [4.66, 4.78] |
| Leaked shots read correctly, "97-99%" | Q3 97.0 [96.9, 97.1]; Q5 98.7 [98.6, 98.9] |
| Q3 leakage at 400 / 600 / 1000 ns, "3.5 / 6.5 / 11.5%" | 3.51 [3.46, 3.56] / 6.45 [6.39, 6.52] / 11.46 [11.38, 11.55] |

## 3. Analysis and Takeaway #5

| Claim in the paper | Value [95% CI] |
|---|---|
| Q1, Q3, Q5 pool: state change 81% (79-81% across references) | 81.27 [80.90, 81.64]. References: e4 core 81.27, r = 1.0 81.36, r = 2.0 80.66, r = 2.5 80.23, half-plane 78.60, e5vrepro 2σ disc 80.73 |
| Same pool: overlap 17% (17-20% across references) | 16.90 [16.54, 17.27]; band 16.8-19.6 |
| Same pool: leakage 1.8% | 1.83 [1.75, 1.91] |
| Q2: "its Gaussian overlap alone predicts 28% error" | 28.26% (Δ = 1.151), against CNN 25.43 [25.35, 25.51] |
| State-change share of the pool, 400 ns → 1 µs: 37% → 81% | 37.35 [36.90, 37.80] → (600 ns: 65.35 [64.92, 65.77]) → 81.27 |
| Per-shot overlap falls "2.5-14×" from 400 ns to 1 µs | Q3 2.52 [2.43, 2.61], Q5 10.5 [9.97, 11.07], Q1 13.9 [13.5, 14.4] |
| Per-shot state change, 400 ns → 1 µs | Q3 4.355 [4.297, 4.413] → 4.484 [4.447, 4.521]; Q5 2.147 [2.105, 2.188] → 2.671 [2.658, 2.685] |
| "leakage triples on Q3" | 11.46 / 3.51 = 3.27 [3.22, 3.32] |
| "rarely as a mid-window step" | See the step row in §2 |
| HERQULES errors that are also CNN errors (1 µs) | Q3 68.23 [67.92, 68.55]; Q5 92.12 [91.83, 92.40] |
| HERQULES Q3 state change | 69.38 [68.70, 70.06] (its overlap is 27.14 [26.48, 27.81]) |
| "MCMit-CNN makes 23% fewer Q3 errors" | 23.1 [22.4, 23.9] (65,425 vs 85,129 errors) |
| Takeaway "~80%" | 81.27, with reference band 78.6-81.4 |
| Takeaway "≤20%" | 16.9; ≤19.6 under every reference |
| Takeaway "0.4-6% of shots silently leak" | Q1 0.41 [0.39, 0.42] up to Q3 6.14 [6.10, 6.19] |

## 4. Appendix (App. error-modes)

**Order swap.** Putting the step test before leakage moves O and S by ≤0.03 pp. On Q3, 1.44 pp of errors move from L to T (L 3.14 → 1.70, T 1.75 → 3.18).

**Reference and model-free checks:**
- Q2's late-window cores are 0.5 σ apart.
- Across references, the pooled state change stays at 79-81%.
- Half-plane check: 89-95% of prepared-|1> errors on Q1 and Q3-Q5 are on the |0> side, against 6-33% of correctly read shots. Per qubit, errors vs correct shots:
  - Q1: 90.07 [89.71, 90.42] vs 5.81
  - Q3: 90.97 vs 32.96
  - Q4: 89.08 vs 15.88
  - Q5: 94.66 vs 15.33
- That implies a state-change share of 89.1% for Q1/Q3/Q5, against 90.3% attributed.

**Step test (T):**
- The FP correction is conservative on misread stationary shots: on Q1/Q5, 24 flags observed vs 54.1 expected (ratio 0.44).
- Sensitivity: P(flag) ≥ 0.5 over about 225-775 ns. Efficiency given a misread is 16-28% (e2seg detector; 36-56% for the e5vrepro detector).
- Q3's relaxation test fails its null. On quasi-null shots, observed/expected flags are 1.035 / 0.958 when Q1, Q4 and Q5 are all in |0>, and 5.39 / 5.18 when all three are excited (e5vrepro). The neighbour mechanism is INFERRED.

**Leakage at 1 µs:**
- Q3: 11.46 [11.38, 11.55]% of prepared |1>, 0.82 [0.79, 0.85]% of prepared |0>.
- Q5: 4.72 [4.66, 4.78]%. Q2: 1.62 [1.58, 1.65]%. Q1: 0.82 [0.79, 0.84]% (all prepared |1>).
- Leaked shots are 13.39 [12.72, 14.05]% of Q3's prepared-|0> errors and 3.14% of all Q3 errors.
- 99.32 [99.25, 99.40]% of leaked prepared-|1> Q3 shots are read as |1>.

## 5. Verifier corrections: verdicts

All the corrections from both verifiers were accepted. None was rejected.

**e5vstat:**
1. **"300×" claim.** Dropped. It never reached the paper.
2. **Q2 split is circular.** Q2's split is shown as n/m. "SNR-bound" now rests on the LDA, and the all-5 pool is not headlined.
3. **Q4 in the headline.** The headline pool is now Q1, Q3, Q5. The Q4 row is footnoted as indicative.
4. **Reference systematic.** The "79-81% / 17-20% across references" band is now in the text.
5. **Late-window check understated.** It now uses half-plane numbers (89-95% vs 6-33%), with the identity estimate.
6. **Step-test null.**
   - T is now labelled a conservative lower bound, and the sensitivity window is 225-775 ns.
   - The efficiency-corrected values (|1> 10-27%, |0> 19-51%) are kept only in the json.
   - Q3 excitation T stays at 1.7%. e5vrepro's detector passes the errors-only null (1.09) and gives 1.72.
7. **Labels.** O now means "errors on shots that stayed in their state", and T+S absorbs sub-threshold leakage. With k = 1, Q3 state change is ≥75.3%.

**e5vrepro:**
1. **T depends on the detector.** T is now quoted only as a two-detector bracket (2-8%) and is kept out of the table.
2. **FP over-subtraction.** Both verifiers measured it, so T is called conservative.
3. **Q3 relaxation cause.** The cause is now neighbour-driven 1→0→1 excursions; the mechanism is INFERRED.
4. **All-5 pool.** Not headlined.
5. **Core-size dependence.** Replaced by the half-plane definition.
6. **Q3 leakage figure.** 11.5% (11.3-11.6 by frame and IF) is used instead of 11.3%: 3.5 and 6.5 are single line-frame values, so 11.5 keeps one consistent frame.
