# Leakage in the Lienhard 5-qubit readout dataset, and whether MCMit-CNN should flag it

> **Location note (Oraqle repository, added 2026-11-05).** This report was written in the paper repository; its
> body is unchanged. Paths in it map to this repository as follows: `data/leakage/<f>` -> `results/leakage/<f>`
> (the three `em_e3herq_preds_*.npz` are not committed, see `leakage/README.md` S4); `sims/leakage/<f>.py` ->
> `leakage/<f>.py` (superseded versions in `leakage/iterations/`); `sims/qsd`, `sims/soft_readout_cnn.py`,
> `sims/sweep_soft.py`, `data/sims/soft_params.json` -> the QEC/system part of the repository;
> `MCMit/qubit_state_discrimination/...`, `Oraqle/Discriminators/...` -> `discriminators/...`;
> graham `~/oraqle-ics-sims/leakage/<stage>` -> `$ORAQLE_WORK/leakage/<stage>` (`leakage/lkpaths.py`);
> `~/oraqle_reports` -> `results/discriminators`. Absolute `/home/manosgior/...` paths refer to the original machines.
> Numbers in this report predate the final error-mode attribution: the paper uses `results/leakage/em_e8fix_final_numbers.json`.

Final report of the leakage study (tag `f`), written 2026-11-04 for the Oraqle ICS'27 paper.
It merges four measurement agents (m1tail, m2mix, m3time, m4conf), three surveys (s1local, s2cluster, s3lit),
the reconciliation (r), three adversarial verifiers (vstatistics, vphysics, vreproduction) and one new run of mine
(`f_cnn_leak_xtab`). Where a verifier corrected a number, the corrected value is used here, not the reconciled one.

**Conventions used throughout**

- **Qubit names.** Paper Q1..Q5 = code q0..q4. Bit q of the 5-bit label y is qubit q, and `(y>>q)&1` is its prepared bit.
  Q2 (q1) is the low-SNR "hardware" qubit.
- **Sampling.** 500 MS/s, 2 ns per sample. The stored trace is 1000 samples = 2 us.
- **Windows.**
  - The paper's ML "1 us" (the "full trace") is raw samples 0-499. That is the shared `feat1us` bins 0-9: causal FIR, the same information as the CNN input.
  - 600 ns = samples 0-299.
  - 400 ns = samples 0-199.
  - 2 us = samples 0-999. The paper never uses samples 500-999.
- **Data.** TEST split unless stated: 1,120,000 shots, i.e. 560,000 per prepared bit per qubit. TRAIN (240,000 per bit) is a sampling replicate of the same run, not an independent experiment.
- **Statistic.**
  - E3 = [N(z > 3) - N(z < -3)] / N, the one-sided mirror excess.
  - z is the perpendicular distance from the |0>-|1> line through the two density *cores*, measured per neighbour configuration. Its unit is the prepared-|0> noise width of that configuration.
  - E3 is a k = 3 convention. It is a **lower bound** on "was in the third state during the window".
  - CIs are statistical 95% intervals (±1.96 sqrt(N+ + N-)/N, Wilson or Clopper-Pearson). Frame systematics are quoted separately, as a range.
- **MEASURED vs INFERRED.** Off-axis populations, where they sit, how fast they appear, how they decay and how they correlate across qubits are MEASURED. What physical state they are is INFERRED. No shot in this dataset was prepared in |2>.

---

## 1. TL;DR

### Q1. Does the dataset contain real leakage, and how much?

**Yes, but it depends on the qubit, and it is not known to be |2>.**

**Q3 (q2): strong.** A population outside the computational subspace is present.
- Size at 1 us: **11.3% of prepared-|1> shots** (stat. ±0.09 pp; frame range 10.7-11.8%) and **0.83% of prepared-|0> shots** (range 0.81-0.87%), i.e. **6.1% of all Q3 shots** (n = 1.12M).
- How big depending on the definition: 7.7% (k = 5) to 15% (k = 1, or the HMM's "visited by 1 us" 14.5%).
- Most of it is *created during the readout*:
  - only 0.4-2.0% of prepared-|1> shots are in it at the start;
  - over equal-length 500 ns windows it grows 4.8% -> 11.1% -> 13.5% -> 13.7%;
  - at the paper's 400 / 600 ns windows it is 3.5% / 6.5% of prepared |1>.
- It decays back to |1>, rarely to |0>.
- It survives every confounder screen: static crosstalk, simultaneity, raw glitches, DC, bursts and drift (95-96% of candidates survive).

**Q5 (q4) and Q1 (q0): measured, but tied to Q3.**
- Prepared |1>: 4.7% for Q5 and 0.78% for Q1. Prepared |0>: 0.10% and about 0.
- Both appear almost only when Q3 is prepared |1>, and they line up in time with Q3 losing its excitation (10-20x chance).
  - Q5 prepared |1>: 0.3-0.4% when Q3 = Q4 = 0, and 8.6-9.7% when Q3 = 1, Q4 = 0.
- So the 4.7% is an average fixed by the dataset's balanced design, not a property of Q5 alone.

**Q2 (q1): probable.** 1.55% of prepared |1> and 0.18% of prepared |0>, at low SNR and partly correlated with Q3.

**Q4 (q3): inconclusive.** Its |1> response itself splits into several sub-clouds.

**Identity.**
- For Q3, the population sits near the point where the resonator would be if the qubit stopped pulling it. Under linear dispersive readout with the published device parameters, that is 4-7 times the |0>->|1> pull, where |2> should give 0.76.
- That points to a measurement-induced high-excitation state rather than |2>. A genuine |2> would sit only about 0.55 separations from |1>, and that region is **not measured** by anything here.
- Q1's location is compatible with its |2>. Q5's location matches no dispersive level.

**The binary CNN barely sees it.**
- At least 98.8% of leaked prepared-|1> shots are read as |1>.
- Leaked shots are **3.1% [3.0, 3.3] of MCMit-CNN's Q3 errors** (13% of its prepared-|0> errors), 1.05% of Q5's, 0.34% of Q2's and about 0 of Q1's.
- Pooled over Q1, Q2, Q3 and Q5: 0.81% [0.78, 0.84] of errors.

**Existing numbers that are wrong or circular:**
- The paper's Table `tab:error_modes` "Leakage" column (0.0-0.2% of errors) is an amplitude-outlier count of the **HybridCNN's** errors. It is not a leakage measurement, and it undercounts Q3 by **23-31x**.
- The text's "excitation errors 10^-3-10^-2% of shots" is the same metric. The measured populations are 0.4-6% of shots.
- `three_level_labels.hdf5` and the "0.03-1.1%" target are circular.
- The "near-boundary 97.4%" share is a residual by construction (§2).

### Q2. Should MCMit-CNN also discriminate |2>?

**Conditional yes, in a narrower form.** Add a per-qubit **"non-computational (L)" flag** next to the unchanged binary bit. Do not add a "|2> class", and do not report 3-class accuracy as a fidelity.

**There is enough signal to learn.**
- Q3 has about 30k L shots in TRAIN and about 69k in TEST. Q5 and Q2 have a few thousand.
- The cost is tiny: +85 parameters (+2.8%) and +80 MACs (+0.6%). The output fits in the 2-bit code that §7's soft readout already pays for.

**But the value cannot be shown with today's tools.**
- I measured that using L flags as **erasures of the current outcome makes things worse**:
  - pooled over Q1/Q3/Q4/Q5 at 1 us, L-flagged measurements are wrong only 2.5% of the time;
  - erasing them raises e_u from 4.27% to 4.33%;
  - a confidence flag of the same size lowers e_u to 3.43%.
  The same holds at 600 and 400 ns.
- L is valuable only as a **herald of the post-measurement state**: a qubit left in L is not returned to |0> by a conditional reset. On Q3 that happens after about 12% of |1> measurements at 1 us.
- lattice-sim has no leakage, and ECCentric's leakage is frozen at build time and visible to the decoder. So showing the value needs a leakage-aware simulation.

**Without such a simulation:** correct the error-mode table and report leakage as a hidden channel, but put the flag in Discussion/Future work.

**What would make it an unconditional yes:** a calibration run with prepared |2> (and a readout-power/duration sweep), plus a leakage-aware QEC model.

**The prior `train_three_level_cnn.py` should not be reused**, except for its ten-line `CNN3` head (§5.7).

---

## 2. Existing numbers and their status

| # | Existing number (source) | What it actually is | Status |
|---|---|---|---|
| 1 | Table `tab:error_modes` "Leakage" (share of errors): Q1 0.0, Q2 0.2, Q3 0.1, Q4 0.04, Q5 0.0%, pooled 0.2% (`ml_discriminators.tex`; `herqules_error_breakdown_hybridcnn.csv`, 1000 ns rows; `_train_and_analyze_hybridcnn.py`) | **HybridCNN** (not MCMit-CNN) misclassified shots with \|z\| > 3 on the time-mean \|IQ\| amplitude, z taken against a train mean/std pooled over all states, counted after the T1 test. A radial test cannot see an off-axis state, and leaked shots read "correctly" are never counted. Counts: Q2 673/288,485, Q3 91/68,811, Q4 26/63,026, Q1/Q5 0. | **Wrong as a leakage number.** Measured leaked share of MCMit-CNN errors: Q1 ≈0 (consistent), Q2 0.34% (same order), **Q3 3.1% (23x the unrounded 0.132%, 31x the printed 0.1%)**, **Q5 1.05% (table: 0.0)**, Q4 not measurable; pooled 0.81%. |
| 2 | Same table, error rates 3.2 / 25.8 / 6.1 / 5.6 / 3.0% | HybridCNN error rates (3.21 / 25.76 / 6.14 / 5.63 / 3.04%) | **Mislabelled.** Canonical MCMit-CNN (`preds_MCMitCNN_pb.npy`, n = 1.12M per qubit): 2.98 / 25.43 / 5.84 / 5.31 / 2.96%. |
| 3 | Same table, "Near-boundary" 97.4% pooled; Takeaway #5 "99.8% of errors are device-set" | Near-boundary means P(correct) < 0.6, which every error satisfies. `any_n_near_boundary == n_error` in every CSV row. | **Residual by construction**, not a measurement. Takeaway #5 must be reworded. |
| 4 | "Excitation errors stay at 10^-3-10^-2% of shots" (Results paragraph) | The same amplitude-outlier count, as a share of shots (Q2 0.060%, Q3 0.0081%, Q4 0.0023%) | **Wrong/mislabelled.** Measured off-axis populations: Q3 6.1%, Q5 2.4%, Q2 0.87%, Q1 0.39% of shots (1 us). |
| 5 | `three_level_labels.hdf5` (`build_three_level_labels.py`): prepared \|1> 0.93 / 1.77 / 9.43 / 1.94 / 3.65%; prepared \|0> 0.23 / 0.38 / 0.74 / 0.06 / 0.19% | Two-sided \|perp\| > 3·sigma_perp in the HERQULES frame, with sigma pooled over both prepared bits, stats pooled over train+test. Exact reproduction confirms k = 3.0. | **Circular.** The "tuned to 0.03-1.1%" claim is not even met (0.58-5.08%). Q1 prepared-\|0> labels are pure Gaussian tail (0% survive screens). Q4 prepared-\|1> labels are its broad core (0.013% survive). Jaccard overlap with screened candidates 0.008-0.52. The Q2/Q3/Q5 prepared-\|1> magnitudes are close to the measured values only by coincidence. |
| 6 | Mude et al. 2025: 487 (Q1) to 17,642 (Q4) of 1.6M traces = 0.030-1.10% | Size of the smallest spectral cluster of mean IQ; no \|2> calibration; numbering not confirmed | **Not comparable.** If the numbering matches, it is inconsistent: the largest population is Q3 (about 95k of 1.6M), Q1 is 0.39% not 0.030%, and Q4's cluster is most likely its split \|1> response. |
| 7 | Lienhard Tab. V P(2\|pi): Q1 0.001, Q2 0.019, **Q3 0.052**, Q4 0.734, Q5 ≪0.001 | Weight of the third peak in a 1-D matched-filter trimodal fit; no \|2> was prepared | Q3 is **consistent as a third-peak weight** (our 0-500 ns E3 = 5.05%) but is **not** evidence that the state is \|2>. Q2 is roughly consistent. Q1/Q5 are not comparable: a 1-D matched filter is blind to off-axis states. Q4's 0.734 is an artefact (P(2) cannot be 73% with F_pi = 0.970). |
| 8 | Error-mode CSVs (`herqules_error_modes_*`, `herqules_error_breakdown_{full,hybridcnn}`): e.g. HERQULES Q2 641, Q3 162 amplitude outliers among errors | Same radial heuristic | **Wrong as leakage.** About 2,050 of 65,425 canonical Q3 CNN errors fall on leaked shots, against 91-162 in the CSVs. |
| 9 | Synthetic leakage sweeps (`synth_transfer_sweep_leakage_F0.98_105146`): gmean 0.978 -> 0.953 (HERQULES), 0.971 -> 0.942 (CNN) for p = 0-0.2 | Synthetic \|2> placed on the matched-filter boundary and labelled 1 | **Not comparable.** The real third states lie on or beyond the \|1> side and cost little 2-level accuracy. Run 102850 is broken. |
| 10 | HERQULES preclassifier "ket2" bucket: 41-88% of prepared \|1> | Tangent-circle geometry collects the Gaussian tail | **Geometric artefact.** |
| 11 | `iq_error_gallery_q5` cloud (paper Fig. iq_taxonomy): at least 2,500 shots; 1.2% of a 4,000-shot prepared-\|1> subsample | Geometric selection on HERQULES mean IQ | **Consistent** as a strict subset of the Q5 population (4.7%). It contradicts the table's Q5 0.0%. The label "Leakage \|2>" is INFERRED and Q3-conditioned. |
| 12 | s2cluster Gaussian-null excess (prepared \|1>, k = 3): 0.60 / 1.44 / 8.78 / -0.72 / 3.14% | Excess over a Gaussian null with each cluster's own width | **Consistent** in order (slightly low, because the own width is inflated by the population). |
| 13 | s2cluster CNN errors on old labels: pooled 0.85%, Q3 2.73% | Cross-tab against circular labels | **Roughly consistent** with the corrected 0.81% / 3.1%. |
| 14 | `inflate_leakage.py` "~1%/qubit, q0 low, q2 high"; `three_level_common.py` "\|2> is rare (<5%)" | Read off the circular labels | Qualitatively right, **circular in origin**. |
| 15 | Thermal prior (s3lit): at most about 2e-3 of \|1>-prepared shots per 1 us | Boltzmann + detailed balance, upper bound | **Inconsistent for Q3**, whose population is readout-induced (about 2.6% per 100 ns from \|1>). Within the literature's 0.12-7.8% per readout (Hazra 2024). |
| 16 | ECCentric leakage model (Willow 2.5e-4 per op, propagation 2.0e-4) | Drawn once per circuit build (seed 123) as p = 1 Paulis; PyMatching corrects them through the DEM (LER unchanged, 1.5e-5 with and without) | **Cannot represent** the cost of leakage or the value of a flag. |
| 17 | Reconciled summary (r): Q3 prepared \|0> 0.65%, 2 us row (Q3 23.2% / 1.1%), Q3 leaked share 4.06%, pooled 0.96%, "30-40x", "tau_L ≈ T1/2 is the \|2> signature", "\|2>-like" for Q3, Q5 4.47% [4.34, 4.60] | Frame artefact (mean-based rotation tilted 7 deg by the population itself); broken 2 us frame; wrong T1 | **Superseded** by §3 (see §4 for each correction). |
| 18 | Shared-features README: "rotation is correct", "residual IF drift < 0.7 kHz" | Rotation aligns *means*, not cores (Q3 core line tilted -7.3 deg at 1 us, -13 deg at 2 us). Q3's IF is 28 kHz off (biased by the growing population); HERQULES's -24.79 MHz is right. | **Erratum.** E3 at 1 us is low by about 0.2 pp for Q3 and 0.01-0.06 pp for Q2. Any statistic centred per state (instead of on the core line) is biased. |

---

## 3. Independent measurement per qubit

### 3.1 How it was measured

Five independent lines of evidence, all on the paper's 1 us window unless stated:

1. **m1tail.** Per-state centres and widths for all 32 prepared states; the mirror excess E(k); Student-t and mixture models; 2-D distance to the segment; a cut-stability test; sliding windows.
2. **m2mix.** 2-D binned-likelihood mixtures with |0>/|1> cores, a T1 bridge, and extra Gaussians. Flexible cores, survival of the far component, bootstrap, parametric LR nulls.
3. **m3time.** 3-state HMM over the 20 bins of 100 ns, fitted on TRAIN and scored on TEST. A label-switching bug was fixed. Null simulations; replication without the FIR.
4. **m4conf.** Five confounder screens on |z| > 5 candidates: static crosstalk, simultaneity, raw glitch, burst/drift, and DC. Raw-trace probes of all five tones.
5. **Line-frame E3, reproduced four times.**
   - vstatistics and vreproduction (independent demodulation from raw HDF5 on a 384k subsample).
   - My `f_cnn_leak_xtab.py`: line through the per-configuration core modes, fitted on TRAIN and applied to TEST, prepared-|0> width.
   - My frame reproduces the -7.4 to -8.7 deg core-line tilt for Q3 that vstatistics found.

### 3.2 Headline table (1 us = raw samples 0-499, TEST, 560,000 shots per prepared bit)

"Best" is the central value across 6-9 frames. The statistical CI is from my line frame, n = 560,000. The frame range spans m1tail, the r frame where it is not biased, vstatistics A/B, vreproduction box/bw and f. The definition range spans k = 1-5, the m2 mixture weight and the m3 HMM occupancy. A range is not a CI.

| Qubit | Prepared \|0> | Prepared \|1> | All shots | Def. range (prep \|1>) | Verdict |
|---|---|---|---|---|---|
| **Q3 / q2** | **0.83%** (f: 0.82% [0.80, 0.85]; frames 0.81-0.87) | **11.3%** (f: 11.46% [11.37, 11.54]; frames 10.7-11.6, +0.2 pp with the correct IF) | **6.1%** (5.9-6.2) | 7.7% (k = 5) - 15.2% (k = 1); m2 8.9%; HMM visited 14.5% | **Leakage present, strong.** A non-computational state, identity unresolved (geometry disfavours \|2>). |
| **Q5 / q4** | 0.10% (f: 0.095% [0.079, 0.111]; 0.09-0.12) | **4.7%** (f: 4.76% [4.70, 4.82]; frames 4.35-5.0). Random effects over the 16 neighbour configurations: [3.7, 6.3] | 2.4% (2.2-2.5) | 0.72-0.80% (k = 5), 2.3-2.4% (k = 4); m2 3.1% | **Measured, Q3-conditioned.** Q3,Q4 = 00: 0.31-0.42%; 01: 1.3-1.6%; 10: 8.6-9.7%; 11: 6.8-8.2%. Origin unresolved. |
| **Q1 / q0** | ≈0 (f: -0.006% [-0.020, 0.008]; at most 0.02%) | **0.78%** (f: 0.86% [0.83, 0.88]; frames 0.61-0.86; 0.53% with own width) | 0.39% | 0.15-0.18% (k = 4); m2 0.43% | **Measured, Q3-conditioned.** 0.12-0.13% when Q3 is prepared \|0>, 1.3-1.5% when Q3 is prepared \|1>. Location compatible with Q1's \|2>. |
| **Q2 / q1** | 0.18% (f: 0.19% [0.18, 0.21]; 0.16-0.19) | **1.55%** (f: 1.62% [1.59, 1.66]; frames 1.49-1.62) | 0.87% | 0.83% (k = 5) - 2.8% (HMM) | **Probable**, moderate confidence (SNR 1.2; 4.7% when Q3 is off-axis vs 1.6% otherwise). |
| **Q4 / q3** | n/m | n/m | - | The minus side of the line is also in excess (f: 18,633 shots vs about 756 Gaussian) | **Inconclusive.** Its \|1> response has sub-clouds, including a stationary one at (0.41, -0.22) sep (21% of prepared \|1>) whose pull lies between those of \|0> and \|1> (a fluctuator-like reduced-shift \|1>, not \|2>). |

### 3.3 Window dependence (prepared |1> / prepared |0>, E3)

| Window | Q1 | Q2 | Q3 | Q5 |
|---|---|---|---|---|
| 400 ns (samples 0-199) | 0.06-0.09% / 0 | 0.38-0.40% / 0.05-0.07% | **3.5-3.7%** / 0.12-0.13% | 0.40-0.48% / ≈0 |
| 600 ns (samples 0-299; f, boxcar) | 0.24% / 0.01% | 0.79% / 0.09% | **6.46% [6.39, 6.53]** / 0.38% | 1.70% / 0.02% |
| 1 us (samples 0-499) | 0.78% / 0 | 1.55% / 0.18% | **11.3%** / 0.83% | 4.7% / 0.10% |
| 2 us (samples 0-999) | 2.3-3.0% / ≤0.02% | 3.0% / 0.35% | **≈20% (frames 16.4-21.4%)** / 2.3-2.4% | 9.8-10.1% / 0.6-0.7% |

- The 400 ns-2 us growth is partly an SNR effect.
- Equal-length 500 ns windows (prepared |1>; 0-500 / 500-1000 / 1000-1500 / 1500-2000 ns):
  - Q3: 4.8 / 11.1 / 13.5 / 13.7%
  - Q5: 1.1 / 5.5 / 7.9 / 8.9%
  - Q1: 0.11 / 0.75 / 1.09 / 1.32%
  - Q2: 0.56 / 1.65 / 1.89 / 1.68%
- So the population genuinely builds up after ring-up, and Q3 saturates at about 13.5% after 1 us.
- The 2 us values from r (Q3 23.2% / 1.1%) come from a broken frame and are withdrawn.

**Systems consequence.** Readout-induced leakage is a hidden cost of long readout. On Q3 it roughly halves from 1 us to 600 ns (11.3% -> 6.5% of prepared |1>). This favours §7's 600 ns operating point, beyond what fidelity shows.

### 3.4 Dynamics, location, identity (Q3 unless stated)

**Present at the start vs created during the readout (MEASURED).**
- Start-in-L is 1.97% [1.93, 2.01] of prepared |1> (Viterbi over bins 1-2, null 0.005%). The HMM's posterior for bin 0 gives 0.61%; an NNLS fit over entry times gives 0.41%.
- Entry rate: about 2.6% per 100 ns from |1> (HMM). Entries from |0> are about 10x rarer (1.0-1.3% of |0>-start shots within 1 us).
- Entry depends on the qubit's *actual* state at readout start, not its label: from |1>, 13.0% (prepared |1>) and 15.8% (prepared |0>); from |0>, 1.3% and 1.05%.

**Exit (MEASURED; the robust part).**
- L goes back to |1>, not |0>:
  - HMM branch: 99.9%;
  - L-starters: 96.4% go to |1>;
  - segment classifier: about 9:1 for |1>:|0>.
- My late-window check (1-us L-flagged shots seen in the 1.2-2.0 us window, mirror-corrected): 77% still L, 16% in |1>, 7% in |0>. For comparison, 28% of prepared-|1> core shots end in |0>.
- The ratio is the same for prepared-|0> episodes (96.8%), so it is not an artefact that returns to its origin state.

**Lifetime (model-dependent).**
- HMM: tau_L = 1.79 us [1.77, 1.80]. Model-light segment analysis: 2.7-3.5 us.
- Both are far below Lienhard's T1(Q3) = 21.4 us (intrinsic |2> decay would take about 10.7 us). So the decay is **readout-driven**.
- The earlier "tau_L ≈ T1/2, the |2> signature" argument compared against the |1>->|0> lifetime *under readout* (2.47 us) and is withdrawn.

**Location and identity (INFERRED, vphysics).**
- Q3's state sits at (1.9, +1.5) separations: beyond |1> and above the axis. It is in the same place for both prepared bits, all 16 neighbour configurations, start-in-L and mid-readout entries.
- On the resonance circle fixed by c0, c1 and the measured direct-path direction, it pulls the resonator **4.0-6.8x** as far as |0>->|1> does (ring-up fit: 4.2-5.2x).
- Predictions: |2> 0.76x, |3> 1.35x, a decoupled ("bare") resonator 4.77x.
- Forcing 0.76x raises the rms residual from 0.4-0.85 to 3.3-6.6 ADC.
- The direct decay to |0> is about 0.1%. If L were "|1> with an odd resonator response", it would decay straight to |0> in at least 8.4% of exits. So **L is a real non-computational qubit state, most consistent with a measurement-induced high-excitation (MIST/ionisation-type) state**.
- Caveats: this assumes a linear resonator and the published chi/kappa, and the fitted kappa is 2x off.

**Other qubits.**
- Q1's L pulls 0.69-0.83x, compatible with its |2> (0.73x).
- Q5's L matches no dispersive level for any probe detuning.
- Q2's pull is 7-13x, with an unreliable step size.

**The Q3 coupling (MEASURED timing; mechanism INFERRED).**
- Timing: Q5 enters L when Q3 drops |1>->|0> (10.3x chance at lag 0) and leaves when Q3 jumps back (20.4x). Q1: 9.4x and 18.5x.
- Q3's decay rate per 100 ns adds up with its partners: 2.07% (neither) / 3.92% (Q5 = 1) / 4.33% (Q1 = 1) / 6.09% (both).
- Frequencies: Q3 f01 = 5.000 GHz is 35 MHz from Q5 f12 = 4.965 GHz. A dephasing-assisted exchange estimate gives about 0.2/us for Q5, matching the observed 0.185/us. For Q1 (120 MHz away) the same estimate is 13x too low, so Q1's route is open.

**Confounders excluded (m4conf).**
- Static crosstalk predicts at most 1.4e-6 shots.
- Off-axis candidates are single-channel: other tones match the controls, and out-of-band power is 1.00x the controls.
- They are stationary: dispersion index 1.04, 0-2 bursts.
- The DC shift is caused by the Q3 signal, not the reverse.
- The noise null is calibrated: within-shot noise predicts the across-shot width to within 1%, and the minus-side tails match a Gaussian to within 0-6%.

**Not measured.** A genuine Q3 |2> would sit at about (1.47, +0.29) separations, 0.55 separations from |1> and about 2 sigma off-axis, inside the broad |1> cloud. Its population (from pi-pulse leakage, or a cascade through |2>) is unquantified; it is not shown to be absent.

### 3.5 MCMit-CNN error budget at 1 us (MEASURED; corrected)

Source: canonical `preds_MCMitCNN_pb.npy` (also the §7 soft model `cnn_soft_1000ns.npz`; the two agree within 0.1 pp). True bits = prepared bits (verified). n = 1.12M per qubit. Flags come from my line frame, mirror-corrected.

| Qubit | CNN error | Leaked share of errors (canonical) | Prep-\|0> errors only | Prep-\|1> errors only | Paper column |
|---|---|---|---|---|---|
| Q1 | 2.98% | 0.05% [-0.02, 0.11] ≈ 0 | 0.34% | ≈0 | 0.0% |
| Q2 | 25.43% | 0.34% [0.32, 0.37] | 0.77% | 0.06% | 0.2% |
| Q3 | 5.84% | **3.14% [3.00, 3.28]** (soft model 3.09%; vstatistics 3.07-3.14%; vreproduction 3.29% on its subsample) | **13.4%** [12.7, 14.1] | 0.80% | 0.1% |
| Q5 | 2.96% | 1.04% [0.91, 1.18] | 7.7% | ≈0 | 0.0% |
| Pooled Q1, Q2, Q3, Q5 | - | **0.81% [0.78, 0.84]** (3,371 / 416,227; soft model) | | | 0.2% (all 5) |

- Leaked prepared-|1> shots read as |1>: Q3 99.35%, Q2 98.8%, Q1 and Q5 about 100%.
- Leaked prepared-|0> shots misread: Q3 35%, Q5 70%.
- The CNN is *confident* on leaked prepared-|1> shots (median P(1) = 0.993 for Q3, 0.998 for Q5, 0.996 for Q1). So §7's confidence flag does **not** catch them: at f = 2% it holds 1.0% of Q3's prepared-|1> L shots, against a 2.9% base rate. It does catch 27% of Q3's leaked prepared-|0> shots.
- At 600 ns the leaked share of Q3 errors is 1.66% [1.55, 1.76]; at 400 ns it is 0.71% (soft models).
- The earlier 4.06% (Q3) and 0.96% (pooled) came from per-state centres in a frame whose rotation was tilted by the population. That tilt pushed about 900 T1-decayed errors onto the + side.

---

## 4. Verification: what changed and why

| Verifier | Claim attacked | Outcome | Applied here |
|---|---|---|---|
| vstatistics, vreproduction | Leaked share of CNN errors (Q3 4.06%, pooled 0.96%, "30-40x") | **Corrected.** The mean-based rotation is tilted by the population (Q3 -7.3 deg at 1 us), so decayed prepared-\|1> errors sit off-centre and about 900 are counted as leaked. Core-line frames give Q3 3.07-3.29%, pooled 0.79-0.81%, Q1 ≈0. | Re-measured on the full TEST set: Q3 3.14% [3.00, 3.28], pooled 0.81% [0.78, 0.84]; 23-31x vs the paper. |
| vstatistics, vreproduction | Q3 prepared \|0> 0.73% (mean of m1 0.81% and r 0.65%) | **Corrected.** r is biased low by the same tilt. | 0.83% (frames 0.81-0.87). |
| vstatistics, vreproduction | CIs "include the frame systematic" (Q3 11.19 [11.08, 11.31], Q5 4.47 [4.34, 4.60], Q1 0.69 [0.59, 0.79]) | **Corrected.** Two frames do not span it; alternative frames fall outside. | Statistical CI and frame range reported separately (§3.2). |
| vstatistics, vreproduction | 2 us row (Q3 23.2% / 1.1%) | **Withdrawn.** r's 2 us frame is broken (negative tails 3-4x Gaussian). | Q3 ≈20% (16.4-21.4%) / 2.3-2.4%. |
| vstatistics | "Grows with window, so created during readout" | **Refined.** Confounded by SNR. Equal-length windows confirm growth after ring-up, and saturation at about 13.5%. | §3.3 |
| vstatistics, vreproduction | Headline "11.2% are in it" | **Refined.** E3 is a k-convention with no plateau. | "At least about 11% (k = 3); 8-15% by definition." |
| vstatistics | Q5 binomial CI | **Refined.** The 16 configurations differ (tau = 2.68 pp). | Conditional values plus a random-effects CI [3.7, 6.3]. |
| vphysics, vstatistics, vreproduction | "tau_L = 1.8 us < T1, ≈ T1/2: \|2> signature" | **Withdrawn.** The reference was not T1 (Lienhard T1(Q3) = 21.4 us). The lifetime is model-dependent (2.7-3.5 us model-light). | Decay is readout-driven; exit to \|1> kept as the robust evidence. |
| vphysics | Q3 "\|2>-like" | **Downgraded.** The pull ratio is 4-7x, against 0.76x for \|2>; the location matches a decoupled resonator. | "Non-computational, identity unresolved; geometry disfavours \|2>." |
| vphysics | Lienhard P(2\|pi) = 0.052 confirms \|2> | **Downgraded.** It is the same third peak, refitted. | Cited as agreement on population only. |
| vphysics | Q5 "\|2>-like via exchange"; Q1 exchange | **Downgraded.** Q5's location matches no dispersive level. Q1's exchange rate is 13x the estimate. | Measured correlation, open mechanism. |
| vphysics | Shared-features IF and rotation | **Erratum.** Q3's IF is 28 kHz off, so E3 is low by about 0.2 pp. | Noted (§2 row 18). Conclusions unchanged. |
| vreproduction | Q1 prepared \|1> 0.69% | **Minor.** The independent value is 0.75-0.79%. | 0.78% (0.61-0.86). |
| all three | Existence and size of the Q3 population; one-sidedness; confounder screens; Q3 conditioning of Q1/Q5; "about 98% of leaked prepared-\|1> shots read as \|1>"; Q4 inconclusive | **Confirmed.** | Kept. |

No verifier refuted the existence of the populations. My run (`f_cnn_leak_xtab`) reproduces E3 to within 0.2 pp of vstatistics' rotated line frame for every qubit (Q3 11.46 vs 11.49%, Q5 4.76 vs 4.82%, Q1 0.86 vs 0.82%, Q2 1.62 vs 1.62%). TRAIN replicates it: Q3 11.56%.

---

## 5. CNN recommendation

### 5.1 Verdict: conditional yes

Add a per-qubit **L ("non-computational") flag** to MCMit-CNN, primarily for Q3 (Q5 and Q2 secondary, Q1 not worth it, Q4 excluded). Keep the binary bit head and its training unchanged. Present it as a leakage herald validated by label-free tests, never as a "|2> discriminator" with a 3-class accuracy.

Claim QEC value only if the paper adds a leakage-aware simulation (§5.5, item 6). Otherwise, put the flag in Discussion and Future work, and spend the effort on correcting `tab:error_modes`.

### 5.2 Why

1. **Prevalence: enough to learn and to evaluate.**
   - L examples in TRAIN: about 27.7k (Q3, prepared |1>) + 2.0k (prepared |0>); Q5 about 11.5k; Q2 about 4.1k; Q1 about 1.9k.
   - TEST has 2.3x as many.
   - Q3's L is 5.5 sigma per 100 ns bin from |1>, so it is learnable from the 13-sample CNN input. Q1 (1.5 sigma per bin) and Q5 (2.0 sigma) are weakly separated.
2. **No ground truth.**
   - Any label is a pseudo-label from the same traces.
   - Training on a threshold of the 1 us window mean only teaches the CNN to reproduce that threshold, so it cannot be validated against those labels.
   - Hence: labels that use information the CNN does not see (§5.4), and population-level, label-free evaluation (§5.5).
   - The state is probably not |2> anyway, so "|2>" is the wrong class name.
3. **QEC value: not where §7 would put it** (MEASURED, `f_cnn_leak_xtab`).
   - As an erasure of the *current* outcome, an L flag is harmful. Pooled over Q1/Q3/Q4/Q5 at 1 us:
     - 3.18% of measurements are flagged, and those are wrong only 2.5% of the time (hard error 4.27%);
     - erasing them gives e_u 4.33% (worse than hard);
     - a confidence flag of the same size gives e_u 3.43% (flagged wrong 30%);
     - adding the L flag to the f = 2% confidence flag raises e_u from 3.63% to 3.68%, against 3.21% for a confidence flag of the combined size.
   - Same at 600 ns: L-flagged wrong 2.6% vs 41.6% for the confidence flag. At 400 ns: 2.8% vs 48.3%.
   - The reason: L is mostly created *during* a |1> measurement, so the reported bit is right.
   - The real cost of leakage is the **post-measurement state**:
     - on Q3, about 12.6% of |1> measurements end the 1 us readout in L (HMM), about 6.5% at 600 ns;
     - L is not touched by a conditional-X reset, and decays to |1> about 2-3 us later.
     - With measure-and-reset ancillas this is a reset failure. Per round it would be about 6% for Q3 (≈12.6% of the roughly half of measurements that return |1>).
   - A flag can trigger a leakage-removal pulse, a wait, or "erase this qubit's next-round detectors". That is the documented value of multi-level readout (ERASER+M, GLADIATOR, Miao 2023, McEwen 2021).
   - Neither lattice-sim (no leakage) nor ECCentric (frozen, decoder-visible p = 1 Paulis) can model it, and `sweep_soft.py` uses one circuit for both sampling and decoding (perfect heralds only).
4. **FPGA cost is not the obstacle** (§5.6).

### 5.3 Design (concrete change to `sims/qsd/CNN.py` / the MCMit recipe)

**Backbone and input are unchanged.**
- `_colleague_prep.py`: FFT IF calibration on samples 0-499, theta, 101-tap 5 MHz FIR, DS = 40, i.e. 13 samples x 10 channels at 1 us.
- The first conv already mixes all 10 channels, so the Q5/Q1 heads can use Q3's channel. That is needed for the exchange-correlated populations.

**Head.**
- Replace `nn.Linear(m, 1)` per qubit by `nn.Linear(m, 2)` per qubit: [bit logit, L logit]. Output (N, 5, 2).
- That is +85 parameters (3061 -> 3146).
- Do *not* use a 3-way softmax per qubit. A softmax forces an L shot to give up its 0/1 answer, but the bit is still right for 99% of prepared-|1> L shots and is what the decoder needs for this round. A separate L logit also lets the L threshold be set by its own false-flag budget.
- The 3^5 = 243-way joint head is rejected:
  - +3,888 MACs (+31%) and 4,131 head parameters (7,107 total, 2.3x);
  - most joint classes are empty (the L populations of Q1 and Q5 avoid each other, lift 0.6-0.7);
  - the cross-qubit correlation is already available through the shared backbone.

**Loss.** L = BCE(bit, prepared bit) on all shots + lambda · BCE(L logit, p_L) on qubits with an L head.
- The bit target stays the prepared bit, so binary training is unchanged.
- p_L is a soft pseudo-label (§5.4).
- pos_weight = (1 - prevalence)/prevalence from TRAIN (Q3 about 15, Q5 about 40, Q2 about 110). Q4 and Q1 are masked out of the L loss.
- lambda is chosen so that the bit-head fidelity stays within ±0.05 pp.

**Two-step rollout.**
- **(a) Linear probe first:** freeze the trained 2-level CNN and fit only the 5 L logits on the 16-d GAP features. No risk to F5Q, minutes of compute, and it answers whether the existing features already encode L.
- **(b)** Joint fine-tuning only if the probe misses the success criteria.

**Output encoding.** 2 bits per qubit per measurement: {0, 1, E (low-confidence erasure, §7), L}. §7 already pays for 2 bits per measurement. Its erasure model discards the bit of a flagged measurement, so one of the four codes is free and **L costs no extra syndrome bandwidth**. (If a later soft decoder keeps the bit of flagged measurements, L needs a third bit or must replace that code.)

**Zero-training baseline to report alongside.**
- Per qubit: accumulate the 13 FIR samples the front end already computes, project the mean (I, Q) onto a stored line normal, and compare with 3 sigma0.
- About 13 adds, 2 multiplies and 1 comparator per qubit.
- This is the "line-frame E3 flag" of this report. The CNN flag must beat it to justify itself.

### 5.4 Labels (pseudo-labels, ranked by how much they avoid circularity)

1. **Primary: privileged-information labels.**
   - p_L = the m3time HMM posterior of L in any bin 0-9, computed by forward-backward over the *full* 2 us trace (bins 10-19 add evidence the 1 us CNN never sees).
   - The HMM is fitted unsupervised on TRAIN, with no rate imposed. Canonical relabelling fixes its state order.
   - Per-shot arrays: graham `~/oraqle-ics-sims/leakage/m3time/out_re/feat_xt1/`.
   - Use for Q3 (null false-positive rate 0.005%) and Q2 (4.1 sigma per bin from |1>). Q5 only with care (null false positives 0.36%). Not Q1 (1.2%).
2. **Cross-check labels:** per-shot responsibilities of the m2 mixture's far component (2-D window-mean model, different assumptions).
3. **Baseline only:** the line-frame z > 3 flag (§3). Training on it is self-distillation of a linear rule, so it can be a comparison point but not evidence.
4. **Never:** `three_level_labels.hdf5`, a k tuned to a target rate, or anything pooled with TEST.

### 5.5 Evaluation plan (no |2> ground truth)

All on TEST, with frames and thresholds fixed on TRAIN.

1. **Mirror test (label-free precision).**
   - Reflect the qubit's input about its per-sample |0>-|1> core axis, so the physical L population lands on the empty side.
   - The flag rate on mirrored inputs estimates the noise-driven false-flag rate.
   - Precision ≈ 1 - flag(mirror)/flag(real).
   - *Success (Q3, 1 us): ≥ 0.90.*
2. **Population recall bracket.**
   - [flag(real) - flag(mirror)] / N_L, with N_L between E3 at k = 3 (lower, 11.3% of prepared |1>) and the HMM's visited-L (upper, 14.5%).
   - *Success: ≥ 0.80 of the E3 count, and a flag rate below the HMM upper bound.*
3. **Future consistency (data the CNN never sees).**
   - Where flagged shots are in the 1.2-2.0 us window, mirror-corrected.
   - Rule-based baseline measured here (Q3, prepared |1>): 77% still L, 16% in |1>, 7% in |0>.
   - *Success: L + |1> ≥ 90%, and the |0> share well below that of prepared-|1> core shots (28%).*
4. **Physics fingerprints.**
   - The prepared-|1>:|0> flag ratio should be in Q3's measured 12-14:1 band.
   - Q5 and Q1 flags should be at least 5x more frequent with Q3 prepared |1>, and should coincide with Q3 |1>->|0> steps.
   - Q3 flags should depend only weakly on the 16 neighbour configurations (measured E3 range 9-14%).
5. **No regression.**
   - The bit head's per-qubit error must stay within ±0.05 pp of 2.98 / 25.4 / 5.83 / 5.33 / 2.95%.
   - §7's soft-flag e_u at f = 2/5/10% must be unchanged.
6. **QEC (required before any Λ claim).**
   - A leakage-aware simulator branch, with separate sampling and decoding circuits.
   - Leakage per |1> measurement as measured (Q3: about 6.5% at 600 ns, about 12% at 1 us), lifetime 1.8-3 us decaying to |1>, not removed by conditional reset, plus CZ transport.
   - Compare Λ and spacetime for: no flag; a flag-triggered leakage-reduction unit or wait (with the flag's mirror-measured precision and recall); and always-on leakage reduction.
7. **Synthetic injection (sensitivity only).**
   - Inject the measured L class-mean trajectory, plus real |1> noise, at controlled onset times.
   - Gives recall vs onset time and the false-flag rate on uninjected shots. It does not test what the state is.
8. **Definitive (needs new data).** A per-qubit calibration with prepared |2> (pi01 then pi12) and a readout-power/duration sweep. For Q3 this immediately tests the physics prediction: |2> should land at about (1.47, +0.29) separations, not at the observed L (1.9, +1.5).

### 5.6 Cost

Measured parameter and MAC counts for m = 16, 13-sample input. Total ≈ 12,656 MACs per inference: conv1 3,360 + conv2 3,072 + residual block 6,144 + heads 80.

| Option | Params | MACs | Est. FPGA impact (ZCU216; CNN today 29% LUT / 16% FF / 20% DSP / 74 ns) |
|---|---|---|---|
| Binary (today) | 3,061 | 12,656 | - |
| **Bit + L logit (recommended)** | 3,146 (+2.8%) | +80 (+0.6%) | <1 pp of LUT/DSP; the L logit runs in parallel with the bit logit (+0-1 cycle, ≤2 ns) |
| 5 x 3 softmax (prior `CNN3`) | 3,231 (+5.6%) | +160 (+1.3%) | similar, plus 2 comparators per qubit |
| 3^5 joint softmax | 7,107 (2.3x) | +3,888 (+31%) | about +30% of the CNN's DSP/LUT, plus a 243-way argmax (latency grows) |
| Rule-based line flag | 0 | about 10 per qubit | negligible |

These FPGA figures are scaling estimates, not synthesis results. Syndrome bandwidth is unchanged relative to §7's 2-bit soft readout.

### 5.7 The prior `train_three_level_cnn.py`

**What it does.**
- `CNN3` subclasses `networks/CNN.CNN` with 5 heads of `Linear(m, 3)` (3,231 parameters at m = 16) and trains with an unweighted per-head CrossEntropy.
- Optimiser and schedule: Adam 1e-3, batch 512, 40 epochs, patience 10, no LR schedule.
- Labels: y3 from `three_level_labels.hdf5`.
- Front end: Butterworth order 4 at 5 MHz, hard-coded IFs, theta = 0, decimation 20 (25 samples), z-normalisation. That is **not** the paper's colleague FIR / DS = 40 front end, and the recipe differs from the paper's (Adam 5e-4, StepLR 0.95/3, batch 64, 30 epochs, BCE).
- Metrics: 3-class accuracy, balanced accuracy, |2> recall, confusion matrices. No |2> precision, no false-flag rate, no CIs.

**Did it run?** No result survives: no `cnn_3level.json` or survey table anywhere under `~`, no container, no log. Only a comment in `train_three_level_herqules.py` hints at an attempted HERQULES run.

**Reusable?** Only the `CNN3` head class, about 10 lines, and even that should become `Linear(m, 2)` (bit + L) rather than a 3-way softmax. Rewrite the rest:
- its labels are circular and wrong for Q1 and Q4;
- its front end and recipe are not the paper's;
- the softmax takes the bit away from L shots;
- its metrics measure agreement with the labeller;
- the labels use TEST statistics.

### 5.8 Risks

- **Identity.** The flagged state is probably not |2>. Calling it "|2>" in the paper invites a physics objection. Call it "non-computational / leakage", and say that Q3's L most likely comes from a measurement-induced transition (MIST).
- **Missing |2>.** A real |2> close to |1> (about 0.55 separations) would not be flagged and is unmeasured.
- **Labels.** HMM labels for Q1 and Q5 are noisy (L is only 1.5-2 sigma per bin from |1>). Q5's population is mixture-dependent (Q3-conditioned). Q4 cannot be labelled.
- **Over-claiming.** Any 3-class accuracy is agreement with a pseudo-labeller. Only the mirror, future-consistency and fingerprint tests are evidence.
- **QEC claims without a leakage-aware simulator** would rest on the wrong model. §7's erasure model makes L look harmful (§5.2, item 3).
- **Device specificity.** One 2021 device with slow, low-power readout; the L physics (MIST, the Q3/Q5 frequency collision) is device-specific.
- **Shared-feature errata.** IF and rotation (§2 row 18) matter for any rule-based labels built from the shared features. Regenerate them with the HERQULES IF and a core-based rotation.

### 5.9 What would change the answer

- **To an unconditional yes:**
  - a calibration dataset with prepared |2> (and |3>) per qubit, which allows supervised training and evaluation;
  - a readout-power/duration sweep, to confirm the measurement-induced mechanism and pick the readout length;
  - repeated measurements on the same qubit, to measure how long L persists across rounds;
  - a leakage-aware QEC simulation that shows a Λ gain from flag-triggered leakage reduction.
- **To a no:**
  - the linear probe on the frozen backbone fails the mirror-precision bar for Q3;
  - or the leakage-aware simulation shows no Λ gain over always-on leakage reduction at the measured rates.

---

## 6. Open items

1. **Identity of Q3's L.** A |2> vs MIST/ionisation-type state needs prepared-|2> data or Lienhard's readout photon number and power (unreported). Ask B. Lienhard.
2. **Genuine |2> near |1>.** Unmeasured. It needs a 4-state model or a time-resolved test at the predicted (1.47, +0.29) separations.
3. **Q4's multimodal |1> response.** It has shot-to-shot memory and an overdispersed sub-cloud. Fluctuator vs a hidden state is unresolved.
4. **Q1/Q5-Q3 coupling mechanism.** Q5 fits a dephasing-assisted exchange; Q1 needs 13x more coupling than the published design.
5. **Shared features.** Regenerate with the HERQULES IFs for q1/q2 and a core-line rotation, and fix the README claims.
6. **Paper edits.**
   - Relabel and recompute `tab:error_modes` with the canonical MCMit-CNN.
   - Fix the near-boundary residual bug and Takeaway #5.
   - Fix the "excitation 10^-3-10^-2%" sentence.
   - Fix the "Leakage |2>" label in Fig. iq_taxonomy.
   - Optionally add the readout-length dependence of leakage (Q3: 3.5 / 6.5 / 11.3% at 400 / 600 / 1000 ns) to §7.
7. **Leakage-aware QEC simulator** (lattice-sim extension or a Pauli+ style model). It is required for any QEC claim about an L flag.
8. **Linear probe** of the trained CNN's GAP features for L (needs a saved model; `soft_readout_cnn.py` does not save weights).
9. **Q2-Q3 image crosstalk** (IFs +25.37 / -24.79 MHz): test phase-imbalance correction.

---

## 7. Files

**New in this step (tag f):**
- Local:
  - `/home/manosgior/Documents/GitHub/oraqle-ics/sims/leakage/f_cnn_leak_xtab.py`
  - `/home/manosgior/Documents/GitHub/oraqle-ics/data/leakage/f_cnn_leak_xtab.json`
  - this report
- Cluster: graham `~/oraqle-ics-sims/leakage/f/{scripts,in,out,logs}`.
  - `in/` holds copies of `cnn_soft_{1000,600,400}ns.npz`.
  - The container `leak-f-xtab` was removed.
- No existing file was modified.

**Key inputs:**
- `data/leakage/{m1tail_v2_*, m2mix_*, m3time_*, m4conf_*, r_*, vstatistics_*, vphysics_*, vreproduction_*, s1local_*, s2cluster_*, x_*}`
- `sims/leakage/*.py`
- `sims/qsd/{CNN.py,_colleague_prep.py}`
- `MCMit/qubit_state_discrimination/runners/{_cnn_length_sweep.py,_xtalk_cnn.py}`
- `sims/{soft_readout_cnn.py,sweep_soft.py}`
- `data/sims/soft_params.json`
- `chapters/{ml_discriminators,system_bottleneck}.tex`
- `Oraqle/Discriminators/runners/train_three_level_cnn.py`
