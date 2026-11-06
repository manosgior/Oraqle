# Is MCMit-CNN at the accuracy ceiling? (not in the paper)

The error analysis (`leakage/`, Sec. 5.3 of the paper) suggests that few of the CNN's remaining errors can be
fixed by a better model. This check tests that directly: a physics-based discriminator that models decay and
leakage during the readout, on the same test shots and the same 1 us window as the CNN.

**Result: little accuracy is left to gain.** Adding the HMM to the CNN lowers the error by at most 0.07 pp (Q1).

## Method

- **`hmm_ceiling.py`:** per qubit, a hidden Markov model (the `leakage/m3time_hmm.py` model) on 100 ns boxcar bins
  of that qubit's own channel, fitted on TRAIN. On TEST it picks the prepared bit with the higher likelihood. This
  is the Bayes-optimal single-shot decision only if the model is true.
- **Variants:** 2 or 3 states; static crosstalk correction using neighbour bits from the CNN, from the HMM itself,
  or the true bits (not deployable, a bound).
- **Reference points:** LDA on the window mean and on the trajectory, the canonical MCMit-CNN, and a logistic
  regression combining the soft MCMit-CNN with the HMM (2-fold cross-fitted on TEST).
- **`ceiling_followup.py`:** controls. Threshold-only recalibration of the soft CNN, paired CIs, LDA on one channel
  vs all five, and a Viterbi profile of the shots each method wins.

## Results

TEST split, 1.12 M shots. Error in % of shots.

| Qubit | CNN | HMM (own neighbour bits) | Soft CNN | Soft CNN + HMM | Gain over soft CNN, pp [95% CI] |
|---|---|---|---|---|---|
| Q1 | 2.98 | 2.92 | 2.98 | 2.91 | 0.072 [0.061, 0.082] |
| Q2 | 25.43 | 25.77 | 25.40 | 25.34 | 0.056 [0.034, 0.078] |
| Q3 | 5.84 | 7.59 | 5.83 | 5.82 | 0.015 [0.004, 0.026] |
| Q4 | 5.32 | 5.65 | 5.33 | 5.29 | 0.038 [0.026, 0.050] |
| Q5 | 2.96 | 3.05 | 2.95 | 2.94 | 0.012 [0.005, 0.019] |

- The five-qubit geometric-mean fidelity goes from 0.8076 (soft CNN) to 0.8085 (combined).
- 87-94% of CNN errors are also errors of the 3-state HMM without crosstalk correction.
- The true neighbour bits give the HMM no gain over its own estimates.
- Q1 is the only qubit where the HMM alone beats the CNN. 38% of the shots it wins contain an in-window state change
  (Viterbi), against 14% of the shots the CNN wins. About a quarter of Q1's combined gain is a threshold shift.

Full numbers are in `hmm_ceiling.json` and `ceiling_followup.json`.

## Limits

- The HMM is **not an upper bound**: the CNN beats it on four of five qubits, by 1.74 pp on Q3. LDA on all five
  channels instead of one explains only about 0.46 pp of that gap. This is strong evidence of little headroom, not a
  proof, and Q3 is where hidden headroom is most likely.
- The two CNN trainings compared here differ from each other by about 0.01 pp on Q1, Q3, Q4 and Q5 and 0.03 pp on
  Q2, so the Q3 and Q5 gains are no larger than the difference between trainings.
- The fitted Markov model does not describe the data exactly (see the goodness-of-fit notes in
  `results/leakage/em_e1hmm_summary.md`), and the start-state probabilities are weakly determined because the
  resonator is still ringing up in the first bin.

## Running it

Needs the raw dataset, through the features built by `leakage/x_calib.py` and `leakage/x_build_features.py`
(about 2 minutes, 3.2 GB). Use the setup of [`leakage/README.md`](../../../leakage/README.md), section 3.

```bash
python extras/leakage/hmm_ceiling/hmm_ceiling.py        # about 20 min on 5 workers
python extras/leakage/hmm_ceiling/ceiling_followup.py   # a few minutes, needs the output of the line above
```

Outputs go to `$ORAQLE_WORK/leakage/hmm_ceiling/`, including per-shot scores (150 MB). Those are not committed.

## Provenance

New code, written and run on 2026-11-06 on graham in the `oraqle-torch` image. The committed JSON files are from that
run, with `/work/ceiling` as the output directory. After moving the scripts here (only path handling changed),
`ceiling_followup.py` reproduced its JSON byte for byte and `hmm_ceiling.py` was smoke-tested for 60 s, not re-run in
full.
