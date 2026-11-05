# Leakage feature dataset (agent tag `x`)

Per-shot, per-qubit, time-binned demodulated IQ for **every** shot of the Lienhard et al. 2022
five-qubit readout dataset (480,000 train + 1,120,000 test = 1.6 M shots). The goal is a shared,
correctly aligned representation for measuring leakage (unlabeled |2>-like populations).

* Cluster (graham/steve, shared NFS home): `~/oraqle-ics-sims/leakage/features/` = `/home/manosgior/oraqle-ics-sims/leakage/features/`
* Scripts: cluster `~/oraqle-ics-sims/leakage/scripts/x_*.py`; local `oraqle-ics/sims/leakage/x_*.py`
* Small results (local): `oraqle-ics/data/leakage/x_{calib,sanity,crosscheck,build_meta,xtalk}.json`, `x_hist.npz`, `x_iq_q{0..4}.png`

## 1. Files

| file | dtype | shape | content |
|---|---|---|---|
| `feat_train.npy` / `feat_test.npy` | float32 | (N, 5, 20, 2) | **main features**: mix → 101-tap FIR (`same`) → rotate → mean of 20 bins × 50 samples |
| `featbox_train.npy` / `featbox_test.npy` | float32 | (N, 5, 20, 2) | same without the FIR (plain boxcar mean of the mixed, rotated signal). Use it to check for FIR artefacts. |
| `feat1us_train.npy` / `feat1us_test.npy` | float32 | (N, 5, 10, 2) | FIR applied to samples 0..499 **only** (zero padding at sample 500). This is exactly what the CNN sees at the paper's "1 us": no information from after 1 us. |
| `y_train.npy` / `y_test.npy` | int32 | (N,) | prepared 5-bit basis state 0..31, copied from `y_train`/`y_test` of the raw file (verified equal on all rows) |
| `idx_train.npy` / `idx_test.npy` | int64 | (N,) | row index into the raw HDF5 (`X_train[idx]` / `X_test[idx]`); currently `arange(N)` (verified) |
| `calib.json` | | | f_q, theta_q actually used, the verbatim `calibrate()` output for (0,1000) and (0,500), FIR taps, diagnostics |
| `sanity.json` | | | threshold accuracies, centroids, MAD sigmas, perpendicular tails, per-bin mean trajectories |
| `xtalk.json` | | | 32-state conditional centroids, cross-qubit co-occurrence of off-axis shots, per-bin cross-channel transfer |
| `crosscheck.json` | | | equivalence with the verbatim MCMit `preprocess()`, stored-vs-recomputed check |
| `build_meta.json` | | | params, shapes, non-finite counts (all 0), script md5s |
| `x_hist.npz` | | | histogram data behind `x_iq_q*.png` |

N_train = 480,000 (15,000 per prepared state); N_test = 1,120,000 (35,000 per prepared state). Raw rows are stored in 32 contiguous blocks sorted by state (verified).

**Axes**
* axis 1 = qubit, **code index q = 0..4 = paper Q1..Q5**. The prepared bit of qubit q is `(y >> q) & 1`. This mapping is independently confirmed by HERQULES's hard-coded IFs (`FREQ_READOUT`, `MCMit/.../trainers/train_HERQULES.py`; its qubit 1..5 = +64.729, +25.366, −24.79, −70.269, −127.282 MHz), which match the per-bit frequencies calibrated here to within 4.5–28 kHz. Bit 1 is also the low-separation qubit (paper Q2, the "hardware" qubit).
* axis 2 = time bin k: raw samples [50k, 50k+49]. At 500 MS/s that is [100k ns, 100k+100 ns). The FIR is centred (zero group delay) and spans ±50 samples, so FIR bin k is influenced by raw samples 50k−50 … 50k+99. Boxcar bin k uses only its own 50 samples.
* axis 3: `[..., 0]` = component **along** the |0>→|1> axis (Re), `[..., 1]` = **perpendicular** component (Im). Units are raw ADC units: the FIR has unity DC gain and no amplitude normalisation is applied.
* The rotation aligns the *difference* of the two centroids with +real. It does not move the line to perp = 0: each qubit's |0>/|1> line sits at a constant perp offset (see §4). Subtract the centroid perp yourself.
* Edge effect of zero padding: for a constant signal, bin 0 and bin 19 of `feat` (and bin 9 of `feat1us`) have gain 0.870; all other bins have gain 1.000. Bin 0 is mostly ring-up anyway.

## 2. How to load

```python
import numpy as np
D = "/home/manosgior/oraqle-ics-sims/leakage/features"          # inside docker: mount it, e.g. -v $D:/feat:ro
F = np.load(f"{D}/feat_test.npy", mmap_mode="r")                # (1120000, 5, 20, 2) float32, 896 MB
y = np.load(f"{D}/y_test.npy")                                  # (1120000,) int32
bits = (y[:, None] >> np.arange(5)) & 1                         # (N, 5) prepared bits, col q = paper Q(q+1)
w1us  = np.asarray(F[:, :, 0:10]).mean(2)                       # (N, 5, 2) paper "1 us" window mean (along, perp)
w400  = np.asarray(F[:, :, 0:4]).mean(2)                        # paper "400 ns"
w2us  = np.asarray(F).mean(2)                                   # full 2 us trace
cnn1us = np.load(f"{D}/feat1us_test.npy", mmap_mode="r").mean(2)  # strictly causal 1 us window (FIR zero-padded at 500)
# raw trace of shot i:  h5py.File(".../DRaw_C_Te_v0-002")["X_test"][idx[i]]  -> (1000, 2) float64
```
Docker example (graham): `docker run --rm --user $(id -u):$(id -g) -v ~/oraqle-ics-sims/leakage/features:/feat:ro -v ~/qubit_readout_klinq/data/five_qubit_data:/data:ro oraqle-torch:latest python ...`.
The window mean of the bin means equals the mean over the corresponding samples, because all bins have the same width.

## 3. Sampling rate and paper windows

The sampling rate is **500 MS/s, 2 ns per sample**. The raw trace has 1000 samples = **2 us**. Evidence:
1. `_colleague_prep.py`: `SAMPLING_RATE = 500e6`.
2. `runners/_cnn_length_sweep.py`: "500 MHz -> 2 ns/sample", lengths 200 ns=100 … 1000 ns=500 samples, `time_slice=(0, L)` and calibration on the fixed window (0, 500). The paper's "1000 ns" CNN row is therefore samples 0..499.
3. KLiNQ `helpers/data_loader.py` (`sampling_interval_ns = 2  # 500 MHz ADC`) and `train_arxiv_model.py` (2e-9 s). HERQULES / transformer / three-level code use `MAX_LENGTH = 500` (`TRACE_LENGTHS` up to 500 samples). HERQULES `SAMPLING_RATE = 500e6`.
4. Paper `chapters/overview.tex`: "sampled at 500 MSamples/s over a 2 us window". `qubit_readout.tex` sweeps T up to 2.0 us. `ml_discriminators.tex` calls the first 1 us the "full 1 us trace".

The rate cannot be fixed from the data alone; all IFs fall within ±130 MHz, consistent with it. The paper's "1 us" ML window is the **first half** of the stored trace.

| paper window | raw samples | `feat`/`featbox` bins | `feat1us` bins |
|---|---|---|---|
| 200 ns | 0–99 | 0–1 | 0–1 |
| 400 ns | 0–199 | 0–3 | 0–3 |
| 600 ns | 0–299 | 0–5 | 0–5 |
| 1 us ("full" ML trace) | 0–499 | 0–9 (FIR bin 9 also sees samples 500–549) | 0–9 (strictly causal) |
| 2 us (whole stored trace) | 0–999 | 0–19 | – |

`1us` and `1us_cnn` give the same threshold errors to within 0.1 pp (table in §5).

## 4. Calibration (`calib.json`)

**Exact `calibrate()` output.** This is MCMit `_colleague_prep.calibrate`, copied byte-identical to `x_colleague_prep_verbatim.py`, run on TRAIN with 500 shots per state from row offset +5000, i.e. 16,000 shots, 8,000 per prepared bit:

| qubit | f, (0,1000) | f, (0,500) | theta_colleague (0,1000) |
|---|---|---|---|
| Q1/q0 | +64.500 MHz | +65.000 MHz | +2.9000 rad |
| Q2/q1 | +25.500 | +25.000 | +1.0876 |
| Q3/q2 | −25.000 | −25.000 | +0.5819 |
| Q4/q3 | −70.000 | −70.000 | +0.9861 |
| Q5/q4 | −127.500 | −127.000 | +0.8088 |

**Two problems with those values, and the fix.**
1. *Frequency quantisation.* `calibrate()` returns an FFT bin centre: a 0.5 MHz grid for 1000 samples, 1 MHz for 500. All five true IFs lie 0.1–0.26 MHz off that grid. With the (0,1000) values, the demodulated |1>−|0> mean-difference vector rotates linearly by +169, −77, +172, −187, +160 deg over 2 us (Q1..Q5). With the (0,500) values that the CNN uses it rotates by −191, +284, +172, −187, −200 deg. In a fixed frame, along/perp would mix across bins, and any mid-trace state change would land off-axis. That is fake "leakage" (the CNN learns around it; a linear feature cannot). **Fix:** start from the (0,1000) value and iterate `f += slope/2π`, where slope is the |d|-weighted linear fit of the unwrapped phase of d(t) = <z|bit=1>(t) − <z|bit=0>(t) over samples 100..899 (z = mixed + FIR), on the same 16,000 calibration shots. Starting from the (0,500) value converges to the identical f. On **16,000 independent TEST shots** (rows s·35000 … +499) the residual is < 0.7 kHz (< 0.5 deg over 2 us) for every qubit.
2. *Rotation.* `calibrate()` computes theta from the mean of the **raw, undemodulated** trace. Each IF tone averages to ~0 there, so it is not the |0>→|1> axis: it leaves the axis at −69, +134, −89, −153, +8 deg. **Fix:** theta_q = −angle(mean over samples 0..999 of d(t)) at the refined f.

**Parameters used for the features** (`calib.json["used_for_features"]`):

| qubit | f_q (Hz) | theta_q (rad) | HERQULES IF (MHz) |
|---|---|---|---|
| Q1/q0 | +64,734,700.7 | −0.65441 | +64.729 |
| Q2/q1 | +25,393,448.9 | −1.83362 | +25.366 |
| Q3/q2 | −24,761,600.6 | −2.66301 | −24.79 |
| Q4/q3 | −70,259,573.6 | +1.94531 | −70.269 |
| Q5/q4 | −127,277,509.0 | +2.09084 | −127.282 |

The mixing convention is `mixed = Xc · exp(−i2π f t)`, t = n / 500e6, n = 0..999 with t = 0 at raw sample 0, followed by `· exp(i theta)`. This is identical to `_colleague_prep.preprocess`.

**Residual axis angle per 100 ns bin** after calibration (train / test agree within ~1–2 deg): bins 2–19 are flat to within ±3 deg for all qubits. Bins 0–1 (ring-up) are rotated by −31/−7 (Q1), −17/+1 (Q2), +15/+2 (Q3), −21/−17 (Q4) and +21/+4 deg (Q5). This is physical resonator ring-up dynamics, not miscalibration. Treat the early-window "perp" with care: in `early_200ns` the centroid difference is at −13, −5, +5, −17, +7 deg.

**Equivalence check** (`crosscheck.json`): the vectorised chain (`scipy.signal.fftconvolve`, `same`) against the verbatim `preprocess()` per-index FIR loop with the same params and ds_factor = 1 differs by at most 9.5e-7 per sample and 4.8e-6 per bin, i.e. float32 rounding (feature std ≈ 5). Stored files equal a fresh recomputation for 300 random rows per split (max diff 0.0). Labels equal the raw file on all rows. No non-finite values and no all-zero rows in any array.

## 5. Sanity (`sanity.json`)

A 1-D threshold on the along-axis window mean, fitted on TRAIN to minimise the balanced error ½[P(1|0)+P(0|1)] and evaluated on TEST (560,000 shots per prepared bit per qubit). 95% CIs: normal approximation for the balanced error from the two binomial conditional error rates (per-class Wilson intervals are in `sanity.json`).

| qubit | paper CNN err @1 us | thr. err `1us` (bins 0–9) | `1us_cnn` | `400ns` (bins 0–3) | `full_2us` (bins 0–19) | `box_1us` |
|---|---|---|---|---|---|---|
| Q1/q0 | 3.2% | **3.15%** [3.11, 3.18] | 3.17% | 9.13% | 3.53% [3.49, 3.56] | 3.22% |
| Q2/q1 | 25.8% | **26.62%** [26.54, 26.70] | 26.72% | 29.96% | 26.78% [26.70, 26.86] | 26.90% |
| Q3/q2 | 6.1% | **10.45%** [10.39, 10.50] | 10.43% | 8.25% | 13.48% [13.42, 13.54] | 10.53% |
| Q4/q3 | 5.6% | **6.58%** [6.53, 6.62] | 6.60% | 10.68% | 8.45% [8.40, 8.50] | 6.66% |
| Q5/q4 | 3.0% | **4.28%** [4.25, 4.32] | 4.26% | 5.06% | 6.61% [6.57, 6.66] | 4.31% |

The threshold is the same order as the CNN and, as expected, slightly worse. Q3 is worse by the largest margin, because of its large non-Gaussian |1> population (§6). For Q3 the threshold is better at 400 ns than at 1 us. For every qubit the full 2 us is worse than 1 us, most of all for Q3/Q4/Q5. Both match the paper's "late samples hurt" finding.

**Centroids** (TEST, window `1us`, median along / median perp, per prepared bit), MAD sigmas, separation = median-along difference, SNR = sep / mean MAD sigma_along:

| qubit | c0 (along, perp) | c1 (along, perp) | sigma_along MAD p0/p1 | sigma_perp MAD p0/p1 | sep | SNR | axis angle median / mean |
|---|---|---|---|---|---|---|---|
| Q1/q0 | (−4.584, −1.251) | (−1.758, −1.302) | 0.553 / 0.627 | 0.554 / 0.579 | 2.826 | 4.79 | −1.0 / −0.6 deg |
| Q2/q1 | (−1.379, +0.192) | (−0.572, +0.155) | 0.544 / 0.786 | 0.531 / 0.551 | 0.807 | 1.21 | −2.7 / −0.1 deg |
| Q3/q2 | (+4.019, −3.296) | (+8.428, −3.601) | 0.658 / 1.575 | 0.626 / 0.807 | 4.409 | 3.95 | −4.0 / +0.5 deg |
| Q4/q3 | (−0.786, −10.671) | (+5.141, −10.748) | 0.748 / 2.821 | 0.700 / 1.327 | 5.927 | 3.32 | −0.8 / −0.2 deg |
| Q5/q4 | (−0.105, −3.909) | (+5.306, −3.989) | 0.667 / 0.788 | 0.661 / 0.762 | 5.411 | 7.44 | −0.9 / +0.4 deg |

Full 2 us (`full_2us`): sep = 2.951 / 0.623 / 4.306 / 6.765 / 5.862 and sigma_along(p0) = 0.396 / 0.399 / 0.520 / 0.595 / 0.504 (Q1..Q5). All windows, plain means, `std`s and the 400 ns/200 ns numbers are in `sanity.json`.

**Rotation check.** The centroid difference has ~0 perpendicular component. With means (the quantity theta was fitted on), the angle is −0.6, −0.1, +0.5, −0.2, +0.4 deg at 1 us and +0.4, −0.2, −0.4, +0.2, +0.1 deg at 2 us. With medians it is −1 to −5 deg, because the |1> distributions are skewed by the off-axis populations described below. The absolute perp of the |0>/|1> line is not 0 (−1.3, +0.2, −3.3…−3.6, −10.7, −3.9). Downstream code must subtract the per-bit median perp.

**Noise reference.** For prepared |0>, sigma_along ≈ sigma_perp (isotropic Gaussian core). For prepared |1> the MAD sigma is inflated by T1 and off-axis structure (Q3: 1.58 vs 0.66; Q4: 2.82 vs 0.75). **Use the |0> sigma, or the 32-state conditional sigma in `xtalk.json`, as the noise scale, not the pooled |1> sigma.**

**FIR vs boxcar.** The window means correlate at 0.994–0.9999. The perpendicular tail counts (>4 sigma, >5 sigma) agree to within ~3–13%; boxcar is slightly lower because its sigma is slightly larger. The FIR does **not** create the off-axis structure. There is a constant perp offset between the variants (up to 0.22 for Q2 at 1 us), from unfiltered other tones and DC in the boxcar; it is irrelevant for shapes.

## 6. Diagnostics: what the IQ plots show (`x_iq_q{0..4}.png`)

Each figure has three rows: full 2 us, 1 us, and early 200 ns window means. The first two columns are log-density 2-D histograms for prepared |0> and |1>. The last two columns are 1-D log histograms of perp in robust z units (with a Gaussian reference) and of along in units of the |0>–|1> separation. Train and test are pooled: 800,000 shots per prepared bit.

The descriptions below are **MEASURED** (visible structure and counts). Interpreting the structure as |2> is **INFERRED**.

* **Q3 (q2): clear third population.** In prepared |1>, a dense streak leaves the |1> centroid (~8, −4), bends upward and ends in a **distinct far cloud at about along 13–15, perp +1 … +3**. That is ~2.5–3 separations from |0> along the axis and ~+4.5…+6.5 units (~7–10 sigma_perp of |0>) above the |0>/|1> line. In the full 2 us window the far cloud is a separate blob, and the streak looks like shots changing state mid-readout. Prepared |0> shows the same streak and far cloud at a much lower density. Off-axis tails at 1 us (|perp − median| > 5 sigma_MAD): prepared |1> 27,629 / 560,000 = 4.93% (Gaussian expectation 6e-7); prepared |0> 2,333 / 560,000 = 0.42%. The along-axis histogram of prepared |1> has a plateau from 1.2 to 2.5 separations carrying a large share of the shots. Q3 also has the largest threshold-vs-CNN gap. **This is the strongest leakage-like signal in the dataset.** A pure perpendicular cut captures it only partly, because the cloud is displaced both along and perp.
* **Q5 (q4):** prepared |1> has a **streak perpendicular to the axis** rising from the |1> centroid (along ≈ 5.5–6) to perp ≈ 0, i.e. +4 units ≈ 6 sigma. At 2 us it ends in a weak blob. Prepared |0> has a faint diffuse patch in the same direction. Off-axis tails >5 sigma at 1 us: |1> 1,241 / 560,000 = 0.22%; |0> 64 / 560,000 = 0.011%. At 2 us: 1.29% / 0.125%. Leakage-like.
* **Q2 (q1):** a large **off-axis plume** from the clusters up to about (+2.5, +6), i.e. ~11 sigma perp at 1 us and ~15 sigma at 2 us. It is strong in prepared |1>, weaker in |0>. >5 sigma at 1 us: |1> 4,072 / 560,000 = 0.73%; |0> 335 / 560,000 = 0.060%. At 2 us: 1.38% / 0.108%. The |0>/|1> clouds themselves overlap (SNR 1.2), so the "hardware" error of Q2 is mainly low separation; the plume is a separate population.
* **Q4 (q3):** prepared |1> **splits into sub-clouds**: an **off-axis blob at about (2, −12.3)**, i.e. along ≈ 0.45 separations and ~1.6 units ≈ 2.3 sigma below the line, with a high density. The main |1> region is itself elongated or bimodal (≈5 and ≈7.5 along). Faint traces of the blob appear in prepared |0>. Because the blob is close to the line in sigma units, the >5 sigma perp counts are small: |1> 1 / 560,000; |0> 91 / 560,000 at 1 us. The along histogram (peak at 0.45 separations) shows it best. **Origin unclear:** the per-32-state centroids show it is *not* driven by another qubit's prepared bit. Candidates are a third-state response, or a |1>→|0> transient that curves off the axis. Treat with caution.
* **Q1 (q0):** close to two clean Gaussian clouds. Prepared |0> has a small cloud at the |1> position (excitation / preparation error). Prepared |1> has a weak tail beyond |1> and slightly off-axis. >5 sigma at 1 us: |1> 124 / 560,000 = 0.022%; |0> 0 / 560,000. At 2 us: 0.126% / 0.0014%.
* **Early window (bins 0–1, first 200 ns):** for every qubit the perpendicular distribution is Gaussian down to ~1e-5 for prepared |0>. Only a mild excess tail appears for prepared |1> (Q2, Q3), so third-state signatures build up *during* the readout. At 200 ns the separation is only ~1–2 sigma, so a population that was already present at t = 0 cannot be excluded from this window alone.
* **Time growth:** for Q1, Q2 and Q5 the off-axis tail fractions grow from 1 us to 2 us (|1> >5 sigma: Q5 0.22% → 1.29%, Q2 0.73% → 1.38%, Q1 0.022% → 0.126%). That is consistent with transitions accumulating during the measurement (measurement-induced or T1-ladder), not only populations present from the start (INFERRED). For Q3 the |1> >5 sigma fraction drops (4.93% → 3.56%), because its |1> MAD sigma_perp inflates at 2 us (0.81 → 0.92) and the streak spreads along the axis. Q4's perp tails stay ~0. A fixed noise scale is needed before concluding anything about the Q3/Q4 time dependence.

**Crosstalk check** (`xtalk.json`, TEST):
* Other qubits' prepared bits move a qubit's centroid by at most 0.94 sigma (Q4 |0> at 2 us), typically 0.1–0.5 sigma. Per-bin cross-channel transfer is ≤ 0.28 sigma of the target qubit. Mirror-IF pairs (Q2/Q3 at a 0.63 MHz beat; Q1/Q4 at 5.5 MHz, where the FIR attenuates only −5.6 dB) show modest coupling: q2←q1 transfer is 0.36 of q1's own response, but that is only 0.13 sigma of q2. None of this produces multi-sigma off-axis clouds.
* Off-axis shots are mostly **single-channel**. At 1 us with K = 4 (relative to the 32-state conditional centroid), only 2% (Q3), 11% (Q2), 2.5% (Q5), 4% (Q1) and 21% (Q4) of a qubit's flagged shots are also flagged on another qubit. Pairwise lifts are 0–5 (Q3–Q4 5.3, Q2–Q3 2.8). The third populations are therefore per-qubit, not one event seen in two channels.

## 7. Caveats for downstream leakage estimates

* Use `feat` (or `feat1us` for the strict CNN 1 us input) with the calibrated params. Do **not** reuse the raw `calibrate()` f/theta for along/perp analysis (§4).
* A k·sigma_perp cut against the *pooled* per-bit centroid mixes in centroid shifts from the other qubits (≤ 0.5 sigma) and uses an inflated |1> sigma. Prefer the 32-state conditional centroids and the |0> sigma. Q3's third cloud is displaced along the axis as much as perpendicular to it, and Q4's blob is only ~2 sigma off-axis, so a perpendicular-only rule under-counts both.
* Mid-readout transitions put the *window mean* anywhere on the path between clouds. The per-bin features (20 × 100 ns) let you track the trajectory: a |2>-like shot should sit at the far cloud in the later bins.
* Tail counts in §6 are descriptive (MEASURED counts in a stated window and noise model). They are not leakage rates: they include T1/transient tails and exclude in-core |2> shots.

## 8. Reproduce

```bash
# graham, from ~ ; scripts in ~/oraqle-ics-sims/leakage/scripts
docker run --rm --user $(id -u):$(id -g) -v ~/qubit_readout_klinq/data/five_qubit_data:/data:ro \
  -v ~/oraqle-ics-sims/leakage/scripts:/scripts:ro -v ~/oraqle-ics-sims/leakage/features:/out oraqle-torch:latest \
  bash -c "python /scripts/x_calib.py /out && python /scripts/x_build_features.py --out /out --workers 24 && \
           python /scripts/x_crosscheck.py /out && python /scripts/x_sanity.py /out && python /scripts/x_xtalk_diag.py /out"
# local plots: .venv python sims/leakage/x_plot_iq.py  (reads data/leakage/x_hist.npz, x_sanity.json)
```
Runtime on graham with 24 workers: calibration 55 s, features 44 s (1.6 M shots), sanity ~2 min. To build features with the unrefined colleague params (for comparison only), run `x_build_features.py --params colleague --out <other dir>` (with a `calib.json` copy there).
