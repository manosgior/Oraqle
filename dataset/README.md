# dataset/ — the raw five-qubit readout traces (not distributed)

Every readout and discriminator number in the paper is computed from one dataset that is
**not part of this repository and not publicly downloadable**: the frequency-multiplexed
single-shot readout traces of the five-qubit transmon chip of

> B. Lienhard, A. Vepsäläinen, L. C. G. Govia, C. R. Hoffer, J. Y. Qiu, D. Ristè, M. Ware,
> D. Kim, R. Winik, A. Melville, B. Niedzielski, J. Yoder, G. J. Ribeill, T. A. Ohki,
> H. K. Krovi, T. P. Orlando, S. Gustavsson, W. D. Oliver,
> *Deep-Neural-Network Discrimination of Multiplexed Superconducting-Qubit States*,
> Phys. Rev. Applied **17**, 014024 (2022), doi:10.1103/PhysRevApplied.17.014024

(author list as in the paper's references.bib entry `Lienhard2022Deep`; the paper cites the
dataset as `Lienhard2022Reinforcement`, an APS March Meeting 2022 abstract by the same group). HERQULES and KLiNQ
obtained it from the authors as well; no Zenodo/figshare/GitHub release exists. **Ask the
authors (MIT EQuS; B. Lienhard is now at TUM) for a copy.** Without it you can still verify
every paper number from the committed results (`verify/discriminators.py`) and regenerate
all figures and tables (`paper_figures/discriminators/make_all.py`); only re-training needs
the traces.

## What to plug in

Two HDF5 files, no attributes, no chunking, no compression:

| File | Size (bytes) | sha256 | Datasets |
|---|---|---|---|
| `DRaw_C_Tr_v0-001` | 7,681,922,048 | `1aabfa02cb434f2ebbc648d44553643ca7d957ce734111f45a41ad152d815b8c` | `X_train` (480000, 1000, 2) float64, `y_train` (480000,) int32 |
| `DRaw_C_Te_v0-002` | 17,924,482,048 | `d538fcf42c0ed63ffd60eb2c86e0c2edd2a8fe73899c22d81593cff53a4bcea4` | `X_test` (1120000, 1000, 2) float64, `y_test` (1120000,) int32 |

(sha256 computed with `sha256sum` on graham, 2026-11-05, over
`~/qubit_readout_klinq/data/five_qubit_data/DRaw_C_*`; layout probed with h5py. The same
facts are in `results/discriminators/dataset_metadata.json`.)

* Last axis of `X_*` = (I, Q); 1000 samples at 500 MS/s = 2 µs per shot.
* Labels 0..31: bit q of the label is the prepared state of qubit q (code q0..q4 = paper Q1..Q5).
* Each file is stored as 32 contiguous blocks, state 0..31 in order: 15,000 shots per state
  in the train file, 35,000 in the test file (1.6 M shots, 50,000 per state in total).
  Several runners rely on this block layout (`samples_per_state` constants, calibration offsets).

Put both files in one directory and point `ORAQLE_DATA` at it (default: this `dataset/`
directory, which is gitignored):

    export ORAQLE_DATA=/path/to/five_qubit_data      # contains DRaw_C_Tr_v0-001, DRaw_C_Te_v0-002
    sha256sum $ORAQLE_DATA/DRaw_C_*                    # compare with the table above

Everything resolves paths through `oraqle_env.py`; scripts that need the traces call
`require_dataset()` and stop with a pointer to this file if they are missing. In Docker,
mount the directory read-only and set the variable, e.g.
`-v /path/to/five_qubit_data:/data:ro -e ORAQLE_DATA=/data`.

## Derived data (regenerated into `ORAQLE_WORK`, gitignored)

Nothing derived from the traces is committed except small per-model result files
(`results/discriminators/`) and 512-shot example inputs for the FPGA flow (`fpga/hls4ml/inputs/`).
Intermediate data is rebuilt on demand into `$ORAQLE_WORK` (default `work/`):

| Derived data | Made by | Used by | Size / time |
|---|---|---|---|
| `$ORAQLE_WORK/cnn/Qubit_5Channel_ds20_{train,test}.h5` (demodulated, 5 MHz Butterworth low-pass, ↓20, phase-rotated, (N, 50, 5, 2) float16) | `python discriminators/helpers/extract_data.py` | `runners/hyper_optimize.py` (CNN objective), `extras/.../evaluate_truncated.py`, `fpga/hls_input_prep/_make_example_inputs.py`, `fpga/hls4ml/micro-submission/` | 1.6 GB, minutes |
| `$ORAQLE_WORK/cnn/Qubit_5Channel_target50_len{50..500}_{train,test}.h5` | `python discriminators/helpers/extract_data_per_length.py` | `extras/discriminators/runners/hyper_optimize_cnn.py` (CNN_m16 Optuna runs; **not** the Table 2 CNN) | 16 GB |
| demodulated per-qubit traces (in memory) | `trainers/HERQULES_original.py` at import (Butterworth demodulation of all 1.6 M shots at `HERQULES_TRACE_LENGTH` samples) | all HERQULES runners, `readout/iq_clouds_vs_length.py`, `runners/iq_error_gallery.py` | ~24 GB RAM at 500 samples; not written to disk |
| HERQULES state pickles `rmf.pkl`, `preclassifier_state.pkl`, `mf_rmf_nn_train.pkl` | `trainers/HERQULES_original.py` (written to the current directory) | `runners/analyze_herqules_errors.py`, `_train_and_analyze_errors.py` | run HERQULES runners from a scratch directory, e.g. `cd $ORAQLE_WORK` |
| `$ORAQLE_WORK/three_level_labels.hdf5` | `extras/discriminators/runners/build_three_level_labels.py` | three-level extras only (circular labels; not in the paper) | 14 MB |

The Table 2 producers (`_cnn_length_sweep.py`, `_herqules_length_both.py`,
`hyper_optimize*.py`, `reproduce_klinq_paper.py`, `readout/baseline_no_ML.py`) read the raw
files directly and need no derived data. See docs/reproduce/discriminators.md.
