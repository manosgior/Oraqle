# qubit_state_discrimination

This directory implements deep-learning and signal-processing classifiers for
**qubit state discrimination** in superconducting quantum computers.  It covers the
full HERQULES pipeline (matched-filter pre-processing + neural network classifier) as
well as three additional neural network approaches: a simple FNN, a Vision-Transformer,
and KLiNQ — a knowledge-distillation pipeline designed for FPGA deployment.

---

## Table of Contents

1. [Physical Background](#physical-background)
2. [HERQULES Pipeline](#herqules-pipeline)
   - [Overview and Design Philosophy](#overview-and-design-philosophy)
   - [Stage 1 — Frequency Demodulation](#stage-1--frequency-demodulation)
   - [Stage 2 — Pre-classification and Trace Purification](#stage-2--pre-classification-and-trace-purification)
   - [Stage 3 — Matched Filter and Relaxation Matched Filter](#stage-3--matched-filter-and-relaxation-matched-filter)
   - [Stage 4 — Neural Network Classifier](#stage-4--neural-network-classifier)
3. [Matched Filter Module](#matched-filter-module)
4. [Additional Model Architectures](#additional-model-architectures)
   - [arXiv:2406.18807 FNN](#1-arxiv240618807-fnn)
   - [Transformer (QubitClassifierTransformer)](#2-transformer-qubitclassifiertransformer)
   - [KLiNQ — Knowledge Distillation Pipeline](#3-klinq--knowledge-distillation-pipeline)
5. [Repository Structure](#repository-structure)
6. [Data Pipeline](#data-pipeline)
7. [Usage Instructions](#usage-instructions)
8. [Development Story of KLiNQ](#development-story-of-klinq)
9. [Disclaimer & Research Context](#disclaimer--research-context)

---

## Physical Background

In dispersive qubit readout a microwave tone is sent through a resonator coupled to the
qubit.  The qubit state (|0⟩ or |1⟩) shifts the resonator frequency, producing a
characteristic phase and amplitude response.  The signal is **downconverted** to an
intermediate frequency (IF) and digitised by an ADC, yielding two quadrature components:

- **I** (in-phase)
- **Q** (quadrature)

For **multiplexed readout** of 5 qubits, all five resonators are probed simultaneously
at different IF frequencies in a shared bandwidth.  Dataset files contain raw IQ traces
of shape `(N_shots, trace_length, 2)` and integer labels in `[0, 31]` (one bit per qubit,
bit *k* encodes qubit *k+1*).

Dataset hardware parameters:

| Parameter | Value |
|---|---|
| Number of qubits | 5 |
| ADC sampling rate | 500 MHz |
| Readout window | 1 µs (500 samples) |
| IF frequencies | −64.73, −25.37, +24.79, +70.27, +127.28 MHz |

---

## HERQULES Pipeline

**File:** [`HERQULES.py`](HERQULES.py)

HERQULES (**H**ierarchical **E**fficient **R**eadout with **QU**bit **L**earning via
**E**nsemble **S**tages) is the primary classification pipeline in this repository.
It combines signal-processing and machine-learning stages to achieve accurate multi-qubit
state discrimination from the raw IQ traces, using a compact feature set designed to be
deployable on an FPGA.

### Overview and Design Philosophy

Raw multiplexed IQ traces contain all 5 qubit signals superimposed at different IF
frequencies.  Direct classification from the raw trace requires a large model and gives
no interpretability.  HERQULES instead uses a structured 4-stage pipeline:

```
Raw multiplexed IQ traces
         │
         ▼  Stage 1
Frequency Demodulation          ← per-qubit digital down-conversion + LPF
         │
         ▼  Stage 2
Pre-classification / Purification  ← geometric IQ clustering; label clean traces;
                                      catalogue error events (relax, |2⟩, excite)
         │
         ▼  Stage 3
Matched Filter (MF)  +  Relaxation MF (RMF)
  ← learn optimal linear envelope per qubit for |0⟩ vs |1⟩
  ← learn optimal linear envelope per qubit for relax vs |0⟩
         │
         ▼  Stage 4
Compact MLP (Net_rmf: 10→10→20→32)
  ← input: 5 MF scalars + 5 RMF scalars
  ← output: logits over 32 basis states
```

This pipeline yields a **10-dimensional** feature vector (5 MF + 5 RMF scalars) which is
fed into a tiny MLP.  The entire forward pass at inference time involves only a few
hundred multiply-accumulate operations — highly FPGA-friendly.

### Stage 1 — Frequency Demodulation

**Function:** `demodulate_multiplexed_traces(iq_traces, qubit_frequencies, sampling_rate, ...)`

The raw multiplexed IQ stream is processed per qubit:

1. Optional DC-offset removal and I/Q amplitude imbalance correction.
2. **Digital down-conversion**: multiply the IQ phasor by `exp(j 2π f_IF t)` to shift
   the target resonator's contribution to DC.
3. **Low-pass filter** (3rd-order Butterworth, default cut-off 10 MHz) to suppress all
   other resonator signals.
4. Save per-qubit IQ traces to `demodulated_q{k}_.h5`.

After demodulation, each qubit's traces are shape `(N_shots, 500, 2)`.

### Stage 2 — Pre-classification and Trace Purification

**Class:** `preclassifier`  
**Helper:** `get_traces(num_qubits, plot, rscale, data_type)`

Before computing matched-filter envelopes it is critical to build *clean* reference
traces for each qubit state.  `preclassifier` clusters the time-averaged IQ responses
into |0⟩ and |1⟩ clouds.

For each qubit:
1. Compute the IQ centroid of each cloud.
2. Define a *purity radius* = `rscale × (inter-centroid distance / 2)`.
3. Keep only traces within the radius as clean reference traces.
4. Classify traces *outside* the radius into three error categories:

| Error category | Criterion |
|---|---|
| **Relaxation** (|1⟩→|0⟩) | |1⟩-labelled trace whose IQ mean falls inside the |0⟩ cluster |
| **|2⟩ leakage** | |1⟩-labelled trace outside both clusters |
| **Thermal excitation** (|0⟩→|1⟩) | |0⟩-labelled trace outside the |0⟩ cluster |

The classifier state (indices and trace classes) is persisted to `preclassifier_state.pkl`
and loaded back for reuse.

Key methods:

| Method | Description |
|---|---|
| `fit()` | Run the purification pipeline; populate `filtered_indices` and `trace_classes`. |
| `predict(data)` | Extract purified traces from a raw array using stored indices. |
| `save_state(filename)` | Pickle the classifier state. |
| `load_state(filename)` | Restore classifier state from pickle. |
| `get_traces()` | Return the per-qubit trace-class dictionary. |

### Stage 3 — Matched Filter and Relaxation Matched Filter

**Class:** `relaxation_mf_classifier`  
**Module:** [`matched_filter.py`](matched_filter.py)  
**Helper:** `get_mf(traces_0, traces_1)`

Two independent matched filters are computed per qubit:

#### Standard Matched Filter (MF)

Discriminates |0⟩ vs |1⟩.  The Wiener-optimal linear envelope is:

```
h_MF = E[x_0 − x_1] / Var[x_0 − x_1]
```

A **boxcar window** is multiplied with the envelope to truncate integration at a per-qubit
optimal time, reducing sensitivity to late-time noise.  The boxcar widths used in practice
are `[1, 1, 9, 2, 9]` (in units of 50 ADC samples).

The discrimination threshold is set at the 99.5th percentile of the MF output distribution
on |0⟩ traces, providing a high-confidence acceptance region.

#### Relaxation Matched Filter (RMF)

Distinguishes clean |0⟩ traces from |1⟩→|0⟩ *relaxation* traces:

```
h_RMF = E[x_relax − x_0] / Var[x_relax − x_0]
```

This filter has a characteristic shape that reflects the IQ trajectory of a qubit that
started in |1⟩ but decayed to |0⟩ during the readout window.  It provides complementary
information to the standard MF and significantly improves discrimination accuracy in the
presence of qubit relaxation.

Key methods of `relaxation_mf_classifier`:

| Method | Description |
|---|---|
| `fit(trace_classes, num_qubits, boxcars)` | Compute per-qubit RMF envelopes and thresholds. |
| `predict(num_qubits, data_type, trace_length)` | Apply RMF to train/val/test data. |
| `save_state(filename)` | Pickle envelopes, thresholds, and inherited state. |
| `load_state(filename)` | Restore from pickle. |

### Stage 4 — Neural Network Classifier

**Classes:** `Net` (MF-only) and `Net_rmf` (MF + RMF, primary)

The compact MLP takes the concatenated MF and RMF scalar outputs (10 features) and
produces logits over 32 classes:

```
Input (10) → Linear(10→10) → ReLU → Linear(10→20) → ReLU → Linear(20→32)
```

A `CrossEntropyLoss` with Adam (lr = 0.01) is used.  A step-decay LR schedule divides
the learning rate by 10 at epochs 30, 60, and 90.  Training runs for 100 epochs and the
best checkpoint (by validation accuracy) is saved to `checkpoints/mf_nn/best_epoch.pth`.

There is also a **baseline** MLP (`Net_baseline`) that takes the full 1000-sample raw
trace as input (1000 → 500 → 250 → 32).  It serves as an accuracy upper-bound but is not
suitable for FPGA deployment.

#### Entry Points

| Function | Description |
|---|---|
| `train(run_pre_filter, run_semi_sup, run_rmf, dur)` | Full HERQULES pipeline. |
| `test()` | Evaluate `Net_rmf` from saved checkpoint. |
| `train_baseline()` | Train the raw-trace baseline MLP. |

---

## Matched Filter Module

**File:** [`matched_filter.py`](matched_filter.py)

This module implements the matched-filter computations used in Stage 3.  It is imported
by `HERQULES.py` but can also be used independently.

### Core Functions

| Function | Description |
|---|---|
| `MF_meas(X_train, X_test, y_train, y_test, stop_index, bcub, ...)` | Single matched-filter train + evaluate (binary, with optional boxcar and SVM threshold). |
| `MF_SVM_limit(X, y)` | SVM-based threshold optimisation: fits a LinearSVC on 1-D MF outputs and finds the decision boundary. |
| `MF_single_disc(X, y, stop_index, th_limit_C)` | Searches for the optimal boxcar width by iteratively shortening the integration window. |
| `obtain_matched_filter_with_bcub(X, y, stop_index, th_limit_C, best_bc)` | Compute MF envelope with a fixed pre-determined boxcar. |
| `find_best_matched_filter(train_gnd, train_ext, best_bc)` | High-level wrapper: constructs train arrays, calls disc or bcub depending on `best_bc`. |

### Multi-qubit Helpers

| Function | Variant | Description |
|---|---|---|
| `search_matched_filter_for_all_qubits` | Standard | Compute MF for all 5 qubits from a stacked `(32, N, T, 2)` array. |
| `search_matched_filter_for_all_qubits_demux` | Demux | Same, but expects per-qubit data list from `mf_demux_data_prep`. |
| `search_matched_filter_for_all_qubits_preclass` | Pre-class | Same, but selects ground/excited traces by label from a flat purified array. |

### Preprocessing and Evaluation

| Function | Description |
|---|---|
| `matched_filter_preprocess(data, envelopes)` | Apply 5 MF envelopes to stacked data → `(32, N_min, 5)` scalar array. |
| `matched_filter_preprocess_demux(data, envelopes)` | Same but for per-qubit demux structure. |
| `calculate_matched_filter_acc(data, all_mfs, all_thres)` | Evaluate MF classifier on stacked data; print accuracy. |
| `calculate_matched_filter_acc_demux(data, all_mfs, all_thres)` | Same for demux data; also saves `mf_preds.pkl`. |

### Matched Filter Mathematics

For a binary |0⟩ / |1⟩ task the matched filter envelope is:

```
h = E[x_0 − x_1] / Var[x_0 − x_1]
```

where `x_0`, `x_1` are flattened IQ traces (I followed by Q).  Classification:

```
y_pred = (x · h) < threshold
```

The *boxcar* is a rectangular window applied element-wise: it zeros the envelope beyond
a given time index, restricting the integration window.

---

## Additional Model Architectures

### 1. arXiv:2406.18807 FNN

**File:** [`networks/Arxiv240618807FNN.py`](networks/Arxiv240618807FNN.py)  
**Training script:** [`train_arxiv_model.py`](train_arxiv_model.py)

A reproduction of the lightweight FNN described in [arXiv:2406.18807](https://arxiv.org/abs/2406.18807).

#### Architecture

```
Input (2)  ──► Linear(2→8) ──► ReLU ──► Linear(8→4) ──► ReLU ──► Linear(4→1) ──► Sigmoid
```

| Layer | Size | Activation |
|---|---|---|
| Input | 2 | — |
| Hidden 1 | 8 | ReLU |
| Hidden 2 | 4 | ReLU |
| Output | 1 | Sigmoid |

#### Input Processing

The raw 5-qubit multiplexed trace is **demodulated** per qubit before being fed to the network:

1. **Frequency demodulation** — for each of the 5 qubit IF frequencies, rotate the trace by
   `exp(j 2πf_IF t)` and integrate over the readout window.
2. **Min-max normalisation** — scale each (I, Q) column to [0, 1].
3. **Per-qubit labelling** — `y_q = (y >> q) & 1`.
4. **One model per qubit** — 5 separate `Arxiv240618807FNN` instances.

#### Training Configuration

| Parameter | Value |
|---|---|
| Loss | Binary Cross-Entropy (`nn.BCELoss`) |
| Optimiser | Adam (lr = 1e-3) |
| Batch size | 64 |
| Epochs | 40 |

---

### 2. Transformer (QubitClassifierTransformer)

**File:** [`networks/Transfomer.py`](networks/Transfomer.py)

A Vision-Transformer (ViT) inspired encoder for **direct classification from raw IQ traces**
across all 32 states simultaneously.

#### Architecture Overview

```
Raw IQ trace (batch, 500, 2)
        │
        ▼
 PatchEmbedding          ← split into 50 patches of 10 samples each, project to 128-D
 + [CLS] token prepended ← learnable token at position 0; final representation → classifier
        │
        ▼
 PositionalEncoding      ← fixed sinusoidal PE added to all 51 tokens
        │
        ▼
 TransformerEncoder      ← 4 × (MHSA + FFN + LayerNorm + Dropout)
   num_heads = 8
   FFN hidden = 512 (4 × embedding_dim)
        │
        ▼
 [CLS] token at index 0  ← aggregates global context via attention
        │
        ▼
 LayerNorm → Linear(128→32) ← raw logits over 32 qubit states
```

#### Default Hyper-parameters

| Parameter | Value |
|---|---|
| `patch_size` | 10 (samples) |
| `embedding_dim` | 128 |
| `num_heads` | 8 |
| `num_layers` | 4 |
| `dropout` | 0.1 |
| `num_classes` | 32 |
| Loss | `nn.CrossEntropyLoss` |

---

### 3. KLiNQ — Knowledge Distillation Pipeline

KLiNQ (**K**nowledge-**Li**ght **N**eural-network **Q**ubit-readout) is a two-stage
distillation framework designed to produce student models small enough for FPGA deployment.

#### Stage 1 — Teacher Training

**File:** [`networks/SingleQubitFNN.py`](networks/SingleQubitFNN.py)

```
Input (2×T)
  └─► Linear(input, max(T, 500)) → BN → ReLU → Dropout(0.5)
  └─► Linear(h1, h1//2)          → BN → ReLU → Dropout(0.5)
  └─► Linear(h2, h2//2)          → BN → ReLU → Dropout(0.5)
  └─► Linear(h3, output)
```

> **Training:** 300 epochs, Adam lr=1e-4, batch=1024, EarlyStopping patience=15.

#### Intermediate Teacher (KLiNQTeacherModel)

**File:** [`networks/KLiNQ_TeacherModel.py`](networks/KLiNQ_TeacherModel.py)

```
Input (2×T) → Linear(→64) → BN → ReLU → Dropout(0.3)
            → Linear(64→32) → BN → ReLU → Dropout(0.3)
            → Linear(32→output)
```

#### Stage 2 — Student Training (KLiNQStudentModel)

**File:** [`networks/KLiNQ_StudentModel.py`](networks/KLiNQ_StudentModel.py)

A tiny model (~250 parameters) trained via knowledge distillation.  Student input is:

| Feature group | Computation | Dimension |
|---|---|---|
| Full flattened IQ trace | `flatten_iq_dimensions(trace[:500, :])` | 1000 |
| Time-averaged IQ | Bin-average to `target_length` bins | 2 × `target_length` |
| Matched-Filter scalar | `I·MF_I + Q·MF_Q` | 1 |

```
Input (input_size) → Linear(→16) → BN → ReLU
                   → Linear(16→8) → BN → ReLU
                   → Linear(8→1)   (raw logit)
```

**Knowledge Distillation Loss:**

```
L = α × L_soft + (1 − α) × L_hard
L_soft = KL( softmax(student/T) || softmax(teacher/T) )
L_hard = BCEWithLogitsLoss(student, true_labels)
```

---

## Repository Structure

```
qubit_state_discrimination/
│
├── HERQULES.py                   ← Full HERQULES training/evaluation pipeline
├── matched_filter.py             ← Matched-filter computation utilities
├── train_arxiv_model.py          ← Training script for arXiv FNN (5-qubit)
├── test.py                       ← Evaluation / inference script
│
├── data/
│   ├── five_qubit_data/          ← Place raw HDF5 dataset files here
│   └── single_qubit_data/        ← Per-qubit datasets (generated by notebooks)
│
├── helpers/
│   ├── config.py                 ← All hyper-parameter and path configuration
│   ├── data_loader.py            ← QubitData class: HDF5 loading + preprocessing
│   ├── data_utils.py             ← Low-level data utilities (normalisation, MF, etc.)
│   └── nn_utils.py               ← Loss/optimizer setup, DataLoader creation
│
├── networks/
│   ├── Arxiv240618807FNN.py      ← 2-hidden-layer FNN (arXiv:2406.18807)
│   ├── Transfomer.py             ← ViT-style Transformer encoder
│   ├── SingleQubitFNN.py         ← Large adaptive FNN (KLiNQ teacher)
│   ├── SingleQubitFNN_StudentModel.py  ← Intermediate student
│   ├── KLiNQ_TeacherModel.py     ← Compact FNN teacher
│   └── KLiNQ_StudentModel.py     ← Tiny student for FPGA deployment
│
├── trainers/                     ← KD and standard training logic
│
└── runners/                      ← Executable training scripts
    ├── train_SingleQubitFNN.py
    ├── train_KD_with_SingleQubitFNN.py
    └── train_KD_with_KLinQ_TeacherStudent.py
```

---

## Data Pipeline

### `helpers/data_utils.py`

Low-level, stateless utility functions operating on NumPy arrays.

| Function | Description |
|---|---|
| `hdf5_data_load` | Load `X` and `y` from an HDF5 file. |
| `custom_hdf5_data_loader` | Memory-efficient partial HDF5 load. |
| `QubitTraceDataset` | `torch.utils.data.Dataset` wrapper for NumPy arrays. |
| `reduce_trace_duration` | Truncate `(N, T, 2)` → `(N, T', 2)`. |
| `flatten_iq_dimensions` | Reshape `(N, T, 2)` → `(N, 2T)` for FNN models. |
| `stratified_split` | Class-balanced train/val split. |
| `normalize_data` | z-score normalisation (creates new arrays). |
| `normalize_data_inplace` | z-score normalisation in-place. |
| `normalize_data_forb` | Frobenius-norm division. |
| `normalize_data_std_p2` | z-score with std rounded to nearest power of 2 (FPGA-friendly). |
| `apply_mf_rmf` | Compute matched-filter scalar: `output = I·MF_I + Q·MF_Q`. |
| `compute_normalization_params` | Compute `{n, mu}` for fixed-point normalisation. |
| `apply_normalization` | Apply fixed-point-friendly normalisation. |

### `helpers/data_loader.py`

High-level `QubitData` class orchestrating the full preprocessing pipeline.

| Method | Pipeline |
|---|---|
| `load_data()` | Load raw HDF5 → `(X_train, y_train, X_test, y_test)`. |
| `transform(...)` | Truncate → flatten → normalise → split. |
| `load_transform()` | **Standard pipeline** for FNN and Transformer. |
| `load_transform_KLiNQ_KD(target_length)` | **KLiNQ pipeline**: full trace + averaged trace + MF scalar. |
| `average_trace_data_fixed_length(data, n)` | Bin-average traces to `n` time bins. |

**Normalisation strategies** (selected via `data_config['normalize']`):

| Key | Method |
|---|---|
| `'mean/std'` | z-score normalisation |
| `'forb'` | Frobenius-norm division |
| `'forb_s'` | Frobenius subtraction variant |
| `'forb-weighted'` | Frobenius / 4× |
| `'mean/p2std'` | z-score with power-of-2 std (FPGA-friendly) |
| `'no-norm'` | No normalisation |

---

## Usage Instructions

### 1. Place raw data

Put the original HDF5 dataset files in `data/five_qubit_data/`:
- `DRaw_C_Tr_v0-001`  (training)
- `DRaw_C_Te_v0-002`  (testing)

### 2. Demodulate the multiplexed traces

Run the demodulation step to produce per-qubit HDF5 files:

```python
from HERQULES import demodulate_multiplexed_traces
import numpy as np

demodulate_multiplexed_traces(
    iq_traces=all_data,
    qubit_frequencies=freq_readout,
    sampling_rate=500e6
)
# Produces: demodulated_q1_.h5 ... demodulated_q5_.h5
```

### 3. Run the HERQULES training pipeline

```python
from HERQULES import train

# Full pipeline: pre-classifier + MF + RMF + neural network
acc_per_qubit = train(run_semi_sup=True, run_rmf=True)
```

This will:
1. Fit the `preclassifier` and save `preclassifier_state.pkl`.
2. Compute and save matched-filter envelopes.
3. Fit the `relaxation_mf_classifier` and save `rmf.pkl`.
4. Train `Net_rmf` and save `checkpoints/mf_nn/best_epoch.pth`.
5. Print overall and per-qubit test accuracy.

### 4. Evaluate a saved model

```python
from HERQULES import test

overall_acc, per_qubit_acc = test()
```

### 5. Run the baseline or neural network models

**arXiv FNN (5-qubit multiplexed):**
```bash
python train_arxiv_model.py
```

**KLiNQ teacher training (per qubit):**
```bash
python runners/train_SingleQubitFNN.py
```

**KLiNQ knowledge distillation:**
```bash
python runners/train_KD_with_KLinQ_TeacherStudent.py
```

### 6. Generate per-qubit datasets for KLiNQ

```bash
jupyter notebook data/single_qubit_dataset_creator.ipynb
jupyter notebook data/multiplexed_traces_mf_rmf_save.ipynb
```

---

## Development Story of KLiNQ

1. **Data preparation** — multiplexed 5-qubit IQ traces split into 5 individual single-qubit
   datasets using the `single_qubit_dataset_creator` notebook.

2. **Teacher training** — `SingleQubitFNN` models (e.g. layers `[1000, 500, 250]`) trained
   independently per qubit.

3. **Architecture search** — many FNN, CNN, and recurrent architectures tested;
   `SingleQubitFNN` consistently outperformed alternatives.

4. **Stage-1 distillation** — trained `SingleQubitFNN` teachers distilled into smaller
   `SingleQubitStudentModel` networks.  Many student models *outperformed* their teachers
   — a well-known regularisation effect of knowledge distillation.

5. **Best student as new teacher** — top-performing stage-1 students re-used as teachers
   for stage 2.

6. **Stage-2 distillation (KLiNQ)** — student takes compact feature vector (averaged IQ +
   MF scalar).  Tiny architectures `[31, 16, 8, 1]` and `[201, 16, 8, 1]` explored,
   targeting FPGA resource budgets.

---

## Disclaimer & Research Context

> **NOTE:** This repository is not in its final production-ready shape.  The codebase has
> not been fully cleaned or polished due to time constraints.  The raw dataset is not
> uploaded to GitHub for space and policy reasons.

> **REMARK:** This repository does **not** contain the codebase for all experiments
> conducted during KLiNQ development.