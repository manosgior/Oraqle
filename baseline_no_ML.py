"""
IQ Trace Threshold Discriminator — Real QPU Data Pipeline
----------------------------------------------------------
Operates on real HDF5 IQ trace data from a 5-qubit QPU.

Pipeline:
  1. Load + merge train/test HDF5 files
  2. Demodulate per qubit (no averaging — preserves time axis)
  3. Sweep truncation length T → integrate up to T → classify
  4. Compute F(T) per qubit with boundary refit at each T
  5. Critical window: dF/dT and matched filter weights |w(t)|
  6. Visualize: per-qubit fidelity curves, critical windows, IQ scatter

Shapes at each stage:
  raw traces       : (N, T_max, 2)              [I, Q]
  demodulated      : (N, T_max, n_qubits, 2)    [I_demod, Q_demod] per qubit
  integrated at T  : (N, n_qubits, 2)           integrated up to T
  labels           : (N, n_qubits)              one label per qubit per shot
"""

import h5py
import numpy as np
import os
import csv
from scipy.signal import butter, sosfilt
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gmean
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

RAW_TRAIN_FILE = "/data/five_qubit_data/DRaw_C_Tr_v0-001"
RAW_TEST_FILE  = "/data/five_qubit_data/DRaw_C_Te_v0-002"

FREQ_READOUT = -np.array([-64.729e6, -25.366e6, 24.79e6, 70.269e6, 127.282e6])

N_QUBITS     = 5
SAMPLE_SEED  = 42
MAX_SAMPLES  = None          # set to e.g. 5000 to subsample for speed

# Sweep: evaluate every STRIDE-th timestep
DT      = 2e-9   # 2 ns per sample

T_START_NS  = 100    # first evaluation point
T_END_NS    = 2000   # 2 µs max
T_STEP_NS   = 100    # evaluate every 100 ns


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_hdf5(filepath: str, is_test: bool, max_samples: int | None = None):
    """
    Load IQ traces and labels from a single HDF5 file.

    Returns:
      X : (N, T, 2)      raw IQ traces
      y : (N, n_qubits)  per-qubit binary labels
    """
    key_suffix = "test" if is_test else "train"
    with h5py.File(filepath, "r") as hf:
        total = hf[f"X_{key_suffix}"].shape[0]
        if max_samples is not None and max_samples < total:
            rng = np.random.RandomState(SAMPLE_SEED)
            indices = np.sort(rng.choice(total, size=max_samples, replace=False))
            X = hf[f"X_{key_suffix}"][indices]
            y = hf[f"y_{key_suffix}"][indices]
        else:
            X = hf[f"X_{key_suffix}"][:]
            y = hf[f"y_{key_suffix}"][:]
    return X, y


def load_and_merge(
    train_file: str,
    test_file: str,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load both HDF5 files and concatenate into a single dataset.
    We don't need the train/test distinction for Section 1 analysis.

    Returns:
      X : (N_total, T_max, 2)
      y : (N_total, n_qubits)  or (N_total,)
    """
    X_tr, y_tr = load_hdf5(train_file, is_test=False, max_samples=max_samples)
    X_te, y_te = load_hdf5(test_file,  is_test=True,  max_samples=max_samples)

    X = np.concatenate([X_tr, X_te], axis=0)
    y = np.concatenate([y_tr, y_te], axis=0)

    print(f"Loaded: {X.shape[0]} traces, T_max={X.shape[1]}, raw shape={X.shape}")
    print(f"Labels shape: {y.shape}")
    return X, y

def decode_labels(y, n_qubits=5):
    """
    y       : (N,) integer labels 0-31
    returns : (N, n_qubits) binary array
              bit q of label → state of qubit q
    """
    y_binary = np.zeros((len(y), n_qubits), dtype=np.uint8)
    for q in range(n_qubits):
        y_binary[:, q] = (y >> q) & 1
    return y_binary

# ─────────────────────────────────────────────
# 2. DEMODULATION (preserves time axis)
# ─────────────────────────────────────────────

def demodulate(
    traces: np.ndarray,
    freq_readout: np.ndarray,
    dt: float = DT,
) -> np.ndarray:
    """
    Demodulate raw IQ traces per qubit WITHOUT averaging.
    Preserves the full time axis for downstream truncation sweeps.

    The original demodulate_and_average collapses axis=1 with np.mean.
    Here we skip that step — the time axis is kept intact so we can
    integrate up to any truncation point T during the sweep.

    Args:
      traces      : (N, T, 2)        raw [I, Q]
      freq_readout: (n_qubits,)      per-qubit LO frequencies in Hz
      dt          : float            sample period in seconds

    Returns:
      demod : (N, T, n_qubits, 2)   demodulated [I, Q] per qubit per timestep
    """
    N, T, _ = traces.shape
    n_qubits = len(freq_readout)
    t = np.arange(T) * dt                        # (T,)

    I_raw = traces[:, :, 0]                      # (N, T)
    Q_raw = traces[:, :, 1]                      # (N, T)

    # Remove DC offsets and correct IQ imbalance per trace
    I_raw = I_raw - np.mean(I_raw, axis=1, keepdims=True)
    Q_raw = Q_raw - np.mean(Q_raw, axis=1, keepdims=True)
    corr = np.std(I_raw, axis=1, keepdims=True) / (np.std(Q_raw, axis=1, keepdims=True) + 1e-10)
    Q_raw = Q_raw * corr

    demod = np.zeros((N, T, n_qubits, 2), dtype=np.float32)

    # Design a 3rd-order Butterworth low-pass filter (10 MHz cutoff)
    sos = butter(3, 10e6, btype='low', fs=1.0/dt, output='sos')

    for i, freq in enumerate(freq_readout):
        lo_i =  np.cos(2 * np.pi * freq * t)    # (T,)
        lo_q = -np.sin(2 * np.pi * freq * t)    # (T,)

        # Standard IQ mixing
        I_demod = I_raw * lo_i - Q_raw * lo_q   # (N, T)
        Q_demod = I_raw * lo_q + Q_raw * lo_i   # (N, T)

        # Filter out the 2*f component and high-frequency noise
        I_demod = sosfilt(sos, I_demod, axis=1)
        Q_demod = sosfilt(sos, Q_demod, axis=1)

        demod[:, :, i, 0] = I_demod
        demod[:, :, i, 1] = Q_demod

    return demod                                 # (N, T, n_qubits, 2)


# ─────────────────────────────────────────────
# 3. INTEGRATION AT TRUNCATION POINT T
# ─────────────────────────────────────────────

def integrate_to_T(
    demod: np.ndarray,
    T_idx: int,
    dt: float = DT,
) -> np.ndarray:
    """
    Integrate demodulated traces up to timestep T_idx (inclusive).

    This is the discrete equivalent of:
      I_bar(T) = integral_0^T I(t) dt
      Q_bar(T) = integral_0^T Q(t) dt

    Args:
      demod : (N, T_max, n_qubits, 2)
      T_idx : int

    Returns:
      integrated : (N, n_qubits, 2)   one [I_bar, Q_bar] per qubit
    """
    segment = demod[:, :T_idx + 1, :, :]              # (N, T, n_qubits, 2)
    integrated = np.trapezoid(segment, dx=dt, axis=1)  # (N, n_qubits, 2)
    return integrated


# ─────────────────────────────────────────────
# 4. THRESHOLD CLASSIFIER
# ─────────────────────────────────────────────

def fit_threshold(X: np.ndarray, y: np.ndarray):
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000))
    ])
    clf.fit(X, y)
    return clf


def assignment_fidelity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    F = 1 - 0.5 * [P(1_hat|0) + P(0_hat|1)]
    Standard symmetric assignment fidelity.
    """
    mask0 = y_true == 0
    mask1 = y_true == 1
    p10 = np.mean(y_pred[mask0] == 1) if mask0.any() else 0.0
    p01 = np.mean(y_pred[mask1] == 0) if mask1.any() else 0.0
    return 1.0 - 0.5 * (p10 + p01)


# ─────────────────────────────────────────────
# 5. FIDELITY SWEEP — Q1
# ─────────────────────────────────────────────

def sweep_fidelity_per_qubit(
    demod: np.ndarray,
    labels: np.ndarray,
    T_indices: np.ndarray,
    dt: float = DT,
    train_frac: float = 0.7,
    seed: int = SAMPLE_SEED,
) -> np.ndarray:
    """
    Sweep truncation length and compute F(T) for each qubit.
    Boundary is REFIT from scratch at every T — this is essential.
    A fixed boundary would conflate signal quality with threshold placement.

    Args:
      demod     : (N, T_max, n_qubits, 2)
      labels    : (N, n_qubits) or (N,)
      T_indices : 1D array of timestep indices to evaluate

    Returns:
      fidelities : (n_qubits, len(T_indices))
    """
    N = demod.shape[0]
    n_qubits = demod.shape[2]

    # Handle both label shapes
    if labels.ndim == 1:
        labels = np.tile(labels[:, None], (1, n_qubits))

    # Fix train/test split once and reuse across all T
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    split = int(N * train_frac)
    tr_idx, te_idx = perm[:split], perm[split:]

    fidelities = np.zeros((n_qubits, len(T_indices)))

    for t_i, T in enumerate(T_indices):
        X_all = integrate_to_T(demod, T, dt=dt)   # (N, n_qubits, 2)

        for q in range(n_qubits):
            X_q = X_all[:, q, :]
            y_q = labels[:, q]

            if len(np.unique(y_q[tr_idx])) < 2:
                fidelities[q, t_i] = np.nan
                continue

            clf = fit_threshold(X_q[tr_idx], y_q[tr_idx])
            y_pred = clf.predict(X_q[te_idx])
            fidelities[q, t_i] = assignment_fidelity(y_q[te_idx], y_pred)

        if t_i % 10 == 0:
            f_str = "  ".join(f"Q{q}:{fidelities[q, t_i]:.3f}" for q in range(n_qubits))
            print(f"  T={T:4d}  {f_str}")

    return fidelities

def evaluate_sliding_window_fidelity(
    demod: np.ndarray,
    labels: np.ndarray,
    window_ns: float = 250.0,
    step_ns: float = 100.0,
    dt: float = DT,
    train_frac: float = 0.7,
    seed: int = SAMPLE_SEED,
):
    """
    Evaluates the 'local' classification power by moving a fixed-width window
    across the readout trace.
    """
    N, T_max, n_qubits, _ = demod.shape
    window_size = int(window_ns / (dt * 1e9))
    step_size = int(step_ns / (dt * 1e9))
    
    if labels.ndim == 1:
        labels = np.tile(labels[:, None], (1, n_qubits))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    split = int(N * train_frac)
    tr_idx, te_idx = perm[:split], perm[split:]

    # Calculate start indices for sliding windows
    start_indices = np.arange(0, T_max - window_size, step_size)
    window_centers_us = (start_indices + window_size // 2) * dt * 1e6
    
    fidelities_sliding = np.zeros((n_qubits, len(start_indices)))

    print(f"Starting Sliding Window Evaluation (Window: {window_ns}ns, Step: {step_ns}ns)")
    
    for i, start in enumerate(start_indices):
        end = start + window_size
        # Integrate over the local window
        segment = demod[:, start:end, :, :]
        X_window = np.trapezoid(segment, dx=dt, axis=1) # (N, n_qubits, 2)

        for q in range(n_qubits):
            X_q = X_window[:, q, :]
            y_q = labels[:, q]

            if len(np.unique(y_q[tr_idx])) < 2:
                fidelities_sliding[q, i] = np.nan
                continue

            clf = fit_threshold(X_q[tr_idx], y_q[tr_idx])
            y_pred = clf.predict(X_q[te_idx])
            fidelities_sliding[q, i] = assignment_fidelity(y_q[te_idx], y_pred)

    return window_centers_us, fidelities_sliding

def plot_sliding_window_results(
    window_centers_us: np.ndarray,
    fidelities_sliding: np.ndarray,
    window_ns: float
):
    """
    Plots the sliding window fidelity evolution.
    """
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("deep")
    markers = ['o', 's', '^', 'D', 'P']
    n_qubits = fidelities_sliding.shape[0]

    plt.figure(figsize=(10, 6))
    for q in range(n_qubits):
        plt.plot(
            window_centers_us, 
            fidelities_sliding[q], 
            color=palette[q], 
            marker=markers[q],
            markersize=5, 
            lw=2, 
            label=f'Q{q}'
        )

    plt.axhline(0.5, color='black', lw=1, ls='--', alpha=0.5, label='Chance')
    plt.xlabel('Window Center Time (µs)', fontsize=12)
    plt.ylabel(f'Local Fidelity ({window_ns}ns window)', fontsize=12)
    plt.title('Sliding Window Readout Fidelity', fontsize=14)
    plt.xlim(0, 2)
    plt.ylim(0.45, 1.02)
    plt.legend(frameon=True)
    
    out_path = './optimization_reports/sliding_window_fidelity.pdf'
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Sliding window plot saved to: {out_path}")


def sweep_fidelity_incremental(
    demod: np.ndarray,
    labels: np.ndarray,
    T_indices: np.ndarray,
    dt: float = DT,
    train_frac: float = 0.7,
    seed: int = SAMPLE_SEED,
) -> np.ndarray:
    N, T_max, n_qubits, _ = demod.shape

    if labels.ndim == 1:
        labels = np.tile(labels[:, None], (1, n_qubits))

    # Fix split once
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    split = int(N * train_frac)
    tr_idx, te_idx = perm[:split], perm[split:]

    T_set = set(T_indices.tolist())
    fidelities = np.zeros((n_qubits, len(T_indices)))
    accumulated = np.zeros((N, n_qubits, 2), dtype=np.float64)
    t_i = 0

    for t in range(1, max(T_indices) + 1):
        # Incremental trapezoidal step
        accumulated += 0.5 * dt * (demod[:, t-1, :, :] + demod[:, t, :, :])

        if t in T_set:
            for q in range(n_qubits):
                X_q = accumulated[:, q, :]
                y_q = labels[:, q]

                if len(np.unique(y_q[tr_idx])) < 2:
                    fidelities[q, t_i] = np.nan
                    continue

                clf = fit_threshold(X_q[tr_idx], y_q[tr_idx])
                y_pred = clf.predict(X_q[te_idx])
                fidelities[q, t_i] = assignment_fidelity(y_q[te_idx], y_pred)

            f_str = "  ".join(f"Q{q}:{fidelities[q, t_i]:.3f}" for q in range(n_qubits))
            print(f"  T={t:4d} ({t*dt*1e9:.0f} ns)  {f_str}")
            t_i += 1

    return fidelities
# ─────────────────────────────────────────────
# 6. CRITICAL WINDOW — Q2
# ─────────────────────────────────────────────

def derivative_critical_window(
    fidelities: np.ndarray,
    T_values: np.ndarray,
    smooth_sigma: float = 2.0,
) -> np.ndarray:
    """
    dF/dT per qubit — information gain rate per unit time.
    Smoothed before differentiation to suppress numerical noise.

    Returns:
      dFdT : (n_qubits, n_T)
    """
    dFdT = np.zeros_like(fidelities)
    for q in range(fidelities.shape[0]):
        F_q = fidelities[q].copy()
        # Fill NaNs via linear interpolation before smoothing
        mask = ~np.isnan(F_q)
        if mask.sum() < 2:
            continue
        F_q = np.where(mask, F_q,
                       np.interp(np.arange(len(F_q)), np.where(mask)[0], F_q[mask]))
        F_smooth = gaussian_filter1d(F_q, sigma=smooth_sigma)
        dFdT[q] = np.gradient(F_smooth, T_values)
    return dFdT


def matched_filter_weights_per_qubit(
    demod: np.ndarray,
    labels: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Model-free matched filter weights per qubit.

      w_q(t) ∝ |μ1_q(t) - μ0_q(t)| / σ²_q(t)

    This is derived analytically from signal statistics — no classification
    needed. The high-weight region is the critical window.

    Args:
      demod  : (N, T_max, n_qubits, 2)
      labels : (N, n_qubits) or (N,)

    Returns:
      weights : (n_qubits, T_max)  normalized to [0, 1] per qubit
    """
    N, T_max, n_qubits, _ = demod.shape

    if labels.ndim == 1:
        labels = np.tile(labels[:, None], (1, n_qubits))

    weights = np.zeros((n_qubits, T_max))

    for q in range(n_qubits):
        mask0 = labels[:, q] == 0
        mask1 = labels[:, q] == 1

        traces_q = demod[:, :, q, :]             # (N, T_max, 2)
        mu0 = traces_q[mask0].mean(axis=0)       # (T_max, 2)
        mu1 = traces_q[mask1].mean(axis=0)       # (T_max, 2)

        var0 = traces_q[mask0].var(axis=0)       # (T_max, 2)
        var1 = traces_q[mask1].var(axis=0)
        sigma2 = (var0 + var1).mean(axis=1) + epsilon  # (T_max,)

        delta = np.linalg.norm(mu1 - mu0, axis=1)     # (T_max,)
        w = delta / sigma2
        
        # Skip the first 50 samples (100ns) for normalization to avoid filter transients
        norm_max = np.max(w[50:]) if T_max > 50 else w.max()
        weights[q] = gaussian_filter1d(w / (norm_max + epsilon), sigma=5.0)

    return weights


# ─────────────────────────────────────────────
# 7. VISUALIZATION
# ─────────────────────────────────────────────

QUBIT_COLORS = ['steelblue', 'tomato', 'mediumseagreen', 'mediumpurple', 'darkorange']
QUBIT_LABELS = [f'Q{i}' for i in range(N_QUBITS)]

def plot_results(
    demod: np.ndarray,
    labels: np.ndarray,
    T_indices: np.ndarray,
    T_values_us: np.ndarray,
    fidelities: np.ndarray,
    dFdT: np.ndarray,
    weights: np.ndarray,
    dt: float = DT,
):
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("deep")
    markers = ['o', 's', '^', 'D', 'P']
    n_qubits = fidelities.shape[0]
    t_full_us = np.arange(demod.shape[1]) * dt * 1e6   # µs

    # 1. Per-qubit Accuracy Evolution
    plt.figure(figsize=(10, 6))
    for q in range(n_qubits):
        plt.plot(T_values_us, fidelities[q], color=palette[q], marker=markers[q],
                 markersize=5, lw=2, label=QUBIT_LABELS[q])
    plt.axhline(0.5, color='black', lw=1, ls='--', alpha=0.5, label='Chance')
    plt.xlabel('Readout Duration (µs)', fontsize=12)
    plt.ylabel('Assignment Fidelity $F(T)$', fontsize=12)
    plt.title('Readout Fidelity Evolution', fontsize=14)
    plt.xlim(0, 2)
    plt.ylim(0.45, 1.02)
    plt.legend(frameon=True)
    plt.savefig('./optimization_reports/readout_fidelity_evolution.pdf', dpi=600, bbox_inches='tight')
    plt.close()

    # 2. Information Gain per Qubit
    plt.figure(figsize=(10, 6))
    for q in range(n_qubits):
        plt.plot(T_values_us, dFdT[q], color=palette[q], marker=markers[q],
                 markersize=5, lw=2, label=QUBIT_LABELS[q])
        peak_idx = np.nanargmax(dFdT[q])
        plt.axvline(T_values_us[peak_idx], color=palette[q], lw=1.0, ls=':', alpha=0.6)
    plt.xlabel('Readout Duration (µs)', fontsize=12)
    plt.ylabel('Information Extraction Rate', fontsize=12)
    plt.title('Information Gain Rate per Qubit', fontsize=14)
    plt.xlim(0, 2)
    plt.legend(frameon=True)
    plt.savefig('./optimization_reports/information_gain_rate.pdf', dpi=600, bbox_inches='tight')
    plt.close()

    # 3. Matched Filter Weights
    fig, axes = plt.subplots(n_qubits, 1, figsize=(10, 8), sharex=True)
    for q in range(n_qubits):
        ax = axes[q]
        ax.plot(t_full_us, weights[q], color=palette[q], lw=2)
        ax.fill_between(t_full_us, weights[q], color=palette[q], alpha=0.2)
        ax.set_ylabel(f'Q{q}', fontsize=10, rotation=0, labelpad=20, va='center')
        ax.set_yticks([])
        ax.set_ylim(0, 1.1)
        ax.set_xlim(0, 2)
    axes[-1].set_xlabel('Time (µs)', fontsize=12)
    axes[0].set_title('Matched Filter Weights $w(t)$', fontsize=14)
    plt.tight_layout()
    plt.savefig('./optimization_reports/matched_filter_weights.pdf', dpi=600, bbox_inches='tight')
    plt.close()

    # 4. IQ Scatter - standalone
    plt.figure(figsize=(12, 8))
    labels_2d = labels if labels.ndim == 2 else np.tile(labels[:, None], (1, n_qubits))
    X_full = integrate_to_T(demod, T_indices[-1], dt=dt)
    for q in range(n_qubits):
        plt.subplot(2, 3, q+1)
        X_q, y_q = X_full[:, q, :], labels_2d[:, q]
        plt.scatter(X_q[y_q==0, 0], X_q[y_q==0, 1], s=2, alpha=0.15, color=palette[0])
        plt.scatter(X_q[y_q==1, 0], X_q[y_q==1, 1], s=2, alpha=0.15, color=palette[1])
        plt.title(f'Q{q} IQ Clusters')
    plt.tight_layout()
    plt.savefig('./optimization_reports/iq_scatter_clusters.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to ./optimization_reports/")

def save_sweep_results(T_indices, fidelities, model_name="Baseline_Threshold", csv_dir="./optimization_reports"):
    """
    Saves sweep results to CSV files in a format similar to hyper_optimize.py.
    """
    os.makedirs(csv_dir, exist_ok=True)
    n_qubits, n_lengths = fidelities.shape
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, length in enumerate(T_indices):
        # Accuracies as percentages (0-100) to match hyper_optimize.py
        q_accs_list = fidelities[:, i] * 100.0

        # Geometric mean of all qubits
        gm = gmean(q_accs_list)

        # Geometric mean excluding Q1 (the second qubit, index 1)
        mask = np.ones(n_qubits, dtype=bool)
        if n_qubits > 1:
            mask[1] = False
        gm_excl_q1 = gmean(q_accs_list[mask])

        row = {
            "timestamp": timestamp,
            "model_name": model_name,
            "target_qubit": "all",
            "trace_length": int(length),
            "overall_accuracy": f"{gm:.4f}",
            "geometric_mean_accuracy": f"{gm:.4f}",
            "geometric_mean_accuracy_excl_q1": f"{gm_excl_q1:.4f}",
        }
        for q in range(n_qubits):
            row[f"qubit_{q}_accuracy"] = f"{q_accs_list[q]:.4f}"

        filename = f"{model_name}_len{int(length)}_{timestamp}.csv"
        filepath = os.path.join(csv_dir, filename)

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)

def save_derivative_results(T_indices, dFdT, model_name="Baseline_Threshold", csv_dir="./optimization_reports"):
    """
    Saves dF/dT (information gain rate) results to a single CSV file.
    """
    os.makedirs(csv_dir, exist_ok=True)
    n_qubits, _ = dFdT.shape
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(csv_dir, f"{model_name}_dFdT_analysis_{timestamp}.csv")

    with open(filepath, "w", newline="") as f:
        fieldnames = ["trace_length", "trace_length_ns"] + [f"qubit_{q}_dFdT" for q in range(n_qubits)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, t_idx in enumerate(T_indices):
            row = {
                "trace_length": int(t_idx),
                "trace_length_ns": int(t_idx * DT * 1e9),
            }
            for q in range(n_qubits):
                row[f"qubit_{q}_dFdT"] = f"{dFdT[q, i]:.8e}"
            writer.writerow(row)
    print(f"Saved dF/dT results to: {filepath}")

def save_sliding_window_results(
    window_centers_us: np.ndarray,
    fidelities_sliding: np.ndarray,
    window_ns: float,
    model_name: str = "Baseline_Threshold",
    csv_dir: str = "./optimization_reports"
):
    """
    Saves sliding window fidelity results to a CSV file.
    """
    os.makedirs(csv_dir, exist_ok=True)
    n_qubits, _ = fidelities_sliding.shape
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(csv_dir, f"{model_name}_sliding_window_{int(window_ns)}ns_{timestamp}.csv")

    with open(filepath, "w", newline="") as f:
        fieldnames = ["window_center_us"] + [f"qubit_{q}_fidelity" for q in range(n_qubits)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, center_us in enumerate(window_centers_us):
            row = {"window_center_us": f"{center_us:.4f}"}
            for q in range(n_qubits):
                row[f"qubit_{q}_fidelity"] = f"{fidelities_sliding[q, i]:.6f}"
            writer.writerow(row)
    print(f"Saved sliding window numerical results to: {filepath}")

# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    X, y = load_and_merge(RAW_TRAIN_FILE, RAW_TEST_FILE, max_samples=MAX_SAMPLES)
    # X : (N, T_max, 2)
    # y : (N, n_qubits) or (N,)

    y_binary = decode_labels(y)
    T_max = X.shape[1]

    # 2. Demodulate per qubit — keeps time axis intact
    print("Demodulating...")
    demod = demodulate(X, FREQ_READOUT, dt=DT)
       
    del X   # free raw traces from memory

    # Shuffle everything together
    rng  = np.random.default_rng(SAMPLE_SEED)
    perm = rng.permutation(len(y_binary))
    demod    = demod[perm]
    y_binary = y_binary[perm]

    print("\nRunning sliding window analysis...")
    win_centers, sliding_fids = evaluate_sliding_window_fidelity(demod, y_binary, window_ns=250)
    save_sliding_window_results(win_centers, sliding_fids, window_ns=250)
    plot_sliding_window_results(win_centers, sliding_fids, window_ns=250)

    exit()

    # 3. Build sweep: every T_STRIDE-th sample from t=4 onwards
    T_START  = int(T_START_NS / (DT * 1e9))
    T_END    = int(T_END_NS   / (DT * 1e9))
    T_STRIDE = int(T_STEP_NS  / (DT * 1e9))
    T_END    = min(T_END, T_max - 1)

    T_indices   = np.arange(T_START, T_END + 1, T_STRIDE)
    T_values_s  = T_indices * DT
    T_values_us = T_values_s * 1e6

    # 4. Fidelity sweep — refit boundary at each T
    print("Sweeping F(T) per qubit (boundary refit at each T)...")
    fidelities = sweep_fidelity_incremental(demod, y_binary, T_indices, dt=DT)

    # Save results to CSV files in standard format
    save_sweep_results(T_indices, fidelities)

    # 5. Critical window: dF/dT and matched filter weights
    print("Computing critical windows...")
    dFdT    = derivative_critical_window(fidelities, T_values_s)

    # Save derivative results
    save_derivative_results(T_indices, dFdT)

    weights = matched_filter_weights_per_qubit(demod, y_binary)

    # 6. Sliding Window Analysis

    print("\nSummary:")
    for q in range(N_QUBITS):
        peak_T  = T_values_us[np.nanargmax(dFdT[q])]
        sat_idx = np.nanargmax(fidelities[q] >= 0.99 * np.nanmax(fidelities[q]))
        sat_T   = T_values_us[sat_idx]
        max_F   = np.nanmax(fidelities[q])
        print(f"  Q{q}: F_max={max_F:.4f}  "
              f"peak dF/dT at {peak_T:.3f} µs  "
              f"saturation at {sat_T:.3f} µs")

    # 6. Plot
    print("\nPlotting...")
    plot_results(demod, y_binary, T_indices, T_values_us, fidelities, dFdT, weights, dt=DT)