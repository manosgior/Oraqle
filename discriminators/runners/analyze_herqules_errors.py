"""
analyze_herqules_errors.py
==========================
Characterise the HERQULES Net_rmf error modes at full 1us readout (length=500).

Workflow
--------
1. Load the HERQULES_len500.pth checkpoint, demodulate train+test, compute the
   paper-faithful MF/RMF envelopes once.
2. Build the 10-D MF+RMF feature vector per test trace, run Net_rmf to get
   32-class logits, marginalise to per-qubit P(qubit=1).
3. For each qubit q, label each error trace with the MOST diagnostic error
   category (exclusive, in priority order):

   - "t1_pattern"   : the matched filter applied to the first half of the
                      trace and to the second half disagree in sign. The
                      global MF averages this out; physically consistent with
                      a T1 |1>->|0> decay mid-readout.
   - "amp_outlier"  : the time-averaged IQ amplitude in qubit q's demodulated
                      band is > AMPLITUDE_OUTLIER_Z train-std away from the
                      train-set mean. Candidate leakage to |2> or RFI.
   - "near_boundary": Net_rmf's marginal P(qubit_q = true_label) is below
                      NEAR_BOUNDARY_CONF_THRESHOLD (e.g. <0.60). Genuine
                      ambiguity from low SNR; classifier is uncertain.
   - "other"        : confident wrong prediction not explained by the above.
                      Most likely candidates: label noise (state-prep error),
                      cross-talk, slow/late signal arrival.

4. Save a per-qubit summary CSV (counts per category, overlap counts) and one
   PNG per qubit with four diagnostic panels.

Run (inside the container):
    python -m runners.analyze_herqules_errors
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from disc_paths import RAW_TRAIN, RAW_TEST, MODELS_OUT, REPORTS_OUT  # [unify] replaces the old docker mount paths
from networks.HERQULES import Net_rmf
# Use HERQULES_original.py's own functions for data prep, envelope computation,
# and RMF — so the features Net_rmf sees here are exactly the features its
# training pipeline produced, modulo random subsampling (we apply MF/RMF
# per-trace to keep the trace identity needed for the error categorisation).
from trainers.HERQULES_original import (
    mf_demux_data_prep,
    relaxation_mf_classifier,
)
from matched_filter import search_matched_filter_for_all_qubits_demux


# ============================================================================
# Configuration
# ============================================================================

RAW_TRAIN_FILE = str(RAW_TRAIN)
RAW_TEST_FILE  = str(RAW_TEST)
NUM_QUBITS     = 5
NUM_CLASSES    = 2 ** NUM_QUBITS
# Trace length to analyse. Must match the length the loaded HERQULES_lenL.pth
# was trained at AND the length the current rmf.pkl was saved at (each
# HERQULES_original.train() run overwrites rmf.pkl).
# Default picks up HERQULES_TRACE_LENGTH so the import-time load in
# HERQULES_original.py and the analyze pipeline use the same length without
# editing two files. CLI --length overrides everything.
TRACE_LENGTH   = int(os.environ.get("HERQULES_TRACE_LENGTH", 500))

# Pin to the bind-mount target (matches HERQULES_original.py and the docker -v
# spec ~/oraqle_models:/app/saved_models). Relative "./saved_models" silently
# resolves to the baked-in in-image copy when launched from /app/Discriminators,
# which is frozen at docker-build time and may be days/weeks out of date.
SAVE_DIR  = str(MODELS_OUT)  # [unify] was /app/saved_models
PLOTS_DIR = str(REPORTS_OUT)  # [unify] was /app/optimization_reports
CSV_DIR   = PLOTS_DIR

# Thresholds for the exclusive categorisation
NEAR_BOUNDARY_CONF_THRESHOLD = 0.60   # P(correct class) below this => near-boundary
AMPLITUDE_OUTLIER_Z          = 3.0    # |z-score(amp)| above this => outlier
NUM_SAMPLE_TRACES            = 3      # how many example traces per error type to plot

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_SEED = 42

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CSV_DIR,   exist_ok=True)


# ============================================================================
# Data / features  (using HERQULES_original.py's own pipeline)
# ============================================================================

def _demux_to_flat_per_qubit(demux_data) -> dict:
    """Convert mf_demux_data_prep's per-qubit/per-state nesting to flat per-trace.

    Input: length-5 list, demux_data[q] has shape (32, N_per_state, T, 2).
    Output: dict {q: (32*N_per_state, T, 2)} with trace order
            [state_0, state_1, ..., state_31] (matches rmf.predict's flat ordering).
    """
    flat = {}
    for q in range(NUM_QUBITS):
        a = demux_data[q]
        flat[q] = a.reshape(a.shape[0] * a.shape[1], a.shape[2], a.shape[3])
    return flat


def _per_trace_mf(demux_data, mf_envelopes) -> np.ndarray:
    """Apply each per-qubit MF envelope to every test trace (no random subsampling).

    Returns shape (N_total, NUM_QUBITS) where row i is the 5-D MF vector for
    the same physical trace — preserving the trace identity that
    matched_filter_preprocess_demux throws away. Trace order matches the
    [state_0, ..., state_31] flattening used elsewhere.
    """
    N_per_state = demux_data[0].shape[1]
    N_total     = NUM_CLASSES * N_per_state
    F           = 2 * TRACE_LENGTH
    out = np.zeros((N_total, NUM_QUBITS), dtype=np.float32)
    for q in range(NUM_QUBITS):
        flat = demux_data[q].reshape(N_total, F).astype(np.float32, copy=False)
        env  = mf_envelopes[q].astype(np.float32, copy=False)
        out[:, q] = flat @ env
    return out


def _scrambled_mf(demux_data, mf_envelopes, seed: int = SAMPLE_SEED):
    """Replicate matched_filter_preprocess_demux: for each (qubit, state),
    permute the per-state trace order independently before applying the MF
    envelope. Returns (mf_features, perms) where perms[q, s, k] is the index
    into demux_data[q][s] that contributed to row (s * N_per_state + k),
    column q of the MF feature matrix.

    Tracking the permutations is the only departure from the original code
    (which discards them); we need them to attribute each MF score back to a
    physical trace for the per-qubit error categorisation.
    """
    rng = np.random.RandomState(seed)
    N_per_state = demux_data[0].shape[1]
    N_total     = NUM_CLASSES * N_per_state
    F           = 2 * TRACE_LENGTH
    mf_out = np.zeros((N_total, NUM_QUBITS), dtype=np.float32)
    perms  = np.zeros((NUM_QUBITS, NUM_CLASSES, N_per_state), dtype=np.int64)
    for q in range(NUM_QUBITS):
        env = mf_envelopes[q].astype(np.float32, copy=False)
        for s in range(NUM_CLASSES):
            n = demux_data[q].shape[1]
            idx = rng.choice(n, n, replace=False)
            perms[q, s] = idx
            traces = demux_data[q][s, idx].reshape(n, F).astype(np.float32, copy=False)
            mf_out[s * N_per_state:(s + 1) * N_per_state, q] = traces @ env
    return mf_out, perms


def _permute_demod_per_qubit(demux_data, perms) -> dict:
    """Build per-qubit flat demod arrays in the same trace order as the
    scrambled MF features. demux_data[q][s, perms[q, s, k]] becomes the
    k-th trace of state s for qubit q.
    """
    out = {}
    N_per_state = demux_data[0].shape[1]
    T_samples   = demux_data[0].shape[2]
    for q in range(NUM_QUBITS):
        flat = np.empty(
            (NUM_CLASSES * N_per_state, T_samples, 2),
            dtype=demux_data[q].dtype,
        )
        for s in range(NUM_CLASSES):
            flat[s * N_per_state:(s + 1) * N_per_state] = demux_data[q][s, perms[q, s]]
        out[q] = flat
    return out


def _find_herqules_checkpoint() -> str:
    for name in (
        f"HERQULES_len{TRACE_LENGTH}.pth",
        f"HERQULES_best_len{TRACE_LENGTH}.pth",
    ):
        p = os.path.join(SAVE_DIR, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No HERQULES_len{TRACE_LENGTH} checkpoint found under {SAVE_DIR}"
    )


# ============================================================================
# Feature analysis
# ============================================================================

def per_qubit_marginal(logits: torch.Tensor) -> np.ndarray:
    """For 32-class logits, return P(qubit_q = 1) per trace.

    Shape: (N, NUM_QUBITS).
    """
    probs = torch.softmax(logits, dim=-1)
    p_q1 = torch.zeros(probs.shape[0], NUM_QUBITS, dtype=probs.dtype)
    states = torch.arange(NUM_CLASSES)
    for q in range(NUM_QUBITS):
        mask = ((states >> q) & 1).bool()
        p_q1[:, q] = probs[:, mask].sum(dim=-1)
    return p_q1.cpu().numpy()


def half_trace_mf_signs(demod_q: np.ndarray, mf_env_q: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Compute the MF score on the first vs second half of each trace.

    A sign mismatch is a structural indicator of mid-readout transition
    (T1 decay or rare jumps).
    """
    N = demod_q.shape[0]
    F = 2 * TRACE_LENGTH
    flat = demod_q[:, :TRACE_LENGTH, :].reshape(N, F).astype(np.float32, copy=False)
    env  = mf_env_q[:F].astype(np.float32, copy=False)
    half = F // 2
    mf_first  = flat[:, :half] @ env[:half]
    mf_second = flat[:, half:] @ env[half:]
    return mf_first, mf_second


def amplitude_per_trace(demod_q: np.ndarray) -> np.ndarray:
    """Mean |IQ| amplitude per trace within the demodulated band of qubit q."""
    return np.linalg.norm(demod_q, axis=-1).mean(axis=-1).astype(np.float32)


# ============================================================================
# Categorisation
# ============================================================================

def categorize_errors_for_qubit(
    q: int,
    mf_feat: np.ndarray, rmf_feat: np.ndarray,
    demod_test: dict, mf_envs: list[np.ndarray],
    y_test: np.ndarray, p_q1: np.ndarray,
    amp_train_stats: dict[int, tuple[float, float]],
):
    """Return a dict of arrays summarising errors for one qubit."""
    true_q = ((y_test.astype(np.int64) >> q) & 1)
    pred_q = (p_q1[:, q] > 0.5).astype(np.int64)
    is_correct = (pred_q == true_q)
    is_error   = ~is_correct
    error_idx  = np.where(is_error)[0]

    confidence_correct = np.where(true_q == 1, p_q1[:, q], 1.0 - p_q1[:, q])
    near_boundary_mask = confidence_correct < NEAR_BOUNDARY_CONF_THRESHOLD

    amp = amplitude_per_trace(demod_test[q])
    amp_mean, amp_std = amp_train_stats[q]
    amp_z = (amp - amp_mean) / (amp_std + 1e-12)
    amp_outlier_mask = np.abs(amp_z) > AMPLITUDE_OUTLIER_Z

    mf_first, mf_second = half_trace_mf_signs(demod_test[q], mf_envs[q])
    # Only count as t1_pattern if both halves are sufficiently above noise so
    # the sign comparison is meaningful; otherwise the sign of noise is junk.
    mf_first_med  = np.median(np.abs(mf_first))
    mf_second_med = np.median(np.abs(mf_second))
    significant = (np.abs(mf_first) > 0.5 * mf_first_med) & \
                  (np.abs(mf_second) > 0.5 * mf_second_med)
    t1_pattern_mask = significant & (np.sign(mf_first) != np.sign(mf_second))

    # Exclusive categorisation: T1 > amp_outlier > near_boundary > other
    category = np.full(error_idx.shape, "other", dtype=object)
    cat_t1   = t1_pattern_mask[error_idx]
    cat_amp  = amp_outlier_mask[error_idx] & ~cat_t1
    cat_near = near_boundary_mask[error_idx] & ~cat_t1 & ~cat_amp
    category[cat_t1]   = "t1_pattern"
    category[cat_amp]  = "amp_outlier"
    category[cat_near] = "near_boundary"

    return {
        "true_q":             true_q,
        "pred_q":             pred_q,
        "is_correct":         is_correct,
        "is_error":           is_error,
        "error_idx":          error_idx,
        "confidence_correct": confidence_correct,
        "near_boundary_mask": near_boundary_mask,
        "amp":                amp,
        "amp_z":              amp_z,
        "amp_outlier_mask":   amp_outlier_mask,
        "mf_first":           mf_first,
        "mf_second":          mf_second,
        "t1_pattern_mask":    t1_pattern_mask,
        "category":           category,
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_qubit(q: int, mf_feat: np.ndarray, rmf_feat: np.ndarray,
               demod_test: dict, mf_envs: list[np.ndarray],
               analysis: dict, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"HERQULES error modes — qubit {q}  "
        f"(N_test={analysis['is_correct'].size:,}, "
        f"acc={100.0 * analysis['is_correct'].mean():.2f}%, "
        f"errors={int(analysis['is_error'].sum())})",
        fontsize=13,
    )

    is_correct = analysis["is_correct"]
    error_idx  = analysis["error_idx"]
    category   = analysis["category"]
    true_q     = analysis["true_q"]
    pred_q     = analysis["pred_q"]

    mf_q  = mf_feat[:, q]
    rmf_q = rmf_feat[:, q]

    # (0, 0) Histogram of MF_q for correct, FN (true=1 pred=0), FP (true=0 pred=1)
    ax = axes[0, 0]
    mf_correct = mf_q[is_correct]
    mf_fn = mf_q[(true_q == 1) & (pred_q == 0)]
    mf_fp = mf_q[(true_q == 0) & (pred_q == 1)]
    bins = np.linspace(mf_q.min(), mf_q.max(), 80)
    ax.hist(mf_correct, bins=bins, alpha=0.4, density=True, label=f"correct (n={mf_correct.size:,})")
    ax.hist(mf_fn,      bins=bins, alpha=0.7, density=True, label=f"FN, true=1 (n={mf_fn.size})", color="C3")
    ax.hist(mf_fp,      bins=bins, alpha=0.7, density=True, label=f"FP, true=0 (n={mf_fp.size})", color="C1")
    ax.set_xlabel(f"MF_{q}")
    ax.set_ylabel("density")
    ax.set_title("MF_q distribution: correct vs FN vs FP")
    ax.legend(fontsize=8)

    # (0, 1) Scatter MF vs RMF, errors colored by category
    ax = axes[0, 1]
    sub_correct = np.random.RandomState(0).choice(
        np.where(is_correct)[0], size=min(5000, int(is_correct.sum())), replace=False
    )
    ax.scatter(mf_q[sub_correct], rmf_q[sub_correct], s=2, alpha=0.15,
               label="correct (subsample)", color="gray")
    cat_colors = {
        "t1_pattern":    ("C3", "T1-pattern"),
        "amp_outlier":   ("C2", "amplitude outlier"),
        "near_boundary": ("C0", "near boundary"),
        "other":         ("C4", "other"),
    }
    for cat, (color, lbl) in cat_colors.items():
        idx = error_idx[category == cat]
        if idx.size:
            ax.scatter(mf_q[idx], rmf_q[idx], s=10, color=color,
                       label=f"{lbl} (n={idx.size})", alpha=0.7)
    ax.set_xlabel(f"MF_{q}")
    ax.set_ylabel(f"RMF_{q}")
    ax.set_title("MF vs RMF (errors highlighted by category)")
    ax.legend(fontsize=8)

    # (1, 0) Amplitude histogram: correct vs errors split by category
    ax = axes[1, 0]
    amp = analysis["amp"]
    bins = np.linspace(amp.min(), amp.max(), 80)
    ax.hist(amp[is_correct], bins=bins, alpha=0.4, density=True, label="correct")
    for cat, (color, lbl) in cat_colors.items():
        idx = error_idx[category == cat]
        if idx.size:
            ax.hist(amp[idx], bins=bins, alpha=0.7, density=True,
                    color=color, label=f"{lbl} (n={idx.size})", histtype="step", linewidth=1.5)
    ax.set_xlabel(f"mean |IQ| amplitude (qubit {q})")
    ax.set_ylabel("density")
    ax.set_title("amplitude distribution: leakage shows up as a tail")
    ax.legend(fontsize=8)

    # (1, 1) Sample misclassified traces (one per category if available)
    ax = axes[1, 1]
    F = 2 * TRACE_LENGTH
    mf_env_q = mf_envs[q][:F].astype(np.float32, copy=False)
    plotted = []
    for cat, (color, lbl) in cat_colors.items():
        idx = error_idx[category == cat]
        if idx.size == 0:
            continue
        # pick one with highest |MF - boundary| confidence flip for visibility
        pick = idx[0]
        # Plot the I channel of the demodulated trace
        ax.plot(demod_test[q][pick, :, 0], color=color, alpha=0.8,
                label=f"{lbl}: idx={pick}")
        plotted.append(pick)
    # Overlay MF envelope (I component only), rescaled for visibility
    env_I = mf_env_q[0::2]
    if plotted:
        ymax = np.max([np.abs(demod_test[q][p, :, 0]).max() for p in plotted])
        env_scaled = env_I * (ymax / (np.abs(env_I).max() + 1e-12))
        ax.plot(env_scaled, color="black", linestyle=":", linewidth=1.2,
                label="MF envelope (I, scaled)")
    ax.set_xlabel("ADC sample")
    ax.set_ylabel("demod I (a.u.)")
    ax.set_title("example misclassified traces (one per category)")
    ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  saved {out_path}")


# ============================================================================
# Main
# ============================================================================

def run(scrambled: bool = False):
    mode_str = (
        "SCRAMBLED (matches HERQULES_original.test(): per-qubit MFs shuffled "
        "across shots within each state)"
        if scrambled
        else "PER-TRACE (deployment-realistic: each row's MFs come from one shot)"
    )
    logger.info(f"=== HERQULES error-mode analysis (length={TRACE_LENGTH}) ===")
    logger.info(f"Device: {DEVICE}  |  mode={mode_str}")

    # 1. Data prep — use HERQULES_original's own function. This loads the
    #    cached demodulated all_data, calls get_train_val_and_test_set per
    #    qubit, and reshapes each split to per-state (32, N_per_class, T, 2).
    logger.info("mf_demux_data_prep ...")
    data_train, data_val, data_test = mf_demux_data_prep(trace_length=TRACE_LENGTH)
    n_per_state = data_test[0].shape[1]
    n_total     = NUM_CLASSES * n_per_state
    logger.info(
        f"  data_train[q].shape = {data_train[0].shape}, "
        f"data_test[q].shape = {data_test[0].shape}  "
        f"(N_total_test = {n_total:,})"
    )

    # 2. MF envelopes — fit on data_train, exactly like the post-fix test()
    #    branch in HERQULES_original.py.
    best_bc = [1, 1, 9, 2, 9]
    logger.info(f"search_matched_filter_for_all_qubits_demux(data_train, best_bc={best_bc}) ...")
    mf_envs, _ = search_matched_filter_for_all_qubits_demux(data_train, best_bc=best_bc)
    mf_envs = [np.asarray(e) for e in mf_envs]

    # 3. RMF — load the saved state from training. Apply per-trace via
    #    rmf.predict(data_type=2) and reshape to (N_total, 5).
    logger.info("relaxation_mf_classifier.load_state('rmf.pkl') + predict(test) ...")
    rmf = relaxation_mf_classifier()
    rmf.load_state('rmf.pkl')
    test_rmf_data = rmf.predict(data_type=2, trace_length=TRACE_LENGTH)  # (32, N_per_state, 5)
    rmf_feat = test_rmf_data.reshape(n_total, NUM_QUBITS).astype(np.float32, copy=False)

    # 4. MF — either per-trace (deployment-realistic, default) or scrambled
    #    (matches HERQULES_original.test()'s matched_filter_preprocess_demux).
    #    Scrambled tracks per-qubit permutations so we can attribute each MF
    #    score back to the physical trace that produced it for the per-qubit
    #    error categorisation.
    if scrambled:
        logger.info("Building MF features via scrambled per-(qubit, state) permutation ...")
        mf_feat, mf_perms = _scrambled_mf(data_test, mf_envs)
        demod_test = _permute_demod_per_qubit(data_test, mf_perms)
    else:
        mf_feat = _per_trace_mf(data_test, mf_envs)
        demod_test = _demux_to_flat_per_qubit(data_test)

    feat_test = np.concatenate([mf_feat, rmf_feat], axis=1)  # (N_total, 10)

    # 5. Labels — N_per_state copies of state index 0..31 in order
    y_test = np.repeat(np.arange(NUM_CLASSES), n_per_state).astype(np.int64)

    # 6. Per-qubit demodulated training data (flat) for amp stats.
    demod_train = _demux_to_flat_per_qubit(data_train)

    # Per-qubit amplitude stats from training (z-score for test amplitudes)
    amp_train_stats: dict[int, tuple[float, float]] = {}
    for q in range(NUM_QUBITS):
        a = amplitude_per_trace(demod_train[q])
        amp_train_stats[q] = (float(a.mean()), float(a.std()))
    logger.info(f"Per-qubit amp_train_stats: {amp_train_stats}")

    ckpt = _find_herqules_checkpoint()
    logger.info(f"Loading HERQULES checkpoint: {ckpt}")
    model = Net_rmf().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    logger.info("Running Net_rmf inference on test set...")
    with torch.no_grad():
        logits = []
        bs = 4096
        Xt = torch.tensor(feat_test, dtype=torch.float32)
        for s in range(0, Xt.shape[0], bs):
            logits.append(model(Xt[s:s + bs].to(DEVICE)).cpu())
        logits = torch.cat(logits, dim=0)
    p_q1 = per_qubit_marginal(logits)

    pred_class = logits.argmax(dim=-1).numpy()
    overall = 100.0 * np.mean(pred_class == y_test)
    logger.info(f"Joint 32-class test acc: {overall:.2f}%")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    split_tag = "scrambled" if scrambled else "pertrace"
    summary_rows = []
    per_qubit_accs = []  # per-qubit fidelity (%) for the normal report

    for q in range(NUM_QUBITS):
        analysis = categorize_errors_for_qubit(
            q, mf_feat, rmf_feat, demod_test, mf_envs,
            y_test, p_q1, amp_train_stats,
        )
        n_total   = int(analysis["is_correct"].size)
        n_correct = int(analysis["is_correct"].sum())
        n_err     = int(analysis["is_error"].sum())
        cats      = analysis["category"]

        per_qubit_accs.append(100.0 * n_correct / n_total)

        n_t1     = int((cats == "t1_pattern").sum())
        n_amp    = int((cats == "amp_outlier").sum())
        n_near   = int((cats == "near_boundary").sum())
        n_other  = int((cats == "other").sum())

        # Non-exclusive counts (membership in each diagnostic mask, regardless of priority)
        err_idx = analysis["error_idx"]
        n_t1_total   = int(analysis["t1_pattern_mask"][err_idx].sum())
        n_amp_total  = int(analysis["amp_outlier_mask"][err_idx].sum())
        n_near_total = int(analysis["near_boundary_mask"][err_idx].sum())

        logger.info(
            f"qubit {q}  acc={100.0 * n_correct / n_total:.3f}%  "
            f"errors={n_err}  | exclusive: t1={n_t1} amp={n_amp} near={n_near} other={n_other}"
        )

        summary_rows.append({
            "split":                 split_tag,
            "qubit":                q,
            "n_test":                n_total,
            "n_correct":             n_correct,
            "n_error":               n_err,
            "accuracy_pct":          f"{100.0 * n_correct / n_total:.4f}",
            "excl_n_t1_pattern":     n_t1,
            "excl_n_amp_outlier":    n_amp,
            "excl_n_near_boundary":  n_near,
            "excl_n_other":          n_other,
            "any_n_t1_pattern":      n_t1_total,
            "any_n_amp_outlier":     n_amp_total,
            "any_n_near_boundary":   n_near_total,
        })

        plot_path = os.path.join(
            PLOTS_DIR,
            f"herqules_errors_q{q}_len{TRACE_LENGTH}_{split_tag}_{ts}.png",
        )
        plot_qubit(q, mf_feat, rmf_feat, demod_test, mf_envs, analysis, plot_path)

    # CSV summary
    csv_path = os.path.join(
        CSV_DIR,
        f"herqules_error_modes_len{TRACE_LENGTH}_{split_tag}_{ts}.csv",
    )
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    logger.info(f"Summary CSV: {csv_path}")
    logger.info(f"Plots:       {PLOTS_DIR}/herqules_errors_q*_len{TRACE_LENGTH}_{ts}.png")

    # Normal-style optimization report: one wide row with overall + per-qubit
    # fidelities, matching the other trainers' CSVs (sans the hyperparameter
    # columns, which don't apply when evaluating a pre-trained checkpoint).
    fid_row = {
        "timestamp":        ts,
        "model_name":       "HERQULES",
        "trace_length":     TRACE_LENGTH,
        "split":            split_tag,
        "overall_accuracy": f"{overall:.4f}",
        **{f"qubit_{q}_accuracy": f"{per_qubit_accs[q]:.4f}"
           for q in range(NUM_QUBITS)},
        "model_path":       ckpt,
    }
    fid_path = os.path.join(
        CSV_DIR,
        f"HERQULES_len{TRACE_LENGTH}_{split_tag}_{ts}.csv",
    )
    with open(fid_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fid_row.keys()))
        w.writeheader()
        w.writerow(fid_row)
    logger.info(f"Fidelity report: {fid_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    parser.add_argument(
        "--length", type=int, default=TRACE_LENGTH,
        help=(f"Trace length in ADC samples (default {TRACE_LENGTH}). Must match "
              f"the HERQULES_lenL.pth checkpoint AND the current rmf.pkl on disk."),
    )
    parser.add_argument(
        "--scrambled", action="store_true",
        help=("Use HERQULES_original.test()'s exact MF pipeline: per-(qubit, "
              "state) random permutation of trace order before applying the "
              "envelope (= matched_filter_preprocess_demux). Reproduces the "
              "published ~0.98 figures but is deployment-unrealistic. Default "
              "is per-trace (one shot -> one prediction)."),
    )
    args = parser.parse_args()
    TRACE_LENGTH = args.length
    run(scrambled=args.scrambled)
