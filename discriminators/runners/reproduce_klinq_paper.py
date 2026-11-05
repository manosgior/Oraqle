"""
reproduce_klinq_paper.py
========================
Paper-exact reproduction of the KLiNQ evaluation (Guo et al., 2025,
arXiv:2503.03544) — NO Optuna / hyper-optimization.

Everything is fixed to the published recipe:

  Teacher  : SingleQubitFNN  [1000 -> 500 -> 250 -> 1]  (per-qubit binary)
             trained on the full 1 us trace (500 samples -> 1000 flat IQ).
             One teacher per qubit, reused across all student durations
             (the teacher is an OFFLINE distillation signal; only the student
             is constrained to the short readout duration).

  Student  : KLiNQStudentModel  [input -> 16 -> 8 -> 1]
               * FNN-A  (paper Q1, Q4, Q5) : 15 avg bins -> 2*15 + 1 MF = 31
               * FNN-B  (paper Q2, Q3)     : 100 avg bins -> 2*100 + 1 MF = 201
             Input = [time-averaged I/Q | matched-filter scalar]  (paper Fig. 2).
             The number of samples per bin is adjusted per duration so the
             bin count (and hence the input size) stays fixed.

  Loss     : L = alpha_ce * BCE(student, y) + soft * [ MSE(student/T, teacher/T) * T^2 ]
             Distillation setting "T4-ST2-CE8": T=4, soft=0.2, alpha_ce=0.8.

  Training : teacher  -> 300 epochs, Adam(lr=1e-4), batch 1024, BCEWithLogits
             student  ->  50 epochs, Adam(lr=1e-3), batch 1024
             (both with ReduceLROnPlateau + early stopping, as in the repo trainer)

  Metric   : per-qubit assignment fidelity + geometric-mean F_5Q (paper Eq. F_GM).

  Durations: 200, 400, 600, 800, 1000 ns  ==  100, 200, 300, 400, 500 samples.

Qubit indexing: bit position 0..4 in the packed label maps to paper Q1..Q5
(bit+1). FNN-A bins for paper {Q1,Q4,Q5} = bits {0,3,4}; FNN-B bins for
paper {Q2,Q3} = bits {1,2}.

Data: reuses the combined 5-qubit HDF5 files (same as hyper_optimize.py) and
extracts per-qubit binary labels by bit-shift. The matched-filter envelope is
computed from the training split using the paper formula
    MF = mean(T0 - T1) / var(T0 - T1).
"""

import os
import csv
from datetime import datetime

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from loguru import logger
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from disc_paths import RAW_TRAIN, RAW_TEST, CNN_DATA, MODELS_OUT, REPORTS_OUT, require_dataset  # [unify] replaces the old docker mount paths

from networks import SingleQubitFNN, KLiNQStudentModel

# ============================================================================
# Fixed configuration (paper recipe — do not "optimize")
# ============================================================================

RAW_TRAIN_FILE = str(RAW_TRAIN)
RAW_TEST_FILE = str(RAW_TEST)

NUM_QUBITS = 5
FULL_SAMPLES = 500                       # 1 us trace -> 500 samples (teacher input = 1000)

# Readout durations to evaluate. (ns, samples) with 2 ns / sample.
DURATIONS = [(200, 100), (400, 200), (600, 300), (800, 400), (1000, 500)]

# Student architecture per qubit-group (paper Fig. 2 / Sec. III-D).
# Keyed by bit index (0..4).  FNN-A -> 15 bins (Q1,Q4,Q5); FNN-B -> 100 bins (Q2,Q3).
FNN_A_BITS = {0, 3, 4}                    # paper Q1, Q4, Q5
FNN_B_BITS = {1, 2}                       # paper Q2, Q3
BINS_A = 15                               # -> input 2*15 + 1 = 31
BINS_B = 100                              # -> input 2*100 + 1 = 201

# Distillation "T4-ST2-CE8"
TEMPERATURE = 4.0
SOFT_WEIGHT = 0.2                         # weight on the KD (MSE) term
CE_WEIGHT = 0.8                           # weight on the BCE (hard-label) term

# Teacher training (paper / config.py)
TEACHER_EPOCHS = 300
TEACHER_LR = 1e-4
TEACHER_BATCH = 1024
TEACHER_PATIENCE = 15

# Student distillation training (paper / config.py)
STUDENT_EPOCHS = 50
STUDENT_LR = 1e-3
STUDENT_BATCH = 1024
STUDENT_PATIENCE = 5

VAL_FRACTION = 0.1                        # ~ 216k/24k train/val in the paper
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = str(MODELS_OUT)  # [unify] was ./saved_models (= /app/saved_models)
CSV_DIR = str(REPORTS_OUT)  # [unify] was ./optimization_reports
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

EPS = 1e-10


# ============================================================================
# Data helpers
# ============================================================================

def load_full(filepath, is_test):
    """Load the full-length (500-sample) IQ traces and packed labels."""
    key = "test" if is_test else "train"
    with h5py.File(filepath, "r") as hf:
        X = hf[f"X_{key}"][:, :FULL_SAMPLES, :]      # (N, 500, 2)
        y = hf[f"y_{key}"][:]                          # (N,) packed 0..31
    return X.astype(np.float32), y


def qubit_bits(y_packed, bit):
    """Binary label for one qubit (bit position)."""
    return ((y_packed >> bit) & 1).astype(np.float32)


def zscore_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + EPS
    return mu, sd


def average_to_bins(X, n_bins):
    """Average a (N, L, 2) trace into exactly n_bins bins per channel.

    Uses np.array_split so the bin count is fixed regardless of L
    (samples-per-bin adjusts with the duration, as described in the paper).
    Returns (N, 2*n_bins): [I bins | Q bins].
    """
    groups = np.array_split(np.arange(X.shape[1]), n_bins)
    I = np.stack([X[:, g, 0].mean(axis=1) for g in groups], axis=1)
    Q = np.stack([X[:, g, 1].mean(axis=1) for g in groups], axis=1)
    return np.concatenate([I, Q], axis=1).astype(np.float32)


def mf_envelope(X_flat_train, y_bin_train):
    """Matched-filter envelope: mean(T0 - T1) / var(T0 - T1)  (paper Sec. III-B2)."""
    gnd = X_flat_train[y_bin_train == 0]
    ext = X_flat_train[y_bin_train == 1]
    n = min(len(gnd), len(ext))
    diff = gnd[:n] - ext[:n]
    return (np.mean(diff, axis=0) / (np.var(diff, axis=0) + EPS)).astype(np.float32)


# ============================================================================
# Training / evaluation
# ============================================================================

def _loader(tensors, batch, shuffle):
    return DataLoader(TensorDataset(*tensors), batch_size=batch, shuffle=shuffle)


def binary_accuracy(logits, y):
    pred = (torch.sigmoid(logits) >= 0.5).float()
    return (pred == y).float().mean().item() * 100.0


def train_teacher(X_train, y_train, X_val, y_val):
    """Train one SingleQubitFNN [1000,500,250,1] teacher on the full 1 us trace."""
    model = SingleQubitFNN(input_size=2 * FULL_SAMPLES, output_size=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=TEACHER_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.75,
                                                     patience=TEACHER_PATIENCE, cooldown=2)
    tr = _loader((torch.tensor(X_train), torch.tensor(y_train)), TEACHER_BATCH, True)
    va = _loader((torch.tensor(X_val), torch.tensor(y_val)), TEACHER_BATCH, False)

    best_val, best_state, counter = None, None, 0
    for epoch in range(TEACHER_EPOCHS):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
                vloss += criterion(model(xb), yb).item() * xb.size(0)
        vloss /= len(va.dataset)
        scheduler.step(vloss)

        if best_val is None or vloss < best_val - 1e-4:
            best_val, best_state, counter = vloss, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            counter += 1
            if counter >= TEACHER_PATIENCE:
                logger.info(f"  teacher early stop @ epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def distill_student(student, teacher, X_stu_tr, X_tea_tr, y_tr,
                    X_stu_va, X_tea_va, y_va):
    """Knowledge distillation: teacher sees full trace, student sees [avg|MF]."""
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=STUDENT_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.75,
                                                     patience=STUDENT_PATIENCE, cooldown=2)

    tr = _loader((torch.tensor(X_stu_tr), torch.tensor(X_tea_tr), torch.tensor(y_tr)),
                 STUDENT_BATCH, True)
    va = _loader((torch.tensor(X_stu_va), torch.tensor(y_va)), STUDENT_BATCH, False)

    best_val, best_state, counter = None, None, 0
    for epoch in range(STUDENT_EPOCHS):
        student.train()
        for xs, xt, yb in tr:
            xs, xt, yb = xs.to(DEVICE), xt.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            with torch.no_grad():
                # Soft probabilities from the teacher at the chosen temperature.
                # Temperature softens sigmoid (closer to 0.5 for larger T) so
                # the teacher's relative confidence is what the student tries
                # to match, not just the {0,1} hard call. T^2 scaling
                # preserves gradient magnitude when T != 1.
                p_t = torch.sigmoid(teacher(xt) / TEMPERATURE)
            s_logits = student(xs)
            p_s = torch.sigmoid(s_logits / TEMPERATURE)
            kd = mse(p_s, p_t) * (TEMPERATURE ** 2)
            hard = bce(s_logits, yb)
            loss = SOFT_WEIGHT * kd + CE_WEIGHT * hard
            loss.backward()
            optimizer.step()

        student.eval()
        vloss = 0.0
        with torch.no_grad():
            for xs, yb in va:
                xs, yb = xs.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
                vloss += bce(student(xs), yb).item() * xs.size(0)
        vloss /= len(va.dataset)
        scheduler.step(vloss)

        if best_val is None or vloss < best_val - 1e-4:
            best_val, best_state, counter = vloss, {k: v.detach().clone() for k, v in student.state_dict().items()}, 0
        else:
            counter += 1
            if counter >= STUDENT_PATIENCE:
                break

    student.load_state_dict(best_state)
    student.eval()
    return student


@torch.no_grad()
def test_fidelity(model, X_test, y_test):
    model.eval()
    loader = _loader((torch.tensor(X_test),), 4096, False)
    preds = []
    for (xb,) in loader:
        preds.append((torch.sigmoid(model(xb.to(DEVICE))) >= 0.5).float().cpu())
    pred = torch.cat(preds).squeeze().numpy()
    return float(np.mean(pred == y_test))   # fidelity in [0,1]


# ============================================================================
# Main
# ============================================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    logger.info(f"Device: {DEVICE}")

    logger.info("Loading full-length (500-sample) data ...")
    X_train_full, y_train_packed = load_full(RAW_TRAIN_FILE, is_test=False)
    X_test_full, y_test_packed = load_full(RAW_TEST_FILE, is_test=True)
    logger.info(f"  train {X_train_full.shape}, test {X_test_full.shape}")

    # Flattened full trace for the teacher (z-scored with train stats).
    Xtr_flat = X_train_full.reshape(len(X_train_full), -1)
    Xte_flat = X_test_full.reshape(len(X_test_full), -1)
    tea_mu, tea_sd = zscore_fit(Xtr_flat)
    Xtr_flat_n = (Xtr_flat - tea_mu) / tea_sd
    Xte_flat_n = (Xte_flat - tea_mu) / tea_sd

    # results[duration_ns][bit] = student_fidelity
    rows = []
    geomean = {ns: [] for ns, _ in DURATIONS}
    # Param counts are fixed per qubit (architecture does not depend on
    # duration), so one entry per bit suffices for the F5Q summary.
    student_params_by_bit = {}
    teacher_params_by_bit = {}

    for bit in range(NUM_QUBITS):
        paper_q = bit + 1
        n_bins = BINS_A if bit in FNN_A_BITS else BINS_B
        student_input = 2 * n_bins + 1
        y_tr_bit = qubit_bits(y_train_packed, bit)
        y_te_bit = qubit_bits(y_test_packed, bit)

        # ---- Teacher (trained ONCE per qubit on the full 1 us trace) ----
        logger.info(f"=== Qubit bit {bit} (paper Q{paper_q}) : training teacher [1000,500,250,1] ===")
        Xtr_t, Xva_t, ytr_t, yva_t = train_test_split(
            Xtr_flat_n, y_tr_bit, test_size=VAL_FRACTION, random_state=SEED, stratify=y_tr_bit)
        teacher = train_teacher(Xtr_t, ytr_t, Xva_t, yva_t)
        teacher_acc = test_fidelity(teacher, Xte_flat_n, y_te_bit)
        teacher_params = sum(p.numel() for p in teacher.parameters())
        teacher_params_by_bit[bit] = teacher_params
        logger.info(f"  teacher test fidelity = {teacher_acc:.4f}  ({teacher_params} params)")

        # ---- Student distillation per duration ----
        for ns, L in DURATIONS:
            # Student features from the trace TRUNCATED to L samples.
            Xtr_L = X_train_full[:, :L, :]
            Xte_L = X_test_full[:, :L, :]
            Xtr_L_flat = Xtr_L.reshape(len(Xtr_L), -1)
            Xte_L_flat = Xte_L.reshape(len(Xte_L), -1)

            # Averaged bins (fixed count) + MF scalar (paper formula, train-fit).
            avg_tr = average_to_bins(Xtr_L, n_bins)
            avg_te = average_to_bins(Xte_L, n_bins)
            env = mf_envelope(Xtr_L_flat, y_tr_bit)
            mf_tr = (Xtr_L_flat @ env).reshape(-1, 1)
            mf_te = (Xte_L_flat @ env).reshape(-1, 1)

            # Z-score each block with train stats, concatenate -> student input.
            a_mu, a_sd = zscore_fit(avg_tr)
            m_mu, m_sd = zscore_fit(mf_tr)
            Xstu_tr = np.concatenate([(avg_tr - a_mu) / a_sd, (mf_tr - m_mu) / m_sd], axis=1).astype(np.float32)
            Xstu_te = np.concatenate([(avg_te - a_mu) / a_sd, (mf_te - m_mu) / m_sd], axis=1).astype(np.float32)
            assert Xstu_tr.shape[1] == student_input, (Xstu_tr.shape, student_input)

            # Aligned train/val split shared by student features and teacher inputs.
            idx = np.arange(len(Xstu_tr))
            tr_idx, va_idx = train_test_split(idx, test_size=VAL_FRACTION,
                                              random_state=SEED, stratify=y_tr_bit)

            student = KLiNQStudentModel(input_size=student_input).to(DEVICE)
            student = distill_student(
                student, teacher,
                Xstu_tr[tr_idx], Xtr_flat_n[tr_idx], y_tr_bit[tr_idx],
                Xstu_tr[va_idx], Xtr_flat_n[va_idx], y_tr_bit[va_idx],
            )

            fid = test_fidelity(student, Xstu_te, y_te_bit)
            student_params = sum(p.numel() for p in student.parameters())
            student_params_by_bit[bit] = student_params
            geomean[ns].append(fid)
            logger.info(f"  Q{paper_q} {ns}ns (L={L}, {n_bins} bins, in={student_input}): "
                        f"student fidelity = {fid:.4f}  ({student_params} params)")

            model_path = os.path.join(SAVE_DIR, f"KLiNQ_student_paper_q{bit}_{ns}ns.pth")
            torch.save(student.state_dict(), model_path)

            rows.append({
                "duration_ns": ns, "trace_samples": L, "qubit_bit": bit, "paper_qubit": f"Q{paper_q}",
                "student_arch": "FNN-A" if bit in FNN_A_BITS else "FNN-B",
                "n_bins": n_bins, "student_input": student_input,
                "student_params": student_params, "teacher_params": teacher_params,
                "temperature": TEMPERATURE, "soft_weight": SOFT_WEIGHT, "ce_weight": CE_WEIGHT,
                "teacher_test_fidelity": f"{teacher_acc:.4f}",
                "student_test_fidelity": f"{fid:.4f}",
                "model_path": model_path,
            })

    # ---- Write per-(duration,qubit) CSV ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_csv = os.path.join(CSV_DIR, f"KLiNQ_paper_repro_{ts}.csv")
    with open(detail_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Per-qubit report: {detail_csv}")

    # ---- Geometric-mean fidelity per duration (paper F_5Q) ----
    summary_csv = os.path.join(CSV_DIR, f"KLiNQ_paper_repro_F5Q_{ts}.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["duration_ns", "trace_samples", "F5Q_geomean"]
            + [f"Q{q+1}_fidelity" for q in range(NUM_QUBITS)]
            + [f"Q{q+1}_student_params" for q in range(NUM_QUBITS)]
            + [f"Q{q+1}_teacher_params" for q in range(NUM_QUBITS)]
            + ["total_student_params"]
        )
        student_params_row = [student_params_by_bit[q] for q in range(NUM_QUBITS)]
        teacher_params_row = [teacher_params_by_bit[q] for q in range(NUM_QUBITS)]
        total_student_params = sum(student_params_row)
        for ns, L in DURATIONS:
            accs = geomean[ns]
            f5q = float(np.prod(accs) ** (1.0 / NUM_QUBITS))
            w.writerow(
                [ns, L, f"{f5q:.4f}"]
                + [f"{a:.4f}" for a in accs]
                + student_params_row
                + teacher_params_row
                + [total_student_params]
            )
            logger.info(f"F5Q @ {ns}ns = {f5q:.4f}")
    logger.info(f"Geometric-mean summary: {summary_csv}")


if __name__ == "__main__":
    require_dataset()  # [unify]
    main()
