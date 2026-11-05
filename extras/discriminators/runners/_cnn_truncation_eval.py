"""NO-RETRAIN truncation curve for the correct CNN (networks/CNN.py), the exact
analog of the HERQULES truncation eval: train the network ONCE at full length
(500 samples / 1000 ns) with her preprocessing+config, FREEZE it, then evaluate
at shorter readout L by re-preprocessing the test set to the first L samples
(fixed full-window calibration -- the analog of HERQULES 'freeze': MF envelopes
sliced, NN not retrained). GAP over time handles the shorter input length.

Contrast with _cnn_length_sweep.py (per-length RETRAIN). GPU.
"""
import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import os, csv, numpy as np, torch as T
from sklearn.model_selection import train_test_split
from networks.CNN import CNN
import runners._colleague_prep as P

T.manual_seed(1); np.random.seed(1)
REPORTS = ensure_dir(REPORTS_OUT)
TR = (str(RAW_TRAIN), "X_train", "y_train", 15000)
TE = (str(RAW_TEST), "X_test", "y_test", 35000)
DS = 40
DEV = T.device("cuda" if T.cuda.is_available() else "cpu")
LR, DECAY, STEP, BS, EPOCHS = 5e-4, 0.95, 3, 64, 30
FULL = 500
SAMPLES = [100, 125, 200, 250, 300, 375, 400, 500]   # 200..1000 ns
print(f"[cnn-trunc] train-once-freeze @ {FULL} samples; eval {SAMPLES} device={DEV}", flush=True)

prm = P.calibrate(TR[0], TR[1], TR[2], TR[3], (0, FULL))   # fixed calibration

# --- train ONCE at full length, then freeze ---
Xtr_raw, ytr = P.preprocess(TR[0], TR[1], TR[2], prm, DS, (0, FULL))
Xtr = P.to_torch_layout(Xtr_raw); Ytr = P.bits(ytr)
Xtr, Xv, Ytr, Yv = train_test_split(Xtr, Ytr, test_size=0.2, random_state=42)
model = CNN(in_channels=10, m_param=16, num_qubits=5).to(DEV)
crit = T.nn.BCEWithLogitsLoss(); opt = T.optim.Adam(model.parameters(), lr=LR)
sched = T.optim.lr_scheduler.StepLR(opt, step_size=STEP, gamma=DECAY)
loader = T.utils.data.DataLoader(
    T.utils.data.TensorDataset(T.tensor(Xtr), T.tensor(Ytr)), batch_size=BS, shuffle=True)
Xv_t = T.tensor(Xv).to(DEV)
for ep in range(EPOCHS):
    model.train()
    for xb, yb in loader:
        l = crit(model(xb.to(DEV)), yb.to(DEV)); opt.zero_grad(); l.backward(); opt.step()
    sched.step()
model.eval()
with T.no_grad():
    va = float(((T.sigmoid(model(Xv_t)).cpu().numpy() > 0.5).astype(int) == Yv).mean())
print(f"[cnn-trunc] frozen full-length model val-bitacc {va:.4f}", flush=True)


def afid(tb, pb):
    return [1 - ((pb[tb[:, q]==1, q]==0).mean() + (pb[tb[:, q]==0, q]==1).mean())/2 for q in range(5)]


rows = []
for L in SAMPLES:
    Xte_raw, yte = P.preprocess(TE[0], TE[1], TE[2], prm, DS, (0, L))   # fixed calibration, truncated input
    Xte = P.to_torch_layout(Xte_raw); Yte = P.bits(yte).astype(int)
    preds = []
    with T.no_grad():
        Xte_t = T.tensor(Xte)
        for s in range(0, len(Xte_t), 8192):
            preds.append((T.sigmoid(model(Xte_t[s:s+8192].to(DEV))).cpu().numpy() > 0.5).astype(int))
    pb = np.concatenate(preds, 0)
    fq = afid(Yte, pb); g = float(np.prod(fq) ** 0.2)
    print(f"[cnn-trunc] {L*2:4d}ns ({L:3d} smp)  Q1-5 {[round(float(a),3) for a in fq]}  F5Q {g:.4f}", flush=True)
    rows.append(dict(policy="freeze", length_ns=L*2, trace_samples=L,
                     Q1=round(float(fq[0]),4), Q2=round(float(fq[1]),4), Q3=round(float(fq[2]),4),
                     Q4=round(float(fq[3]),4), Q5=round(float(fq[4]),4), F5Q=round(g,4)))
out = os.path.join(REPORTS, "cnn_truncation_noretrain.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"[cnn-trunc] wrote {out}", flush=True)
print("[cnn-trunc] done", flush=True)
