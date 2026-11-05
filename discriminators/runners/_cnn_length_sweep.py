"""Readout-length point for the CORRECT CNN (networks/CNN.py, faithful 3061-param
port of the colleague's reference) using her exact preprocessing + training config.
Per-length RETRAIN. GPU. ONE length per process (env CNN_LEN = #samples), so the
driver can fan out all lengths in parallel; results aggregated afterwards.

Readout length in ns matched to HERQULES (500 MHz -> 2 ns/sample):
  200 ns=100, 250=125, 400=200, 500=250, 600=300, 750=375, 800=400, 1000=500.
Frequencies/phase calibrated on a fixed 500-sample window (qubit IF is a fixed
physical property), so short-window FFT resolution does not bias the curve.
"""
import os, csv, numpy as np, torch as T
from sklearn.model_selection import train_test_split
import os as _os, sys as _sys  # [unify] discriminators/ on sys.path (repo layout)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from disc_paths import RAW_TRAIN, RAW_TEST, REPORTS_OUT, ensure_dir, require_dataset  # [unify] replaces the old docker mount paths
from networks.CNN import CNN
import runners._colleague_prep as P

T.manual_seed(1); np.random.seed(1)
REPORTS = ensure_dir(REPORTS_OUT)  # [unify]
require_dataset()  # [unify]
TR = (str(RAW_TRAIN), "X_train", "y_train", 15000)
TE = (str(RAW_TEST), "X_test", "y_test", 35000)
DS = 40
DEV = T.device("cuda" if T.cuda.is_available() else "cpu")
LR, DECAY, STEP, BS, EPOCHS = 5e-4, 0.95, 3, 64, 30
L = int(os.environ["CNN_LEN"])           # number of raw samples for this run
print(f"[cnn-len] L={L} samples ({L*2} ns) device={DEV}", flush=True)

prm = P.calibrate(TR[0], TR[1], TR[2], TR[3], (0, 500))   # fixed calibration window

Xtr_raw, ytr = P.preprocess(TR[0], TR[1], TR[2], prm, DS, (0, L))
Xte_raw, yte = P.preprocess(TE[0], TE[1], TE[2], prm, DS, (0, L))
Xtr = P.to_torch_layout(Xtr_raw); Xte = P.to_torch_layout(Xte_raw)
Ytr = P.bits(ytr); Yte = P.bits(yte)
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
preds = []
with T.no_grad():
    Xte_t = T.tensor(Xte)
    for s in range(0, len(Xte_t), 8192):
        preds.append((T.sigmoid(model(Xte_t[s:s+8192].to(DEV))).cpu().numpy() > 0.5).astype(int))
pb = np.concatenate(preds, 0); tb = Yte.astype(int)
fq = [1 - ((pb[tb[:, q]==1, q]==0).mean() + (pb[tb[:, q]==0, q]==1).mean())/2 for q in range(5)]
g = float(np.prod(fq) ** 0.2)
print(f"[cnn-len] {L*2:4d}ns ({L:3d} smp)  Q1-5 {[round(float(a),3) for a in fq]}  F5Q {g:.4f}", flush=True)
out = os.path.join(REPORTS, f"cnn_length_{L*2}ns.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["length_ns", "trace_samples", "Q1", "Q2", "Q3", "Q4", "Q5", "F5Q"])
    w.writerow([L*2, L] + [f"{float(a):.4f}" for a in fq] + [f"{g:.4f}"])
print(f"[cnn-len] wrote {out}", flush=True)
print("[cnn-len] done", flush=True)
