"""Cross-fidelity + accuracy-vs-N for the CORRECT CNN (networks/CNN.py, the
faithful 3061-param port of the colleague's reference), using her exact
preprocessing and training config. GPU.
"""
import os, csv, numpy as np, torch as T
from itertools import combinations
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
DS, SLICE = 40, (0, 500)   # 500 samples @ 500 MHz = 1000 ns, matched to HERQULES
DEV = T.device("cuda" if T.cuda.is_available() else "cpu")
LR, DECAY, STEP, BS, EPOCHS = 5e-4, 0.95, 3, 64, 30
print(f"[xtalk:CNN] device={DEV}", flush=True)


def xfid(tb, pb):
    F = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if i == j: continue
            m0 = tb[:, j] == 0; m1 = tb[:, j] == 1
            F[i, j] = 1 - ((pb[m0, i] == 1).mean() + (pb[m1, i] == 0).mean())
    return {d: float(np.mean([abs(F[i, j]) for i in range(5) for j in range(5)
                              if abs(i-j) == d])) for d in (1, 2, 3, 4)}


def fvsN(tb, pb):
    """Geometric mean: within each driven subset, geomean of its per-qubit
    fidelities; arithmetic mean over subsets. N=5 => F5Q exactly."""
    res = {}
    for N in range(1, 6):
        gms = []
        for S in combinations(range(5), N):
            spec = [q for q in range(5) if q not in S]
            mask = np.all(tb[:, spec] == 0, 1) if spec else np.ones(len(tb), bool)
            t, p = tb[mask], pb[mask]
            fqs = []
            for i in S:
                m1 = t[:, i] == 1; m0 = t[:, i] == 0
                fqs.append(1 - ((p[m1, i] == 0).mean() + (p[m0, i] == 1).mean())/2)
            gms.append(float(np.prod(fqs) ** (1.0/len(fqs))))
        res[N] = float(np.mean(gms))
    return res


prm = P.calibrate(TR[0], TR[1], TR[2], TR[3], SLICE)
Xtr_raw, ytr = P.preprocess(TR[0], TR[1], TR[2], prm, DS, SLICE)
Xte_raw, yte = P.preprocess(TE[0], TE[1], TE[2], prm, DS, SLICE)
Xtr = P.to_torch_layout(Xtr_raw); Xte = P.to_torch_layout(Xte_raw)
Ytr = P.bits(ytr); Yte = P.bits(yte)
Xtr, Xv, Ytr, Yv = train_test_split(Xtr, Ytr, test_size=0.2, random_state=42)

model = CNN(in_channels=10, m_param=16, num_qubits=5).to(DEV)
crit = T.nn.BCEWithLogitsLoss()
opt = T.optim.Adam(model.parameters(), lr=LR)
sched = T.optim.lr_scheduler.StepLR(opt, step_size=STEP, gamma=DECAY)
Xt = T.tensor(Xtr); Yt = T.tensor(Ytr)
loader = T.utils.data.DataLoader(T.utils.data.TensorDataset(Xt, Yt), batch_size=BS, shuffle=True)
Xv_t = T.tensor(Xv).to(DEV)
for ep in range(EPOCHS):
    model.train()
    for xb, yb in loader:
        l = crit(model(xb.to(DEV)), yb.to(DEV)); opt.zero_grad(); l.backward(); opt.step()
    sched.step()
    model.eval()
    with T.no_grad():
        pv = (T.sigmoid(model(Xv_t)).cpu().numpy() > 0.5).astype(int)
    va = float((pv == Yv).mean())
    print(f"[xtalk:CNN] epoch {ep+1}/{EPOCHS} val-bitacc {va:.4f}", flush=True)
model.eval()

preds = []
with T.no_grad():
    Xte_t = T.tensor(Xte)
    for s in range(0, len(Xte_t), 8192):
        preds.append((T.sigmoid(model(Xte_t[s:s+8192].to(DEV))).cpu().numpy() > 0.5).astype(int))
pb = np.concatenate(preds, 0); tb = Yte.astype(int)
np.save(os.path.join(REPORTS, "preds_MCMitCNN_tb.npy"), tb.astype(np.int8))
np.save(os.path.join(REPORTS, "preds_MCMitCNN_pb.npy"), pb.astype(np.int8))
gmean = float(np.prod([1 - ((pb[tb[:, q]==1, q]==0).mean() + (pb[tb[:, q]==0, q]==1).mean())/2 for q in range(5)]) ** 0.2)
cf = xfid(tb, pb); fn = fvsN(tb, pb)
print(f"[xtalk:CNN] test F5Q (afid geomean) {gmean:.4f}", flush=True)
print("[xtalk:CNN] cross-fidelity |F^CF| by dist:", {k: round(v, 4) for k, v in cf.items()}, flush=True)
print("[xtalk:CNN] <F> vs N:", {k: round(v, 4) for k, v in fn.items()}, flush=True)
with open(os.path.join(REPORTS, "xtalk_CNN.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["metric", "k", "value"])
    w.writerow(["F5Q_afid", 5, f"{gmean:.5f}"])
    for k, v in cf.items(): w.writerow(["crossfid_dist", k, f"{v:.5f}"])
    for k, v in fn.items(): w.writerow(["F_vs_N", k, f"{v:.5f}"])
print("[xtalk:CNN] done", flush=True)
