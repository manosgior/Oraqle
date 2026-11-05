"""Cross-fidelity + accuracy-vs-N for HybridCNN (proposed model).
One inference pass over the full test set -> per-qubit predicted bits -> both metrics.
"""
import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import os, csv, numpy as np, torch as T
from itertools import combinations
from networks.HybridCNN import HybridCNN
import runners.train_hybrid_cnn as TH

T.manual_seed(1); np.random.seed(1)
REPORTS = ensure_dir(REPORTS_OUT)
L, M, DS, FT, FC = 500, 16, 40, 101, 5e6
DEV = T.device("cuda" if T.cuda.is_available() else "cpu")


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
    res = {}
    for N in range(1, 6):
        a = []
        for S in combinations(range(5), N):
            spec = [q for q in range(5) if q not in S]
            mask = np.all(tb[:, spec] == 0, 1) if spec else np.ones(len(tb), bool)
            t, p = tb[mask], pb[mask]
            for i in S:
                m1 = t[:, i] == 1; m0 = t[:, i] == 0
                a.append(1 - ((p[m1, i] == 0).mean() + (p[m0, i] == 1).mean())/2)
        res[N] = float(np.mean(a))
    return res


Xtr, ytr = TH._load_raw(TH.RAW_TRAIN_FILE, "train", L)
Xte, yte = TH._load_raw(TH.RAW_TEST_FILE, "test", L)
Xtr = TH._build_input(Xtr, DS, FT, FC); Xte = TH._build_input(Xte, DS, FT, FC)
mu = Xtr.mean((0, 2), keepdims=True); sd = Xtr.std((0, 2), keepdims=True) + 1e-10
Xtr = (Xtr - mu)/sd; Xte = (Xte - mu)/sd
Ytr = TH._per_qubit_bits(ytr); Yte = TH._per_qubit_bits(yte)
tri, vai = TH._val_split(Xtr.shape[0], 0.1)
model = HybridCNN(n_channels=10, m_param=M, num_qubits=5, n_levels=2).to(DEV)
crit = T.nn.BCEWithLogitsLoss(); opt = T.optim.Adam(model.parameters(), lr=1e-3)
Xt = T.tensor(Xtr[tri], dtype=T.float32); Yt = T.tensor(Ytr[tri], dtype=T.float32)
loader = T.utils.data.DataLoader(T.utils.data.TensorDataset(Xt, Yt), batch_size=512, shuffle=True)
Xv = T.tensor(Xtr[vai], dtype=T.float32).to(DEV); Yv = Ytr[vai]
best, bs, bad = -1, None, 0
for ep in range(40):
    model.train()
    for xb, yb in loader:
        l = crit(model(xb.to(DEV)), yb.to(DEV)); opt.zero_grad(); l.backward(); opt.step()
    model.eval()
    with T.no_grad():
        pv = (T.sigmoid(model(Xv)).cpu().numpy() > 0.5).astype(int)
    va = float((pv == Yv).mean())
    if va >= best: best, bs, bad = va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
    else: bad += 1
    if bad >= 8: break
model.load_state_dict(bs); model.eval()
print(f"[xtalk:HybridCNN] trained (val bit-acc {best:.4f})", flush=True)

P = []
with T.no_grad():
    Xte_t = T.tensor(Xte, dtype=T.float32)
    for s in range(0, len(Xte_t), 4096):
        P.append((T.sigmoid(model(Xte_t[s:s+4096].to(DEV))).cpu().numpy() > 0.5).astype(int))
pb = np.concatenate(P, 0); tb = Yte.astype(int)
cf = xfid(tb, pb); fn = fvsN(tb, pb)
print("[xtalk:HybridCNN] cross-fidelity |F^CF| by dist:", {k: round(v, 4) for k, v in cf.items()}, flush=True)
print("[xtalk:HybridCNN] <F> vs N:", {k: round(v, 4) for k, v in fn.items()}, flush=True)
with open(os.path.join(REPORTS, "xtalk_HybridCNN.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["metric", "k", "value"])
    for k, v in cf.items(): w.writerow(["crossfid_dist", k, f"{v:.5f}"])
    for k, v in fn.items(): w.writerow(["F_vs_N", k, f"{v:.5f}"])
print("[xtalk:HybridCNN] done", flush=True)
