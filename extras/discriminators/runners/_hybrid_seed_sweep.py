"""Seed sweep of MCMit-CNN (HybridCNN) at 1000 ns / 500 samples.
Mirrors _seed_sweep.py (HERQULES): same data/split every run, only the seed
changes. Reports per-qubit assignment fidelity, F5Q (gmean), and run-to-run std.

Env:
  SWEEP_N     : number of seeds, runs seeds 1..N (default 21)
  FAIL_GMEAN  : gmean below this counts as a catastrophic failure (default 0.85)
"""
import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import os, csv, numpy as np, torch as T
from networks.HybridCNN import HybridCNN
import runners.train_hybrid_cnn as TH

N = int(os.environ.get("SWEEP_N", "21"))
FAIL = float(os.environ.get("FAIL_GMEAN", "0.85"))
REPORTS = ensure_dir(REPORTS_OUT)
L, M, DS, FT, FC = 500, 16, 40, 101, 5e6
DEV = T.device("cuda" if T.cuda.is_available() else "cpu")
print(f"[hsweep] MCMit-CNN length={L*2}ns seeds=1..{N} device={DEV}", flush=True)

# data prep ONCE (deterministic) -- only the training seed varies across runs
Xtr, ytr = TH._load_raw(TH.RAW_TRAIN_FILE, "train", L)
Xte, yte = TH._load_raw(TH.RAW_TEST_FILE, "test", L)
Xtr = TH._build_input(Xtr, DS, FT, FC); Xte = TH._build_input(Xte, DS, FT, FC)
mu = Xtr.mean((0, 2), keepdims=True); sd = Xtr.std((0, 2), keepdims=True) + 1e-10
Xtr = (Xtr - mu)/sd; Xte = (Xte - mu)/sd
Ytr = TH._per_qubit_bits(ytr); Yte = TH._per_qubit_bits(yte)


def per_qubit_fid(tb, pb):
    """Assignment fidelity per qubit: 1 - (P(pred0|true1)+P(pred1|true0))/2."""
    f = []
    for q in range(5):
        m1 = tb[:, q] == 1; m0 = tb[:, q] == 0
        fnr = (pb[m1, q] == 0).mean(); fpr = (pb[m0, q] == 1).mean()
        f.append(1 - (fnr + fpr)/2)
    return [float(x) for x in f]


def run_one(seed):
    T.manual_seed(seed); np.random.seed(seed)
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
    P = []
    with T.no_grad():
        Xte_t = T.tensor(Xte, dtype=T.float32)
        for s in range(0, len(Xte_t), 4096):
            P.append((T.sigmoid(model(Xte_t[s:s+4096].to(DEV))).cpu().numpy() > 0.5).astype(int))
    pb = np.concatenate(P, 0); tb = Yte.astype(int)
    return per_qubit_fid(tb, pb)


rows = []
for s in range(1, N + 1):
    fq = run_one(s); g = float(np.prod(fq) ** 0.2)
    rows.append(fq + [g])
    flag = "  <-- FAIL" if g < FAIL else ""
    print(f"[hsweep] seed {s:2d}  Q1-5 {[round(a,3) for a in fq]}  F5Q {g:.4f}{flag}", flush=True)

arr = np.array(rows); ok = arr[arr[:, 5] >= FAIL]; n_fail = len(arr) - len(ok)
lab = ["Q1", "Q2", "Q3", "Q4", "Q5", "F5Q"]
print(f"[hsweep] failures: {n_fail}/{len(arr)}", flush=True)
m, sd2 = ok.mean(0), ok.std(0)
print("[hsweep] SUCCESS mean: " + "  ".join(f"{lab[i]}={m[i]:.4f}" for i in range(6)), flush=True)
print("[hsweep] SUCCESS std : " + "  ".join(f"{lab[i]}={sd2[i]:.4f}" for i in range(6)), flush=True)
with open(os.path.join(REPORTS, "hybrid_seed_sweep_len500.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["seed"] + lab)
    for i, r in enumerate(rows): w.writerow([i+1] + [f"{x:.4f}" for x in r])
    w.writerow(["mean_success"] + [f"{m[i]:.4f}" for i in range(6)])
    w.writerow(["std_success"] + [f"{sd2[i]:.4f}" for i in range(6)])
print("[hsweep] done", flush=True)
