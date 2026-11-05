"""Strategy 3 on CLEAN HERQULES (32-class softmax head, no scramble): does a
TIME-RESOLVED matched-filter feature help vs the single full-window MF?

Same demod + envelopes + data + seeds; only the FEATURES / first layer differ.
  clean10   : [MF x5 | RMF x5] (N,10), published Net_rmf trunk 10->10->20->32
  clean_wide: same 10 features, wide trunk 10->64->64->32   (capacity control)
  mwK_wide  : [MF over K time-windows x5 | RMF x5] (N,5K+5), wide trunk ->64->64->32
The K window-MFs are disjoint slices that SUM to the full MF -> strictly more info.
Isolates: clean10 vs clean_wide = capacity; clean_wide vs mwK_wide = features.
1000 ns (500 samples). Reports per-qubit F + F5Q geomean, mean/std over seeds.
"""
import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import os, csv, numpy as np, torch as T, torch.nn as nn, h5py
from trainers.train_HERQULES import demodulate_all_qubits, compute_all_envelopes, build_features

REPORTS = ensure_dir(REPORTS_OUT); DEV = T.device("cuda" if T.cuda.is_available() else "cpu")
L = 500; NSEED = int(os.environ.get("NSEED", "5")); KS = [5, 10]
N_TR, N_TE = 80000, 60000
RAW_TR = str(RAW_TRAIN); RAW_TE = str(RAW_TEST)
print(f"[mw] device={DEV} seeds=1..{NSEED} K={KS}", flush=True)


def load(path, key, n):
    with h5py.File(path, "r") as f:
        tot = f[f"X_{key}"].shape[0]; idx = np.sort(np.random.RandomState(0).choice(tot, n, replace=False))
        return f[f"X_{key}"][idx, :L, :].astype(np.float32), f[f"y_{key}"][idx].astype(np.int64)


def build_mw(demod, mf_e, rmf_e, K):
    N = demod[0].shape[0]; S = L // K
    mf = np.zeros((N, 5, K), np.float32); rmf = np.zeros((N, 5), np.float32)
    for q in range(5):
        for k in range(K):
            a, b = k*S, (k+1)*S
            mf[:, q, k] = demod[q][:, a:b, :].reshape(N, 2*(b-a)) @ mf_e[q][2*a:2*b].astype(np.float32)
        rmf[:, q] = demod[q][:, :L, :].reshape(N, 2*L) @ rmf_e[q][:2*L].astype(np.float32)
    return np.concatenate([mf.reshape(N, 5*K), rmf], 1)


class NetOrig(nn.Module):                     # published Net_rmf
    def __init__(s):
        super().__init__(); s.h1 = nn.Linear(10, 10); s.h2 = nn.Linear(10, 20); s.o = nn.Linear(20, 32); s.r = nn.ReLU()
    def forward(s, x): return s.o(s.r(s.h2(s.r(s.h1(x)))))


class NetWide(nn.Module):
    def __init__(s, d):
        super().__init__(); s.h1 = nn.Linear(d, 64); s.h2 = nn.Linear(64, 64); s.o = nn.Linear(64, 32); s.r = nn.ReLU()
    def forward(s, x): return s.o(s.r(s.h2(s.r(s.h1(x)))))


def bits(y): return np.stack([(y >> q) & 1 for q in range(5)], 1)


def zfit(a): return a.mean(0), a.std(0) + 1e-8


def train_eval(Xtr, ytr, Xte, yte, make_net, seed):
    T.manual_seed(seed); np.random.seed(seed)
    mu, sd = zfit(Xtr); Xtr = (Xtr-mu)/sd; Xte = (Xte-mu)/sd
    ntr = int(len(Xtr)*0.85); perm = np.random.RandomState(seed).permutation(len(Xtr))
    tri, vai = perm[:ntr], perm[ntr:]
    net = make_net().to(DEV); opt = T.optim.Adam(net.parameters(), lr=0.01); crit = nn.CrossEntropyLoss()
    xt = T.tensor(Xtr[tri]); yt = T.tensor(ytr[tri])
    ld = T.utils.data.DataLoader(T.utils.data.TensorDataset(xt, yt), batch_size=512, shuffle=True)
    xv = T.tensor(Xtr[vai]).to(DEV); yvb = bits(ytr[vai])
    best, bs = -1, None
    for ep in range(100):
        for g in opt.param_groups: g['lr'] = 0.01 * (0.1 ** sum(ep >= m for m in (30, 60, 90)))
        net.train()
        for xb, yb in ld:
            loss = crit(net(xb.to(DEV)), yb.to(DEV)); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with T.no_grad():
            pv = net(xv).argmax(1).cpu().numpy()
        pb = np.stack([(pv >> q) & 1 for q in range(5)], 1); va = (pb == yvb).mean()
        if va >= best: best = va; bs = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(bs); net.eval()
    with T.no_grad():
        pt = net(T.tensor(Xte).to(DEV)).argmax(1).cpu().numpy()
    pb = np.stack([(pt >> q) & 1 for q in range(5)], 1); tb = bits(yte)
    fq = [float(1-((pb[tb[:, q]==1, q]==0).mean()+(pb[tb[:, q]==0, q]==1).mean())/2) for q in range(5)]
    return fq, float(np.prod(fq) ** 0.2)


# ---- build everything once ----
Xtr_raw, ytr = load(RAW_TR, "train", N_TR); Xte_raw, yte = load(RAW_TE, "test", N_TE)
dtr = demodulate_all_qubits(Xtr_raw); dte = demodulate_all_qubits(Xte_raw)
mf_e, rmf_e = compute_all_envelopes(dtr, ytr)
feats = {"clean": (build_features(dtr, mf_e, rmf_e, L), build_features(dte, mf_e, rmf_e, L))}
for K in KS: feats[f"mw{K}"] = (build_mw(dtr, mf_e, rmf_e, K), build_mw(dte, mf_e, rmf_e, K))
print("[mw] features built:", {k: v[0].shape[1] for k, v in feats.items()}, flush=True)

configs = [("clean10", "clean", lambda: NetOrig()),
           ("clean_wide", "clean", lambda: NetWide(10))]
for K in KS: configs.append((f"mw{K}_wide", f"mw{K}", (lambda d: (lambda: NetWide(d)))(5*K+5)))

res = {}
for name, fk, mk in configs:
    Xtr, Xte = feats[fk]; gs = []
    for s in range(1, NSEED+1):
        fq, g = train_eval(Xtr.copy(), ytr, Xte.copy(), yte, mk, s)
        gs.append(fq+[g]); print(f"[mw] {name:11s} seed {s} F5Q {g:.4f} Q {[round(a,3) for a in fq]}", flush=True)
    res[name] = np.array(gs)

lab = ["Q1", "Q2", "Q3", "Q4", "Q5", "F5Q"]; out = []
print("\n[mw] === SUMMARY (mean over seeds) ===")
for name, _, _ in configs:
    m, sd = res[name].mean(0), res[name].std(0)
    print(f"[mw] {name:11s} " + "  ".join(f"{lab[i]}={m[i]:.4f}±{sd[i]:.4f}" for i in range(6)), flush=True)
    out.append([name] + [f"{m[i]:.4f}" for i in range(6)] + [f"{sd[i]:.4f}" for i in range(6)])
with open(os.path.join(REPORTS, "herqules_mw_features.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["config"]+[f"{x}_mean" for x in lab]+[f"{x}_std" for x in lab]); w.writerows(out)
print("[mw] done", flush=True)
