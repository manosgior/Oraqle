"""Controlled deployment comparison: HERQULES (per-trace) vs MCMit-CNN on the
IDENTICAL full test file (DRaw_C_Te, read in order so rows align), per-trace,
trained on full DRaw train, 1000 ns. Same per-qubit assignment-fidelity metric.

HERQULES  : Butterworth demod -> per-trace MF+RMF (build_features) -> Net_rmf (32-class)
MCMit-CNN : her demod+FIR+ds20 preprocessing (_colleague_prep) -> CNN.py (5 heads)
Both deployment-faithful (each input row = one physical shot; no scramble, no subsample).
"""
import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import os, gc, numpy as np, torch as T, torch.nn as nn, h5py
from sklearn.model_selection import train_test_split
from networks.CNN import CNN
from trainers.HERQULES_original import Net_rmf, adjust_learning_rate
from trainers.train_HERQULES import demodulate_all_qubits, compute_all_envelopes, build_features
import runners._colleague_prep as P

L = 500; DEV = T.device("cuda" if T.cuda.is_available() else "cpu")
RAW_TR = str(RAW_TRAIN); RAW_TE = str(RAW_TEST)
REPORTS = ensure_dir(REPORTS_OUT)
print(f"[cmp] device={DEV}", flush=True)


def load(path, key):
    with h5py.File(path, "r") as f:
        return f[f"X_{key}"][:, :L, :].astype(np.float32), f[f"y_{key}"][:].astype(np.int64)


def bits(y): return np.stack([(y >> q) & 1 for q in range(5)], 1).astype(int)


def fid(pb, tb):
    return [float(1 - ((pb[tb[:, q]==1, q]==0).mean() + (pb[tb[:, q]==0, q]==1).mean())/2) for q in range(5)]


Xtr, ytr = load(RAW_TR, "train"); Xte, yte = load(RAW_TE, "test")
tb = bits(yte)
print(f"[cmp] train {len(ytr)} test {len(yte)} (full DRaw, in-order)", flush=True)

# ================= HERQULES (per-trace) =================
dtr = demodulate_all_qubits(Xtr); dte = demodulate_all_qubits(Xte)
mf_e, rmf_e = compute_all_envelopes(dtr, ytr)
Ftr = build_features(dtr, mf_e, rmf_e, L); Fte = build_features(dte, mf_e, rmf_e, L)
del dtr, dte; gc.collect()
mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8; Ftr = (Ftr - mu)/sd; Fte = (Fte - mu)/sd
T.manual_seed(1); np.random.seed(1)
net = Net_rmf().to(DEV); opt = T.optim.Adam(net.parameters(), lr=0.01); crit = nn.CrossEntropyLoss()
tri, vai = train_test_split(np.arange(len(ytr)), test_size=0.15, random_state=1)
ld = T.utils.data.DataLoader(T.utils.data.TensorDataset(T.tensor(Ftr[tri]), T.tensor(ytr[tri])), batch_size=512, shuffle=True)
Xv = T.tensor(Ftr[vai]).to(DEV); yvb = bits(ytr[vai])
best, bs = -1, None
for ep in range(100):
    adjust_learning_rate(0.01, opt, ep); net.train()
    for xb, yb in ld:
        loss = crit(net(xb.to(DEV)), yb.to(DEV)); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with T.no_grad():
        pv = net(Xv).argmax(1).cpu().numpy()
    va = (np.stack([(pv >> q) & 1 for q in range(5)], 1) == yvb).mean()
    if va >= best: best = va; bs = {k: v.detach().clone() for k, v in net.state_dict().items()}
net.load_state_dict(bs); net.eval()
with T.no_grad():
    P_ = [net(T.tensor(Fte[s:s+8192]).to(DEV)).argmax(1).cpu().numpy() for s in range(0, len(Fte), 8192)]
hp = np.concatenate(P_); herq_pb = np.stack([(hp >> q) & 1 for q in range(5)], 1)
herq_fq = fid(herq_pb, tb); herq_g = float(np.prod(herq_fq) ** 0.2)
print(f"[cmp] HERQULES per-trace  F5Q={herq_g:.4f}  Q={[round(a,4) for a in herq_fq]}", flush=True)
del Ftr, Fte; gc.collect()

# ================= MCMit-CNN =================
prm = P.calibrate(RAW_TR, "X_train", "y_train", 15000, (0, L))
Xc_tr, yc_tr = P.preprocess(RAW_TR, "X_train", "y_train", prm, 40, (0, L))
Xc_te, yc_te = P.preprocess(RAW_TE, "X_test", "y_test", prm, 40, (0, L))
Xc_tr = P.to_torch_layout(Xc_tr); Xc_te = P.to_torch_layout(Xc_te)
assert np.array_equal(yc_te, yte), "CNN test labels must match HERQULES test labels (same file/order)"
Ytr = P.bits(yc_tr)
tri, vai = train_test_split(np.arange(len(yc_tr)), test_size=0.1, random_state=1)
T.manual_seed(1); np.random.seed(1)
m = CNN(in_channels=10, m_param=16, num_qubits=5).to(DEV)
crit = nn.BCEWithLogitsLoss(); opt = T.optim.Adam(m.parameters(), lr=5e-4)
sch = T.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.95)
ld = T.utils.data.DataLoader(T.utils.data.TensorDataset(T.tensor(Xc_tr[tri]), T.tensor(Ytr[tri])), batch_size=64, shuffle=True)
Xv = T.tensor(Xc_tr[vai]).to(DEV); Yv = Ytr[vai]
best, bs, bad = -1, None, 0
for ep in range(30):
    m.train()
    for xb, yb in ld:
        loss = crit(m(xb.to(DEV)), yb.to(DEV)); opt.zero_grad(); loss.backward(); opt.step()
    sch.step(); m.eval()
    with T.no_grad():
        va = (((T.sigmoid(m(Xv)).cpu().numpy() > 0.5).astype(int)) == Yv).mean()
    if va >= best: best, bs, bad = va, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, 0
    else: bad += 1
    if bad >= 6: break
m.load_state_dict(bs); m.eval()
with T.no_grad():
    P2 = [(T.sigmoid(m(T.tensor(Xc_te[s:s+8192]).to(DEV))).cpu().numpy() > 0.5).astype(int) for s in range(0, len(Xc_te), 8192)]
cnn_pb = np.concatenate(P2)
cnn_fq = fid(cnn_pb, tb); cnn_g = float(np.prod(cnn_fq) ** 0.2)
print(f"[cmp] MCMit-CNN         F5Q={cnn_g:.4f}  Q={[round(a,4) for a in cnn_fq]}", flush=True)

import csv
with open(os.path.join(REPORTS, "deploy_compare_1000ns.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "Q1", "Q2", "Q3", "Q4", "Q5", "F5Q", "n_test"])
    w.writerow(["HERQULES_pertrace"] + [f"{x:.4f}" for x in herq_fq] + [f"{herq_g:.4f}", len(yte)])
    w.writerow(["MCMit_CNN"] + [f"{x:.4f}" for x in cnn_fq] + [f"{cnn_g:.4f}", len(yte)])
print(f"\n[cmp] SAME {len(yte)} test shots | HERQULES {herq_g:.4f}  vs  CNN {cnn_g:.4f}  (Δ={cnn_g-herq_g:+.4f})", flush=True)
print("[cmp] done", flush=True)
