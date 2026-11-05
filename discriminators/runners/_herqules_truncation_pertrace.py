"""NO-RETRAIN truncation test for DEPLOYMENT-FAITHFUL (per-trace) HERQULES, full
data. Train Net_rmf ONCE at full 1000 ns (500 samples), FREEZE it, then evaluate
at shorter readout L by integrating only the first L samples. Two envelope policies:
  freeze : MF/RMF envelopes from the 500-sample fit, SLICED to L (nothing refit)
  refit  : MF/RMF envelopes RE-FIT at L (matched filters recalibrated; NN frozen)
This is the apples-to-apples test of the OG paper's "needs no re-training" claim.
Per-trace MF (each row = one physical shot). Output per-qubit + gmean5 + gmean4(no Q2).
"""
import os, csv, numpy as np, torch as T, torch.nn as nn
import os as _os, sys as _sys  # [unify] discriminators/ on sys.path (repo layout)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from disc_paths import REPORTS_OUT, ensure_dir  # [unify] replaces the old docker mount paths
import trainers.HERQULES_original as H
from trainers.HERQULES_original import (
    preclassifier, relaxation_mf_classifier, mf_demux_data_prep,
    Net_rmf, adjust_learning_rate)
from matched_filter import search_matched_filter_for_all_qubits_demux

H.SUBSET_NUM_TRAIN_VAL = 20000; H.SUBSET_NUM_TEST = 30000
_d = [0.35 if (isinstance(x, float) and abs(x-0.53) < 1e-9) else x
      for x in H.get_train_val_and_test_set.__defaults__]
H.get_train_val_and_test_set.__defaults__ = tuple(_d)
REPORTS = ensure_dir(REPORTS_OUT); BOX = [1, 1, 9, 2, 9]; NUM_CLASSES = 32  # [unify] was REPORTS = \"/app/optimization_reports\"
DEV = T.device("cuda" if T.cuda.is_available() else "cpu")
FULL = H.TRACE_LENGTH
LENGTHS_NS = [200, 250, 400, 500, 600, 750, 800, 1000]
print(f"[trunc-pt] device={DEV} full={FULL} ({FULL*2}ns) lengths={LENGTHS_NS}", flush=True)

pre = preclassifier(); pre.fit(); tc = pre.get_traces()
rmf = relaxation_mf_classifier(); rmf.fit(tc, boxcars=BOX)
dtr, dval, dte = mf_demux_data_prep(trace_length=FULL)
mf_env_full, _ = search_matched_filter_for_all_qubits_demux(dtr, best_bc=BOX)
mf_env_full = [np.asarray(e) for e in mf_env_full]


def bits(y): return np.stack([(y >> q) & 1 for q in range(5)], 1).astype(int)


def per_trace_mf(data, envs, L):
    N = NUM_CLASSES * data[0].shape[1]; F = 2 * L
    out = np.zeros((N, 5), np.float32)
    for q in range(5):
        out[:, q] = data[q][:, :, :L, :].reshape(N, F).astype(np.float32) @ envs[q][:F].astype(np.float32)
    return out


def feats(data, dtype, L, envs, rmf_clf):
    mf = per_trace_mf(data, envs, L)
    rf = rmf_clf.predict(data_type=dtype, trace_length=L).reshape(mf.shape[0], 5)
    y = np.repeat(np.arange(NUM_CLASSES), data[0].shape[1]).astype(np.int64)
    return np.concatenate([mf, rf], 1).astype(np.float32), y


def refit_rmf_at_L(L):
    tcL = {q: dict(tc[q]) for q in tc}
    for q in tcL:
        tcL[q]['traces_relax'] = tc[q]['traces_relax'][:, :L, :]
        tcL[q]['traces_0'] = tc[q]['traces_0'][:, :L, :]
    r = relaxation_mf_classifier(); r.fit(tcL, boxcars=BOX); return r


# ---- train ONCE at full length (per-trace), then freeze ----
Xtr, ytr = feats(dtr, 0, FULL, mf_env_full, rmf)
Xva, yva = feats(dval, 1, FULL, mf_env_full, rmf)
T.manual_seed(1); np.random.seed(1)
net = Net_rmf().to(DEV); opt = T.optim.Adam(net.parameters(), lr=0.01); crit = nn.CrossEntropyLoss()
ld = T.utils.data.DataLoader(T.utils.data.TensorDataset(T.tensor(Xtr), T.tensor(ytr)), batch_size=512, shuffle=True)
Xv = T.tensor(Xva).to(DEV); yvb = bits(yva); best, bs = -1, None
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
print(f"[trunc-pt] frozen 500-model trained (val bit-acc {best:.4f})", flush=True)


def evalfeat(X, y):
    with T.no_grad():
        pp = np.concatenate([net(T.tensor(X[s:s+8192]).to(DEV)).argmax(1).cpu().numpy() for s in range(0, len(X), 8192)])
    pb = np.stack([(pp >> q) & 1 for q in range(5)], 1); tb = bits(y)
    fq = [float(1 - ((pb[tb[:, q]==1, q]==0).mean() + (pb[tb[:, q]==0, q]==1).mean())/2) for q in range(5)]
    return fq, float(np.prod(fq) ** 0.2), float(np.prod([fq[0], fq[2], fq[3], fq[4]]) ** 0.25)


rows = []
for ns in LENGTHS_NS:
    L = ns // 2
    # (a) freeze: full envelopes sliced, first L samples
    Xf, yf = feats(dte, 2, L, mf_env_full, rmf)
    fqa, g5a, g4a = evalfeat(Xf, yf)
    # (b) refit: re-fit MF + RMF at L, NN still frozen
    dtr_L = [dtr[q][:, :, :L, :] for q in range(5)]
    env_L, _ = search_matched_filter_for_all_qubits_demux(dtr_L, best_bc=BOX); env_L = [np.asarray(e) for e in env_L]
    rmfL = refit_rmf_at_L(L)
    Xr, yr = feats(dte, 2, L, env_L, rmfL)
    fqb, g5b, g4b = evalfeat(Xr, yr)
    print(f"[trunc-pt] {ns:4d}ns  freeze g5={g5a:.4f} g4={g4a:.4f} | refit g5={g5b:.4f} g4={g4b:.4f}", flush=True)
    for pol, fq, g5, g4 in [("freeze", fqa, g5a, g4a), ("refit", fqb, g5b, g4b)]:
        rows.append([pol, ns, L] + [round(a, 4) for a in fq] + [round(g5, 4), round(g4, 4)])

out = os.path.join(REPORTS, "herqules_truncation_pertrace_deployment.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["policy", "length_ns", "trace_samples", "Q1", "Q2", "Q3", "Q4", "Q5", "gmean5", "gmean4_no_q2"])
    w.writerows(rows)
print(f"[trunc-pt] wrote {out}", flush=True); print("[trunc-pt] done", flush=True)
