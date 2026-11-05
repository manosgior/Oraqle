"""Cross-fidelity + accuracy-vs-N for the DEPLOYMENT-FAITHFUL HERQULES (per-trace,
full data, no scramble -- the 0.905 model). One inference pass over the full test
set -> per-shot predicted 5-bit -> both crosstalk metrics.
"""
import os, csv, numpy as np, torch as T, torch.nn as nn
from itertools import combinations
import os as _os, sys as _sys  # [unify] discriminators/ on sys.path (repo layout)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from disc_paths import REPORTS_OUT, ensure_dir  # [unify] replaces the old docker mount paths
import trainers.HERQULES_original as H
from trainers.HERQULES_original import (
    preclassifier, relaxation_mf_classifier, mf_demux_data_prep,
    Net_rmf, adjust_learning_rate)
from matched_filter import search_matched_filter_for_all_qubits_demux

H.SUBSET_NUM_TRAIN_VAL = 20000; H.SUBSET_NUM_TEST = 30000          # full
_d = [0.35 if (isinstance(x, float) and abs(x-0.53) < 1e-9) else x
      for x in H.get_train_val_and_test_set.__defaults__]
H.get_train_val_and_test_set.__defaults__ = tuple(_d)
REPORTS = ensure_dir(REPORTS_OUT); BOX = [1, 1, 9, 2, 9]; NUM_CLASSES = 32  # [unify] was REPORTS = \"/app/optimization_reports\"
DEV = T.device("cuda" if T.cuda.is_available() else "cpu"); L = H.TRACE_LENGTH
print(f"[xtd] device={DEV} full data per-trace, length={L}", flush=True)


def bits(y): return np.stack([(y >> q) & 1 for q in range(5)], 1).astype(int)


def xfid(tb, pb):
    F = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if i == j: continue
            m0 = tb[:, j] == 0; m1 = tb[:, j] == 1
            F[i, j] = 1 - ((pb[m0, i] == 1).mean() + (pb[m1, i] == 0).mean())
    return {d: float(np.mean([abs(F[i, j]) for i in range(5) for j in range(5) if abs(i-j) == d])) for d in (1, 2, 3, 4)}


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


pre = preclassifier(); pre.fit()
rmf = relaxation_mf_classifier(); rmf.fit(pre.get_traces(), boxcars=BOX)
dtr, dval, dte = mf_demux_data_prep(trace_length=L)
envs, _ = search_matched_filter_for_all_qubits_demux(dtr, best_bc=BOX); envs = [np.asarray(e) for e in envs]


def per_trace_mf(data):
    N = NUM_CLASSES * data[0].shape[1]; F = 2 * L
    out = np.zeros((N, 5), np.float32)
    for q in range(5):
        out[:, q] = data[q][:, :, :L, :].reshape(N, F).astype(np.float32) @ envs[q][:F].astype(np.float32)
    return out


def feats(data, dtype):
    mf = per_trace_mf(data); rf = rmf.predict(data_type=dtype, trace_length=L).reshape(mf.shape[0], 5)
    y = np.repeat(np.arange(NUM_CLASSES), data[0].shape[1]).astype(np.int64)
    return np.concatenate([mf, rf], 1).astype(np.float32), y


Xtr, ytr = feats(dtr, 0); Xva, yva = feats(dval, 1); Xte, yte = feats(dte, 2)
print(f"[xtd] features built (train {len(ytr)}, test {len(yte)})", flush=True)
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
with T.no_grad():
    pp = np.concatenate([net(T.tensor(Xte[s:s+8192]).to(DEV)).argmax(1).cpu().numpy() for s in range(0, len(Xte), 8192)])
pb = bits(pp); tb = bits(yte)
np.save(os.path.join(REPORTS, "preds_HERQULESdeploy_tb.npy"), tb.astype(np.int8))
np.save(os.path.join(REPORTS, "preds_HERQULESdeploy_pb.npy"), pb.astype(np.int8))
fq = [1 - ((pb[tb[:, q]==1, q]==0).mean() + (pb[tb[:, q]==0, q]==1).mean())/2 for q in range(5)]
print(f"[xtd] F5Q (sanity) {float(np.prod(fq)**0.2):.4f}", flush=True)
cf = xfid(tb, pb); fn = fvsN(tb, pb)
print("[xtd] cross-fidelity |F^CF| by dist:", {k: round(v, 4) for k, v in cf.items()}, flush=True)
print("[xtd] <F> vs N:", {k: round(v, 4) for k, v in fn.items()}, flush=True)
with open(os.path.join(REPORTS, "xtalk_HERQULES_deployment.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["metric", "k", "value"])
    for k, v in cf.items(): w.writerow(["crossfid_dist", k, f"{v:.5f}"])
    for k, v in fn.items(): w.writerow(["F_vs_N", k, f"{v:.5f}"])
print("[xtd] done", flush=True)
