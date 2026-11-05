"""Clean HERQULES length sweep, full data, no scramble, retrained per length.
Two variants from the SAME envelopes/data/seed per length -- only the MF feature
builder differs:
  deployment_pertrace : each row = one physical shot (build per-trace MF)   -> ~0.904
  demux_subsample     : matched_filter_preprocess_demux(scramble=False)      -> ~0.925
Output: variant, length_ns, trace_samples, Q1..Q5, gmean5, gmean4(excl Q2).
"""
import os, csv, numpy as np, torch as T, torch.nn as nn
import os as _os, sys as _sys  # [unify] discriminators/ on sys.path (repo layout)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from disc_paths import REPORTS_OUT, ensure_dir  # [unify] replaces the old docker mount paths
import trainers.HERQULES_original as H
from trainers.HERQULES_original import (
    preclassifier, relaxation_mf_classifier, mf_demux_data_prep,
    Net_rmf, MFOutputDataset, adjust_learning_rate)
from matched_filter import search_matched_filter_for_all_qubits_demux, matched_filter_preprocess_demux

H.SUBSET_NUM_TRAIN_VAL = 20000; H.SUBSET_NUM_TEST = 30000           # full
_d = [0.35 if (isinstance(x, float) and abs(x-0.53) < 1e-9) else x
      for x in H.get_train_val_and_test_set.__defaults__]
H.get_train_val_and_test_set.__defaults__ = tuple(_d)
REPORTS = ensure_dir(REPORTS_OUT); BOX = [1, 1, 9, 2, 9]  # [unify] was REPORTS = \"/app/optimization_reports\"
DEV = T.device("cuda" if T.cuda.is_available() else "cpu")
NUM_CLASSES = 32
LENGTHS_NS = [200, 250, 400, 500, 600, 750, 800, 1000]
print(f"[ls] device={DEV} lengths={LENGTHS_NS}", flush=True)

pre = preclassifier(); pre.fit()
rmf = relaxation_mf_classifier(); rmf.fit(pre.get_traces(), boxcars=BOX)


def bits(y): return np.stack([(y >> q) & 1 for q in range(5)], 1).astype(int)


def per_trace_mf(data, envs, S):
    N = NUM_CLASSES * data[0].shape[1]; F = 2 * S
    out = np.zeros((N, 5), np.float32)
    for q in range(5):
        out[:, q] = data[q].reshape(N, F).astype(np.float32) @ envs[q][:F].astype(np.float32)
    return out


def train_eval(Xtr, ytr, Xva, yva, Xte, yte, seed=1):
    T.manual_seed(seed); np.random.seed(seed)
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
    pb = np.stack([(pp >> q) & 1 for q in range(5)], 1); tb = bits(yte)
    fq = [float(1 - ((pb[tb[:, q]==1, q]==0).mean() + (pb[tb[:, q]==0, q]==1).mean())/2) for q in range(5)]
    g5 = float(np.prod(fq) ** 0.2); g4 = float(np.prod([fq[0], fq[2], fq[3], fq[4]]) ** 0.25)
    return fq, g5, g4


def demux_feats(data, dtype, S):
    mf = matched_filter_preprocess_demux(data, mf_env_S, scramble=False)   # (32, min_ind, 5)
    rf = rmf.predict(data_type=dtype, trace_length=S)                      # (32, n_per_state, 5)
    m = min(mf.shape[1], rf.shape[1])
    feat = np.concatenate([mf[:, :m], rf[:, :m]], 2).reshape(NUM_CLASSES * m, 10).astype(np.float32)
    y = np.repeat(np.arange(NUM_CLASSES), m).astype(np.int64)
    return feat, y


rows = []
for ns in LENGTHS_NS:
    S = ns // 2
    dtr, dval, dte = mf_demux_data_prep(trace_length=S)
    mf_env_S, _ = search_matched_filter_for_all_qubits_demux(dtr, best_bc=BOX)
    mf_env_S = [np.asarray(e) for e in mf_env_S]
    # ---- deployment per-trace ----
    def pt(data, dtype):
        mf = per_trace_mf(data, mf_env_S, S)
        rf = rmf.predict(data_type=dtype, trace_length=S).reshape(mf.shape[0], 5)
        y = np.repeat(np.arange(NUM_CLASSES), data[0].shape[1]).astype(np.int64)
        return np.concatenate([mf, rf], 1).astype(np.float32), y
    Xtr, ytr = pt(dtr, 0); Xva, yva = pt(dval, 1); Xte, yte = pt(dte, 2)
    fq, g5, g4 = train_eval(Xtr, ytr, Xva, yva, Xte, yte)
    print(f"[ls] {ns:4d}ns pertrace  g5={g5:.4f} g4={g4:.4f}  Q={[round(a,3) for a in fq]}", flush=True)
    rows.append(["deployment_pertrace", ns, S] + [round(a, 4) for a in fq] + [round(g5, 4), round(g4, 4)])
    # ---- demux subsample ----
    Xtr, ytr = demux_feats(dtr, 0, S); Xva, yva = demux_feats(dval, 1, S); Xte, yte = demux_feats(dte, 2, S)
    fq, g5, g4 = train_eval(Xtr, ytr, Xva, yva, Xte, yte)
    print(f"[ls] {ns:4d}ns demux     g5={g5:.4f} g4={g4:.4f}  Q={[round(a,3) for a in fq]}", flush=True)
    rows.append(["demux_subsample", ns, S] + [round(a, 4) for a in fq] + [round(g5, 4), round(g4, 4)])

out = os.path.join(REPORTS, "herqules_length_faithful_vs_demux.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["variant", "length_ns", "trace_samples", "Q1", "Q2", "Q3", "Q4", "Q5", "gmean5", "gmean4_no_q2"])
    w.writerows(rows)
print(f"[ls] wrote {out}", flush=True); print("[ls] done", flush=True)
