"""Error-mode breakdown for the FIXED (no-scramble) FULL-dataset HERQULES (~0.925).
Train clean Net_rmf on full data, then categorize the remaining test errors with
the SAME exclusive logic as analyze_herqules_errors.py:
  t1_pattern   : MF sign flips first->second half (mid-readout relaxation/jump)
  amp_outlier  : |IQ| amplitude z-outlier (leakage to |2> / RFI)
  near_boundary: P(correct) < 0.60 (genuine low-SNR ambiguity)
  other        : remainder (most likely state-prep / label noise)
"""
import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import os, csv, numpy as np, torch as T
import trainers.HERQULES_original as H
from trainers.HERQULES_original import (
    preclassifier, relaxation_mf_classifier, mf_demux_data_prep,
    Net_rmf, adjust_learning_rate)
from matched_filter import search_matched_filter_for_all_qubits_demux
from runners.analyze_herqules_errors import (
    per_qubit_marginal, categorize_errors_for_qubit,
    _per_trace_mf, _demux_to_flat_per_qubit, amplitude_per_trace,
    NUM_QUBITS, NUM_CLASSES)

H.SUBSET_NUM_TRAIN_VAL = 20000; H.SUBSET_NUM_TEST = 30000      # full
_d = [0.35 if (isinstance(x, float) and abs(x-0.53) < 1e-9) else x
      for x in H.get_train_val_and_test_set.__defaults__]
H.get_train_val_and_test_set.__defaults__ = tuple(_d)
REPORTS = ensure_dir(REPORTS_OUT); BOX = [1, 1, 9, 2, 9]; TL = H.TRACE_LENGTH
DEV = T.device("cuda" if T.cuda.is_available() else "cpu")
print(f"[err] device={DEV} full data, length={TL}", flush=True)

dtr, dval, dte = mf_demux_data_prep(trace_length=TL)
mf_envs, _ = search_matched_filter_for_all_qubits_demux(dtr, best_bc=BOX)
mf_envs = [np.asarray(e) for e in mf_envs]
pre = preclassifier(); pre.fit()
rmf = relaxation_mf_classifier(); rmf.fit(pre.get_traces(), boxcars=BOX)


def feats(data, dtype):
    mf = _per_trace_mf(data, mf_envs)
    rf = rmf.predict(data_type=dtype, trace_length=TL).reshape(mf.shape[0], NUM_QUBITS)
    return np.concatenate([mf, rf], 1).astype(np.float32)


def ylab(data): return np.repeat(np.arange(NUM_CLASSES), data[0].shape[1]).astype(np.int64)


Xtr, ytr = feats(dtr, 0), ylab(dtr)
Xva, yva = feats(dval, 1), ylab(dval)
Xte = feats(dte, 2); yte = ylab(dte)
demod_test = _demux_to_flat_per_qubit(dte)
demod_train = _demux_to_flat_per_qubit(dtr)
amp_stats = {q: (float(amplitude_per_trace(demod_train[q]).mean()),
                 float(amplitude_per_trace(demod_train[q]).std())) for q in range(NUM_QUBITS)}
print(f"[err] features built (train {len(ytr)}, test {len(yte)})", flush=True)

# ---- train clean Net_rmf on full data (seed 1) ----
T.manual_seed(1); np.random.seed(1)
net = Net_rmf().to(DEV); opt = T.optim.Adam(net.parameters(), lr=0.01); crit = T.nn.CrossEntropyLoss()
trl = T.utils.data.DataLoader(T.utils.data.TensorDataset(T.tensor(Xtr), T.tensor(ytr)), batch_size=512, shuffle=True)
yvab = np.stack([(yva >> q) & 1 for q in range(5)], 1)
Xva_t = T.tensor(Xva).to(DEV)
best, bs = -1, None
for ep in range(100):
    adjust_learning_rate(0.01, opt, ep); net.train()
    for xb, yb in trl:
        loss = crit(net(xb.to(DEV)), yb.to(DEV)); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with T.no_grad():
        pv = net(Xva_t).argmax(1).cpu().numpy()
    pvb = np.stack([(pv >> q) & 1 for q in range(5)], 1); va = (pvb == yvab).mean()
    if va >= best: best = va; bs = {k: v.detach().clone() for k, v in net.state_dict().items()}
net.load_state_dict(bs); net.eval()

with T.no_grad():
    logits = T.cat([net(T.tensor(Xte[s:s+4096]).to(DEV)).cpu() for s in range(0, len(Xte), 4096)], 0)
p_q1 = per_qubit_marginal(logits)
mf_te = Xte[:, :5]
tb = np.stack([(yte >> q) & 1 for q in range(5)], 1); pb = (p_q1 > 0.5).astype(int)
fq = [1 - ((pb[tb[:, q]==1, q]==0).mean() + (pb[tb[:, q]==0, q]==1).mean())/2 for q in range(5)]
gm = float(np.prod(fq) ** 0.2)
print(f"[err] full-data Net_rmf test F5Q (geomean) {gm:.4f}  per-qubit {[round(float(x),4) for x in fq]}", flush=True)

rows = []
print("[err] === per-qubit error breakdown (exclusive categories) ===", flush=True)
for q in range(NUM_QUBITS):
    a = categorize_errors_for_qubit(q, mf_te, Xte[:, 5:], demod_test, mf_envs, yte, p_q1, amp_stats)
    n = int(a["is_correct"].size); nerr = int(a["is_error"].sum()); cats = a["category"]
    c = {k: int((cats == k).sum()) for k in ("t1_pattern", "amp_outlier", "near_boundary", "other")}
    acc = 100.0 * (n - nerr) / n
    pct = {k: (100.0 * v / nerr if nerr else 0) for k, v in c.items()}
    print(f"[err] Q{q+1} acc={acc:.3f}% err={nerr} ({100.0*nerr/n:.2f}%)  | "
          f"t1={c['t1_pattern']}({pct['t1_pattern']:.0f}%) amp={c['amp_outlier']}({pct['amp_outlier']:.0f}%) "
          f"near={c['near_boundary']}({pct['near_boundary']:.0f}%) other={c['other']}({pct['other']:.0f}%)", flush=True)
    rows.append(dict(qubit=q+1, n_test=n, accuracy_pct=round(acc, 4), n_error=nerr, err_rate_pct=round(100.0*nerr/n, 4),
                     t1=c["t1_pattern"], amp=c["amp_outlier"], near=c["near_boundary"], other=c["other"],
                     t1_pct=round(pct["t1_pattern"], 1), amp_pct=round(pct["amp_outlier"], 1),
                     near_pct=round(pct["near_boundary"], 1), other_pct=round(pct["other"], 1)))

with open(os.path.join(REPORTS, "herqules_error_breakdown_full.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
# aggregate excluding Q2 (qubit index 1)
agg = {k: sum(r[k] for i, r in enumerate(rows) if i != 1) for k in ("t1", "amp", "near", "other", "n_error")}
te = agg["n_error"]
print(f"\n[err] EXCLUDING Q2 — of {te} errors: t1={100*agg['t1']/te:.0f}%  amp={100*agg['amp']/te:.0f}%  "
      f"near={100*agg['near']/te:.0f}%  other={100*agg['other']/te:.0f}%", flush=True)
print("[err] done", flush=True)
