"""Paper-faithful 'without re-training' readout-duration curve for HERQULES.

Train ONE no-scramble Net_rmf at full length (500 samples / 1000 ns) and FREEZE
the network. Then sweep readout length L, integrating only the first L samples,
under two envelope policies:
  (a) freeze   : MF/RMF envelopes fit at 500 and SLICED to 2L (nothing refit)
  (b) refit    : MF/RMF envelopes RE-FIT at L (NN still frozen) -- the likely
                 paper protocol ("without re-training" = the NETWORK isn't
                 retrained, but the matched filters are recalibrated)

Contrast with our earlier per-length *retraining*, which over-states short
readout. Published split (3000/7000, val 0.35). Seed 1. Env HERQULES_TRACE_LENGTH=500.
"""
import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import csv
import numpy as np
import torch as T
import trainers.HERQULES_original as H
from trainers.HERQULES_original import (
    preclassifier, relaxation_mf_classifier, mf_demux_data_prep,
    Net_rmf, MFOutputDataset, adjust_learning_rate, accuracy,
)
from matched_filter import (
    search_matched_filter_for_all_qubits_demux, matched_filter_preprocess_demux,
)

H.SUBSET_NUM_TRAIN_VAL = 3000
H.SUBSET_NUM_TEST = 7000
_d = [0.35 if (isinstance(x, float) and abs(x - 0.53) < 1e-9) else x
      for x in H.get_train_val_and_test_set.__defaults__]
H.get_train_val_and_test_set.__defaults__ = tuple(_d)
REPORTS = ensure_dir(REPORTS_OUT)

import os
SEED = int(os.environ.get("HERQ_SEED", "1"))
FULL = H.TRACE_LENGTH
BOX = [1, 1, 9, 2, 9]
LENGTHS = [int(x) for x in os.environ.get("HERQ_TRUNC_SAMPLES", "100,200,300,400,500").split(",")]
OUT_NAME = os.environ.get("HERQ_TRUNC_OUT", "herqules_truncation_noretrain.csv")
POLICIES = os.environ.get("HERQ_POLICIES", "freeze,refit").split(",")
DEV = T.device("cuda" if T.cuda.is_available() else "cpu")
T.manual_seed(SEED); np.random.seed(SEED)
print(f"[trunc] seed={SEED} device={DEV} policies={POLICIES}", flush=True)
print(f"[trunc] full={FULL} ({FULL*2} ns); train-once-freeze, two envelope policies", flush=True)

# --- fit preclassifier + RMF + MF envelopes ONCE at full length ---
pre = preclassifier(); pre.fit()
tc = pre.get_traces()
rmf = relaxation_mf_classifier(); rmf.fit(tc, boxcars=BOX)
dtr, dval, dte = mf_demux_data_prep(FULL)
mf_env, _ = search_matched_filter_for_all_qubits_demux(dtr, best_bc=BOX)

# --- train the network ONCE at full length (no-scramble), then freeze ---
tr_feat = np.concatenate((matched_filter_preprocess_demux(dtr, mf_env, scramble=False), rmf.predict(data_type=0)), axis=2)
va_feat = np.concatenate((matched_filter_preprocess_demux(dval, mf_env, scramble=False), rmf.predict(data_type=1)), axis=2)
tr_ld = T.utils.data.DataLoader(MFOutputDataset(tr_feat), batch_size=512, shuffle=True)
va_ld = T.utils.data.DataLoader(MFOutputDataset(va_feat), batch_size=512, shuffle=False)
net = Net_rmf().to(DEV).train(); opt = T.optim.Adam(net.parameters(), lr=0.01); crit = T.nn.CrossEntropyLoss()


def evaluate(feat):
    ld = T.utils.data.DataLoader(MFOutputDataset(feat), batch_size=4096, shuffle=False)
    net.eval(); preds, labs = [], []
    with T.no_grad():
        for b in ld:
            preds.append(net(b['predictors'].to(DEV)).argmax(1).cpu().numpy())
            labs.append(b['target'].numpy())
    net.train()
    pred = np.concatenate(preds); lab = np.concatenate(labs)
    acc = float((pred == lab).mean())
    per_q = []; pi = pred.copy(); ll = lab.copy()
    for _ in range(5):
        per_q.append(float((pi % 2 == ll % 2).mean())); pi = pi >> 1; ll = ll >> 1
    return acc, per_q


best, best_state = -1.0, None
for ep in range(100):
    adjust_learning_rate(0.01, opt, ep)
    for b in tr_ld:
        o = net(b['predictors'].to(DEV)); l = crit(o, b['target'].to(DEV))
        opt.zero_grad(); l.backward(); opt.step()
    av, _ = evaluate(va_feat)
    if av >= best:
        best, best_state = av, {k: v.detach().clone() for k, v in net.state_dict().items()}
net.load_state_dict(best_state); net.eval()
print(f"[trunc] seed={SEED} frozen 500-model best val acc {best:.4f}", flush=True)


def refit_rmf_at_L(L):
    tcL = {q: dict(tc[q]) for q in tc}
    for q in tcL:
        tcL[q]['traces_relax'] = tc[q]['traces_relax'][:, :L, :]
        tcL[q]['traces_0'] = tc[q]['traces_0'][:, :L, :]
    r = relaxation_mf_classifier(); r.fit(tcL, boxcars=BOX)
    return r


rows = []
print("[trunc] readout curve:", flush=True)
for L in LENGTHS:
    dte_L = [dte[q][:, :, :L, :] for q in range(5)]
    dtr_L = [dtr[q][:, :, :L, :] for q in range(5)]
    results = []
    # (a) freeze everything
    if "freeze" in POLICIES:
        mf_a = matched_filter_preprocess_demux(dte_L, [mf_env[q][:2*L] for q in range(5)], scramble=False)
        rmf_a = rmf.predict(data_type=2, trace_length=L)
        acc_a, q_a = evaluate(np.concatenate((mf_a, rmf_a), axis=2))
        results.append(("freeze", q_a, float(np.prod(q_a) ** 0.2), acc_a))
    # (b) refit envelopes at L, NN frozen
    if "refit" in POLICIES:
        mf_env_L, _ = search_matched_filter_for_all_qubits_demux(dtr_L, best_bc=BOX)
        rmfL = refit_rmf_at_L(L)
        mf_b = matched_filter_preprocess_demux(dte_L, mf_env_L, scramble=False)
        rmf_b = rmfL.predict(data_type=2, trace_length=L)
        acc_b, q_b = evaluate(np.concatenate((mf_b, rmf_b), axis=2))
        results.append(("refit", q_b, float(np.prod(q_b) ** 0.2), acc_b))
    msg = "  ".join(f"{tag} gmean={g:.4f}" for tag, _, g, _ in results)
    print(f"[trunc] seed={SEED} L={L:3d} ({L*2:4d}ns)  {msg}", flush=True)
    for tag, q, g, acc in results:
        rows.append(dict(policy=tag, length_ns=L*2, trace_samples=L,
                         Q1=round(q[0],4), Q2=round(q[1],4), Q3=round(q[2],4),
                         Q4=round(q[3],4), Q5=round(q[4],4),
                         gmean=round(g,4), joint=round(acc,4)))

out = os.path.join(REPORTS, OUT_NAME)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"[trunc] wrote {out}", flush=True)
