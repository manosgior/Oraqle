"""em_e3herq_regen_scrambled.py  (leakage study, tag e3herq)

Regenerate the "fake"/scrambled HERQULES of the error-mode CSVs
(~/oraqle_reports/herqules_error_modes_len{500,200}_scrambled_*.csv ->
herqules_error_breakdown_fake.csv) through the SAME code path:
Oraqle/Discriminators/runners/_train_and_analyze_errors.py, scramble=True branch =
  H.train(run_semi_sup=False, run_pre_filter=False, run_rmf=True, scramble=True)
  followed by analyze_herqules_errors.run(scrambled=True).
The train() body is replicated step by step (same calls, same order, same seeds)
but WITHOUT writing preclassifier_state.pkl / rmf.pkl / HERQULES_len*.pth / profiler
traces, so no existing file is touched. The analyze step's per-(qubit,state)
permutation (RandomState(42)) is reproduced and kept, so each row's MF source shot
is known per qubit; the RMF source shot is the unpermuted row.
Extra (inference only): the scrambled-trained model on shot-aligned rows (MF and
RMF of all five qubits from one physical shot) = its true single-shot predictions.
Trace length via HERQULES_TRACE_LENGTH (read at import by HERQULES_original).
Default split (no SUBSET_*): test = the full raw TEST file (35,000 per state).
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
_sys.path.insert(0, str(LP.DISCRIMINATORS))  # was PYTHONPATH=<Oraqle/Discriminators>
import os, sys, json, time, types, argparse, logging, gc

import numpy as np


def _install_stubs():
    if "loguru" not in sys.modules:
        lg = logging.getLogger("loguru-stub")

        class _Logger:
            def __getattr__(self, name):
                if name in ("info", "debug", "warning", "error", "critical", "success", "trace"):
                    return lambda msg="", *a, **k: lg.info(str(msg))
                return lambda *a, **k: None

        m = types.ModuleType("loguru")
        m.logger = _Logger()
        sys.modules["loguru"] = m
    if "pandas" not in sys.modules:
        p = types.ModuleType("pandas")

        def _ga(name):
            raise AttributeError(f"pandas stub: '{name}' used but pandas is not installed")
        p.__getattr__ = _ga
        sys.modules["pandas"] = p


_install_stubs()
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(LP.stage("em/e3herq") / "out"))
ap.add_argument("--threads", type=int, default=8)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

import torch as T
T.set_num_threads(args.threads)
t_imp = time.time()
import trainers.HERQULES_original as H
from trainers.HERQULES_original import (
    preclassifier, relaxation_mf_classifier, mf_demux_data_prep, Net_rmf,
    MFOutputDataset, adjust_learning_rate, accuracy)
from matched_filter import search_matched_filter_for_all_qubits_demux, matched_filter_preprocess_demux
# the pandas stub is only needed while HERQULES_original imports it (unused there);
# drop it so sklearn/narwhals do not mistake it for a real pandas
if getattr(sys.modules.get("pandas"), "__getattr__", None) is not None and not hasattr(sys.modules["pandas"], "DataFrame"):
    del sys.modules["pandas"]
L = H.TRACE_LENGTH
ns = 2 * L
NUM_CLASSES = 32
print(f"[e3B] import+demod {time.time()-t_imp:.0f}s TRACE_LENGTH={L}", flush=True)

N_TRAIN_FILE = int(len(H.y_train_raw))
y_test_raw = np.asarray(H.y_test_raw).astype(np.int64)
y_data = np.asarray(H.y_data).astype(np.int64)
for _n in ("all_data", "train_semi_sup_data", "X_train_raw", "X_test_raw"):
    if hasattr(H, _n):
        setattr(H, _n, None)
gc.collect()

# default split, replicated: NUM_TRAIN_VAL=int(min*0.3), NUM_TEST=min-NUM_TRAIN_VAL, val ratio 0.53
mn = min(int(np.sum(y_data == i)) for i in range(NUM_CLASSES))
TV = int(mn * 0.3); TE = mn - TV
G_TE = np.concatenate([np.where(y_data == i)[0][TV:TV+TE] for i in range(NUM_CLASSES)])
RAW_TE = np.where(G_TE >= N_TRAIN_FILE, G_TE - N_TRAIN_FILE, -1)
assert (RAW_TE >= 0).all()
assert np.array_equal(y_test_raw[RAW_TE], np.repeat(np.arange(NUM_CLASSES), TE))
print(f"[e3B] default split: per state TV={TV} TE={TE}; test = {len(G_TE)} raw TEST rows "
      f"(identity order: {bool(np.array_equal(RAW_TE, np.arange(len(RAW_TE))))})", flush=True)

# ======================= replicate H.train(..., scramble=True, seed=1, save=False) =================
T.manual_seed(1); np.random.seed(1)
semi = preclassifier(); rmf = relaxation_mf_classifier()
semi.fit()                                   # (save_state skipped: pickle only)
boxcars = [1, 1, 9, 2, 9]
trace_classes = semi.get_traces()
rmf.fit(trace_classes, boxcars=boxcars)      # (save_state skipped)
train_rmf = rmf.predict(data_type=0); val_rmf = rmf.predict(data_type=1); test_rmf = rmf.predict(data_type=2)
dtr, dva, dte = mf_demux_data_prep(L)
# data-level alignment: split test (s,k) == demodulated raw TEST row
n_te = dte[0].shape[1]
for q in range(5):
    for s in range(NUM_CLASSES):
        g = G_TE[s*n_te:(s+1)*n_te]
        assert np.array_equal(H._DEMOD_CACHE[q][g, :L, :], dte[q][s]), (q, s)
print("[e3B] data alignment verified", flush=True)
mf_envelopes, _ = search_matched_filter_for_all_qubits_demux(dtr, best_bc=boxcars)
train_feat = np.concatenate((matched_filter_preprocess_demux(dtr, mf_envelopes, scramble=True), train_rmf), axis=2)
val_feat = np.concatenate((matched_filter_preprocess_demux(dva, mf_envelopes, scramble=True), val_rmf), axis=2)
test_feat = np.concatenate((matched_filter_preprocess_demux(dte, mf_envelopes, scramble=True), test_rmf), axis=2)
train_ds = MFOutputDataset(train_feat); val_ds = MFOutputDataset(val_feat); test_ds = MFOutputDataset(test_feat)
train_ldr = T.utils.data.DataLoader(train_ds, batch_size=512, shuffle=True)
val_ldr = T.utils.data.DataLoader(val_ds, batch_size=512, shuffle=False)
test_ldr = T.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False)
net = Net_rmf().train()
loss_obj = T.nn.CrossEntropyLoss(); optimizer = T.optim.Adam(net.parameters(), lr=0.01)
best_acc, best_state, best_ep = -1, None, -1
t0 = time.time()
for epoch in range(100):
    epoch_loss = 0.0
    lr = adjust_learning_rate(0.01, optimizer, epoch)
    for batch in train_ldr:
        oupt = net(batch['predictors']); loss_val = loss_obj(oupt, batch['target'])
        epoch_loss += loss_val.item()
        optimizer.zero_grad(); loss_val.backward(); optimizer.step()
    acc_train, _ = accuracy(net, train_ldr)       # kept: consumes the torch RNG exactly as train() does
    acc_val, acc_val_q = accuracy(net, val_ldr)
    if acc_val >= best_acc:
        best_acc = acc_val; best_ep = epoch
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    if epoch % 10 == 0:
        print(f"[e3B] epoch {epoch} loss {epoch_loss:.2f} lr {lr} val {acc_val:.4f} ({time.time()-t0:.0f}s)", flush=True)
net.load_state_dict(best_state)
acc_test_train_fn, acc_test_q_train_fn = accuracy(net, test_ldr)
print(f"[e3B] train()-style test acc {acc_test_train_fn:.4f} per qubit {[round(a,4) for a in acc_test_q_train_fn]} "
      f"(best val {best_acc:.4f} @ep{best_ep})", flush=True)
net.eval()

# ======================= replicate analyze_herqules_errors.run(scrambled=True) =====================
mf_envs = [np.asarray(e) for e in mf_envelopes]
n_per_state = dte[0].shape[1]; n_total = NUM_CLASSES * n_per_state; F = 2 * L
rmf_feat = test_rmf.reshape(n_total, 5).astype(np.float32, copy=False)
rng = np.random.RandomState(42)                                   # SAMPLE_SEED
mf_scr = np.zeros((n_total, 5), dtype=np.float32)
perms = np.zeros((5, NUM_CLASSES, n_per_state), dtype=np.int64)
for q in range(5):
    env = mf_envs[q].astype(np.float32, copy=False)
    for s in range(NUM_CLASSES):
        n = dte[q].shape[1]
        idx = rng.choice(n, n, replace=False)
        perms[q, s] = idx
        traces = dte[q][s, idx].reshape(n, F).astype(np.float32, copy=False)
        mf_scr[s*n_per_state:(s+1)*n_per_state, q] = traces @ env
mf_pt = np.zeros((n_total, 5), dtype=np.float32)                  # analyze's _per_trace_mf
for q in range(5):
    mf_pt[:, q] = dte[q].reshape(n_total, F).astype(np.float32, copy=False) @ mf_envs[q].astype(np.float32, copy=False)
y_test = np.repeat(np.arange(NUM_CLASSES), n_per_state).astype(np.int64)
states = np.arange(NUM_CLASSES)


def run_net(X):
    lg = []
    with T.no_grad():
        Xt = T.tensor(X, dtype=T.float32)
        for s in range(0, Xt.shape[0], 4096):
            lg.append(net(Xt[s:s+4096]).cpu())
    lg = T.cat(lg, 0)
    pr = T.softmax(lg, dim=-1)
    p1 = T.zeros(pr.shape[0], 5, dtype=pr.dtype)
    for q in range(5):
        p1[:, q] = pr[:, ((T.tensor(states) >> q) & 1).bool()].sum(-1)
    return lg.argmax(-1).numpy().astype(np.uint8), p1.numpy().astype(np.float32)


def rates(cls, p1, y):
    tb = np.stack([(y >> q) & 1 for q in range(5)], 1)
    pm = (p1 > 0.5).astype(int)                                   # analyze's per-qubit decision
    pa = np.stack([(cls.astype(np.int64) >> q) & 1 for q in range(5)], 1)
    return {"n_error_marginal": [int((pm[:, q] != tb[:, q]).sum()) for q in range(5)],
            "acc_marginal_pct": [float(100*(pm[:, q] == tb[:, q]).mean()) for q in range(5)],
            "acc_argmax_pct": [float(100*(pa[:, q] == tb[:, q]).mean()) for q in range(5)],
            "joint_acc_pct": float(100*(cls.astype(np.int64) == y).mean())}


scr_cls, scr_p1 = run_net(np.concatenate([mf_scr, rmf_feat], 1))
ss_cls, ss_p1 = run_net(np.concatenate([mf_pt, rmf_feat], 1))
R = {"length_ns": ns, "trace_samples": L, "best_val_joint_acc": float(best_acc), "best_epoch": best_ep,
     "train_fn_test_acc_per_qubit": [float(a) for a in acc_test_q_train_fn],
     "scrambled_analyze": rates(scr_cls, scr_p1, y_test), "singleshot_rows": rates(ss_cls, ss_p1, y_test)}
print(f"[e3B] {ns}ns scrambled (analyze) : {R['scrambled_analyze']}", flush=True)
print(f"[e3B] {ns}ns same model, 1-shot rows: {R['singleshot_rows']}", flush=True)
mf_src_raw = np.zeros((n_total, 5), dtype=np.int32)
for q in range(5):
    mf_src_raw[:, q] = RAW_TE[(np.arange(NUM_CLASSES)[:, None] * n_per_state + perms[q]).ravel()]
np.savez_compressed(os.path.join(args.out, f"em_e3herq_B_{ns}ns.npz"),
                    length_ns=ns, trace_samples=L, test_rawrow=RAW_TE.astype(np.int32),
                    test_state=y_test.astype(np.uint8),
                    scr_cls=scr_cls, scr_p1=scr_p1, scr_mf_src_rawrow=mf_src_raw,
                    scr_rmf_src_rawrow=RAW_TE.astype(np.int32),
                    ss_cls=ss_cls, ss_p1=ss_p1)
T.save({"net": best_state, "mf_env": [e.astype(np.float32) for e in mf_envs],
        "rmf_env": [np.asarray(e, np.float32) for e in rmf.envelopes]},
       os.path.join(args.out, f"em_e3herq_B_{ns}ns_model.pt"))
with open(os.path.join(args.out, f"em_e3herq_B_{ns}ns_summary.json"), "w") as f:
    json.dump(R, f, indent=1)
print("[e3B] done", flush=True)
