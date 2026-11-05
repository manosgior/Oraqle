"""em_e3herq_regen_pertrace_ckpt.py  (leakage study, tag e3herq)  -- "C"

INFERENCE ONLY (no training): exact per-shot TEST predictions of the HERQULES
model behind ~/oraqle_reports/herqules_error_breakdown_realistic.csv, i.e. the
"(b) HERQULES (Per-Trace)" panel of the (cut) paper figure
plots/cnn_vs_herqules_error_breakdown.pdf.

Provenance (cluster timestamps, 2026-06-02 UTC), from
Oraqle/Discriminators/runners/_train_and_analyze_errors.py, scramble=False pass:
  len500: HERQULES_len500.pth 08:26:28 -> rmf_len500.pkl 08:28:05 ->
          herqules_error_modes_len500_pertrace_20260602_082923.csv
  len200: HERQULES_len200.pth 08:18:53 -> rmf_len200.pkl 08:20:44 ->
          herqules_error_modes_len200_pertrace_20260602_082117.csv
The checkpoint is read from ~/oraqle_models (mounted ro at /models) and the RMF
state from ~/oraqle_reports/rmf_len{L}.pkl (mounted ro at /reports); nothing is
written outside --out.

Replicates runners/analyze_herqules_errors.run(scrambled=False) step by step:
mf_demux_data_prep(L) (default split: test = full raw TEST file, identity order),
MF envelopes from search_matched_filter_for_all_qubits_demux(train, [1,1,9,2,9]),
per-trace MF on test, RMF from the saved state, Net_rmf (networks.HERQULES) logits,
per-qubit marginal P(q=1), decision p>0.5, and the exclusive error categorisation
(t1_pattern > amp_outlier > near_boundary > other) copied verbatim (the original
module imports matplotlib, absent from oraqle-torch:latest).
Trace length via HERQULES_TRACE_LENGTH (read at import by HERQULES_original).
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
ap.add_argument("--ckpt_dir", default=str(LP.MODELS))
ap.add_argument("--rmf_dir", default=str(LP.REPORTS))
ap.add_argument("--threads", type=int, default=8)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

import torch
torch.set_num_threads(args.threads)
t_imp = time.time()
import trainers.HERQULES_original as H
from trainers.HERQULES_original import mf_demux_data_prep, relaxation_mf_classifier
from matched_filter import search_matched_filter_for_all_qubits_demux
from networks.HERQULES import Net_rmf          # the class analyze_herqules_errors loads into
if getattr(sys.modules.get("pandas"), "__getattr__", None) is not None and not hasattr(sys.modules["pandas"], "DataFrame"):
    del sys.modules["pandas"]
L = H.TRACE_LENGTH
ns = 2 * L
NUM_QUBITS, NUM_CLASSES = 5, 32
NEAR_BOUNDARY_CONF_THRESHOLD = 0.60
AMPLITUDE_OUTLIER_Z = 3.0
print(f"[e3C] import+demod {time.time()-t_imp:.0f}s TRACE_LENGTH={L}", flush=True)

N_TRAIN_FILE = int(len(H.y_train_raw))
y_test_raw = np.asarray(H.y_test_raw).astype(np.int64)
y_data = np.asarray(H.y_data).astype(np.int64)
for _n in ("all_data", "train_semi_sup_data", "X_train_raw", "X_test_raw"):
    if hasattr(H, _n):
        setattr(H, _n, None)
gc.collect()

mn = min(int(np.sum(y_data == i)) for i in range(NUM_CLASSES))
TV = int(mn * 0.3); TE = mn - TV
G_TE = np.concatenate([np.where(y_data == i)[0][TV:TV+TE] for i in range(NUM_CLASSES)])
RAW_TE = np.where(G_TE >= N_TRAIN_FILE, G_TE - N_TRAIN_FILE, -1)
assert (RAW_TE >= 0).all()
assert np.array_equal(y_test_raw[RAW_TE], np.repeat(np.arange(NUM_CLASSES), TE))
print(f"[e3C] default split TV={TV} TE={TE}; identity order: {bool(np.array_equal(RAW_TE, np.arange(len(RAW_TE))))}", flush=True)

data_train, data_val, data_test = mf_demux_data_prep(trace_length=L)
del data_val; gc.collect()
n_per_state = data_test[0].shape[1]; n_total = NUM_CLASSES * n_per_state
for q in range(NUM_QUBITS):
    for s in range(NUM_CLASSES):
        g = G_TE[s*n_per_state:(s+1)*n_per_state]
        assert np.array_equal(H._DEMOD_CACHE[q][g, :L, :], data_test[q][s]), (q, s)
print("[e3C] data alignment verified", flush=True)

best_bc = [1, 1, 9, 2, 9]
mf_envs, _ = search_matched_filter_for_all_qubits_demux(data_train, best_bc=best_bc)
mf_envs = [np.asarray(e) for e in mf_envs]

rmf = relaxation_mf_classifier()
rmf.load_state(os.path.join(args.rmf_dir, f"rmf_len{L}.pkl"))
test_rmf = rmf.predict(data_type=2, trace_length=L)
rmf_feat = test_rmf.reshape(n_total, NUM_QUBITS).astype(np.float32, copy=False)

F = 2 * L
mf_feat = np.zeros((n_total, NUM_QUBITS), dtype=np.float32)
for q in range(NUM_QUBITS):
    mf_feat[:, q] = data_test[q].reshape(n_total, F).astype(np.float32, copy=False) @ mf_envs[q].astype(np.float32, copy=False)
feat = np.concatenate([mf_feat, rmf_feat], axis=1)
y_test = np.repeat(np.arange(NUM_CLASSES), n_per_state).astype(np.int64)

ckpt = os.path.join(args.ckpt_dir, f"HERQULES_len{L}.pth")
model = Net_rmf()
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()
with torch.no_grad():
    Xt = torch.tensor(feat, dtype=torch.float32)
    logits = torch.cat([model(Xt[s:s+4096]).cpu() for s in range(0, Xt.shape[0], 4096)], 0)
probs = torch.softmax(logits, dim=-1)
p_q1 = torch.zeros(probs.shape[0], NUM_QUBITS, dtype=probs.dtype)
states = torch.arange(NUM_CLASSES)
for q in range(NUM_QUBITS):
    p_q1[:, q] = probs[:, ((states >> q) & 1).bool()].sum(dim=-1)
p_q1 = p_q1.numpy()
cls = logits.argmax(-1).numpy().astype(np.uint8)
print(f"[e3C] joint acc {100*np.mean(cls == y_test):.4f}%", flush=True)


def amplitude_per_trace(d):
    return np.linalg.norm(d, axis=-1).mean(axis=-1).astype(np.float32)


amp_stats = {}
for q in range(NUM_QUBITS):
    a = amplitude_per_trace(data_train[q].reshape(-1, data_train[q].shape[2], 2))
    amp_stats[q] = (float(a.mean()), float(a.std()))
del data_train; gc.collect()

CAT = {"correct": 0, "near_boundary": 1, "t1_pattern": 2, "amp_outlier": 3, "other": 4}
category = np.zeros((n_total, NUM_QUBITS), np.uint8)
masks = {k: np.zeros((n_total, NUM_QUBITS), bool) for k in ("near", "amp", "t1")}
amp_z_all = np.zeros((n_total, NUM_QUBITS), np.float32)
rows = []
for q in range(NUM_QUBITS):
    demod_q = data_test[q].reshape(n_total, L, 2)
    true_q = (y_test >> q) & 1
    pred_q = (p_q1[:, q] > 0.5).astype(np.int64)
    is_err = pred_q != true_q
    conf = np.where(true_q == 1, p_q1[:, q], 1.0 - p_q1[:, q])
    near = conf < NEAR_BOUNDARY_CONF_THRESHOLD
    amp = amplitude_per_trace(demod_q)
    amp_z = (amp - amp_stats[q][0]) / (amp_stats[q][1] + 1e-12)
    ampo = np.abs(amp_z) > AMPLITUDE_OUTLIER_Z
    flat = demod_q[:, :L, :].reshape(n_total, F).astype(np.float32, copy=False)
    env = mf_envs[q][:F].astype(np.float32, copy=False)
    half = F // 2
    m1 = flat[:, :half] @ env[:half]; m2 = flat[:, half:] @ env[half:]
    sig = (np.abs(m1) > 0.5 * np.median(np.abs(m1))) & (np.abs(m2) > 0.5 * np.median(np.abs(m2)))
    t1 = sig & (np.sign(m1) != np.sign(m2))
    c = np.where(is_err, CAT["other"], CAT["correct"]).astype(np.uint8)
    c[is_err & near] = CAT["near_boundary"]
    c[is_err & ampo] = CAT["amp_outlier"]
    c[is_err & t1] = CAT["t1_pattern"]          # priority t1 > amp > near > other
    category[:, q] = c
    masks["near"][:, q] = near; masks["amp"][:, q] = ampo; masks["t1"][:, q] = t1
    amp_z_all[:, q] = amp_z
    rows.append({"qubit": q, "n_test": int(n_total), "n_error": int(is_err.sum()),
                 "accuracy_pct": float(100 * (1 - is_err.mean())),
                 "excl_n_t1_pattern": int((c == CAT["t1_pattern"]).sum()),
                 "excl_n_amp_outlier": int((c == CAT["amp_outlier"]).sum()),
                 "excl_n_near_boundary": int((c == CAT["near_boundary"]).sum()),
                 "excl_n_other": int((c == CAT["other"]).sum())})
    print(f"[e3C] {ns}ns q{q} {rows[-1]}", flush=True)

np.savez_compressed(os.path.join(args.out, f"em_e3herq_C_{ns}ns.npz"),
                    length_ns=ns, trace_samples=L, test_rawrow=RAW_TE.astype(np.int32),
                    test_state=y_test.astype(np.uint8), cls=cls, p1=p_q1.astype(np.float32),
                    category=category, category_codes=json.dumps(CAT),
                    near_mask=masks["near"], amp_outlier_mask=masks["amp"], t1_mask=masks["t1"],
                    amp_z=amp_z_all)
with open(os.path.join(args.out, f"em_e3herq_C_{ns}ns_summary.json"), "w") as f:
    json.dump({"length_ns": ns, "trace_samples": L, "ckpt": ckpt,
               "rmf_state": os.path.join(args.rmf_dir, f"rmf_len{L}.pkl"),
               "joint_acc_pct": float(100 * np.mean(cls == y_test)),
               "amp_train_stats": amp_stats, "per_qubit": rows}, f, indent=1)
print("[e3C] done", flush=True)
