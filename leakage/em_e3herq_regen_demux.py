"""em_e3herq_regen_demux.py  (leakage study, tag e3herq)

Regenerate per-shot TEST predictions of HERQULES through the SAME code path as
Oraqle/Discriminators/runners/_herqules_length_both.py, which produced
  * paper Table tab:fidelity HERQULES rows       (variant deployment_pertrace)
  * paper Table tab:herqules-pertrace "As released" (variant demux_subsample)
(-> ~/oraqle_reports/herqules_length_faithful_vs_demux.csv).

Differences from the original runner (none change the numerics):
  * the shot indices used by matched_filter_preprocess_demux(scramble=False) are
    recovered by replaying the identical numpy RNG draws (checked: RNG state after
    replay == state after the call, and MF features recomputed from the recovered
    indices are bit-identical);
  * predictions (argmax class + per-qubit marginals) are saved, not only fidelities;
  * two inference-only diagnostics: demux-trained model on shot-aligned rows, and
    per-trace model on demux rows;
  * loguru/pandas are stubbed (oraqle-torch:latest lacks them; logging only / unused);
  * the raw-data module globals are dropped after demodulation to save RAM.
Run inside oraqle-torch:latest with PYTHONPATH=<Oraqle/Discriminators>, cwd = a
fresh work dir (HERQULES_original writes accuracy/ stats/ logs/ into cwd).
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
_sys.path.insert(0, str(LP.DISCRIMINATORS))  # was PYTHONPATH=<Oraqle/Discriminators>
import os, sys, json, time, types, argparse, logging, gc

import numpy as np


# ---------------------------------------------------------------- stubs
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
ap.add_argument("--lengths", default="1000,400,600")
ap.add_argument("--out", default=str(LP.stage("em/e3herq") / "out"))
ap.add_argument("--threads", type=int, default=8)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

import torch as T
import torch.nn as nn
T.set_num_threads(args.threads)

t_imp = time.time()
import trainers.HERQULES_original as H  # module import loads + demodulates train+test
from trainers.HERQULES_original import (
    preclassifier, relaxation_mf_classifier, mf_demux_data_prep,
    Net_rmf, adjust_learning_rate)
from matched_filter import search_matched_filter_for_all_qubits_demux, matched_filter_preprocess_demux
# the pandas stub is only needed while HERQULES_original imports it (unused there);
# drop it so sklearn/narwhals do not mistake it for a real pandas
if getattr(sys.modules.get("pandas"), "__getattr__", None) is not None and not hasattr(sys.modules["pandas"], "DataFrame"):
    del sys.modules["pandas"]
print(f"[e3A] import+demod {time.time()-t_imp:.0f}s TRACE_LENGTH={H.TRACE_LENGTH}", flush=True)

# ---- identical patches to _herqules_length_both.py
H.SUBSET_NUM_TRAIN_VAL = 20000; H.SUBSET_NUM_TEST = 30000           # full
_d = [0.35 if (isinstance(x, float) and abs(x-0.53) < 1e-9) else x
      for x in H.get_train_val_and_test_set.__defaults__]
H.get_train_val_and_test_set.__defaults__ = tuple(_d)
BOX = [1, 1, 9, 2, 9]
DEV = T.device("cuda" if T.cuda.is_available() else "cpu")
NUM_CLASSES = 32

# ---- bookkeeping for raw-row alignment
N_TRAIN_FILE = int(len(H.y_train_raw))
y_test_raw = np.asarray(H.y_test_raw).astype(np.int64)
y_data = np.asarray(H.y_data).astype(np.int64)
for _n in ("all_data", "train_semi_sup_data", "X_train_raw", "X_test_raw"):
    if hasattr(H, _n):
        setattr(H, _n, None)
gc.collect()


def split_globals(TV, TE, ratio):
    """Replicates get_train_val_and_test_set's indexing (SUBSET_SEED=None):
    returns global (train-file ++ test-file) indices of train/val/test, state-major."""
    counts = [int(np.sum(y_data == i)) for i in range(NUM_CLASSES)]
    mn = min(counts)
    assert TV + TE <= mn
    NV = int(TV * ratio); NT = TV - NV
    tr, va, te = [], [], []
    for i in range(NUM_CLASSES):
        ind = np.where(y_data == i)[0]
        tr.append(ind[:NT]); va.append(ind[NT:TV]); te.append(ind[TV:TV+TE])
    return np.concatenate(tr), np.concatenate(va), np.concatenate(te)


def to_rawtest(g):
    """global index -> raw TEST row (X_test of DRaw_C_Te_v0-002), -1 if a train-file row."""
    g = np.asarray(g)
    return np.where(g >= N_TRAIN_FILE, g - N_TRAIN_FILE, -1)


G_TR, G_VA, G_TE = split_globals(20000, 30000, 0.35)
RAW_TE = to_rawtest(G_TE); RAW_VA = to_rawtest(G_VA)
assert (RAW_TE >= 0).all()
assert np.array_equal(y_test_raw[RAW_TE], np.repeat(np.arange(NUM_CLASSES), 30000))
print(f"[e3A] split: train {len(G_TR)} val {len(G_VA)} (of which test-file {int((RAW_VA>=0).sum())}) test {len(G_TE)}", flush=True)

# ---- identical to original: preclassifier + RMF fitted once (unseeded, as in the original)
pre = preclassifier(); pre.fit()
rmf = relaxation_mf_classifier(); rmf.fit(pre.get_traces(), boxcars=BOX)
print("[e3A] preclassifier + RMF fitted", flush=True)


def bits(y): return np.stack([(y >> q) & 1 for q in range(5)], 1).astype(int)


def per_trace_mf(data, envs, S):
    N = NUM_CLASSES * data[0].shape[1]; F = 2 * S
    out = np.zeros((N, 5), np.float32)
    for q in range(5):
        out[:, q] = data[q].reshape(N, F).astype(np.float32) @ envs[q][:F].astype(np.float32)
    return out


def train_net(Xtr, ytr, Xva, yva, seed=1):
    """Identical training loop to train_eval() of the original runner."""
    T.manual_seed(seed); np.random.seed(seed)
    net = Net_rmf().to(DEV); opt = T.optim.Adam(net.parameters(), lr=0.01); crit = nn.CrossEntropyLoss()
    ld = T.utils.data.DataLoader(T.utils.data.TensorDataset(T.tensor(Xtr), T.tensor(ytr)), batch_size=512, shuffle=True)
    Xv = T.tensor(Xva).to(DEV); yvb = bits(yva); best, bs, best_ep = -1, None, -1
    for ep in range(100):
        adjust_learning_rate(0.01, opt, ep); net.train()
        for xb, yb in ld:
            loss = crit(net(xb.to(DEV)), yb.to(DEV)); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with T.no_grad():
            pv = net(Xv).argmax(1).cpu().numpy()
        va = (np.stack([(pv >> q) & 1 for q in range(5)], 1) == yvb).mean()
        if va >= best:
            best = va; best_ep = ep; bs = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(bs); net.eval()
    return net, float(best), best_ep


def infer(net, X):
    cls, p1 = [], []
    states = np.arange(NUM_CLASSES)
    masks = [((states >> q) & 1).astype(bool) for q in range(5)]
    with T.no_grad():
        for s in range(0, len(X), 8192):
            lo = net(T.tensor(X[s:s+8192]).to(DEV)).cpu()
            cls.append(lo.argmax(1).numpy())
            pr = T.softmax(lo.double(), dim=-1).numpy()
            p1.append(np.stack([pr[:, m].sum(1) for m in masks], 1).astype(np.float32))
    return np.concatenate(cls).astype(np.uint8), np.concatenate(p1)


def fid_from_cls(cls, y):
    pb = bits(cls.astype(np.int64)); tb = bits(y)
    fq = [float(1 - ((pb[tb[:, q] == 1, q] == 0).mean() + (pb[tb[:, q] == 0, q] == 1).mean())/2) for q in range(5)]
    g5 = float(np.prod(fq) ** 0.2); g4 = float(np.prod([fq[0], fq[2], fq[3], fq[4]]) ** 0.25)
    return fq, g5, g4


def rng_state_equal(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


summary = {"tag": "e3herq", "runner_replicated": "Oraqle/Discriminators/runners/_herqules_length_both.py",
           "N_TRAIN_FILE": N_TRAIN_FILE, "lengths": {}}
for ns in [int(x) for x in args.lengths.split(",")]:
    t0 = time.time()
    S = ns // 2
    dtr, dval, dte = mf_demux_data_prep(trace_length=S)
    # ---- data-level alignment check: split test row (s,k) == demodulated raw TEST row
    n_te = dte[0].shape[1]
    for q in range(5):
        for s in range(NUM_CLASSES):
            g = G_TE[s*n_te:(s+1)*n_te]
            assert np.array_equal(H._DEMOD_CACHE[q][g, :S, :], dte[q][s]), (q, s)
    print(f"[e3A] {ns}ns data alignment verified for all 5 qubits x 32 states", flush=True)
    mf_env_S, _ = search_matched_filter_for_all_qubits_demux(dtr, best_bc=BOX)
    mf_env_S = [np.asarray(e) for e in mf_env_S]

    rf_all = {}
    def pt(data, dtype):
        mf = per_trace_mf(data, mf_env_S, S)
        rf3 = rmf.predict(data_type=dtype, trace_length=S)          # (32, n_per_state, 5)
        rf_all[dtype] = rf3
        rf = rf3.reshape(mf.shape[0], 5)
        y = np.repeat(np.arange(NUM_CLASSES), data[0].shape[1]).astype(np.int64)
        return np.concatenate([mf, rf], 1).astype(np.float32), y
    Xtr, ytr = pt(dtr, 0); Xva, yva = pt(dval, 1); Xte, yte = pt(dte, 2)
    net_pt, vbest_pt, ep_pt = train_net(Xtr, ytr, Xva, yva)
    pt_te_cls, pt_te_p1 = infer(net_pt, Xte)
    pt_va_cls, pt_va_p1 = infer(net_pt, Xva)
    fq, g5, g4 = fid_from_cls(pt_te_cls, yte)
    print(f"[e3A] {ns:4d}ns pertrace  g5={g5:.4f} g4={g4:.4f} Q={[round(a,4) for a in fq]} ({time.time()-t0:.0f}s)", flush=True)
    res = {"pertrace": {"fq": fq, "g5": g5, "g4": g4, "val_best": vbest_pt, "best_epoch": ep_pt}}

    # ---- demux subsample (identical feature builder + recovered indices)
    def demux_feats_tracked(data, dtype, check):
        st0 = np.random.get_state()
        mf = matched_filter_preprocess_demux(data, mf_env_S, scramble=False)   # (32, min_ind, 5)
        st1 = np.random.get_state()
        np.random.set_state(st0)
        min_ind = min([traces.shape[0] for traces in data[0]])
        shared = [np.random.choice(data[0][s].shape[0], min_ind, replace=False) for s in range(len(data[0]))]
        st2 = np.random.get_state()
        assert rng_state_equal(st1, st2), "RNG replay mismatch"
        np.random.set_state(st1)
        if check:   # bit-exact recomputation of the MF table from the recovered indices
            for q in range(5):
                for s in range(NUM_CLASSES):
                    x = data[q][s][shared[s]]
                    x = x.reshape([x.shape[0], -1])
                    assert np.array_equal(np.sum(x * mf_env_S[q], axis=1), mf[s, :, q]), (q, s)
        rf = rmf.predict(data_type=dtype, trace_length=S)                      # (32, n_per_state, 5)
        assert np.array_equal(rf, rf_all[dtype])
        m = min(mf.shape[1], rf.shape[1])
        feat = np.concatenate([mf[:, :m], rf[:, :m]], 2).reshape(NUM_CLASSES * m, 10).astype(np.float32)
        y = np.repeat(np.arange(NUM_CLASSES), m).astype(np.int64)
        mf_src = np.stack([shared[s][:m] for s in range(NUM_CLASSES)])          # split-local index in state block
        rf_src = np.tile(np.arange(m), (NUM_CLASSES, 1))
        return feat, y, mf_src, rf_src
    # NB: original order = train, val, test, right after train_eval(pertrace) seeded np.random(1)
    np.random.seed(1)   # == state left by train_eval's np.random.seed(1) (training uses no numpy RNG)
    Dtr, dytr, _, _ = demux_feats_tracked(dtr, 0, False)
    Dva, dyva, _, _ = demux_feats_tracked(dval, 1, False)
    Dte, dyte, mf_src, rf_src = demux_feats_tracked(dte, 2, True)
    n = mf_src.shape[1]
    off = (np.arange(NUM_CLASSES) * n)[:, None]
    dm_mf_raw = RAW_TE[(off + mf_src).ravel()]
    dm_rf_raw = RAW_TE[(off + rf_src).ravel()]
    same_shot = float((dm_mf_raw == dm_rf_raw).mean())
    assert np.array_equal(y_test_raw[dm_mf_raw], dyte) and np.array_equal(y_test_raw[dm_rf_raw], dyte)
    net_dm, vbest_dm, ep_dm = train_net(Dtr, dytr, Dva, dyva)
    dm_te_cls, dm_te_p1 = infer(net_dm, Dte)
    fq, g5, g4 = fid_from_cls(dm_te_cls, dyte)
    print(f"[e3A] {ns:4d}ns demux     g5={g5:.4f} g4={g4:.4f} Q={[round(a,4) for a in fq]} "
          f"(MF/RMF same shot in {100*same_shot:.4f}% of rows) ({time.time()-t0:.0f}s)", flush=True)
    res["demux_subsample"] = {"fq": fq, "g5": g5, "g4": g4, "val_best": vbest_dm, "best_epoch": ep_dm,
                              "frac_rows_mf_rmf_same_shot": same_shot}
    # ---- inference-only diagnostics
    a_cls, a_p1 = infer(net_dm, Xte)          # demux model, one physical shot per row
    av_cls, av_p1 = infer(net_dm, Xva)
    b_cls, b_p1 = infer(net_pt, Dte)          # per-trace model, demux (two-shot) rows
    for nm, c, yy in (("demuxmodel_on_singleshot_rows", a_cls, yte), ("pertracemodel_on_demux_rows", b_cls, dyte)):
        fq, g5, g4 = fid_from_cls(c, yy)
        res[nm] = {"fq": fq, "g5": g5, "g4": g4}
        print(f"[e3A] {ns:4d}ns {nm:32s} g5={g5:.4f} g4={g4:.4f} Q={[round(a,4) for a in fq]}", flush=True)
    summary["lengths"][str(ns)] = res
    np.savez_compressed(
        os.path.join(args.out, f"em_e3herq_A_{ns}ns.npz"),
        length_ns=ns, trace_samples=S,
        test_rawrow=RAW_TE.astype(np.int32), test_state=yte.astype(np.uint8),
        val_rawrow=RAW_VA.astype(np.int32), val_state=yva.astype(np.uint8),
        pt_test_cls=pt_te_cls, pt_test_p1=pt_te_p1, pt_val_cls=pt_va_cls, pt_val_p1=pt_va_p1,
        dm_test_cls=dm_te_cls, dm_test_p1=dm_te_p1, dm_test_state=dyte.astype(np.uint8),
        dm_mf_src_rawrow=dm_mf_raw.astype(np.int32), dm_rmf_src_rawrow=dm_rf_raw.astype(np.int32),
        dmmodel_ss_test_cls=a_cls, dmmodel_ss_test_p1=a_p1, dmmodel_ss_val_cls=av_cls, dmmodel_ss_val_p1=av_p1,
        ptmodel_on_dm_test_cls=b_cls, ptmodel_on_dm_test_p1=b_p1,
    )
    T.save({"pertrace": net_pt.state_dict(), "demux": net_dm.state_dict(),
            "mf_env": [e.astype(np.float32) for e in mf_env_S],
            "rmf_env": [np.asarray(e, np.float32) for e in rmf.envelopes]},
           os.path.join(args.out, f"em_e3herq_A_{ns}ns_models.pt"))
    with open(os.path.join(args.out, "em_e3herq_A_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    del dtr, dval, dte, Xtr, Xva, Xte, Dtr, Dva, Dte; gc.collect()
    print(f"[e3A] {ns}ns done in {time.time()-t0:.0f}s", flush=True)
print("[e3A] done", flush=True)
