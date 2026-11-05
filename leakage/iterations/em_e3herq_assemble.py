"""em_e3herq_assemble.py  (leakage study, tag e3herq)

Turn the cluster outputs of em_e3herq_regen_demux.py (A) and
em_e3herq_regen_scrambled.py (B) into per-window TEST-order files
  data/leakage/em_e3herq_preds_{1us,600ns,400ns}.npz
and a summary JSON with per-qubit error rates (Wilson 95% CIs), the comparison
with the paper / source CSVs, and the MCMit-CNN vs HERQULES rate gap.

Usage: python em_e3herq_assemble.py <dir with em_e3herq_{A,B}_*ns.npz> <out dir>
"""
import sys, os, json
import numpy as np

IN, OUT = sys.argv[1], sys.argv[2]
REP = "/home/manosgior/Documents/oraqle_reports"
SOFT = "/home/manosgior/Documents/GitHub/oraqle-ics/data/sims/soft"
N = 1_120_000
WIN = {1000: "1us", 600: "600ns", 400: "400ns"}
Z = 1.959964

# ---- reference numbers
PAPER_PERTRACE = {1000: [0.969, 0.740, 0.924, 0.944, 0.970], 600: [0.950, 0.720, 0.922, 0.932, 0.969],
                  400: [0.906, 0.695, 0.917, 0.899, 0.945]}            # ml_discriminators.tex tab:fidelity
PAPER_DEMUX_F = {1000: (0.925, 0.973), 600: (0.927, 0.972), 400: (0.914, 0.961)}   # appendix tab:herqules-pertrace
CSV_DEMUX = {1000: [0.9794, 0.7556, 0.9631, 0.9639, 0.986], 600: [0.9729, 0.7671, 0.9657, 0.9632, 0.9876],
             400: [0.9573, 0.7482, 0.9597, 0.9504, 0.9769]}           # herqules_length_faithful_vs_demux.csv
CSV_FAKE_ACC = {1000: [98.1919, 75.1277, 96.3055, 96.2051, 98.6751],
                400: [94.3406, 70.4554, 95.4331, 94.6968, 97.2314]}    # herqules_error_breakdown_fake.csv
CSV_FAKE_NERR = {1000: [20251, 278570, 41378, 42503, 14839], 400: [63385, 330899, 51149, 59396, 31008]}
PAPER_CNN = {1000: [0.970, 0.746, 0.942, 0.947, 0.970], 600: [0.953, 0.725, 0.938, 0.937, 0.970],
             400: [0.907, 0.697, 0.926, 0.901, 0.951]}


def bits(c):
    c = np.asarray(c).astype(np.int64)
    return np.stack([(c >> q) & 1 for q in range(5)], 1).astype(np.uint8)


def wilson(k, n):
    if n == 0:
        return [float("nan")] * 3
    p = k / n
    d = 1 + Z**2 / n
    c = (p + Z**2 / (2*n)) / d
    h = Z * np.sqrt(p*(1-p)/n + Z**2/(4*n*n)) / d
    return [100*p, 100*(c-h), 100*(c+h)]


def err_tab(pred, tb, mask=None):
    if mask is not None:
        pred, tb = pred[mask], tb[mask]
    n = len(tb)
    return [wilson(int((pred[:, q] != tb[:, q]).sum()), n) + [n] for q in range(5)]


def fid(pred, tb, mask=None):
    if mask is not None:
        pred, tb = pred[mask], tb[mask]
    fq = [float(1 - ((pred[tb[:, q] == 1, q] == 0).mean() + (pred[tb[:, q] == 0, q] == 1).mean())/2) for q in range(5)]
    return fq, float(np.prod(fq)**0.2), float(np.prod([fq[i] for i in (0, 2, 3, 4)])**0.25)


def ratio_ci(k1, n1, k2, n2):
    """rate1/rate2 with a delta-method CI on the log ratio (independent binomials)."""
    r1, r2 = k1/n1, k2/n2
    se = np.sqrt((1-r1)/k1 + (1-r2)/k2)
    R = r1/r2
    return [float(R), float(R*np.exp(-Z*se)), float(R*np.exp(Z*se))]


def paired_ratio_ci(e1, e2, B=400, seed=0):
    """rate(e1)/rate(e2) on the SAME shots; multinomial bootstrap over the 2x2 table."""
    a = int((e1 & e2).sum()); b = int((e1 & ~e2).sum()); c = int((~e1 & e2).sum()); n = len(e1)
    d = n - a - b - c
    rng = np.random.default_rng(seed)
    m = rng.multinomial(n, np.array([a, b, c, d]) / n, size=B)
    R = (m[:, 0] + m[:, 1]) / (m[:, 0] + m[:, 2])
    return [float((a+b)/(a+c)), float(np.quantile(R, 0.025)), float(np.quantile(R, 0.975))]


# ---- TEST labels (raw order) from the CNN soft file, cross-checked below against the cluster arrays
lab = np.load(f"{SOFT}/cnn_soft_1000ns.npz")["label"].astype(np.uint8)
y_state = (lab.astype(np.int64) * (1 << np.arange(5))).sum(1)
cnn_pred = {1000: np.load(f"{REP}/preds_MCMitCNN_pb.npy").astype(np.uint8)}
assert np.array_equal(np.load(f"{REP}/preds_MCMitCNN_tb.npy").astype(np.uint8), lab)
for ns in (600, 400):
    d = np.load(f"{SOFT}/cnn_soft_{ns}ns.npz")
    assert np.array_equal(d["label"].astype(np.uint8), lab)
    cnn_pred[ns] = (d["prob"].astype(np.float32) > 0.5).astype(np.uint8)
cnn_soft1000 = (np.load(f"{SOFT}/cnn_soft_1000ns.npz")["prob"].astype(np.float32) > 0.5).astype(np.uint8)

summary = {"tag": "e3herq", "N_test": N, "windows": {}}
for ns, w in WIN.items():
    fA = os.path.join(IN, f"em_e3herq_A_{ns}ns.npz")
    if not os.path.exists(fA):
        print("missing", fA); continue
    A = np.load(fA)
    S = {"length_ns": ns, "trace_samples": ns // 2}
    raw_te, raw_va = A["test_rawrow"].astype(np.int64), A["val_rawrow"].astype(np.int64)
    assert np.array_equal(y_state[raw_te], A["test_state"]), "test rows misaligned"
    vv = raw_va >= 0
    assert np.array_equal(y_state[raw_va[vv]], A["val_state"][vv]), "val rows misaligned"
    in_test = np.zeros(N, bool); in_test[raw_te] = True
    in_val = np.zeros(N, bool); in_val[raw_va[vv]] = True
    assert not (in_test & in_val).any() and (in_test | in_val).all()
    out = {"length_ns": ns, "trace_samples": ns // 2, "true_bits": lab, "y_state": y_state.astype(np.uint8),
           "herq_test_mask": in_test}

    def scatter_ss(cls_te, p1_te, cls_va, p1_va):
        pred = np.full((N, 5), 255, np.uint8); p1 = np.full((N, 5), np.nan, np.float16)
        pred[raw_te] = bits(cls_te); p1[raw_te] = p1_te
        pred[raw_va[vv]] = bits(cls_va[vv]); p1[raw_va[vv]] = p1_va[vv]
        return pred, p1
    out["pertrace_pred"], out["pertrace_p1"] = scatter_ss(A["pt_test_cls"], A["pt_test_p1"], A["pt_val_cls"], A["pt_val_p1"])
    out["demuxmodel_1shot_pred"], out["demuxmodel_1shot_p1"] = scatter_ss(
        A["dmmodel_ss_test_cls"], A["dmmodel_ss_test_p1"], A["dmmodel_ss_val_cls"], A["dmmodel_ss_val_p1"])
    # demux (paper App. config): every row mixes two shots; attribute to the MF-source row
    mf_src, rf_src = A["dm_mf_src_rawrow"].astype(np.int64), A["dm_rmf_src_rawrow"].astype(np.int64)
    assert len(np.unique(mf_src)) == len(mf_src) and set(np.unique(mf_src)) == set(raw_te)
    assert np.array_equal(y_state[mf_src], A["dm_test_state"]) and np.array_equal(y_state[rf_src], A["dm_test_state"])
    dp = np.full((N, 5), 255, np.uint8); dp1 = np.full((N, 5), np.nan, np.float16); drs = np.full(N, -1, np.int32)
    dp[mf_src] = bits(A["dm_test_cls"]); dp1[mf_src] = A["dm_test_p1"]; drs[mf_src] = rf_src
    out["demux_pred_by_mfsrc"], out["demux_p1_by_mfsrc"], out["demux_rmf_src_of_mfsrc"] = dp, dp1, drs
    S["demux_frac_rows_mf_eq_rmf_shot"] = float((mf_src == rf_src).mean())

    tb_te = lab[in_test]
    E = {}
    E["pertrace"] = (out["pertrace_pred"], in_test)
    E["demux_paper_config_2shot"] = (dp, in_test)
    E["demuxmodel_1shot"] = (out["demuxmodel_1shot_pred"], in_test)
    S["fidelity"] = {}
    for k, (p, m) in E.items():
        fq, g5, g4 = fid(p, lab, m)
        S["fidelity"][k] = {"fq": fq, "F5Q": g5, "F4Q": g4}
    S["fidelity"]["pertracemodel_on_demux_rows"] = dict(zip(("fq", "F5Q", "F4Q"), fid(
        bits(A["ptmodel_on_dm_test_cls"]), bits(A["dm_test_state"]))))
    S["paper_pertrace_fq"] = PAPER_PERTRACE[ns]
    S["paper_demux_F5Q_F4Q"] = PAPER_DEMUX_F[ns]
    S["csv_demux_fq"] = CSV_DEMUX[ns]
    S["paper_cnn_fq"] = PAPER_CNN[ns]

    # ---- B (scrambled "fake" CSV config), full TEST, identity order
    fB = os.path.join(IN, f"em_e3herq_B_{ns}ns.npz")
    if os.path.exists(fB):
        B = np.load(fB)
        assert np.array_equal(B["test_rawrow"], np.arange(N)) and np.array_equal(B["test_state"], y_state)
        sp = (B["scr_p1"] > 0.5).astype(np.uint8)          # analyze's per-qubit decision
        msrc = B["scr_mf_src_rawrow"].astype(np.int64)
        spm = np.zeros((N, 5), np.uint8); sp1 = np.zeros((N, 5), np.float16); srs = np.zeros((N, 5), np.int32)
        for q in range(5):
            assert len(np.unique(msrc[:, q])) == N and np.array_equal(y_state[msrc[:, q]], y_state)
            spm[msrc[:, q], q] = sp[:, q]; sp1[msrc[:, q], q] = B["scr_p1"][:, q]; srs[msrc[:, q], q] = np.arange(N)
        out["scr_pred_by_mfsrc"], out["scr_p1_by_mfsrc"], out["scr_rmf_src_of_mfsrc"] = spm, sp1, srs
        out["scrmodel_1shot_pred"] = (B["ss_p1"] > 0.5).astype(np.uint8)
        out["scrmodel_1shot_p1"] = B["ss_p1"].astype(np.float16)
        E["scrambled_fake_2shot"] = (spm, None)
        E["scrmodel_1shot"] = (out["scrmodel_1shot_pred"], None)
        nerr = [int((sp[:, q] != bits(B["test_state"])[:, q]).sum()) for q in range(5)]
        S["scrambled_n_error"] = nerr
        if ns in CSV_FAKE_NERR:
            S["csv_fake_n_error"] = CSV_FAKE_NERR[ns]; S["csv_fake_acc_pct"] = CSV_FAKE_ACC[ns]
        for k in ("scrambled_fake_2shot", "scrmodel_1shot"):
            fq, g5, g4 = fid(E[k][0], lab)
            S["fidelity"][k] = {"fq": fq, "F5Q": g5, "F4Q": g4}
    # existing per-trace predictions (1 us only)
    if ns == 1000:
        pb = np.load(f"{REP}/preds_HERQULESdeploy_pb.npy").astype(np.uint8)
        tbx = np.load(f"{REP}/preds_HERQULESdeploy_tb.npy").astype(np.uint8)
        m = np.concatenate([np.where(y_state == s)[0][5000:35000] for s in range(32)])
        assert np.array_equal(lab[m], tbx) and np.array_equal(np.sort(m), np.sort(raw_te))
        ex = np.full((N, 5), 255, np.uint8); ex[m] = pb
        out["existing_deploy_pred"] = ex
        E["existing_deploy_pertrace"] = (ex, in_test)
        fq, g5, g4 = fid(ex, lab, in_test)
        S["fidelity"]["existing_deploy_pertrace"] = {"fq": fq, "F5Q": g5, "F4Q": g4}
        S["agree_regen_vs_existing_pertrace"] = [float((ex[in_test, q] == out["pertrace_pred"][in_test, q]).mean()) for q in range(5)]

    # ---- error rates + rate gap (CNN / HERQULES), matched rows
    cp = cnn_pred[ns]
    S["cnn_err_full_test"] = err_tab(cp, lab)
    S["cnn_err_on_herq_test_rows"] = err_tab(cp, lab, in_test)
    S["cnn_fq_full_test"] = fid(cp, lab)[0]
    if ns == 1000:
        S["cnn_soft1000_vs_canonical_agree"] = [float((cnn_soft1000[:, q] == cp[:, q]).mean()) for q in range(5)]
    S["herq_err"] = {}; S["rate_gap_cnn_over_herq"] = {}
    for k, (p, m) in E.items():
        mm = np.ones(N, bool) if m is None else m
        S["herq_err"][k] = err_tab(p, lab, mm)
        gaps = []
        for q in range(5):
            eh = p[mm, q] != lab[mm, q]; ec = cp[mm, q] != lab[mm, q]
            if k.endswith("2shot"):        # two-shot rows: no shot pairing -> independent-binomial CI
                gaps.append(ratio_ci(int(ec.sum()), int(mm.sum()), int(eh.sum()), int(mm.sum())))
            else:
                gaps.append(paired_ratio_ci(ec, eh))
        S["rate_gap_cnn_over_herq"][k] = gaps
    summary["windows"][w] = S
    np.savez_compressed(os.path.join(OUT, f"em_e3herq_preds_{w}.npz"), **out)
    print(f"[asm] wrote em_e3herq_preds_{w}.npz", flush=True)

with open(os.path.join(OUT, "em_e3herq_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print("[asm] done")
