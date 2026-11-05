"""em_e3herq_assemble_v2.py  (leakage study, tag e3herq)

v2 of em_e3herq_assemble.py (kept unchanged): adds the inference-only "C" outputs
(exact per-shot predictions of the per-trace HERQULES checkpoints behind
herqules_error_breakdown_realistic.csv), defines a primary prediction per window,
and adds the HybridCNN (paper tab:error_modes source) rate gaps.

Inputs (copied from graham ~/oraqle-ics-sims/leakage/em/e3herq/out/):
  em_e3herq_A_{1000,600,400}ns.npz  em_e3herq_regen_demux.py         (retrained; _herqules_length_both.py split)
  em_e3herq_B_{1000,600,400}ns.npz  em_e3herq_regen_scrambled.py     (retrained; "fake"/scrambled CSV config)
  em_e3herq_C_{1000,400}ns.npz      em_e3herq_regen_pertrace_ckpt.py (inference; "realistic" CSV config)
Outputs: <out>/em_e3herq_preds_{1us,600ns,400ns}.npz, <out>/em_e3herq_summary_v2.json

Every array in a preds file has N = 1,120,000 rows in raw TEST order
(X_test of DRaw_C_Te_v0-002); true_bits[:, q] = (y >> q) & 1 = code qubit q = paper Q(q+1).
Usage: python em_e3herq_assemble_v2.py <in dir> <out dir>
"""
import sys, os, json
import numpy as np

IN, OUT = sys.argv[1], sys.argv[2]
REP = "/home/manosgior/Documents/oraqle_reports"
SOFT = "/home/manosgior/Documents/GitHub/oraqle-ics/data/sims/soft"
N = 1_120_000
WIN = {1000: "1us", 600: "600ns", 400: "400ns"}
Z = 1.959964

# ---- reference numbers (paper / source CSVs)
PAPER_PERTRACE = {1000: [0.969, 0.740, 0.924, 0.944, 0.970], 600: [0.950, 0.720, 0.922, 0.932, 0.969],
                  400: [0.906, 0.695, 0.917, 0.899, 0.945]}     # chapters/ml_discriminators.tex tab:fidelity (l.158-162)
PAPER_PERTRACE_F = {1000: (0.905, 0.951), 600: (0.894, 0.943), 400: (0.867, 0.917)}   # appendix_herqules.tex tab:herqules-pertrace
PAPER_DEMUX_F = {1000: (0.925, 0.973), 600: (0.927, 0.972), 400: (0.914, 0.961)}      # appendix_herqules.tex tab:herqules-pertrace "As released"
CSV_DEMUX = {1000: [0.9794, 0.7556, 0.9631, 0.9639, 0.9860], 600: [0.9729, 0.7671, 0.9657, 0.9632, 0.9876],
             400: [0.9573, 0.7482, 0.9597, 0.9504, 0.9769]}    # herqules_length_faithful_vs_demux.csv demux_subsample
CSV_FAKE_NERR = {1000: [20251, 278570, 41378, 42503, 14839], 400: [63385, 330899, 51149, 59396, 31008]}       # herqules_error_breakdown_fake.csv
CSV_REAL_NERR = {1000: [35236, 291943, 85129, 65643, 34149], 400: [105646, 340980, 93894, 113619, 61656]}     # herqules_error_breakdown_realistic.csv
CSV_REAL_MODES = {  # (t1, amp, near) exclusive counts, herqules_error_breakdown_realistic.csv
    1000: [(1, 0, 35235), (9786, 641, 281516), (0, 162, 84967), (721, 18, 64904), (2719, 0, 31430)],
    400: [(30, 35, 105581), (40170, 832, 299978), (0, 76, 93818), (36, 87, 113496), (2368, 11, 59277)]}
CSV_HYB_NERR = {1000: [35908, 288485, 68811, 63026, 34060], 400: [120291, 346773, 93465, 125567, 71123]}     # herqules_error_breakdown_hybridcnn.csv
PAPER_CNN = {1000: [0.970, 0.746, 0.942, 0.947, 0.970], 600: [0.953, 0.725, 0.938, 0.937, 0.970],
             400: [0.907, 0.697, 0.926, 0.901, 0.951]}


def bits(c):
    c = np.asarray(c).astype(np.int64)
    return np.stack([(c >> q) & 1 for q in range(5)], 1).astype(np.uint8)


def wilson(k, n):
    p = k / n
    d = 1 + Z**2 / n
    c = (p + Z**2 / (2*n)) / d
    h = Z * np.sqrt(p*(1-p)/n + Z**2/(4*n*n)) / d
    return [float(100*p), float(100*(c-h)), float(100*(c+h))]


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
    """rate1/rate2, delta-method CI on log ratio (independent binomials)."""
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


# ---- TEST labels (raw order), MCMit-CNN predictions
lab = np.load(f"{SOFT}/cnn_soft_1000ns.npz")["label"].astype(np.uint8)
y_state = (lab.astype(np.int64) * (1 << np.arange(5))).sum(1)
cnn_pred = {1000: np.load(f"{REP}/preds_MCMitCNN_pb.npy").astype(np.uint8)}
assert np.array_equal(np.load(f"{REP}/preds_MCMitCNN_tb.npy").astype(np.uint8), lab)
cnn_src = {1000: "oraqle_reports/preds_MCMitCNN_pb.npy (canonical)"}
for ns in (600, 400):
    d = np.load(f"{SOFT}/cnn_soft_{ns}ns.npz")
    assert np.array_equal(d["label"].astype(np.uint8), lab)
    cnn_pred[ns] = (d["prob"].astype(np.float32) > 0.5).astype(np.uint8)
    cnn_src[ns] = f"data/sims/soft/cnn_soft_{ns}ns.npz prob>0.5"

summary = {"tag": "e3herq", "N_test": N, "windows": {},
           "category_codes_C": {"correct": 0, "near_boundary": 1, "t1_pattern": 2, "amp_outlier": 3, "other": 4}}
for ns, w in WIN.items():
    S = {"length_ns": ns, "trace_samples": ns // 2, "cnn_source": cnn_src[ns]}
    out = {"length_ns": np.int64(ns), "trace_samples": np.int64(ns // 2), "true_bits": lab, "y_state": y_state.astype(np.uint8)}
    E = {}           # name -> (pred (N,5) uint8, mask or None, two_shot flag)
    S["fidelity"] = {}

    # ---------------- B: scrambled ("fake"; paper-text "demux-pipeline"), full TEST
    fB = os.path.join(IN, f"em_e3herq_B_{ns}ns.npz")
    if os.path.exists(fB):
        B = np.load(fB)
        assert np.array_equal(B["test_rawrow"], np.arange(N)) and np.array_equal(B["test_state"], y_state)
        sp = (B["scr_p1"] > 0.5).astype(np.uint8)             # analyze's per-qubit decision
        msrc = B["scr_mf_src_rawrow"].astype(np.int64)
        assert np.array_equal(B["scr_rmf_src_rawrow"], np.arange(N))
        spm = np.zeros((N, 5), np.uint8); sp1 = np.zeros((N, 5), np.float32); srs = np.zeros((N, 5), np.int32)
        for q in range(5):
            assert len(np.unique(msrc[:, q])) == N and np.array_equal(y_state[msrc[:, q]], y_state)
            spm[msrc[:, q], q] = sp[:, q]; sp1[msrc[:, q], q] = B["scr_p1"][:, q]; srs[msrc[:, q], q] = np.arange(N)
        S["scr_frac_mf_eq_rmf_shot"] = [float((msrc[:, q] == np.arange(N)).mean()) for q in range(5)]
        S["scr_frac_rows_all5_mf_same_shot"] = float(np.all(msrc == msrc[:, :1], axis=1).mean())
        out["scr_pred_by_mfsrc"], out["scr_p1_by_mfsrc"], out["scr_rmf_src_of_mfsrc"] = spm, sp1.astype(np.float16), srs
        out["scr_row_pred"], out["scr_row_p1"], out["scr_row_mf_src"] = sp, B["scr_p1"].astype(np.float16), msrc.astype(np.int32)
        out["scrmodel_1shot_pred"] = (B["ss_p1"] > 0.5).astype(np.uint8)
        out["scrmodel_1shot_p1"] = B["ss_p1"].astype(np.float16)
        E["scrambled_fake_2shot"] = (spm, None, True)
        E["scrmodel_1shot"] = (out["scrmodel_1shot_pred"], None, False)
        S["scrambled_n_error_rowwise"] = [int((sp[:, q] != lab[:, q]).sum()) for q in range(5)]
        assert S["scrambled_n_error_rowwise"] == [int((spm[:, q] != lab[:, q]).sum()) for q in range(5)]
        if ns in CSV_FAKE_NERR:
            S["csv_fake_n_error"] = CSV_FAKE_NERR[ns]
            S["scrambled_minus_csv_pp"] = [100*(a-b)/N for a, b in zip(S["scrambled_n_error_rowwise"], CSV_FAKE_NERR[ns])]

    # ---------------- C: per-trace checkpoints of the realistic CSV (inference only), full TEST
    fC = os.path.join(IN, f"em_e3herq_C_{ns}ns.npz")
    if os.path.exists(fC):
        C = np.load(fC)
        assert np.array_equal(C["test_rawrow"], np.arange(N)) and np.array_equal(C["test_state"], y_state)
        cp_ = (C["p1"] > 0.5).astype(np.uint8)
        out["pertrace_ckpt_pred"], out["pertrace_ckpt_p1"] = cp_, C["p1"].astype(np.float32)
        out["pertrace_ckpt_cls"] = C["cls"].astype(np.uint8)
        out["pertrace_ckpt_category"] = C["category"].astype(np.uint8)
        E["pertrace_ckpt"] = (cp_, None, False)
        nerr = [int((cp_[:, q] != lab[:, q]).sum()) for q in range(5)]
        cat = C["category"]
        modes = [(int((cat[:, q] == 2).sum()), int((cat[:, q] == 3).sum()), int((cat[:, q] == 1).sum())) for q in range(5)]
        S["pertrace_ckpt_n_error"] = nerr; S["pertrace_ckpt_modes_t1_amp_near"] = modes
        S["csv_realistic_n_error"] = CSV_REAL_NERR[ns]; S["csv_realistic_modes_t1_amp_near"] = CSV_REAL_MODES[ns]
        S["pertrace_ckpt_exact_match_csv"] = bool(nerr == CSV_REAL_NERR[ns] and [list(m) for m in modes] == [list(m) for m in CSV_REAL_MODES[ns]])

    # ---------------- A: _herqules_length_both.py split (test = per-state raw TEST rows 5000..34999)
    fA = os.path.join(IN, f"em_e3herq_A_{ns}ns.npz")
    if os.path.exists(fA):
        A = np.load(fA)
        raw_te, raw_va = A["test_rawrow"].astype(np.int64), A["val_rawrow"].astype(np.int64)
        assert np.array_equal(y_state[raw_te], A["test_state"])
        vv = raw_va >= 0
        assert np.array_equal(y_state[raw_va[vv]], A["val_state"][vv])
        in_test = np.zeros(N, bool); in_test[raw_te] = True
        in_val = np.zeros(N, bool); in_val[raw_va[vv]] = True
        assert not (in_test & in_val).any() and (in_test | in_val).all()
        out["A_test_mask"] = in_test

        def scatter_ss(cls_te, p1_te, cls_va, p1_va):
            pred = np.full((N, 5), 255, np.uint8); p1 = np.full((N, 5), np.nan, np.float16)
            pred[raw_te] = bits(cls_te); p1[raw_te] = p1_te
            pred[raw_va[vv]] = bits(cls_va[vv]); p1[raw_va[vv]] = p1_va[vv]
            return pred, p1
        out["A_pertrace_pred"], out["A_pertrace_p1"] = scatter_ss(A["pt_test_cls"], A["pt_test_p1"], A["pt_val_cls"], A["pt_val_p1"])
        out["A_demuxmodel_1shot_pred"], out["A_demuxmodel_1shot_p1"] = scatter_ss(
            A["dmmodel_ss_test_cls"], A["dmmodel_ss_test_p1"], A["dmmodel_ss_val_cls"], A["dmmodel_ss_val_p1"])
        mf_src, rf_src = A["dm_mf_src_rawrow"].astype(np.int64), A["dm_rmf_src_rawrow"].astype(np.int64)
        assert len(np.unique(mf_src)) == len(mf_src) and np.array_equal(np.sort(mf_src), np.sort(raw_te))
        assert np.array_equal(y_state[mf_src], A["dm_test_state"]) and np.array_equal(y_state[rf_src], A["dm_test_state"])
        dp = np.full((N, 5), 255, np.uint8); dp1 = np.full((N, 5), np.nan, np.float16); drs = np.full(N, -1, np.int32)
        dp[mf_src] = bits(A["dm_test_cls"]); dp1[mf_src] = A["dm_test_p1"]; drs[mf_src] = rf_src
        out["A_demux_pred_by_mfsrc"], out["A_demux_p1_by_mfsrc"], out["A_demux_rmf_src_of_mfsrc"] = dp, dp1, drs
        S["A_demux_frac_rows_mf_eq_rmf_shot"] = float((mf_src == rf_src).mean())
        E["A_pertrace"] = (out["A_pertrace_pred"], in_test, False)
        E["A_demux_asreleased_2shot"] = (dp, in_test, True)
        E["A_demuxmodel_1shot"] = (out["A_demuxmodel_1shot_pred"], in_test, False)
        S["fidelity"]["A_pertracemodel_on_demux_rows"] = dict(zip(("fq", "F5Q", "F4Q"), fid(
            bits(A["ptmodel_on_dm_test_cls"]), bits(A["dm_test_state"]))))
        S["paper_pertrace_fq"] = PAPER_PERTRACE[ns]; S["paper_pertrace_F5Q_F4Q"] = PAPER_PERTRACE_F[ns]
        S["paper_demux_F5Q_F4Q"] = PAPER_DEMUX_F[ns]; S["csv_demux_fq"] = CSV_DEMUX[ns]
    S["paper_cnn_fq"] = PAPER_CNN[ns]

    # ---------------- existing per-trace deployment predictions (1 us only)
    if ns == 1000:
        pb = np.load(f"{REP}/preds_HERQULESdeploy_pb.npy").astype(np.uint8)
        tbx = np.load(f"{REP}/preds_HERQULESdeploy_tb.npy").astype(np.uint8)
        m = np.concatenate([np.where(y_state == s)[0][5000:35000] for s in range(32)])
        assert np.array_equal(lab[m], tbx)
        ex = np.full((N, 5), 255, np.uint8); ex[m] = pb
        exm = np.zeros(N, bool); exm[m] = True
        out["existing_deploy_pred"] = ex
        E["existing_deploy_pertrace"] = (ex, exm, False)
        if "A_test_mask" in out:
            assert np.array_equal(exm, out["A_test_mask"])
            S["agree_A_pertrace_vs_existing_deploy"] = [float((ex[exm, q] == out["A_pertrace_pred"][exm, q]).mean()) for q in range(5)]

    # ---------------- primary prediction = paper-text config ("demux-pipeline" = scrambled/fake CSV)
    if "scr_pred_by_mfsrc" in out:
        out["pred_bits"], out["p1"] = out["scr_pred_by_mfsrc"], out["scr_p1_by_mfsrc"]
        out["primary_config"] = np.array("scrambled_fake_2shot: H.train(scramble=True)+analyze(scrambled=True) "
                                         "regenerated; row r = MF-source shot of qubit q (as analyze attributes it); "
                                         "RMF features come from shot scr_rmf_src_of_mfsrc[r,q]")

    # ---------------- fidelities, error rates, rate gaps
    cp = cnn_pred[ns]
    S["cnn_err_full_test"] = err_tab(cp, lab)
    S["cnn_fq_full_test"] = fid(cp, lab)[0]
    S["herq_err"], S["rate_gap_cnn_over_herq"], S["herq_err_rate_pct"] = {}, {}, {}
    for k, (p, msk, two) in E.items():
        mm = np.ones(N, bool) if msk is None else msk
        fq, g5, g4 = fid(p, lab, mm)
        S["fidelity"][k] = {"fq": fq, "F5Q": g5, "F4Q": g4, "n_rows": int(mm.sum())}
        S["herq_err"][k] = err_tab(p, lab, mm)
        S["herq_err_rate_pct"][k] = [round(S["herq_err"][k][q][0], 3) for q in range(5)]
        gaps = []
        for q in range(5):
            eh = p[mm, q] != lab[mm, q]; ec = cp[mm, q] != lab[mm, q]
            gaps.append(ratio_ci(int(ec.sum()), int(mm.sum()), int(eh.sum()), int(mm.sum())) if two
                        else paired_ratio_ci(ec, eh))
        S["rate_gap_cnn_over_herq"][k] = gaps
        if msk is not None:
            S.setdefault("cnn_err_on_rows", {})[k] = err_tab(cp, lab, mm)
    # HybridCNN (tab:error_modes / cut-figure CNN panel; aggregate counts only)
    if ns in CSV_HYB_NERR:
        S["rate_gap_hybridcnn_over_herq_aggregate"] = {}
        for k, ref in (("scrambled_fake_csv", CSV_FAKE_NERR[ns]), ("pertrace_realistic_csv", CSV_REAL_NERR[ns])):
            S["rate_gap_hybridcnn_over_herq_aggregate"][k] = [ratio_ci(CSV_HYB_NERR[ns][q], N, ref[q], N) for q in range(5)]
        S["hybridcnn_err_rate_pct"] = [100*x/N for x in CSV_HYB_NERR[ns]]
    summary["windows"][w] = S
    np.savez_compressed(os.path.join(OUT, f"em_e3herq_preds_{w}.npz"), **out)
    print(f"[asm2] wrote em_e3herq_preds_{w}.npz keys={sorted(out)}", flush=True)
    for k in E:
        print(f"[asm2] {w} {k:28s} err% {S['herq_err_rate_pct'][k]}  gap CNN/HERQ "
              f"{[round(g[0], 2) for g in S['rate_gap_cnn_over_herq'][k]]}", flush=True)

with open(os.path.join(OUT, "em_e3herq_summary_v2.json"), "w") as f:
    json.dump(summary, f, indent=1)
print("[asm2] done")
