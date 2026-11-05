"""em_e3herq_assemble_v3.py  (leakage study, tag e3herq)

v3 of em_e3herq_assemble{,_v2}.py (both kept unchanged). Builds the per-window
HERQULES per-shot prediction files used for the failure-analysis cross-tabs.

Inputs (copies of graham ~/oraqle-ics-sims/leakage/em/e3herq/out/):
  em_e3herq_B_{1000,600,400}ns.npz  <- em_e3herq_regen_scrambled.py
      retrained through the code path behind herqules_error_breakdown_fake.csv
      (_train_and_analyze_errors.py, scramble=True: H.train(scramble=True) +
      analyze_herqules_errors.run(scrambled=True)). This is the paper-text
      "demux-pipeline HERQULES ... fewest errors of any configuration we logged".
  em_e3herq_C_{1000,400}ns.npz      <- em_e3herq_regen_pertrace_ckpt.py
      inference with the saved checkpoints behind herqules_error_breakdown_realistic.csv
      = panel "(b) HERQULES (Per-Trace)" of plots/cnn_vs_herqules_error_breakdown.pdf.
  em_e3herq_A_{1000,600,400}ns.npz  <- em_e3herq_regen_demux.py
      retrained through _herqules_length_both.py (per-trace + demux_subsample) on its
      own split (test = raw TEST rows 5000..34999 of each state, 960,000 rows).
Outputs: <out>/em_e3herq_preds_{1us,600ns,400ns}.npz, <out>/em_e3herq_summary_v3.json

Every per-shot array has N = 1,120,000 rows in raw TEST order (X_test of
DRaw_C_Te_v0-002). true_bits[:, q] = (y_test >> q) & 1 = code qubit q = paper Q(q+1).
Rows not covered by a configuration hold 255 (pred) / NaN (prob).

Primary keys
  pred_bits, p1            : demux-pipeline ("fake"/scrambled) HERQULES, row r = shot whose
                             qubit-q MF score produced the decision (analyze's attribution).
                             CAVEAT: that decision also used the RMF features of a DIFFERENT
                             shot (scr_rmf_src_of_mfsrc) and the other qubits' MF scores of yet
                             other shots -> not a single-shot prediction.
  pred_bits_pertrace, p1_pertrace (1us, 400ns only): exact single-shot predictions of the
                             per-trace checkpoints (cut-figure panel b).
Usage: python em_e3herq_assemble_v3.py <in dir> <out dir>
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import sys, os, json
import numpy as np

IN, OUT = sys.argv[1], LP.outdir(sys.argv[2])  # Oraqle unify: create the output dir
REP = str(LP.REPORTS)
SOFT = str(LP.SOFT)
N = 1_120_000
WIN = {1000: "1us", 600: "600ns", 400: "400ns"}
Z = 1.959964

# ---------------- reference numbers
# paper tab:fidelity HERQULES rows (chapters/ml_discriminators.tex l.158-162), = per-trace
PAPER_PERTRACE = {1000: [0.969, 0.740, 0.924, 0.944, 0.970], 600: [0.950, 0.720, 0.922, 0.932, 0.969],
                  400: [0.906, 0.695, 0.917, 0.899, 0.945]}
# appendix_herqules.tex tab:herqules-pertrace (l.133-141): (F5Q,F4Q) per-trace | as released (demux)
PAPER_PERTRACE_F = {1000: (0.905, 0.951), 600: (0.894, 0.943), 400: (0.867, 0.917)}
PAPER_DEMUX_F = {1000: (0.925, 0.973), 600: (0.927, 0.972), 400: (0.914, 0.961)}
PAPER_CNN = {1000: [0.970, 0.746, 0.942, 0.947, 0.970], 600: [0.953, 0.725, 0.938, 0.937, 0.970],
             400: [0.907, 0.697, 0.926, 0.901, 0.951]}
# oraqle_reports/herqules_length_faithful_vs_demux.csv (source of the two paper tables above)
CSV_PERTRACE = {1000: [0.9686, 0.7398, 0.9237, 0.9437, 0.9696], 600: [0.9501, 0.7204, 0.9223, 0.9324, 0.9685],
                400: [0.9058, 0.6953, 0.9171, 0.8993, 0.945]}
CSV_DEMUX = {1000: [0.9794, 0.7556, 0.9631, 0.9639, 0.9860], 600: [0.9729, 0.7671, 0.9657, 0.9632, 0.9876],
             400: [0.9573, 0.7482, 0.9597, 0.9504, 0.9769]}
# oraqle_reports/herqules_error_breakdown_{fake,realistic,hybridcnn}.csv  n_error of 1,120,000
CSV_FAKE_NERR = {1000: [20251, 278570, 41378, 42503, 14839], 400: [63385, 330899, 51149, 59396, 31008]}
CSV_REAL_NERR = {1000: [35236, 291943, 85129, 65643, 34149], 400: [105646, 340980, 93894, 113619, 61656]}
CSV_REAL_MODES = {1000: [(1, 0, 35235), (9786, 641, 281516), (0, 162, 84967), (721, 18, 64904), (2719, 0, 31430)],
                  400: [(30, 35, 105581), (40170, 832, 299978), (0, 76, 93818), (36, 87, 113496), (2368, 11, 59277)]}
CSV_HYB_NERR = {1000: [35908, 288485, 68811, 63026, 34060], 400: [120291, 346773, 93465, 125567, 71123]}


def bits(c):
    c = np.asarray(c).astype(np.int64)
    return np.stack([(c >> q) & 1 for q in range(5)], 1).astype(np.uint8)


def wilson(k, n):
    p = k / n
    d = 1 + Z**2 / n
    c = (p + Z**2 / (2*n)) / d
    h = Z * np.sqrt(p*(1-p)/n + Z**2/(4*n*n)) / d
    return [round(100*p, 4), round(100*(c-h), 4), round(100*(c+h), 4)]


def fid(pred, tb):
    fq = [float(1 - ((pred[tb[:, q] == 1, q] == 0).mean() + (pred[tb[:, q] == 0, q] == 1).mean())/2) for q in range(5)]
    return [round(f, 4) for f in fq], round(float(np.prod(fq)**0.2), 4), round(float(np.prod([fq[i] for i in (0, 2, 3, 4)])**0.25), 4)


def paired_ratio_ci(e1, e2, B=400, seed=0):
    """rate(e1)/rate(e2) on the same rows; multinomial bootstrap over the 2x2 table."""
    a = int((e1 & e2).sum()); b = int((e1 & ~e2).sum()); c = int((~e1 & e2).sum()); n = len(e1)
    d = n - a - b - c
    m = np.random.default_rng(seed).multinomial(n, np.array([a, b, c, d]) / n, size=B)
    R = (m[:, 0] + m[:, 1]) / (m[:, 0] + m[:, 2])
    return [round((a+b)/(a+c), 3), round(float(np.quantile(R, 0.025)), 3), round(float(np.quantile(R, 0.975)), 3)]


def indep_ratio_ci(k1, n1, k2, n2):
    r1, r2 = k1/n1, k2/n2
    se = np.sqrt((1-r1)/k1 + (1-r2)/k2)
    R = r1/r2
    return [round(R, 3), round(R*np.exp(-Z*se), 3), round(R*np.exp(Z*se), 3)]


def overlap(eh, ec):
    """co-failure: lift = P(both)/(P(h)P(c)); frac of HERQ errors that are also CNN errors (Wilson)."""
    nb, nh, nc, n = int((eh & ec).sum()), int(eh.sum()), int(ec.sum()), len(eh)
    return {"lift": round(nb * n / (nh * nc), 3), "frac_herq_err_also_cnn_err_pct": wilson(nb, nh),
            "frac_cnn_err_also_herq_err_pct": wilson(nb, nc)}


# ---------------- TEST labels (raw order) and MCMit-CNN predictions
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

summary = {"tag": "e3herq", "version": 3, "N_test": N,
           "label_check": "y_state == raw y_test of DRaw_C_Te_v0-002 verified on graham (em_e3herq_ycheck.py)",
           "category_codes_pertrace": {"correct": 0, "near_boundary": 1, "t1_pattern": 2, "amp_outlier": 3, "other": 4},
           "windows": {}}

for ns, w in WIN.items():
    S = {"length_ns": ns, "trace_samples": ns // 2, "cnn_source": cnn_src[ns], "configs": {}}
    out = {"length_ns": np.int64(ns), "trace_samples": np.int64(ns // 2),
           "true_bits": lab, "y_state": y_state.astype(np.uint8)}
    E = {}           # name -> (pred (N,5), row mask or None)
    cp = cnn_pred[ns]
    S["cnn_err_pct_wilson"] = [wilson(int((cp[:, q] != lab[:, q]).sum()), N) for q in range(5)]
    S["cnn_fq"] = fid(cp, lab)[0]
    S["paper_cnn_fq"] = PAPER_CNN[ns]

    # ---- B: demux-pipeline / scrambled ("fake"), full TEST
    fB = os.path.join(IN, f"em_e3herq_B_{ns}ns.npz")
    if os.path.exists(fB):
        B = np.load(fB)
        assert np.array_equal(B["test_rawrow"], np.arange(N)) and np.array_equal(B["test_state"], y_state)
        assert np.array_equal(B["scr_rmf_src_rawrow"], np.arange(N))
        sp = (B["scr_p1"] > 0.5).astype(np.uint8)
        msrc = B["scr_mf_src_rawrow"].astype(np.int64)
        spm = np.zeros((N, 5), np.uint8); sp1 = np.zeros((N, 5), np.float32); srs = np.zeros((N, 5), np.int32)
        for q in range(5):
            assert len(np.unique(msrc[:, q])) == N and np.array_equal(y_state[msrc[:, q]], y_state)
            spm[msrc[:, q], q] = sp[:, q]; sp1[msrc[:, q], q] = B["scr_p1"][:, q]; srs[msrc[:, q], q] = np.arange(N)
        out["pred_bits"], out["p1"] = spm, sp1
        out["scr_rmf_src_of_mfsrc"] = srs                       # shot whose RMF features fed that decision
        out["scr_row_mf_src"] = msrc.astype(np.int32)           # model input row -> MF source shot per qubit
        out["scrmodel_1shot_pred"] = (B["ss_p1"] > 0.5).astype(np.uint8)   # same model, shot-aligned inputs
        out["scrmodel_1shot_p1"] = B["ss_p1"].astype(np.float32)
        E["demuxpipe_scrambled"] = (spm, None)
        E["demuxpipe_scrambled_model_on_1shot_rows"] = (out["scrmodel_1shot_pred"], None)
        nerr = [int((spm[:, q] != lab[:, q]).sum()) for q in range(5)]
        assert nerr == [int((sp[:, q] != lab[:, q]).sum()) for q in range(5)]
        S["scr_frac_rows_mf_eq_rmf_shot"] = [float((msrc[:, q] == np.arange(N)).mean()) for q in range(5)]
        S["scr_frac_rows_all5_mf_same_shot"] = float(np.all(msrc == msrc[:, :1], axis=1).mean())
        S["scr_best_epoch"] = json.load(open(os.path.join(IN, f"em_e3herq_B_{ns}ns_summary.json")))["best_epoch"]
        if ns in CSV_FAKE_NERR:
            S["repro_scrambled_vs_fake_csv"] = {
                "regen_n_error": nerr, "csv_n_error": CSV_FAKE_NERR[ns],
                "diff_pp": [round(100*(a-b)/N, 4) for a, b in zip(nerr, CSV_FAKE_NERR[ns])],
                "max_abs_diff_pp": round(max(abs(100*(a-b)/N) for a, b in zip(nerr, CSV_FAKE_NERR[ns])), 4)}
        S["repro_scrambled_vs_paper_demux_F"] = {"regen": fid(spm, lab)[1:], "paper_as_released": PAPER_DEMUX_F[ns],
                                                 "csv_demux_subsample_fq": CSV_DEMUX[ns]}

    # ---- C: per-trace checkpoints (cut-figure panel b), full TEST, inference only
    fC = os.path.join(IN, f"em_e3herq_C_{ns}ns.npz")
    if os.path.exists(fC):
        C = np.load(fC)
        assert np.array_equal(C["test_rawrow"], np.arange(N)) and np.array_equal(C["test_state"], y_state)
        cpp = (C["p1"] > 0.5).astype(np.uint8)
        out["pred_bits_pertrace"], out["p1_pertrace"] = cpp, C["p1"].astype(np.float32)
        out["pertrace_cls32"] = C["cls"].astype(np.uint8)
        out["pertrace_category"] = C["category"].astype(np.uint8)
        E["pertrace_ckpt"] = (cpp, None)
        nerr = [int((cpp[:, q] != lab[:, q]).sum()) for q in range(5)]
        cat = C["category"]
        modes = [[int((cat[:, q] == 2).sum()), int((cat[:, q] == 3).sum()), int((cat[:, q] == 1).sum())] for q in range(5)]
        S["repro_pertrace_ckpt_vs_realistic_csv"] = {
            "regen_n_error": nerr, "csv_n_error": CSV_REAL_NERR[ns], "modes_t1_amp_near": modes,
            "exact_match": bool(nerr == CSV_REAL_NERR[ns] and modes == [list(m) for m in CSV_REAL_MODES[ns]])}
        f5 = fid(cpp, lab)
        S["repro_pertrace_ckpt_vs_paper_tabfidelity"] = {
            "regen_fq": f5[0], "paper_fq": PAPER_PERTRACE[ns],
            "diff_pp": [round(100*(a-b), 2) for a, b in zip(f5[0], PAPER_PERTRACE[ns])],
            "regen_F5Q_F4Q": f5[1:], "paper_F5Q_F4Q": PAPER_PERTRACE_F[ns]}

    # ---- A: _herqules_length_both.py retrain (its own 960k-row test split)
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

        def scatter(cls_te, p1_te):
            pred = np.full((N, 5), 255, np.uint8); p1 = np.full((N, 5), np.nan, np.float32)
            pred[raw_te] = bits(cls_te); p1[raw_te] = p1_te
            return pred, p1
        out["A_pertrace_pred"], out["A_pertrace_p1"] = scatter(A["pt_test_cls"], A["pt_test_p1"])
        mf_src, rf_src = A["dm_mf_src_rawrow"].astype(np.int64), A["dm_rmf_src_rawrow"].astype(np.int64)
        assert len(np.unique(mf_src)) == len(mf_src) and np.array_equal(np.sort(mf_src), np.sort(raw_te))
        assert np.array_equal(y_state[mf_src], A["dm_test_state"]) and np.array_equal(y_state[rf_src], A["dm_test_state"])
        dp = np.full((N, 5), 255, np.uint8); dp1 = np.full((N, 5), np.nan, np.float32); drs = np.full(N, -1, np.int32)
        dp[mf_src] = bits(A["dm_test_cls"]); dp1[mf_src] = A["dm_test_p1"]; drs[mf_src] = rf_src
        out["A_demuxsub_pred_by_mfsrc"], out["A_demuxsub_p1_by_mfsrc"], out["A_demuxsub_rmf_src_of_mfsrc"] = dp, dp1, drs
        S["A_demuxsub_frac_rows_mf_eq_rmf_shot"] = float((mf_src == rf_src).mean())
        E["A_pertrace"] = (out["A_pertrace_pred"], in_test)
        E["A_demux_subsample"] = (dp, in_test)
        fpt, fdm = fid(out["A_pertrace_pred"][in_test], lab[in_test]), fid(dp[in_test], lab[in_test])
        S["repro_A_pertrace_vs_csv"] = {"regen_fq": fpt[0], "csv_fq": CSV_PERTRACE[ns],
                                        "diff_pp": [round(100*(a-b), 2) for a, b in zip(fpt[0], CSV_PERTRACE[ns])]}
        S["repro_A_demuxsub_vs_csv"] = {"regen_fq": fdm[0], "csv_fq": CSV_DEMUX[ns],
                                        "diff_pp": [round(100*(a-b), 2) for a, b in zip(fdm[0], CSV_DEMUX[ns])],
                                        "regen_F5Q_F4Q": fdm[1:], "paper_as_released_F5Q_F4Q": PAPER_DEMUX_F[ns]}

    # ---- existing per-trace deployment predictions (found; 1 us, 960k rows)
    if ns == 1000:
        pb = np.load(f"{REP}/preds_HERQULESdeploy_pb.npy").astype(np.uint8)
        tbx = np.load(f"{REP}/preds_HERQULESdeploy_tb.npy").astype(np.uint8)
        m = np.concatenate([np.where(y_state == s)[0][5000:35000] for s in range(32)])
        assert np.array_equal(lab[m], tbx)
        ex = np.full((N, 5), 255, np.uint8); ex[m] = pb
        exm = np.zeros(N, bool); exm[m] = True
        out["existing_deploy_pred"] = ex
        E["existing_deploy_pertrace"] = (ex, exm)
        f5 = fid(pb, tbx)
        S["repro_existing_deploy_vs_paper_tabfidelity"] = {
            "fq": f5[0], "paper_fq": PAPER_PERTRACE[ns], "diff_pp": [round(100*(a-b), 2) for a, b in zip(f5[0], PAPER_PERTRACE[ns])],
            "F5Q_F4Q": f5[1:], "source": "oraqle_reports/preds_HERQULESdeploy_{pb,tb}.npy (runners/_xtalk_herqules_deploy.py)"}
        if "A_test_mask" in out:
            assert np.array_equal(exm, out["A_test_mask"])

    # ---- per-config error rates, fidelities, rate gaps vs canonical MCMit-CNN, co-failure
    for k, (p, msk) in E.items():
        mm = np.ones(N, bool) if msk is None else msk
        pk, lk, ck = p[mm], lab[mm], cp[mm]
        R = {"n_rows": int(mm.sum())}
        R["fq"], R["F5Q"], R["F4Q"] = fid(pk, lk)
        R["herq_err_pct_wilson"] = [wilson(int((pk[:, q] != lk[:, q]).sum()), len(lk)) for q in range(5)]
        R["cnn_err_pct_wilson_same_rows"] = [wilson(int((ck[:, q] != lk[:, q]).sum()), len(lk)) for q in range(5)]
        R["gap_cnn_over_herq_paired"] = [paired_ratio_ci(ck[:, q] != lk[:, q], pk[:, q] != lk[:, q]) for q in range(5)]
        R["cofailure_vs_cnn"] = [overlap(pk[:, q] != lk[:, q], ck[:, q] != lk[:, q]) for q in range(5)]
        S["configs"][k] = R
        print(f"[asm3] {w:6s} {k:42s} err% {[x[0] for x in R['herq_err_pct_wilson']]}  "
              f"CNN/HERQ {[g[0] for g in R['gap_cnn_over_herq_paired']]}  lift {[o['lift'] for o in R['cofailure_vs_cnn']]}",
              flush=True)
    if ns in CSV_HYB_NERR:
        S["hybridcnn_err_pct"] = [round(100*x/N, 4) for x in CSV_HYB_NERR[ns]]
        S["gap_hybridcnn_over_herq_aggregate"] = {
            "vs_fake_csv": [indep_ratio_ci(CSV_HYB_NERR[ns][q], N, CSV_FAKE_NERR[ns][q], N) for q in range(5)],
            "vs_realistic_csv": [indep_ratio_ci(CSV_HYB_NERR[ns][q], N, CSV_REAL_NERR[ns][q], N) for q in range(5)]}
    out["notes"] = np.array(
        "pred_bits/p1 = demux-pipeline (scrambled, 'fake' CSV) HERQULES regenerated, attributed per qubit to the "
        "MF-source shot; decision ALSO used RMF of shot scr_rmf_src_of_mfsrc and other qubits' MF of other shots "
        "(not single-shot). pred_bits_pertrace/p1_pertrace = exact per-trace checkpoint predictions (cut figure "
        "panel b; 1us/400ns only). A_* = _herqules_length_both.py retrain on its 960k-row split (A_test_mask). "
        "255/NaN = row not covered. See em_e3herq_summary_v3.json.")
    summary["windows"][w] = S
    np.savez_compressed(os.path.join(OUT, f"em_e3herq_preds_{w}.npz"), **out)
    print(f"[asm3] wrote em_e3herq_preds_{w}.npz keys={sorted(out)}", flush=True)

with open(os.path.join(OUT, "em_e3herq_summary_v3.json"), "w") as f:
    json.dump(summary, f, indent=1)
print("[asm3] done")
