"""em_e4recon_analyze.py (tag e4recon): final error-mode attribution, identical for MCMit-CNN and HERQULES.

Inputs (all from em_e4recon_glrt.py, which re-ran the e2seg v3 detector on every TEST shot and reproduces
em_e2seg_v3_raw_main.json bit-for-bit for every canonical/soft attribution):
  /work/out/em_e4recon_allshots_box{1us,600,400}_q{q}.npz  per-shot Sd/Su, alpha thresholds, FP, mhat, z, cfg, prep
  /work/out/em_e4recon_latecore.npz                       late-window (1-2 us) |0>/|1> core membership per shot
  /work/out/em_e4recon_glrt_raw.json                       null calibration on quasi-null shots, injection efficiency
  /reports/preds_MCMitCNN_pb.npy, /soft/cnn_soft_*.npz, /work/in/em_e4recon_herq_in.npz
  /e1hmm/em_e1hmm_v6_allshots_featbox_q{q}.npz             (cross-check only: e1hmm CNN-surrogate distance s_sur)

Final attribution (order A; each error gets exactly one primary mode; no mode is true by construction):
  1. LEAKAGE: line-frame z > 3 (1 us: feat1us bins 0-9; 600/400 ns: featbox 0-5/0-3), mirror-corrected
     (minus z < -3). Q4: n/m.
  2. MID-READOUT TRANSITION (resolved step): e2seg v3 GLRT at alpha = 0.01, relaxation test on prepared-|1> errors,
     excitation test on prepared-|0> errors; FP-corrected (sum of per-shot FP from the stationary conditional null);
     e2seg gating (quasi-null obs/exp in [0.6, 1.3] and efficiency given a misread >= 5%), otherwise n/m and those
     errors stay in the residual. Sensitive window ~150-850 ns at 1 us; MEASURED lower bound.
  3. RESIDUAL (no resolved event), split per error by a MODEL-NATIVE, INDEPENDENT near-boundary test:
       reference = the model's own errors on "stationary" shots, i.e. prepared-|b> shots whose LATE window
       (1-2 us, never seen by any model) sits in the |b> core. Their misread rate p_ov(b, cfg) is the empirical
       overlap error rate at the model's own boundary (no surrogate, no Gaussian assumption; an UPPER bound,
       because shots that changed state early and came back, or prepared-|0> shots that started in |1> and
       decayed before 1 us, also sit in the late core). Their soft-output P(correct) distribution defines the
       overlap zone: p*_b = 5th percentile of P(correct) among these reference errors.
       3a NEAR-BOUNDARY (overlap): residual errors with P(correct) >= p*, divided by 0.95 (the 5% of overlap
          errors that fall below p* are moved back from 3b).
       3b DEEP, NO RESOLVED STEP: the rest = errors read with high confidence in the wrong state: the qubit was in
          the other computational state for most of the window (transition before ~150-200 ns or before the
          window, preparation error, thermal population: INFERRED identity).
  Order B (robustness): transition first on all errors, leakage among the non-flagged errors.
CIs: Wilson for rates; mirror/FP-corrected counts +/-1.96 sqrt(N+ + N-) or sqrt(flags + MC var(FP sum));
     weighted residual splits +/-1.96 sqrt(sum w^2 + propagated terms); late-core expectation via binomial per cfg.
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import os

import numpy as np

W = str(LP.stage("em/e4recon") / "out")
ALPHA_IDX = 1          # alpha = 0.01
NB_BANK = 16384
Q_STAR = float(os.environ.get("E4_QSTAR", "0.05"))
MMID0, MSTEP, NMID = -1.49, 0.02, 200
QNAMES = ["Q1", "Q2", "Q3", "Q4", "Q5"]
VAR = {"1us": "box1us", "600ns": "box600", "400ns": "box400"}


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def pct(x, n):
    return None if x is None or n == 0 else float(100.0 * x / n)


def ci_pct(x, var, n):
    if x is None or n == 0:
        return None
    h = 1.96 * np.sqrt(max(var, 0.0))
    return [float(100 * (x - h) / n), float(100 * (x + h) / n)]


def var_fp(S, var, cell):
    if not S.any():
        return 0.0
    keys, inv, cnt = np.unique(cell[S], return_inverse=True, return_counts=True)
    vk = np.bincount(inv, weights=var[S]) / cnt
    return float((cnt.astype(float) ** 2 * vk).sum())


def load_shots(v, q):
    d = np.load(f"{W}/em_e4recon_allshots_{v}_q{q}.npz")
    assert np.array_equal(d["idx"], np.arange(len(d["idx"])))
    Fd = d["Sd"] > d["thrd"][:, ALPHA_IDX]
    Fu = d["Su"] > d["thru"][:, ALPHA_IDX]
    FPd = d["FPd"][:, ALPHA_IDX].astype(np.float64)
    FPu = d["FPu"][:, ALPHA_IDX].astype(np.float64)
    ci_ = np.clip(np.round((d["mhat"].astype(np.float64) - MMID0) / MSTEP).astype(int), 0, NMID - 1)
    cell = d["cfg"].astype(np.int64) * 1000 + ci_
    return dict(z=d["z"].astype(np.float64), cfg=d["cfg"].astype(np.int64), prep=d["prep"].astype(np.int64),
                Fd=Fd, Fu=Fu, FPd=FPd, FPu=FPu, cell=cell, mhat=d["mhat"])


def gate_info(raw, v, q):
    o = raw["variants"][v][f"q{q}"]
    g = {}
    for b, mode in ((1, "down"), (0, "up")):
        cal = o["calibration_all_shots"][f"{mode}_on_prep{b}_latecore"]["0.01"]
        ratio = cal["observed"] / cal["expected"] if cal["expected"] > 0 else np.inf
        eff = o["efficiency_by_alpha_tau_fit"][mode]["0.01"]["given_proxy_misread"]
        eff_mf = o["efficiency_by_alpha_tau_fit"][mode]["0.01"]["given_mf_misread"]
        ok = (0.6 <= ratio <= 1.3) and (eff >= 0.05)       # e2seg gating rule, applied to every qubit
        sens = o["sensitivity"][mode]["tau_fit"]
        ts = np.array(sens["ts_ns"])
        pf = np.array(sens["p_flag"])
        above = ts[pf >= 0.5]
        g[b] = {"test": mode, "quasinull_obs": cal["observed"], "quasinull_exp": cal["expected"], "quasinull_ratio": ratio,
                "eff_given_proxy_misread": eff, "eff_given_mf_misread": eff_mf, "measurable": bool(ok),
                "sens_ts_ns": ts.tolist(), "sens_p_flag": pf.tolist(),
                "sens_window_p_ge_0.5_ns": [float(above.min()), float(above.max())] if len(above) else None,
                "sens_max": float(pf.max())}
    return g


def attribute(S, pred, pc, lc, q, gate, s_sur=None):
    """Attribution of the errors of one model on one qubit. pc = model P(correct) per shot (soft), lc = late-core (N,2)."""
    prep, z, cfg = S["prep"], S["z"], S["cfg"]
    out = {}
    for b in (0, 1):
        m = prep == b
        nb = int(m.sum())
        E = m & (pred != prep)
        N = int(E.sum())
        F = S["Fd"] if b == 1 else S["Fu"]
        FP = S["FPd"] if b == 1 else S["FPu"]
        VR = FP * (1 - FP) / NB_BANK
        meas_T = gate[b]["measurable"]
        if q == 3:
            Pz = np.zeros_like(E)
            Mz = np.zeros_like(E)
        else:
            Pz = E & (z > 3)
            Mz = E & (z < -3)
        nP, nM = int(Pz.sum()), int(Mz.sum())
        notP = E & ~Pz
        leak = None if q == 3 else nP - nM
        leak_var = None if q == 3 else nP + nM
        # ---- order A transition
        pool_flag = int((F & notP).sum() + (F & Mz).sum())
        pool_fp = float(FP[notP].sum() + FP[Mz].sum())
        pool_var = var_fp(notP, VR, S["cell"]) + var_fp(Mz, VR, S["cell"])
        T_raw = pool_flag - pool_fp
        T = T_raw if meas_T else None
        T_var = pool_flag + pool_var
        # per-shot residual weights (order A)
        w = np.zeros(len(prep))
        if meas_T:
            w[notP] += (1.0 - F[notP]) + FP[notP]
            w[Mz] += (1.0 - F[Mz]) + FP[Mz]
        else:
            w[notP] += 1.0
            w[Mz] += 1.0
        R = float(w.sum())
        # ---- model-native near-boundary reference (late-window |b> core = stationary shots)
        LC = m & lc[:, b]
        Elc = LC & (pred != prep)
        n_lc, n_elc = int(LC.sum()), int(Elc.sum())
        pstar = float(np.quantile(pc[Elc], Q_STAR)) if n_elc >= 20 else None
        pc_lc_err = pc[Elc]
        lc_hist = np.histogram(pc_lc_err, bins=[0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0001])[0].tolist()
        Oexp, Ovar = 0.0, 0.0
        for c in range(16):
            mc = m & (cfg == c)
            nl = int((LC & (cfg == c)).sum())
            ne = int((Elc & (cfg == c)).sum())
            if nl == 0:
                continue
            p = ne / nl
            Nc = int(mc.sum())
            Oexp += Nc * p
            Ovar += Nc * Nc * max(p * (1 - p), 1.0 / nl ** 2) / nl
        # ---- residual split by the overlap zone
        if pstar is not None:
            inz = pc >= pstar
            NBraw = float(w[inz].sum())
            NBvar = float((w[inz] ** 2).sum())
            deepraw = float(w[~inz].sum())
            near = NBraw / (1 - Q_STAR)
            deep = R - near
            near_var = NBvar / (1 - Q_STAR) ** 2
            deep_var = float((w[~inz] ** 2).sum()) + (Q_STAR / (1 - Q_STAR)) ** 2 * NBvar
        else:
            near = deep = near_var = deep_var = NBraw = deepraw = None
        # residual P(correct) histogram (weighted)
        res_hist = np.histogram(pc, bins=[0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0001], weights=w)[0].round(2).tolist()
        # late-window confirmation: where does the qubit sit at 1-2 us?
        conf = {}
        cls = {"all_errors": E, "flagged_transition": E & F & ~Pz,
               "residual_deep": E & ~Pz & ~F & (pc < pstar) if pstar is not None else None,
               "residual_near": E & ~Pz & ~F & (pc >= pstar) if pstar is not None else None,
               "correct_shots": m & (pred == prep)}
        for k_, sel in cls.items():
            if sel is None or not sel.any():
                conf[k_] = None
                continue
            ns = int(sel.sum())
            conf[k_] = {"n": ns, "late_in_misread_state_core": float(lc[sel, 1 - b].mean()),
                        "late_in_prepared_state_core": float(lc[sel, b].mean())}
        # cross-check: e1hmm surrogate distance of the residual (weights), within 1 / 2 sigma of the CNN boundary
        sur = None
        if s_sur is not None:
            sur = {"frac_resid_within_1sigma": float(w[s_sur >= -1].sum() / max(R, 1e-9)),
                   "frac_resid_within_2sigma": float(w[s_sur >= -2].sum() / max(R, 1e-9))}
        # ---- order B (transition first, on all errors; leakage among non-flagged)
        flag_all = int((F & E).sum())
        fp_all = float(FP[E].sum())
        T_B = (flag_all - fp_all) if meas_T else None
        T_B_var = flag_all + var_fp(E, VR, S["cell"])
        if q == 3:
            L_B, L_B_var = None, None
        else:
            if meas_T:
                L_B = int((Pz & ~F).sum()) - int((Mz & ~F).sum())
                L_B_var = int((Pz & ~F).sum()) + int((Mz & ~F).sum())
            else:
                L_B, L_B_var = leak, leak_var
        R_B = N - (T_B or 0.0) - (L_B or 0)
        # overlap of leakage and transition flags (mirror- and FP-corrected)
        ov = None
        if q != 3 and meas_T:
            ov = (int((F & Pz).sum()) - float(FP[Pz].sum())) - (int((F & Mz).sum()) - float(FP[Mz].sum()))
        out[f"prep{b}"] = {
            "n_shots": nb, "n_err": N, "err_rate_pct": pct(N, nb), "err_rate_ci95_pct": [100 * x for x in wilson(N, nb)],
            "leak_plus": nP, "leak_minus": nM, "leak": leak, "leak_var": leak_var,
            "T_measurable": meas_T, "T": T, "T_var": T_var, "T_ungated": T_raw, "pool_flag": pool_flag, "pool_fp": pool_fp,
            "R": R, "near": near, "near_var": near_var, "deep": deep, "deep_var": deep_var,
            "near_raw_in_zone": NBraw, "deep_raw_below_zone": deepraw,
            "pstar": pstar, "n_latecore": n_lc, "n_latecore_err": n_elc,
            "latecore_err_rate_pct": pct(n_elc, n_lc), "latecore_err_pc_hist": lc_hist,
            "overlap_expected_latecore_upper": Oexp, "overlap_expected_var": Ovar,
            "resid_pc_hist_weighted": res_hist, "late_window_confirmation": conf, "e1hmm_surrogate_depth": sur,
            "orderB": {"T": T_B, "T_var": T_B_var, "L": L_B, "L_var": L_B_var, "R": R_B},
            "overlap_leak_and_transition": ov}
    return out


def combine(parts, keys_q):
    """Pool per-(qubit, bit) dicts: sum counts and variances."""
    tot = {"n_shots": 0, "n_err": 0}
    sums = {k: 0.0 for k in ("leak", "leak_var", "T", "T_var", "R", "near", "near_var", "deep", "deep_var",
                             "overlap_expected_latecore_upper", "overlap_expected_var", "T_ungated")}
    flags = {"leak_nm": [], "T_nm": [], "near_nm": []}
    B = {"T": 0.0, "T_var": 0.0, "L": 0.0, "L_var": 0.0, "R": 0.0}
    for qn, d in parts:
        if qn not in keys_q:
            continue
        for b in (0, 1):
            x = d[f"prep{b}"]
            tot["n_shots"] += x["n_shots"]
            tot["n_err"] += x["n_err"]
            for k in ("leak", "leak_var"):
                if x[k] is None:
                    flags["leak_nm"].append(f"{qn}") if k == "leak" else None
                else:
                    sums[k] += x[k]
            for k in ("T", "T_var"):
                if x["T"] is None:
                    if k == "T":
                        flags["T_nm"].append(f"{qn}-prep{b}")
                else:
                    sums[k] += x[k]
            for k in ("R", "overlap_expected_latecore_upper", "overlap_expected_var", "T_ungated"):
                sums[k] += x[k]
            if x["near"] is None:
                flags["near_nm"].append(f"{qn}-prep{b}")
            else:
                for k in ("near", "near_var", "deep", "deep_var"):
                    sums[k] += x[k]
            for k in ("T", "T_var", "L", "L_var", "R"):
                if x["orderB"][k] is not None:
                    B[k] += x["orderB"][k]
    tot.update(sums)
    tot["orderB"] = B
    tot["not_measured_counted_as_0"] = {k: sorted(set(v)) for k, v in flags.items()}
    return tot


def shares(x, N, nshots):
    """Shares of errors (%) and of shots (%) with CIs for a combined/per-bit dict."""
    o = {"n_err": N, "n_shots": nshots, "err_rate_pct": pct(N, nshots), "err_rate_ci95_pct": [100 * v for v in wilson(N, nshots)]}
    for k, vk in (("leak", "leak_var"), ("T", "T_var"), ("near", "near_var"), ("deep", "deep_var")):
        if x.get(k) is None:
            o[k] = None
            continue
        o[k] = {"count": x[k], "pct_err": pct(x[k], N), "ci95_pct_err": ci_pct(x[k], x[vk], N),
                "pct_shots": pct(x[k], nshots), "ci95_pct_shots": ci_pct(x[k], x[vk], nshots)}
    o["R"] = {"count": x["R"], "pct_err": pct(x["R"], N)}
    o["overlap_expected_latecore_upper"] = {"count": x["overlap_expected_latecore_upper"],
                                            "pct_err": pct(x["overlap_expected_latecore_upper"], N),
                                            "ci95_pct_err": ci_pct(x["overlap_expected_latecore_upper"], x["overlap_expected_var"], N)}
    o["T_ungated_pct_err"] = pct(x.get("T_ungated"), N)
    if "orderB" in x:
        B = x["orderB"]
        o["orderB"] = {"T_pct_err": pct(B["T"], N) if B["T"] is not None else None,
                       "T_ci95": ci_pct(B["T"], B["T_var"], N) if B["T"] is not None else None,
                       "L_pct_err": pct(B["L"], N) if B["L"] is not None else None,
                       "L_ci95": ci_pct(B["L"], B["L_var"], N) if B["L"] is not None else None,
                       "R_pct_err": pct(B["R"], N)}
    return o


def per_qubit_summary(d):
    """Sum prep0/prep1 of one qubit."""
    x = combine([("q", d)], ["q"])
    return shares(x, x["n_err"], x["n_shots"]) | {"not_measured_counted_as_0": x["not_measured_counted_as_0"]}


def hidden(S, pred, q):
    if q == 3:
        return None
    z, prep = S["z"], S["prep"]
    o = {}
    tot = {"Lp": 0, "Lm": 0, "eP": 0, "eM": 0, "n": 0}
    for b in (0, 1):
        m = prep == b
        Lp, Lm = m & (z > 3), m & (z < -3)
        err = pred != prep
        nL = int(Lp.sum()) - int(Lm.sum())
        eL = int((Lp & err).sum()) - int((Lm & err).sum())
        n = int(m.sum())
        o[f"prep{b}"] = {"leaked_pct_shots": pct(nL, n), "ci95": ci_pct(nL, int(Lp.sum()) + int(Lm.sum()), n),
                         "leaked_misread": eL, "frac_leaked_read_correctly": (1 - eL / nL) if nL > 0 else None}
        tot["Lp"] += int(Lp.sum()); tot["Lm"] += int(Lm.sum()); tot["n"] += n
        tot["eP"] += int((Lp & err).sum()); tot["eM"] += int((Lm & err).sum())
    nL = tot["Lp"] - tot["Lm"]
    eL = tot["eP"] - tot["eM"]
    o["all"] = {"leaked_pct_shots": pct(nL, tot["n"]), "ci95": ci_pct(nL, tot["Lp"] + tot["Lm"], tot["n"]),
                "leaked_shots": nL, "leaked_misread": eL, "frac_leaked_read_correctly": 1 - eL / nL,
                "frac_read_correctly_ci95": wilson(nL - eL, nL)}
    return o


def main():
    raw = json.load(open(f"{W}/em_e4recon_glrt_raw.json"))
    LC = np.load(f"{W}/em_e4recon_latecore.npz")
    yte = np.load(str(LP.FEAT / "y_test.npy"))
    Bte = ((yte[:, None] >> np.arange(5)) & 1).astype(np.int64)
    pb = np.load(str(LP.REPORTS / "preds_MCMitCNN_pb.npy")).astype(np.int64)
    soft = {T: np.load(f"{LP.SOFT}/cnn_soft_{T}ns.npz")["prob"].astype(np.float64) for T in (1000, 600, 400)}
    H = np.load(str(LP.stage("em/e4recon") / "in" / "em_e4recon_herq_in.npz"))
    models = {
        "1us": {"cnn_canonical": (pb, soft[1000]), "cnn_soft1000": ((soft[1000] > 0.5).astype(np.int64), soft[1000]),
                "herq_pertrace": (H["pt_pred_1us"].astype(np.int64), H["pt_p1_1us"].astype(np.float64)),
                "herq_scrambled_mfsrc": (H["scr_pred_1us"].astype(np.int64), H["scr_p1_1us"].astype(np.float64))},
        "600ns": {"cnn_soft600": ((soft[600] > 0.5).astype(np.int64), soft[600])},
        "400ns": {"cnn_soft400": ((soft[400] > 0.5).astype(np.int64), soft[400]),
                  "herq_pertrace": (H["pt_pred_400ns"].astype(np.int64), H["pt_p1_400ns"].astype(np.float64)),
                  "herq_scrambled_mfsrc": (H["scr_pred_400ns"].astype(np.int64), H["scr_p1_400ns"].astype(np.float64))},
    }
    res = {"note": __doc__, "q_star": Q_STAR, "windows": {}}
    for wn, v in VAR.items():
        res["windows"][wn] = {"gating": {}, "models": {}, "leak_E3": {}}
        shots = {q: load_shots(v, q) for q in range(5)}
        for q in range(5):
            assert np.array_equal(shots[q]["prep"], Bte[:, q])
            res["windows"][wn]["gating"][QNAMES[q]] = {f"prep{b}": g for b, g in gate_info(raw, v, q).items()}
            z, prep = shots[q]["z"], shots[q]["prep"]
            res["windows"][wn]["leak_E3"][QNAMES[q]] = {
                f"prep{b}": {"plus": int(((prep == b) & (z > 3)).sum()), "minus": int(((prep == b) & (z < -3)).sum()),
                             "n": int((prep == b).sum()),
                             "E3_pct": pct(int(((prep == b) & (z > 3)).sum()) - int(((prep == b) & (z < -3)).sum()), int((prep == b).sum())),
                             "ci95": ci_pct(int(((prep == b) & (z > 3)).sum()) - int(((prep == b) & (z < -3)).sum()),
                                            int(((prep == b) & (z > 3)).sum()) + int(((prep == b) & (z < -3)).sum()), int((prep == b).sum()))}
                for b in (0, 1)} if q != 3 else None
        e1 = {}
        if wn in ("1us", "600ns", "400ns"):
            for q in range(5):
                f = f"{LP.stage('em/e1hmm')}/full_v6/em_e1hmm_v6_allshots_featbox_q{q}.npz"  # was /e1hmm
                if os.path.exists(f):
                    e1[q] = np.load(f)[f"{wn}_s_sur"].astype(np.float64)
        for mname, (P, P1) in models[wn].items():
            per_q, parts = {}, []
            for q in range(5):
                pred = P[:, q]
                p1 = P1[:, q]
                pc = np.where(Bte[:, q] == 1, p1, 1 - p1)
                gate = {b: res["windows"][wn]["gating"][QNAMES[q]][f"prep{b}"] for b in (0, 1)}
                s_sur = e1.get(q) if mname.startswith("cnn") else None
                d = attribute(shots[q], pred, pc, LC[f"latecore_q{q}"], q, gate, s_sur)
                parts.append((QNAMES[q], d))
                per_q[QNAMES[q]] = {"per_bit": {b_: shares(d[b_], d[b_]["n_err"], d[b_]["n_shots"]) | {"detail": d[b_]}
                                                for b_ in ("prep0", "prep1")},
                                    "all": per_qubit_summary(d), "hidden_leakage": hidden(shots[q], pred, q)}
            pooled = {}
            for pname, ks in (("all5", QNAMES), ("Q1Q3Q4Q5", ["Q1", "Q3", "Q4", "Q5"]), ("Q1Q2Q3Q5", ["Q1", "Q2", "Q3", "Q5"]),
                              ("Q1Q3Q5", ["Q1", "Q3", "Q5"]), ("Q3Q5", ["Q3", "Q5"])):
                x = combine(parts, ks)
                pooled[pname] = shares(x, x["n_err"], x["n_shots"]) | {"not_measured_counted_as_0": x["not_measured_counted_as_0"]}
            res["windows"][wn]["models"][mname] = {"per_qubit": per_q, "pooled": pooled}
            print(wn, mname, "done", flush=True)
    json.dump(res, open(f"{W}/em_e4recon_analysis{os.environ.get('E4_TAG', '')}.json", "w"), indent=1, default=float)
    print("written")


if __name__ == "__main__":
    main()
