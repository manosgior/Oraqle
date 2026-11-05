#!/usr/bin/env python3
"""m3time_reanalyze.py -- post-hoc TEST analysis of the m3time HMM fits with CANONICAL state labels
(agent tag m3time).

Why this exists: m3time_hmm.py picks the best of several EM restarts but never re-labels the states
afterwards. The 3-state likelihood is invariant under state permutations, so for Q3 (feat runs) the
winning restart had "state 1" = the far third cloud and "state 2" = the true |1> cloud, and all of the
downstream statistics (which hard-code L = 2) are mislabelled. Q4 is ambiguous (two |1>-like clouds).
This script loads the TRAIN-fitted parameters from m3time_q{q}.json, relabels them, and redoes the
held-out TEST analysis with the frozen parameters:

  canonical labels:  |0> := argmax_s pi[prep0, s]   (state where prepared-|0> shots start)
                     |1> := argmax over the other two of pi[prep1, s]
                     L   := the remaining state

Added on top of m3time_hmm.py's analysis:
  * cascade check: after an L-starter's first exit into |1> (filtered Viterbi, >= 2 bins in |1>),
    how long does it stay in |1> and where does it go next (|2> -> |1> -> |0> would look like a T1 dwell);
  * time-resolved soft transition rates per bin (is mid-readout entry into L constant or rising?);
  * a NULL simulation: synthetic trajectories from the same fitted model with L removed (no transitions
    into L, pi_L = 0), MA(1) bin-to-bin noise matching the measured lag-1 correlation, 10 sub-steps per
    bin so a jump inside a bin produces an intermediate bin value, run through the identical
    Viterbi + blip-filter pipeline. Its start-L / ever-L / mid-entry rates are the false-positive floor.

Run (graham):
  docker run -d --name leak-m3time-re_feat_xt1 --user $(id -u):$(id -g) -e OMP_NUM_THREADS=8 \
    -v ~/oraqle-ics-sims/leakage/features:/feat:ro -v ~/oraqle-ics-sims/leakage/m3time:/m3 \
    oraqle-torch:latest python /m3/scripts/m3time_reanalyze.py --run /m3/out/feat_xt1 --out /m3/out_re/feat_xt1
"""
import argparse, json, os, sys, time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3time_hmm import (FEAT, NB, DT_NS, L, NAMES, jsonable, cp_ci, frac, tau_from_p, cfg_other,  # noqa: E402
                        log_emis, fb, viterbi, blip_filter, start_state, exit_stats, mid_entries,
                        delta_ratio_ci)


def log(q, msg):
    print(f"[m3time-re q{q}/{NAMES[q]}] {msg}", flush=True)


def canon_perm(P):
    pi0, pi1 = np.asarray(P["pi_prep0"]), np.asarray(P["pi_prep1"])
    s0 = int(np.argmax(pi0))
    rest = [s for s in range(3) if s != s0]
    s1 = rest[int(np.argmax(pi1[rest]))]
    sL = [s for s in rest if s != s1][0]
    return [s0, s1, sL]


def permute(P, perm):
    mu = np.asarray(P["mu"])[perm]
    A = np.asarray(P["A"])[np.ix_(perm, perm)]
    pi = np.stack([np.asarray(P["pi_prep0"])[perm], np.asarray(P["pi_prep1"])[perm]])
    return {"mu": mu, "Sig": np.asarray(P["Sig"]), "A": A, "pi": pi}


def fb_perbin_xi(logB, A, pi0, groups):
    """forward-backward returning gamma, per-shot xi sums, ll, and per-bin xi/gamma sums per group."""
    N, K, S = logB.shape
    mx = logB.max(2, keepdims=True)
    B = np.exp(logB - mx)
    alpha = np.empty_like(B)
    c = np.empty((N, K))
    a = pi0 * B[:, 0]
    c[:, 0] = a.sum(1)
    alpha[:, 0] = a / c[:, 0, None]
    for k in range(1, K):
        a = (alpha[:, k - 1] @ A) * B[:, k]
        c[:, k] = a.sum(1)
        alpha[:, k] = a / c[:, k, None]
    beta = np.empty_like(B)
    beta[:, K - 1] = 1.0
    xis = np.zeros((N, S, S))
    G = len(groups)
    xik = np.zeros((G, K - 1, S, S))
    for k in range(K - 2, -1, -1):
        bb = B[:, k + 1] * beta[:, k + 1] / c[:, k + 1, None]
        beta[:, k] = bb @ A.T
        t = alpha[:, k, :, None] * A[None] * bb[:, None, :]
        xis += t
        for g, m in enumerate(groups):
            xik[g, k] = t[m].sum(0)
    gamma = alpha * beta
    gamma /= gamma.sum(2, keepdims=True)
    ll = np.log(c).sum(1) + mx[..., 0].sum(1)
    return gamma, xis, ll, xik


def cascade(p, sel, Kw):
    """L-starters (bins 1-2) in sel whose first exit (bin>=3) goes to |1>: dwell in |1> and next state."""
    P = p[sel][:, :Kw]
    n = len(P)
    if n == 0:
        return None
    notL = P[:, 3:] != L
    has = notL.any(1)
    e = np.where(has, notL.argmax(1) + 3, Kw)
    ii = np.where(has)[0]
    d = P[ii, e[ii]]
    j1 = ii[d == 1]
    e1 = e[j1]
    sub = P[j1]
    cols = np.arange(Kw)[None]
    after = (cols >= e1[:, None]) & (sub != 1)
    hend = after.any(1)
    end = np.where(hend, after.argmax(1), Kw)
    dwell = end - e1
    nxt = np.where(hend, sub[np.arange(len(j1)), np.minimum(end, Kw - 1)], -1)
    exposure = int(np.where(hend, dwell, dwell - 1).sum())  # bin-steps spent in 1 with a successor in window
    n0 = int((nxt == 0).sum())
    nL = int((nxt == L).sum())
    p0 = n0 / exposure if exposure else None
    ci = cp_ci(n0, exposure) if exposure else [None, None]
    return {"n_Lstart_exit_to1": int(len(j1)), "dwell_in_1_hist_bins": [int((dwell == k).sum()) for k in range(Kw + 1)],
            "next_after_1": {"0": n0, "L": nL, "censored": int((nxt == -1).sum())},
            "exposure_bin_steps": exposure, "p_1to0_per_100ns": p0, "p_1to0_ci95": ci,
            "tau_1to0_ns": tau_from_p(p0), "tau_1to0_ns_ci95": [tau_from_p(ci[1]), tau_from_p(ci[0])] if exposure else None}


# ----------------------------------------------------------------------------- null simulation
def simulate_null(P, bits, rho_along, rho_perp, rng, nsub=10):
    """Synthetic shots from the fitted model with L switched off.
    P: canonical params. bits: prepared bit per shot. Returns X (n,K,2) and true per-bin majority state."""
    mu, Sig, A, pi = P["mu"], P["Sig"], P["A"].copy(), P["pi"].copy()
    K = mu.shape[1]
    n = len(bits)
    # 2-state chain without L
    A2 = A[:2, :2].copy()
    A2[0, 0] += A[0, L]
    A2[1, 1] += A[1, L]
    pi2 = pi[:, :2] / pi[:, :2].sum(1, keepdims=True)
    # per-substep transition probabilities (rates split evenly over nsub sub-steps)
    p01 = 1 - (1 - A2[0, 1]) ** (1 / nsub)
    p10 = 1 - (1 - A2[1, 0]) ** (1 / nsub)
    s = (rng.random(n) < pi2[bits, 1]).astype(np.int8)       # state at t=0
    occ1 = np.zeros((n, K))                                  # fraction of the bin spent in |1>
    for k in range(K):
        acc = np.zeros(n)
        for _ in range(nsub):
            u = rng.random(n)
            flip = np.where(s == 0, u < p01, u < p10)
            s = np.where(flip, 1 - s, s).astype(np.int8)
            acc += s
        occ1[:, k] = acc / nsub
    X = (1 - occ1)[:, :, None] * mu[0][None] + occ1[:, :, None] * mu[1][None]
    # MA(1) noise with the measured lag-1 correlation, per-bin covariance Sig[k]
    noise = np.empty((n, K, 2))
    for d, rho in ((0, rho_along), (1, rho_perp)):
        rho = float(np.clip(rho, 0.0, 0.49))
        th = (1 - np.sqrt(1 - 4 * rho * rho)) / (2 * rho) if rho > 1e-6 else 0.0
        e = rng.standard_normal((n, K + 1))
        noise[:, :, d] = (e[:, 1:] + th * e[:, :-1]) / np.sqrt(1 + th * th)
    Lc = np.linalg.cholesky(Sig)                             # (K,2,2)
    X += np.einsum("kij,nkj->nki", Lc, noise)
    return X, (occ1 > 0.5).astype(np.int8)


def null_stats(P, rho, rng, n_per_bit):
    out = {}
    bits = np.repeat(np.array([0, 1]), n_per_bit)
    X, _ = simulate_null(P, bits, rho["along"], rho["perp"], rng)
    logB = log_emis(X, P["mu"], P["Sig"])
    path = blip_filter(viterbi(logB, P["A"], P["pi"][bits]))
    gamma, _, _ = fb(logB, P["A"], P["pi"][bits])
    st = start_state(path)
    for wn, Kw in (("1us", 10), ("2us", 20)):
        W = {}
        for b in (0, 1):
            sb = bits == b
            nb = int(sb.sum())
            me, _, _, _ = mid_entries(path, sb, st, Kw)
            W[f"prep{b}"] = {"start_L": frac(((st == L) & sb).sum(), nb),
                             "ever_L": frac(((path[:, :Kw] == L).any(1) & sb).sum(), nb),
                             "mid_entry_L": me["enter_L"],
                             "soft_P_L_mean_bins": float(gamma[sb, :Kw, L].mean()),
                             "soft_pi_L": float(gamma[sb, 0, L].mean())}
        out[wn] = W
    out["n_per_bit"] = int(n_per_bit)
    out["note"] = ("synthetic shots from the canonical TRAIN fit with L switched off (no entries into L, pi_L=0), "
                   "MA(1) noise with measured lag-1 corr, 10 sub-steps per bin; identical Viterbi+blip pipeline")
    return out


# ----------------------------------------------------------------------------- per qubit
def run_qubit(args):
    q, a = args
    t0 = time.time()
    rng = np.random.default_rng(777 + q)
    d = json.load(open(os.path.join(a.run, f"m3time_q{q}.json")))
    kind, xt = d["kind"], d["xt_correction"]
    perm = canon_perm(d["params_train"])
    P = permute(d["params_train"], perm)
    Pte_refit = permute(d["params_test"], canon_perm(d["params_test"]))
    log(q, f"perm {perm} (old index of canonical 0,1,L)")
    out = {"qubit_code": f"q{q}", "qubit_paper": NAMES[q], "kind": kind, "xt_correction": xt,
           "source_json": os.path.join(a.run, f"m3time_q{q}.json"), "perm_old_index_of_0_1_L": perm,
           "relabelled": perm != [0, 1, 2], "bin_ns": DT_NS, "n_bins": NB,
           "noise_lag1_corr_prep0_core": d["noise_lag1_corr_prep0_core"],
           "heldout_ll_3vs2_from_source": d["heldout_ll"]}
    F = np.load(f"{FEAT}/{kind}_test.npy", mmap_mode="r")
    X = np.asarray(F[:, q], dtype=np.float64)
    y = np.load(f"{FEAT}/y_test.npy").astype(np.int64)
    idx = np.load(f"{FEAT}/idx_test.npy")
    b = (y >> q) & 1
    cfg = cfg_other(y, q)
    offs = np.asarray(d["offsets_xt"])
    if xt:
        X = X - offs[cfg]
    # parameters / lifetimes from the canonical matrices (TRAIN fit and the TEST refit)
    for nm, PP in (("train", P), ("test_refit", Pte_refit)):
        A = PP["A"]
        o = {"A": A, "pi_prep0": PP["pi"][0], "pi_prep1": PP["pi"][1]}
        for s in range(3):
            o[f"tau_state{s}_ns"] = tau_from_p(1 - A[s, s])
            for dd in range(3):
                if dd != s:
                    o[f"tau_{s}to{dd}_ns"] = tau_from_p(A[s, dd])
        out[f"params_{nm}_canonical"] = o
    out["mu_canonical"] = P["mu"]
    # geometry in physical units (bins 3..9 and 3..19)
    geo = {}
    for wn, (k0, k1) in {"1us": (3, 10), "2us": (3, 20)}.items():
        mu = P["mu"][:, k0:k1].mean(1)
        sep = mu[1] - mu[0]
        sepn = np.linalg.norm(sep)
        u = sep / sepn
        v = np.array([-u[1], u[0]])
        sb = np.sqrt(np.mean([np.trace(S_) / 2 for S_ in P["Sig"][k0:k1]]))
        d2 = mu[2] - mu[0]
        geo[wn] = {"mu0": mu[0], "mu1": mu[1], "muL": mu[2], "sep": sepn, "sigma_bin": sb,
                   "L_along_in_sep": float(d2 @ u / sepn), "L_perp_in_sep": float(d2 @ v / sepn),
                   "L_perp_in_sigma_bin": float(d2 @ v / sb),
                   "L_dist_to_1_in_sigma_bin": float(np.linalg.norm(mu[2] - mu[1]) / sb),
                   "L_dist_to_0_in_sigma_bin": float(np.linalg.norm(mu[2] - mu[0]) / sb),
                   "sep_in_sigma_bin": float(sepn / sb)}
    out["geometry"] = geo
    # held-out posteriors with frozen TRAIN params
    groups = [b == 0, b == 1]
    logB = log_emis(X, P["mu"], P["Sig"])
    gamma, xis, ll3, xik = fb_perbin_xi(logB, P["A"], P["pi"][b], groups)
    occ = {}
    for bb in (0, 1):
        g = gamma[b == bb]
        n = len(g)
        occ[f"prep{bb}"] = {"n": int(n), "P_L_per_bin": g[:, :, L].mean(0), "P_1_per_bin": g[:, :, 1].mean(0),
                            "P_0_per_bin": g[:, :, 0].mean(0),
                            "pi_L_onestep": float(g[:, 0, L].mean()),
                            "pi_L_onestep_ci95": [float(g[:, 0, L].mean() - 1.96 * g[:, 0, L].std() / np.sqrt(n)),
                                                  float(g[:, 0, L].mean() + 1.96 * g[:, 0, L].std() / np.sqrt(n))],
                            "P_L_bins1_2_mean": float(g[:, 1:3, L].mean()),
                            "P_L_bins1_2_ci95": [float(g[:, 1:3, L].mean(1).mean() - 1.96 * g[:, 1:3, L].mean(1).std() / np.sqrt(n)),
                                                 float(g[:, 1:3, L].mean(1).mean() + 1.96 * g[:, 1:3, L].mean(1).std() / np.sqrt(n))]}
        # time-resolved soft rates per bin: P(s->d at k->k+1) / P(s at k)
        gk = g[:, :-1].sum(0)                    # (K-1,S)
        xk = xik[bb]                             # (K-1,S,S)
        occ[f"prep{bb}"]["rate_per_bin"] = {f"{s}to{dd}": (xk[:, s, dd] / np.maximum(gk[:, s], 1e-12))
                                            for s in range(3) for dd in range(3) if s != dd}
    out["soft_occupancy_test"] = occ
    # HMM transition rates on TEST with delta-method CIs
    gnl = gamma[:, :-1].sum(1)
    rates = {}
    for s in range(3):
        out_s = xis[:, s, :].sum(1) - xis[:, s, s]
        R, se, ci = delta_ratio_ci(out_s, gnl[:, s])
        rates[f"from{s}"] = {"p_exit_per_100ns": R, "ci95": ci, "tau_ns": tau_from_p(R),
                             "tau_ns_ci95": [tau_from_p(ci[1]), tau_from_p(ci[0])]}
        for dd in range(3):
            if dd == s:
                continue
            R2, _, ci2 = delta_ratio_ci(xis[:, s, dd], gnl[:, s])
            Rb, _, cib = delta_ratio_ci(xis[:, s, dd], out_s)
            rates[f"from{s}"].update({f"p_to{dd}": R2, f"p_to{dd}_ci95": ci2, f"tau_to{dd}_ns": tau_from_p(R2),
                                      f"tau_to{dd}_ns_ci95": [tau_from_p(ci2[1]), tau_from_p(ci2[0])],
                                      f"branch_to{dd}": Rb, f"branch_to{dd}_ci95": cib})
        for bb in (0, 1):
            m = b == bb
            R3, _, ci3 = delta_ratio_ci(out_s[m], gnl[m, s])
            rates[f"from{s}"][f"prep{bb}_p_exit"] = R3
            rates[f"from{s}"][f"prep{bb}_p_exit_ci95"] = ci3
            for dd in range(3):
                if dd != s:
                    R4, _, ci4 = delta_ratio_ci(xis[m, s, dd], gnl[m, s])
                    rates[f"from{s}"][f"prep{bb}_p_to{dd}"] = R4
                    rates[f"from{s}"][f"prep{bb}_p_to{dd}_ci95"] = ci4
    out["hmm_rates_test_onestep"] = rates
    del xis
    # Viterbi
    path_raw = viterbi(logB, P["A"], P["pi"][b])
    path = blip_filter(path_raw)
    hard = {}
    for pname, pp in (("viterbi_raw", path_raw), ("viterbi_filtered", path)):
        st = start_state(pp)
        H = {}
        for wn, Kw in (("1us", 10), ("2us", 20)):
            W = {}
            for bb in (0, 1):
                sb_ = b == bb
                nb = int(sb_.sum())
                W[f"prep{bb}"] = {"start_L": frac(((st == L) & sb_).sum(), nb),
                                  "start_0": frac(((st == 0) & sb_).sum(), nb),
                                  "start_1": frac(((st == 1) & sb_).sum(), nb),
                                  "start_ambiguous": frac(((st == -1) & sb_).sum(), nb),
                                  "ever_L": frac(((pp[:, :Kw] == L).any(1) & sb_).sum(), nb),
                                  "L_at_bin": [int(((pp[:, k] == L) & sb_).sum()) for k in range(Kw)]}
                for s in range(3):
                    r = exit_stats(pp, sb_ & (st == s), s, Kw)
                    W[f"prep{bb}"][f"exit_from{s}"] = r[0] if r else None
                me, _, _, _ = mid_entries(pp, sb_, st, Kw)
                W[f"prep{bb}"]["mid_entry_L"] = me
                W[f"prep{bb}"]["cascade_L_to_1"] = cascade(pp, sb_ & (st == L), Kw)
            for s in range(3):
                r = exit_stats(pp, st == s, s, Kw)
                W[f"pooled_exit_from{s}"] = r[0] if r else None
            W["pooled_cascade_L_to_1"] = cascade(pp, st == L, Kw)
            H[wn] = W
        hard[pname] = H
    out["hard_test"] = hard
    # location of L (filtered Viterbi, bins 3..19 in L)
    st = start_state(path)
    mask = path[:, 3:] == L
    Xl = X[:, 3:]
    g2 = geo["2us"]
    u = (g2["mu1"] - g2["mu0"]) / g2["sep"]
    v = np.array([-u[1], u[0]])

    def locate(sel):
        mm = mask[sel]
        n = int(mm.sum())
        if n < 30:
            return {"n_bins": n}
        z = Xl[sel][mm]
        dd = z.mean(0) - g2["mu0"]
        return {"n_bins": n, "n_shots": int(mm.any(1).sum()), "mean": z.mean(0),
                "along_in_sep": float(dd @ u / g2["sep"]), "perp_in_sep": float(dd @ v / g2["sep"])}
    loc = {}
    for bb in (0, 1):
        loc[f"prep{bb}"] = locate(b == bb)
        loc[f"prep{bb}_startL"] = locate((b == bb) & (st == L))
        loc[f"prep{bb}_midentry"] = locate((b == bb) & (st != L))
        loc[f"per_config_prep{bb}"] = {str(c): locate((b == bb) & (cfg == c)) for c in range(16)}
    out["L_location_test"] = loc
    out["start_L_per_config"] = {f"prep{bb}": {str(c): frac(((st == L) & (b == bb) & (cfg == c)).sum(),
                                                            ((b == bb) & (cfg == c)).sum()) for c in range(16)}
                                 for bb in (0, 1)}
    # examples (TEST rows), ~20 per qubit
    ex = []
    gL, g1, g0 = gamma[:, :, L], gamma[:, :, 1], gamma[:, :, 0]
    rr = exit_stats(path, st == L, L, NB)
    iL = np.where(st == L)[0]
    e, dest = (rr[1], rr[2]) if rr else (np.zeros(0, int), np.zeros(0, int))

    def add(cat, rows, score, n):
        if len(rows) == 0:
            return
        for j in np.argsort(-score)[:n]:
            i = rows[j]
            ex.append({"cat": cat, "test_row": int(i), "raw_idx": int(idx[i]), "y": int(y[i]),
                       "prep_bit": int(b[i]), "cfg_other": int(cfg[i]),
                       "path_raw": "".join("01L"[s] for s in path_raw[i]),
                       "path": "".join("01L"[s] for s in path[i]),
                       "postL": np.round(gL[i], 3), "score": float(score[j])})
    for dcat, gd, nmax in ((1, g1, 7), (0, g0, 3)):
        m = (dest == dcat) & (e >= 4)
        rows = iL[m]
        sc = np.array([gL[i, 1:ee].mean() * (ee - 1) + gd[i, ee:].mean() * (NB - ee) for i, ee in zip(rows, e[m])])
        add(f"startL_to{dcat}", rows, sc, nmax)
    rows = iL[dest == -1]
    add("startL_stays", rows, gL[rows, 1:].sum(1), 3)
    _, has, kk, frm = mid_entries(path, np.ones(len(path), bool), st, NB)
    for s0, n in ((1, 3), (0, 2)):
        rows = np.where(has & (frm == s0) & (kk >= 4))[0]
        g_s = g1 if s0 == 1 else g0
        sc = np.array([g_s[i, 1:k].mean() * (k - 1) + gL[i, k:].mean() * (NB - k) for i, k in zip(rows, kk[rows])])
        add(f"mid_{s0}toL", rows, sc, n)
    rr1 = exit_stats(path, st == 1, 1, NB)
    if rr1:
        _, e1, d1 = rr1
        i1 = np.where(st == 1)[0]
        m = (d1 == 0) & (e1 >= 6) & (e1 <= 14)
        rows = i1[m]
        sc = np.array([g1[i, 1:ee].mean() * (ee - 1) + g0[i, ee:].mean() * (NB - ee) for i, ee in zip(rows, e1[m])])
        add("T1_1to0_ref", rows, sc, 2)
    out["examples"] = ex
    out["offsets_xt"] = offs
    np.savez_compressed(os.path.join(a.out, f"m3time_re_q{q}_arrays.npz"), path=path, path_raw=path_raw,
                        gL=gL.astype(np.float16), st=st.astype(np.int8))
    del gamma, logB
    # null simulation
    rho = d["noise_lag1_corr_prep0_core"]
    out["null_sim"] = null_stats(P, rho, rng, a.nnull)
    out["null_sim_white"] = null_stats(P, {"along": 0.0, "perp": 0.0}, rng, a.nnull)
    out["runtime_s"] = time.time() - t0
    with open(os.path.join(a.out, f"m3time_re_q{q}.json"), "w") as f:
        json.dump(jsonable(out), f, indent=1)
    log(q, f"done in {time.time()-t0:.0f}s")
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qubits", default="0,1,2,3,4")
    ap.add_argument("--nnull", type=int, default=200000)
    ap.add_argument("--procs", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    qs = [int(x) for x in a.qubits.split(",")]
    with Pool(min(a.procs, len(qs))) as pool:
        for q in pool.imap_unordered(run_qubit, [(q, a) for q in qs]):
            print(f"[m3time-re] finished q{q}", flush=True)
    if len(qs) == 5:
        flags = {}
        for q in qs:
            z = np.load(os.path.join(a.out, f"m3time_re_q{q}_arrays.npz"))
            flags[q] = {"startL": z["st"] == L, "everL_1us": (z["path"][:, :10] == L).any(1)}
        co = {}
        for f in ("startL", "everL_1us"):
            M = {}
            for q in qs:
                other = np.zeros(len(flags[q][f]), bool)
                for r in qs:
                    if r == q:
                        continue
                    a_, b_ = flags[q][f], flags[r][f]
                    pa, pb, pab = a_.mean(), b_.mean(), (a_ & b_).mean()
                    M[f"{NAMES[q]}&{NAMES[r]}"] = {"P_q": float(pa), "P_r": float(pb), "P_both": float(pab),
                                                  "lift": float(pab / (pa * pb)) if pa * pb > 0 else None}
                    other |= b_
                M[f"{NAMES[q]}_share_any_other"] = float((flags[q][f] & other).sum() / max(flags[q][f].sum(), 1))
            co[f] = M
        with open(os.path.join(a.out, "m3time_re_cooccurrence.json"), "w") as fh:
            json.dump(jsonable(co), fh, indent=1)
    print("[m3time-re] all done", flush=True)


if __name__ == "__main__":
    main()
