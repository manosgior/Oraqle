"""vstatistics_line.py -- follow-up (tag vstatistics): LINE-referenced z.

The shared frame aligns the MEAN |0>->|1> difference, which the off-axis population pulls; the core-to-core
line is tilted (Q3: c1 core ~0.3 ADC below c0).  A per-state-centre z then puts decayed prepared-|1> shots
and excited prepared-|0> shots off-centre.  Here z is measured from the core-to-core line of each
neighbour configuration (TRAIN-fitted, applied to TEST), two variants:
  clamp : perp_line(a) = p0 + (p1-p0)*clip((a-a0)/(a1-a0),0,1)
  rot   : rotate about c0 so that c0->c1 is horizontal (line extended)
c0/c1 = (along histogram mode, perp centre by truncated-normal ML on [mu-4s, mu+1s] with s fixed to the
|0> width, using shots within 1 sigma of the along mode).  sigma0 from the prepared-|0> state (same fit).
Outputs E(k) scans, CNN cross-tab per prepared bit with error-core centring check, drift amplitude,
cross-mirror, Q3 prepared-|0> decomposition by start state.
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import math
import time

import numpy as np
from scipy import optimize
from scipy.special import ndtr

T0 = time.time()
D = str(LP.FEAT)
OUT = LP.outdir(LP.stage("vstatistics"))
PAPER = [f"Q{q+1}" for q in range(5)]
KS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
Z95 = 1.959963984540054


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def Qtail(k):
    return 0.5 * math.erfc(k / math.sqrt(2))


def mirror(npl, nmi, n):
    n = max(n, 1)
    e = (npl - nmi) / n
    h = Z95 * math.sqrt(npl + nmi) / n
    return [e, e - h, e + h]


def wilson(k, n):
    if n == 0:
        return [None, None, None]
    p = k / n
    den = 1 + Z95 ** 2 / n
    c = (p + Z95 ** 2 / (2 * n)) / den
    h = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n)) / den
    return [p, c - h, c + h]


def tnfit(x, mu, s, lo_k=4.0, hi_k=1.0, fix_s=None, iters=40):
    x = np.asarray(x, np.float64)
    if fix_s is not None:
        s = fix_s
    for _ in range(iters):
        a, b = mu - lo_k * s, mu + hi_k * s
        xi = x[(x > a) & (x < b)]
        n = len(xi)
        c = xi.mean(); d = xi - c
        S1, S2 = d.sum(), (d * d).sum()
        ac, bc = a - c, b - c
        if fix_s is None:
            def nll(p):
                m, ls = p; sd = math.exp(ls)
                Z = ndtr((bc - m) / sd) - ndtr((ac - m) / sd)
                return 0.5 * (S2 - 2 * m * S1 + n * m * m) / sd ** 2 + n * ls + n * math.log(max(Z, 1e-300))
            r = optimize.minimize(nll, [mu - c, math.log(s)], method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-9, maxiter=4000))
            m2, s2 = r.x[0] + c, math.exp(r.x[1])
        else:
            sd = fix_s
            def nll1(p):
                m = p[0]
                Z = ndtr((bc - m) / sd) - ndtr((ac - m) / sd)
                return 0.5 * (S2 - 2 * m * S1 + n * m * m) / sd ** 2 + n * math.log(max(Z, 1e-300))
            r = optimize.minimize(nll1, [mu - c], method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-10, maxiter=4000))
            m2, s2 = r.x[0] + c, sd
        conv = abs(m2 - mu) < 1e-6 * s and abs(s2 - s) < 1e-6 * s
        mu, s = float(m2), float(s2)
        if conv:
            break
    return mu, s


def amode(aa, s):
    lo, hi = np.percentile(aa, [0.5, 99.5])
    nb = max(10, int((hi - lo) / (s / 4)) + 1)
    h, e = np.histogram(aa, bins=nb, range=(lo, hi))
    k = np.exp(-0.5 * (np.arange(-8, 9) / 2.0) ** 2)
    hs = np.convolve(h, k / k.sum(), mode="same")
    i = int(np.argmax(hs))
    return 0.5 * (e[i] + e[i + 1])


log("loading")
DAT = {}
for sp in ("train", "test"):
    DAT[sp] = dict(F1=np.load(f"{D}/feat1us_{sp}.npy"), F=np.load(f"{D}/feat_{sp}.npy"),
                   y=np.load(f"{D}/y_{sp}.npy").astype(np.int64))
    DAT[sp]["bits"] = ((DAT[sp]["y"][:, None] >> np.arange(5)) & 1).astype(np.int8)
log("loaded")


def line_frame(W, y, q):
    A, P = W[:, 0], W[:, 1]
    fr = {}
    for s in range(32):
        if (s >> q) & 1:
            continue
        s1 = s | (1 << q)
        m0 = y == s
        p0all = P[m0]
        med = float(np.median(p0all)); mad = float(1.4826 * np.median(np.abs(p0all - med)))
        _, sg0 = tnfit(p0all, med, mad)
        a0 = amode(A[m0], sg0)
        n0 = m0 & (np.abs(A - a0) < sg0)
        p0, _ = tnfit(P[n0], float(np.median(P[n0])), sg0, fix_s=sg0)
        m1 = y == s1
        a1 = amode(A[m1], sg0)
        n1 = m1 & (np.abs(A - a1) < sg0)
        p1, sg1 = tnfit(P[n1], float(np.median(P[n1])), sg0, fix_s=sg0)
        _, sg1own = tnfit(P[n1], p1, sg0)
        fr[s] = dict(a0=a0, p0=p0, a1=a1, p1=p1, sig=sg0, sig1own=sg1own)
    return fr


def zline(W, y, q, fr):
    A, P = W[:, 0], W[:, 1]
    zc = np.empty(len(y)); zr = np.empty(len(y)); zs = np.empty(len(y))
    for s in range(32):
        s0 = s & ~(1 << q)
        f = fr[s0]
        m = y == s
        t = np.clip((A[m] - f["a0"]) / (f["a1"] - f["a0"]), 0, 1)
        zc[m] = (P[m] - (f["p0"] + (f["p1"] - f["p0"]) * t)) / f["sig"]
        ang = math.atan2(f["p1"] - f["p0"], f["a1"] - f["a0"])
        zr[m] = (-(A[m] - f["a0"]) * math.sin(ang) + (P[m] - f["p0"]) * math.cos(ang)) / f["sig"]
        zs[m] = (P[m] - (f["p1"] if (s >> q) & 1 else f["p0"])) / f["sig"]    # per-state centre (r-like)
    return zc, zr, zs


def scan(z, n):
    t = {}
    for k in KS:
        npl, nmi = int((z > k).sum()), int((z < -k).sum())
        t[f"k{k}"] = dict(n_plus=npl, n_minus=nmi, gauss=n * Qtail(k), E=mirror(npl, nmi, n))
    return t


res = {"tag": "vstatistics", "what": "line-referenced z, TRAIN frame -> TEST", "windows": {}}
pb = np.load(str(LP.REPORTS / "preds_MCMitCNN_pb.npy")).astype(np.int8)
tb = np.load(str(LP.REPORTS / "preds_MCMitCNN_tb.npy")).astype(np.int8)
bte = DAT["test"]["bits"]; yte = DAT["test"]["y"]
assert np.array_equal(tb, bte)
err = pb != tb
ZK = {}
WIN = {"fir1us": lambda d, q: d["F1"][:, q].mean(1), "fir400": lambda d, q: d["F"][:, q, 0:4].mean(1),
       "fir2us": lambda d, q: d["F"][:, q].mean(1)}
for wn, fn in WIN.items():
    rw = {}
    for q in range(5):
        Wtr = np.asarray(fn(DAT["train"], q), np.float64)
        Wte = np.asarray(fn(DAT["test"], q), np.float64)
        fr = line_frame(Wtr, DAT["train"]["y"], q)
        zc, zr, zs = zline(Wte, yte, q, fr)
        tilt = [math.degrees(math.atan2(f["p1"] - f["p0"], f["a1"] - f["a0"])) for f in fr.values()]
        dp = [(f["p1"] - f["p0"]) / f["sig"] for f in fr.values()]
        rq = {"tilt_deg_median": float(np.median(tilt)), "tilt_deg_range": [float(min(tilt)), float(max(tilt))],
              "p1_minus_p0_over_sig0_median": float(np.median(dp)),
              "sig1own_over_sig0_median": float(np.median([f["sig1own"] / f["sig"] for f in fr.values()]))}
        for vn, z in (("clamp", zc), ("rot", zr), ("statecentre", zs)):
            for b in (0, 1):
                m = bte[:, q] == b
                rq[f"{vn}_prep{b}"] = scan(z[m], int(m.sum()))
            if wn == "fir1us":
                ZK[(vn, q)] = z
        rw[PAPER[q]] = rq
        log(f"{wn} {PAPER[q]} tilt {rq['tilt_deg_median']:+.2f} deg dp {rq['p1_minus_p0_over_sig0_median']:+.3f} sig "
            + " ".join(f"{vn}: p1 {100*rq[f'{vn}_prep1']['k3.0']['E'][0]:.3f}% (n- {rq[f'{vn}_prep1']['k3.0']['n_minus']}) p0 {100*rq[f'{vn}_prep0']['k3.0']['E'][0]:.3f}%"
                       for vn in ("clamp", "rot", "statecentre")))
    res["windows"][wn] = rw

# CNN cross-tab
cnn = {}
for vn in ("clamp", "rot", "statecentre"):
    cv = {}
    totE, totP, totM = 0, 0, 0
    for q in range(5):
        z = ZK[(vn, q)]
        e = err[:, q]
        d = {"n_err": int(e.sum())}
        for b in (0, 1, "all"):
            m = np.ones(len(z), bool) if b == "all" else bte[:, q] == b
            eb = e & m
            ne = int(eb.sum())
            core = eb & (np.abs(z) < 2.5)
            npl, nmi = int((eb & (z > 3)).sum()), int((eb & (z < -3)).sum())
            lp, lm = int((m & (z > 3)).sum()), int((m & (z < -3)).sum())
            d[f"prep{b}"] = dict(n_err=ne, err_core_median_z=float(np.median(z[core])) if core.any() else None,
                                 share_mirror3=mirror(npl, nmi, ne), n_plus=npl, n_minus=nmi,
                                 leaked_shots_mirror3=mirror(lp, lm, int(m.sum())),
                                 misread_on_leaked_mirror=(npl - nmi) / max(lp - lm, 1))
        cv[PAPER[q]] = d
        if q != 3:
            totE += d["prepall"]["n_err"]; totP += d["prepall"]["n_plus"]; totM += d["prepall"]["n_minus"]
    cv["pooled_Q1Q2Q3Q5"] = dict(n_err=totE, n_plus=totP, n_minus=totM, share=mirror(totP, totM, totE))
    cnn[vn] = cv
    log(f"CNN {vn}: " + " ".join(f"{PAPER[q]} {100*cv[PAPER[q]]['prepall']['share_mirror3'][0]:.2f}% (p0 {100*cv[PAPER[q]]['prep0']['share_mirror3'][0]:.2f} c{cv[PAPER[q]]['prep0']['err_core_median_z']:+.2f}, p1 {100*cv[PAPER[q]]['prep1']['share_mirror3'][0]:.2f} c{cv[PAPER[q]]['prep1']['err_core_median_z']:+.2f}; misread p1 {100*cv[PAPER[q]]['prep1']['misread_on_leaked_mirror']:.2f} p0 {100*cv[PAPER[q]]['prep0']['misread_on_leaked_mirror']:.1f})" for q in (0, 1, 2, 4))
        + f" pooled {100*cv['pooled_Q1Q2Q3Q5']['share'][0]:.3f}%")
res["cnn"] = cnn

# Q3 prepared-|0>: split by start state (feat bins 1-2 along vs midpoint), fir1us clamp z
q = 2
F = DAT["test"]["F"]
early = F[:, q, 1:3, 0].mean(1).astype(np.float64)
z = ZK[("clamp", q)]
dec = {}
for b in (0, 1):
    m = bte[:, q] == b
    # midpoint from early-window medians of prepared bits
    mid = 0.5 * (np.median(early[bte[:, q] == 0]) + np.median(early[bte[:, q] == 1]))
    st1 = early > mid
    for nm, mm in (("start1like", m & st1), ("start0like", m & ~st1)):
        dec[f"prep{b}_{nm}"] = dict(n=int(mm.sum()), frac_of_prep=float(mm.sum() / m.sum()),
                                    E3=mirror(int((z[mm] > 3).sum()), int((z[mm] < -3).sum()), int(mm.sum())),
                                    E3_contrib_of_prep=mirror(int((z[mm] > 3).sum()), int((z[mm] < -3).sum()), int(m.sum())))
res["Q3_decomposition_by_early_state"] = dec
log("Q3 decomposition", json.dumps({k: [round(v['frac_of_prep'], 4), round(100 * v['E3'][0], 3), round(100 * v['E3_contrib_of_prep'][0], 3)] for k, v in dec.items()}))

# drift amplitude within blocks (train and test), along of prepared-|0> qubit, 20 slices
dr = {}
for q in range(5):
    out = []
    for sp in ("train", "test"):
        d = DAT[sp]
        A = d["F1"][:, q].mean(1)[:, 0].astype(np.float64)
        ratios = []
        for s in range(32):
            if (s >> q) & 1:
                continue
            a = A[d["y"] == s]
            sl = np.array_split(a, 20)
            med = np.array([np.median(x) for x in sl])
            se = 1.2533 * 1.4826 * np.median(np.abs(a - np.median(a))) / math.sqrt(len(a) / 20)
            ratios.append(float(med.std(ddof=1) / se))
        out.append(dict(split=sp, sd_slice_medians_over_noise_se_median=float(np.median(ratios))))
    dr[PAPER[q]] = out
log("drift", json.dumps(dr))
res["drift_amplitude"] = dr

with open(OUT + "/vstatistics_line.json", "w") as f:
    json.dump(res, f, indent=1)
log("done")
