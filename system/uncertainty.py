"""Error bars on the §7 fits (I11): how sure are Lambda, the required distance and the spacetime?

Parametric bootstrap: for every (point, d) the failures are redrawn as Binomial(shots, fails/shots) and the fit of
analyze.py is redone (ln eps = a + b d over distances with >= 30 failures and P_L < 0.3); a spacetime ratio is taken
within one replicate, so numerator and baseline are drawn together. Fit-form sensitivity: the same data fitted over
every distance (baseline), without d = 5, over d >= 9, over the last three distances, and weighted by the failures.

Sets: disc_hi (§7.2-7.5), soft (§7.6), decoders (§7.7, union-find left out), and large once ler_large.jsonl exists
(sweep_large.py, d up to 49): there the fit over d <= 15 is compared with the fit over every distance, and each
large-d eps is compared with what the d <= 15 fit predicts.

Usage: python system/uncertainty.py [--reps 2000] [--data DIR]   (prints tables, writes DIR/uncertainty.json;
       DIR defaults to $ORAQLE_WORK/system)
"""
import argparse
import collections
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import MIN_FAILS, load, required_d
from oraqle_env import WORK  # analyze.py puts the repo root on sys.path

DATA = str(WORK / "system")  # was <oraqle-ics>/data/sims; override with --data
GATES_NS = 340
TARGET = 1e-9
DECODERS = ["mwpm_w", "mwpm_c", "mwpm_q16", "mwpm_q4", "mwpm_u"]
VARIANTS = {"all": lambda ds: ds, "no d5": lambda ds: [d for d in ds if d > 5], "d>=9": lambda ds: [d for d in ds if d >= 9],
            "last 3": lambda ds: ds[-3:], "weighted": lambda ds: ds}


def eps_of(fails, shots, R):
    P = fails / shots
    return (1 - (1 - 2 * P) ** (1 / R)) / 2 if 0 < P < 0.5 else float("nan")


def fit(pts, variant="all", dmax=None):
    """pts: {d: (shots, fails, R)}. Returns (a, b) or None."""
    ok = [d for d in sorted(pts) if pts[d][1] >= MIN_FAILS and pts[d][1] / pts[d][0] < 0.3 and (dmax is None or d <= dmax)]
    ok = VARIANTS[variant](ok)
    if len(ok) < 2:
        return None
    x = np.array(ok, dtype=float)
    y = np.log([eps_of(pts[d][1], pts[d][0], pts[d][2]) for d in ok])
    w = np.sqrt([pts[d][1] for d in ok]) if variant == "weighted" else None
    b, a = np.polyfit(x, y, 1, w=w)
    return a, b


def metrics(ab, T):
    if not ab or ab[1] >= 0:
        return dict(Lambda=math.exp(-2 * ab[1]) if ab else float("nan"), d=None, spacetime=None)
    d = required_d(ab[0], ab[1], TARGET)
    st = (2 * d * d - 1) * (d + 1) * (T + GATES_NS) if d else None
    return dict(Lambda=math.exp(-2 * ab[1]), d=d, spacetime=st)


def groups_hard(path):
    g = collections.defaultdict(dict)
    for (label, T, p, d), r in load(path).items():
        g[(label, T, p)][d] = (r["shots"], r["fails"], r["R"])
    return g


def groups_decoders(paths):
    acc, seen = collections.defaultdict(collections.Counter), set()
    for line in (l for p in paths for l in open(p)):
        r = json.loads(line)
        if (r["d"], r["T"], r["f"], r["pattern"]) in seen:
            continue
        seen.add((r["d"], r["T"], r["f"], r["pattern"]))
        a = acc[(r["T"], r["f"], r["d"])]
        a["shots"] += r["shots"]
        a["R"] = r["rounds"]
        for dec in DECODERS:
            a[dec] += r["fails"][dec]
    g = collections.defaultdict(dict)
    for (T, f, d), a in acc.items():
        for dec in DECODERS:
            g[(dec, T, f)][d] = (a["shots"], a[dec], a["R"])
    return g


def bootstrap(groups, base_key, reps, rng):
    """Median and 95% interval of Lambda, d and spacetime relative to base_key, per group."""
    keys = list(groups)
    draws = {k: collections.defaultdict(list) for k in keys}
    for _ in range(reps):
        res = {}
        for k in keys:
            pts = {d: (s, rng.binomial(s, f / s), R) for d, (s, f, R) in groups[k].items()}
            res[k] = metrics(fit(pts), k[1])
        base = res[base_key]["spacetime"]
        for k in keys:
            m = res[k]
            draws[k]["Lambda"].append(m["Lambda"])
            draws[k]["d"].append(m["d"] if m["d"] else np.nan)
            draws[k]["st_rel"].append(m["spacetime"] / base if m["spacetime"] and base else np.nan)
    out = {}
    for k in keys:
        out[k] = {}
        for q, v in draws[k].items():
            v = np.array(v, dtype=float)
            ok = v[np.isfinite(v)]
            out[k][q] = (np.median(ok), np.percentile(ok, 2.5), np.percentile(ok, 97.5), 1 - len(ok) / len(v)) if len(ok) else None
    return out


def sensitivity(groups, base_key):
    out = {}
    base = {v: metrics(fit(groups[base_key], v), base_key[1])["spacetime"] for v in VARIANTS}
    for k, pts in groups.items():
        out[k] = {}
        for v in VARIANTS:
            m = metrics(fit(pts, v), k[1])
            out[k][v] = dict(Lambda=m["Lambda"], d=m["d"], st_rel=m["spacetime"] / base[v] if m["spacetime"] and base[v] else None)
    return out


def fmt(t, f="{:.2f}"):
    if not t:
        return "     -"
    s = f"{f.format(t[0])} [{f.format(t[1])}, {f.format(t[2])}]"
    return s + (f" ({100 * t[3]:.0f}% none)" if t[3] > 0.005 else "")


def report(name, groups, base_key, reps, rng, select):
    print(f"\n== {name}: median [95% interval] over {reps} bootstrap replicates; spacetime rel. to {base_key}")
    boot = bootstrap(groups, base_key, reps, rng)
    sens = sensitivity(groups, base_key)
    print(f"{'point':<28}{'Lambda':>24}{'d @ 1e-9':>24}{'spacetime rel':>26}   d by fit form ({', '.join(VARIANTS)})")
    for k in sorted(groups, key=str):
        if not select(k):
            continue
        b = boot[k]
        dv = " ".join(str(sens[k][v]["d"] or "-") for v in VARIANTS)
        print(f"{str(k):<28}{fmt(b['Lambda']):>24}{fmt(b['d'], '{:.0f}'):>24}{fmt(b['st_rel']):>26}   {dv}")
    return {str(k): dict(bootstrap=boot[k], forms=sens[k]) for k in groups}


def large_check(hard, dec, path):
    """Fit over d <= 15 against every distance, and each large-d eps against the d <= 15 prediction."""
    out = {}
    print("\n== large distances: Lambda and d @ 1e-9 from d <= 15 vs from every distance; measured / predicted eps")
    big = collections.defaultdict(lambda: collections.Counter())
    for line in open(path):
        r = json.loads(line)
        k = (r["decoder"], r["label"], r["T"], round(r["p"], 9), r["d"])
        big[k]["shots"] += r["shots"]
        big[k]["fails"] += r["fails"]
        big[k]["R"] = r["rounds"]
    for dec_name, label, T, p in sorted({k[:4] for k in big}, key=str):
        small = dict(hard[(label, T, p)]) if dec_name == "w" else dict(dec[("mwpm_u", T, 0.0)])
        allp = dict(small)
        for (dn, lb, TT, pp, d), c in big.items():
            if (dn, lb, TT, pp) == (dec_name, label, T, p) and c["shots"]:
                allp[d] = (c["shots"], c["fails"], c["R"])
        f_small, f_all = fit(small, dmax=15), fit(allp)
        m_small, m_all = metrics(f_small, T), metrics(f_all, T)
        ratios = {}
        for d in sorted(allp):
            if d > 15 and allp[d][1] > 0:
                s, f, R = allp[d]
                lo, hi = (max(f - 1.96 * math.sqrt(f), 0.5), f + 1.96 * math.sqrt(f))
                pred = math.exp(f_small[0] + f_small[1] * d)
                ratios[d] = (eps_of(f, s, R) / pred, eps_of(lo, s, R) / pred, eps_of(hi, s, R) / pred, f)
        key = f"{dec_name}/{label}/{T}"
        out[key] = dict(small=m_small, all=m_all, ratios=ratios)
        print(f"{key:<22} Lambda {m_small['Lambda']:.2f} -> {m_all['Lambda']:.2f}   d {m_small['d']} -> {m_all['d']}   "
              + "  ".join(f"d{d}: {r[0]:.2f} [{r[1]:.2f},{r[2]:.2f}] ({r[3]} f)" for d, r in ratios.items()))
    return out


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--data", default=DATA)
    a = ap.parse_args()
    DATA = a.data
    rng = np.random.default_rng(20261029)
    out = {}
    hard = groups_hard(os.path.join(DATA, "ler_disc_hi.jsonl"))
    base = next(k for k in hard if k[0] == "MCMit-CNN" and k[1] == 1000)
    out["disc_hi"] = report("disc_hi (§7.2-7.5)", hard, base, a.reps, rng,
                            lambda k: k[0] in ("MCMit-CNN", "QubiCML", "HERQULES") and k[1] >= 400)
    soft = groups_hard(os.path.join(DATA, "ler_soft.jsonl"))
    base = next(k for k in soft if k[0] == "soft_f0" and k[1] == 1000)
    out["soft"] = report("soft (§7.6)", soft, base, a.reps, rng,
                         lambda k: (k[1], k[0]) in {(T, "soft_f0") for T in (400, 600, 800, 1000)}
                         | {(400, "soft_f0.1"), (600, "soft_f0.05"), (800, "soft_f0.02"), (1000, "soft_f0.02")})
    dec = groups_decoders([os.path.join(DATA, f) for f in ("ler_decoders.jsonl", "ler_decoders_soft13.jsonl",
                                                          "ler_decoders_soft11.jsonl")])
    out["decoders"] = report("decoders (§7.7)", dec, ("mwpm_w", 1000, 0.0), a.reps, rng, lambda k: True)
    large = os.path.join(DATA, "ler_large.jsonl")
    if os.path.exists(large):
        out["large"] = large_check(hard, dec, large)
    json.dump(out, open(os.path.join(DATA, "uncertainty.json"), "w"), indent=1, default=lambda o: None if o is None else float(o))


if __name__ == "__main__":
    main()
