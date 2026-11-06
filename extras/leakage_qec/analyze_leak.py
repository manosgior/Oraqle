#!/usr/bin/env python3
"""analyze_leak.py -- Lambda, required distance and spacetime from the sweep_leak.py outputs, same definitions as system/analyze.py.

Each decoder of a job gives one curve. Fits use distances with >= MIN_FAILS failures and P_L < 0.3. A curve whose error does not fall
with distance has Lambda <= 1 and is above threshold. Round time: T + 340 ns, plus the reduction-unit slot for policies that have one.
Intervals on Lambda are 95% bootstrap intervals: leakage histories are resampled within each distance, with Poisson noise on the
failures of each history, so they cover both the choice of histories and the shot noise.
"""
import argparse, collections, json, math, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
MIN_FAILS, GATES_NS, TARGET, N_BOOT = 30, 340, 1e-9, 400
import numpy as np


def eps(P, R):
    return (1 - (1 - 2 * P) ** (1 / R)) / 2 if P < 0.5 else float("nan")


def fit(pts):
    ok = [(d, r["eps"]) for d, r in sorted(pts.items()) if r["fails"] >= MIN_FAILS and r["P"] < 0.3]
    if len(ok) < 2:
        return None
    b, a = np.polyfit([d for d, _ in ok], np.log([e for _, e in ok]), 1)
    return a, b, math.exp(-2 * b), [d for d, _ in ok]


def req_d(a, b):
    if b >= 0:
        return None
    for d in range(3, 2001, 2):
        e = math.exp(a + b * d)
        if (1 - (1 - 2 * e) ** (d + 1)) / 2 <= TARGET:
            return d
    return None


def boot_lambda(pts, used, rng):
    vals = []
    for _ in range(N_BOOT):
        xs, ys = [], []
        for d in used:
            f, s, R = pts[d]["pat_fails"], pts[d]["pat_shots"], pts[d]["rounds"]
            i = rng.integers(len(f), size=len(f))
            P = rng.poisson(f[i]).sum() / s[i].sum()
            if not 0 < P < 0.5:
                break
            xs.append(d)
            ys.append(math.log(eps(P, R)))
        else:
            vals.append(math.exp(-2 * np.polyfit(xs, ys, 1)[0]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def curves(path, keys):
    rng = np.random.default_rng(1)
    G = collections.defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        for dec, f in r["fails"].items():
            P = f / r["shots"]
            k = tuple((r.get("leak") or {}).get("p_create") if x == "p_create" else r.get(x) for x in keys) + (dec,)
            G[k][r["d"]] = dict(P=P, fails=f, shots=r["shots"], eps=eps(P, r["rounds"]), t_extra=r["t_extra_ns"], rounds=r["rounds"],
                                pat_fails=np.array(r["pat_fails"][dec], float), pat_shots=np.array(r["pat_shots"], float))
    out = {}
    for k, pts in G.items():
        f = fit(pts)
        s = dict(eps={d: p["eps"] for d, p in sorted(pts.items())}, P={d: p["P"] for d, p in sorted(pts.items())}, fails={d: p["fails"] for d, p in sorted(pts.items())})
        s["t_r_us"] = (k[keys.index("T")] + next(iter(pts.values()))["t_extra"] + GATES_NS) / 1000
        if f:
            a, b, lam, used = f
            s.update(Lambda=lam, fit_ds=used, ci=boot_lambda(pts, used, rng))
            d = req_d(a, b)
            if d and s["ci"][0] > 1:       # a Lambda compatible with 1 has no meaningful required distance
                s.update(d=d, qubits=2 * d * d - 1, spacetime=(2 * d * d - 1) * (d + 1) * s["t_r_us"])
        out[k] = s
    return out


def paired_ratio(rows, dec, rng, n=300):
    """Lambda(dec) / Lambda(base) for decoders of one job, which decode the same shots: the same histories are resampled for both."""
    def lam(dc, idx):
        xs, ys = [], []
        for d, r in sorted(rows.items()):
            f, sh = np.array(r["pat_fails"][dc], float)[idx[d]].sum(), np.array(r["pat_shots"], float)[idx[d]].sum()
            if f >= MIN_FAILS and f / sh < 0.3:
                xs.append(d)
                ys.append(math.log(eps(f / sh, r["rounds"])))
        return math.exp(-2 * np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else float("nan")
    v = []
    for _ in range(n):
        idx = {d: rng.integers(len(r["pat_shots"]), size=len(r["pat_shots"])) for d, r in rows.items()}
        v.append(lam(dec, idx) / lam("base", idx))
    v = np.array(v)
    v = v[np.isfinite(v)]
    return [float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > n / 3 else None


def L(s):
    if not s or "Lambda" not in s:
        return "      above"
    lo, hi = s["ci"]
    return f"{s['Lambda']:4.2f}[{lo:4.2f},{hi:4.2f}]"


def ST(s, ref):
    return f"d={s['d']:>3} ST={s['spacetime'] / ref:5.2f}" if s and "spacetime" in s else "d=  - ST=    -"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "leak_summary.json"))
    out = ap.parse_args().out
    d = HERE / "sweeps"
    print("=== soft readout, no leakage: Lambda [95%], d and spacetime relative to hard 1 us")
    S = curves(d / "ler_soft.jsonl", ["T", "level"])
    ref = S[(1000, "hard", "base")]["spacetime"]
    for T in (400, 600, 800, 1000):
        for lv in ("hard", "flag0.02", "flag0.05", "flag0.1", "bits1", "bits2", "bits3", "bits4"):
            s = S.get((T, lv, "base"))
            print(f"  T={T} {lv:9s} {L(s)} {ST(s, ref)}")
    print("\n=== tolerance: Lambda by leakage rate per measurement (survival, tpr and fpr of q3_like)")
    Tl = curves(d / "ler_tolerance.jsonl", ["T", "p_ld", "p_create", "policy"])
    rates = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
    for T in (600, 1000):
        for ld in (0.75, 0.25):
            print(f"T={T} p_ld={ld}    " + " ".join(f"{p:>15g}" for p in rates))
            for pol, dec in (("none", "base"), ("none", "uniform"), ("none", "flag"), ("none", "oracle"), ("flag", "base"), ("flag", "flag"), ("always", "base")):
                print(f"   {pol:7s}{dec:8s}", " ".join(f"{L(Tl.get((T, ld, pc, pol, dec))):>15s}" for pc in rates))
    print("\n=== measured scenarios")
    Sc = curves(d / "ler_scenarios.jsonl", ["scenario", "T", "p_ld", "policy"])
    for sc in ("pooled", "q3_like"):
        for ld in (0.75, 0.25):
            print(f"{sc} p_ld={ld}")
            for pol, dec in (("none", "base"), ("none", "uniform"), ("none", "flag"), ("none", "oracle"), ("flag", "base"), ("flag", "flag"), ("always", "base")):
                print(f"   {pol:7s}{dec:8s}", " | ".join(f"{L(Sc.get((sc, T, ld, pol, dec)))} {ST(Sc.get((sc, T, ld, pol, dec)), ref)}" for T in (400, 600, 800, 1000)))
    print("\n=== combined: always-on reduction with soft readout, and the slot alone without leakage")
    C = curves(d / "ler_combined.jsonl", ["scenario", "T", "level"])
    for sc in ("no_leak_with_slot", "pooled", "q3_like"):
        print(sc)
        for lv in ("hard", "bits2", "bits3", "bits4"):
            print(f"   {lv:6s}", " | ".join(f"{L(C.get((sc, T, lv, 'base')))} {ST(C.get((sc, T, lv, 'base')), ref)}" for T in (400, 600, 800, 1000)))
    print("\n=== reduction unit: flag against always-on (p_ld 0.75, hard readout)")
    U = curves(d / "ler_lru.jsonl", ["scenario", "T", "e_lru", "t_lru", "fpr_target"])
    for sc in ("pooled", "q3_like"):
        for T in (600, 1000):
            print(f"{sc} T={T}   e_lru t_lru  always / flag@1% / 5% / 20% / 50% FPR (decoder base, then flag-aware)")
            for e, t in ((0.002, 100), (0.002, 200), (0.002, 400), (0.01, 200), (0.03, 200)):
                row = [L(U.get((sc, T, e, t, "always", "base")))]
                row += [L(U.get((sc, T, e, t, f, "base"))) for f in ("0.01", "0.05", "0.2", "0.5")]
                print(f"   {e:6g} {t:4d} base ", " ".join(row))
                print(f"   {'':11s} flag ", " " * 16 + " ".join(L(U.get((sc, T, e, t, f, "flag"))) for f in ("0.01", "0.05", "0.2", "0.5")))
    print("\n=== removal probability of the unit: Lambda, always-on and flag at 50% FPR (p_ld 0.75, hard readout)")
    E = curves(d / "ler_eta.jsonl", ["scenario", "T", "eta", "policy"])
    etas = (0.5, 0.9, 0.95, 0.99, 0.999)
    print("                        " + " ".join(f"{x:>15g}" for x in etas))
    for sc in ("pooled", "q3_like"):
        for T in (600, 1000):
            for pol in ("always", "flag"):
                print(f"   {sc:8s}{T:5d} {pol:7s}", " ".join(f"{L(E.get((sc, T, x, pol, 'base'))):>15s}" for x in etas))
    print("\n=== decoder knowledge without removal: Lambda(decoder) / Lambda(base), same shots, paired")
    G, rng, Pr = collections.defaultdict(dict), np.random.default_rng(2), {}
    for line in open(d / "ler_tolerance.jsonl"):
        r = json.loads(line)
        if r["policy"] == "none":
            G[(r["T"], r["p_ld"], r["leak"]["p_create"])][r["d"]] = r
    for k in sorted(G):
        if k[2] in (1e-3, 3e-3, 1e-2):
            Pr[k] = {dc: paired_ratio(G[k], dc, rng) for dc in ("uniform", "flag", "oracle")}
            print("  ", k, {dc: (f"{v[0]:.2f}[{v[1]:.2f},{v[2]:.2f}]" if v else "n/a") for dc, v in Pr[k].items()})
    print("\n=== fit range, no leakage, hard readout: Lambda falls with the distances fitted")
    FR = {}
    rows = [json.loads(x) for x in open(d / "ler_soft.jsonl")]
    for T in (600, 1000):
        for ds in ((5, 7, 9, 11), (5, 7, 9, 11, 13), (7, 9, 11, 13), (9, 11, 13)):
            pts = {x["d"]: dict(P=x["fails"]["base"] / x["shots"], fails=x["fails"]["base"], eps=eps(x["fails"]["base"] / x["shots"], x["rounds"]))
                   for x in rows if x["T"] == T and x["level"] == "hard" and x["d"] in ds}
            FR[str((T, ds))] = fit(pts)[2]
            print(f"   T={T} d={ds}: {FR[str((T, ds))]:.2f}")
    def strs(D):
        return {str(k): v for k, v in D.items()}
    json.dump(dict(soft=strs(S), tolerance=strs(Tl), scenarios=strs(Sc), combined=strs(C), lru=strs(U), eta=strs(E), paired_decoders={str(k): v for k, v in Pr.items()}, fit_range=FR, reference_spacetime=ref),
              open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
