#!/usr/bin/env python3
"""sweep_leak.py -- logical error with readout-induced leakage, flags, reduction units and soft readout.

Each job is one (experiment, d, T, leakage, policy, soft level) point. A job draws patterns until the best decoder has
TARGET_FAILS failures or the shot cap is reached, but never on fewer than the minimum number of patterns, and appends one JSON line with shots and failures per decoder. All decoders
of a job decode the same shots, so comparisons within a job are paired.

Experiments (--exp):
  soft        no leakage: hard, k-bit soft and 1-bit flags that keep the bit, with per-measurement weights
  tolerance   leakage rate as a free axis, which does not rely on the rates of this device
  scenarios   the measured rates of make_params.py at each readout length
  lru         reduction-unit error and duration, and the flag's operating point: flag against always-on
  combined    always-on reduction with soft readout, and what the unit costs without any leakage
  eta         how well the reduction unit must remove leakage, for always-on and for a permissive flag

Usage (oraqle-latsim image): python sweep_leak.py --exp soft --params leak_params.json --out DIR/ler_soft.jsonl
"""
import argparse, json, os, pathlib, sys, time, zlib
from multiprocessing import Pool

import numpy as np
import pymatching
import stim

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "system"), str(pathlib.Path(__file__).resolve().parent)]
from sweep_ler import build  # noqa: E402
import leaksim as S  # noqa: E402

DIST = [5, 7, 9, 11, 13]
CAP = {5: 3e6, 7: 3e6, 9: 2e6, 11: 1.5e6, 13: 1e6}
MAX_PER_PATTERN = {5: 20000, 7: 20000, 9: 20000, 11: 20000, 13: 20000}
TARGET_FAILS = 400
MIN_PATTERNS_LEAK = 48                                  # a history is one random draw: never stop on fewer
MIN_PATTERNS_SOFT = 16
PILOT_SHOTS = 500
T_LRU_NS = 200
P_LD = [0.75, 0.25]
POLICIES = {"none": ("base", "uniform", "flag", "oracle"), "flag": ("base", "flag"), "always": ("base",)}


def matcher(c):
    return pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True))


def fails(m, det, obs):
    return int(np.sum(np.any(m.decode_batch(det, bit_packed_shots=True, bit_packed_predictions=True) != obs, axis=1)))


def run(job):
    d, T = job["d"], job["T"]
    t_extra = job.get("t_lru", T_LRU_NS) if job["policy"] in ("flag", "always") else 0
    e0 = job["e"]
    c0, _, t_circ = build(d, T + t_extra, e0, "ibm_boston")
    lay = S.Layout(c0)
    n_meas = c0.num_measurements
    lk = job.get("leak") or dict(p_create=0.0, survive=0.0)
    p = lk["p_create"]
    occ = p / (1 - lk["survive"] + p) if p > 0 else 0.0
    tpr, fpr = lk.get("tpr", 1.0), lk.get("fpr", 0.0)
    prec = tpr * occ / (tpr * occ + fpr * (1 - occ)) if occ > 0 and (tpr > 0 or fpr > 0) else 1.0
    model = S.LeakModel(p_create=p, survive=lk["survive"], frac_from_0=lk.get("frac_from_0", 0.0), p_ld=job.get("p_ld", 0.75),
                        tpr=tpr, fpr=fpr, precision=prec, lru=job["policy"], eta=job.get("eta", 0.99), e_lru=job.get("e_lru", 0.002))
    levels = job.get("levels")
    seed = zlib.crc32(json.dumps({k: v for k, v in job.items() if k != "levels"}, sort_keys=True).encode())
    rng = np.random.default_rng(seed)
    decs = POLICIES[job["policy"]] if p > 0 else ("base",)
    if p > 0 or job["policy"] != "none":
        min_pat = MIN_PATTERNS_LEAK
    elif levels:
        min_pat = MIN_PATTERNS_SOFT
    else:
        min_pat = 1
    shared = {}                                         # decoders that do not depend on the pattern
    counts = {k: 0 for k in decs}
    pat_fails = {k: [] for k in decs}
    pat_shots = []
    per_pattern = levels is not None or job["policy"] != "none"
    cap, n, t0 = CAP[d], PILOT_SHOTS, time.time()
    while True:
        h = S.draw_history(lay, model, rng)
        soft = S.draw_soft(n_meas, levels, rng) if levels else None
        truth = S.truth_circuit(lay, h, model, soft=soft)
        det, obs = truth.compile_detector_sampler(seed=int(rng.integers(2**31))).sample(n, separate_observables=True, bit_packed=True)
        pat_shots.append(n)
        for k in decs:
            if k in ("base", "uniform") and not per_pattern:
                if k not in shared:
                    shared[k] = matcher(S.belief_circuit(lay, h, model, soft=soft, mode=k))
                m = shared[k]
            else:
                m = matcher(S.belief_circuit(lay, h, model, soft=soft, mode=k))
            f = fails(m, det, obs)
            counts[k] += f
            pat_fails[k].append(f)
        shots, best = sum(pat_shots), max(1, min(counts.values()))
        if len(pat_shots) >= min_pat and (best >= TARGET_FAILS or shots >= cap or best > 0.4 * shots):
            break
        # spread the remaining work over the remaining patterns, at least the shots that keep a pattern worthwhile
        remaining = max(1, min_pat - len(pat_shots))
        need = (TARGET_FAILS - best) * shots / best if best < TARGET_FAILS else 0
        n = int(np.clip(need / remaining, 200, MAX_PER_PATTERN[d]))
    return dict(job, shots=shots, patterns=len(pat_shots), fails=counts, pat_fails=pat_fails, pat_shots=pat_shots, rounds=2 * (d + 1),
                circuit_ns=t_circ, t_extra_ns=t_extra, precision=prec, occupancy=occ, wall_s=time.time() - t0)


def jobs_soft(P):
    out = []
    for T, v in P["T"].items():
        tab = v["confidence_table"]
        for d in DIST:
            base = dict(exp="soft", d=d, T=int(T), policy="none", e=tab["hard_err"], levels=None, level="hard")
            out.append(base)
            for b in (1, 2, 3, 4):
                out.append(dict(base, levels=S.soft_table(tab, b), level=f"bits{b}"))
            for f in (0.02, 0.05, 0.10):
                out.append(dict(base, levels=S.flag_fraction_table(tab, f), level=f"flag{f:g}"))
    return out


def jobs_tolerance(P):
    out = []
    for T in ("600", "1000"):
        v = P["T"][T]
        q3 = v["q3_like"]
        for pc in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2):
            for ld in P_LD:
                for pol in POLICIES:
                    for d in DIST[:4]:
                        out.append(dict(exp="tolerance", d=d, T=int(T), policy=pol, e=v["confidence_table"]["hard_err"], p_ld=ld,
                                        leak=dict(p_create=pc, survive=q3["survive"], frac_from_0=q3["frac_from_0"], tpr=q3["tpr"], fpr=q3["fpr"]),
                                        level="hard"))
    return out


def jobs_scenarios(P):
    out = []
    for T, v in P["T"].items():
        for sc in ("pooled", "q3_like"):
            lk = v[sc]
            for ld in P_LD:
                for pol in POLICIES:
                    for d in DIST:
                        out.append(dict(exp="scenarios", scenario=sc, d=d, T=int(T), policy=pol, e=v["confidence_table"]["hard_err"], p_ld=ld,
                                        leak=_with_roc(lk, "0.01"), level="hard"))
    return out


def _with_roc(lk, t):
    return dict({k: v for k, v in lk.items() if k != "roc"}, tpr=lk["roc"][t]["tpr"], fpr=lk["roc"][t]["fpr"])


def jobs_lru(P):
    """How the reduction unit's error and duration, and the flag's operating point, decide flag against always-on."""
    out = []
    for T in ("600", "1000"):
        v = P["T"][T]
        for sc in ("pooled", "q3_like"):
            for e_lru in (0.002, 0.01, 0.03):
                for t_lru in (100, 200, 400):
                    if e_lru != 0.002 and t_lru != 200:
                        continue                         # one-at-a-time around the default
                    for d in DIST[:4]:
                        out.append(dict(exp="lru", scenario=sc, d=d, T=int(T), policy="always", e=v["confidence_table"]["hard_err"], p_ld=0.75, e_lru=e_lru,
                                        t_lru=t_lru, leak=_with_roc(v[sc], "0.01"), level="hard", fpr_target="always"))
                        for t in ("0.01", "0.05", "0.2", "0.5"):
                            out.append(dict(exp="lru", scenario=sc, d=d, T=int(T), policy="flag", e=v["confidence_table"]["hard_err"], p_ld=0.75, e_lru=e_lru,
                                            t_lru=t_lru, leak=_with_roc(v[sc], t), level="hard", fpr_target=t))
    return out


def jobs_combined(P):
    """Always-on reduction with soft readout, the candidate operating points."""
    out = []
    for T, v in P["T"].items():
        tab = v["confidence_table"]
        for b in (0, 2, 3, 4):
            lv = S.soft_table(tab, b) if b else None
            for sc in ("pooled", "q3_like"):
                for d in DIST:
                    out.append(dict(exp="combined", scenario=sc, d=d, T=int(T), policy="always", e=tab["hard_err"], p_ld=0.75,
                                    leak=_with_roc(v[sc], "0.01"), levels=lv, level=f"bits{b}" if b else "hard"))
            for d in DIST:                               # what the slot and the unit cost when there is no leakage at all
                out.append(dict(exp="combined", scenario="no_leak_with_slot", d=d, T=int(T), policy="always", e=tab["hard_err"], p_ld=0.75,
                                leak=dict(p_create=0.0, survive=0.0), levels=lv, level=f"bits{b}" if b else "hard"))
    return out


def jobs_eta(P):
    """Removal probability of the unit, the assumption the always-on result leans on most."""
    out = []
    for T in ("600", "1000"):
        v = P["T"][T]
        for sc in ("pooled", "q3_like"):
            for eta in (0.5, 0.9, 0.95, 0.99, 0.999):
                for d in DIST[:4]:
                    base = dict(exp="eta", scenario=sc, d=d, T=int(T), e=v["confidence_table"]["hard_err"], p_ld=0.75, eta=eta, level="hard")
                    out.append(dict(base, policy="always", leak=_with_roc(v[sc], "0.01"), fpr_target="always"))
                    out.append(dict(base, policy="flag", leak=_with_roc(v[sc], "0.5"), fpr_target="0.5"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["soft", "tolerance", "scenarios", "lru", "combined", "eta"])
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=100)
    ap.add_argument("--max-d", type=int, default=13)
    a = ap.parse_args()
    P = json.load(open(a.params))
    jobs = dict(soft=jobs_soft, tolerance=jobs_tolerance, scenarios=jobs_scenarios, lru=jobs_lru, combined=jobs_combined, eta=jobs_eta)[a.exp](P)
    jobs = [j for j in jobs if j["d"] <= a.max_d]
    key = lambda j: json.dumps({k: j.get(k) for k in ("exp", "scenario", "d", "T", "policy", "p_ld", "level", "leak", "e_lru", "t_lru", "fpr_target", "eta")}, sort_keys=True)
    done = set()
    if os.path.exists(a.out):
        done = {key(json.loads(l)) for l in open(a.out)}
    jobs = [j for j in jobs if key(j) not in done]
    jobs.sort(key=lambda j: j["d"])
    print(f"{a.exp}: {len(jobs)} jobs, {a.workers} workers", flush=True)
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a") as fo, Pool(a.workers, maxtasksperchild=4) as pool:
        for i, r in enumerate(pool.imap_unordered(run, jobs, chunksize=1)):
            r.pop("levels", None)
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            if i % 20 == 0:
                print(f"[{time.time()-t0:6.0f}s] {i+1}/{len(jobs)} d={r['d']} T={r['T']} {r['exp']} {r['policy']} {r['level']} {r['fails']}/{r['shots']}", flush=True)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
