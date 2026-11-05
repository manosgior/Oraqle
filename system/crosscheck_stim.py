"""Simulator cross-check (I11): does lattice-sim's Lambda agree with a standard circuit-level noise model?

MCMit-CNN readout at T in {400, 600, 1000} ns, d = 5..13, 2(d+1) rounds of Z memory, decoded with PyMatching:
  latsim_mem     lattice-sim, the two patches of sweep_ler.build without the merge (memory); the lattice-surgery
                 runs of §7 (ler_disc_hi.jsonl) are the reference
  stim_matched   stim's rotated_memory_z with lattice-sim's noise (gate_lib.py): DEPOLARIZE2(cz) after CX,
                 DEPOLARIZE1(1e-4) after H, X_ERROR(p) before every measurement, no reset error, and X_ERROR(q) on
                 the data qubits each round, q = lattice-sim's idle flip for T + 60 ns (measurement plus two H layers;
                 T1/T2 284.95/322.68 us, idle multiplier 3; Z-basis memory keeps only X/Y idle errors)
  stim_standard  stim's own circuit-level model at our parameters: DEPOLARIZE1/2(cz) after every Clifford,
                 X_ERROR(cz) after every reset, X_ERROR(p) before every measurement, DEPOLARIZE1(1.5 q) on the data
                 qubits each round (the same X/Y idle flip plus Z)

Usage (oraqle-latsim:dec image): python system/crosscheck_stim.py --out $ORAQLE_WORK/system/ler_crosscheck.jsonl [--workers 24]
       python system/crosscheck_stim.py --analyze results/system/ler_crosscheck.jsonl results/system/ler_disc_hi.jsonl
"""
import argparse
import collections
import itertools
import json
import math
import os
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np
import pymatching
import stim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discriminators import measure_error
from sweep_ler import CNOT_NS, IDLE_MUL, LATSIM, PRESETS

TS = [400, 600, 1000]
DISTANCES = [5, 7, 9, 11, 13]
VARIANTS = ["latsim_mem", "stim_matched", "stim_standard"]
H_NS, H_ERR = 30, 1e-4
TARGET, CHUNKS = 300, {5: 1, 7: 1, 9: 2, 11: 3, 13: 4}
SHOT_CAP = {5: 2e6, 7: 2e6, 9: 2e6, 11: 1.5e6, 13: 1.2e6}
_cache = {}


def idle_flip(t_ns, preset="ibm_boston"):
    """lattice-sim's idling_model in the Z basis: X_ERROR(1 - (1 - 3 px)^2), px = (1 - exp(-t/T1)) / 4."""
    px = (1 - math.exp(-t_ns * 1e-9 / (PRESETS[preset]["t1"] * 1e-6))) / 4
    return 1 - (1 - px * IDLE_MUL) ** 2


def circuit_for(variant, d, T, p):
    R = 2 * (d + 1)
    cz, q = PRESETS["ibm_boston"]["cz"], idle_flip(T + 2 * H_NS)
    if variant == "latsim_mem":
        if LATSIM not in sys.path:
            sys.path.insert(0, LATSIM)
        from circuit_4 import circuit
        n = PRESETS["ibm_boston"]
        sim = circuit(distance=d, num_patches_x=20, num_patches_y=20, spacing=1, disable_noise=False,
                      fixed_cnot_latency=CNOT_NS, fixed_cnot_noise=n["cz"], fixed_measure_latency=T, fixed_measure_noise=p,
                      fixed_t1=n["t1"], fixed_t2=n["t2"], idle_multiplier=IDLE_MUL, merge=False, basis="Z",
                      ls_basis="X", sync=None, rounds_per_op=d + 1).from_string("qreg q[2];")
        return sim.ckt if isinstance(sim.ckt, stim.Circuit) else stim.Circuit(str(sim.ckt))
    # sentinel probabilities, then rewritten per variant
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=R,
                               after_clifford_depolarization=0.0111, before_round_data_depolarization=0.0222,
                               before_measure_flip_probability=0.0333, after_reset_flip_probability=0.0444)
    s = str(c)
    if variant == "stim_matched":
        rep = {"DEPOLARIZE2(0.0111)": f"DEPOLARIZE2({cz})", "DEPOLARIZE1(0.0111)": f"DEPOLARIZE1({H_ERR})",
               "DEPOLARIZE1(0.0222)": f"X_ERROR({q})", "X_ERROR(0.0333)": f"X_ERROR({p})", "X_ERROR(0.0444)": "X_ERROR(0)"}
    else:
        rep = {"DEPOLARIZE2(0.0111)": f"DEPOLARIZE2({cz})", "DEPOLARIZE1(0.0111)": f"DEPOLARIZE1({cz})",
               "DEPOLARIZE1(0.0222)": f"DEPOLARIZE1({1.5 * q})", "X_ERROR(0.0333)": f"X_ERROR({p})",
               "X_ERROR(0.0444)": f"X_ERROR({cz})"}
    for k, v in rep.items():
        s = s.replace(k, v)
    assert "0.0111" not in s and "0.0222" not in s and "0.0333" not in s and "0.0444" not in s
    return stim.Circuit(s)


def run(job):
    variant, d, T, p, chunk, cap, target = job
    key = (variant, d, T)
    if key not in _cache:
        c = circuit_for(variant, d, T, p)
        _cache.clear()
        _cache[key] = (c, pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True)))
    c, m = _cache[key]
    sampler = c.compile_detector_sampler(seed=zlib.crc32(repr((variant, d, T, chunk, "xcheck")).encode()))
    shots = fails = obs_fails = 0
    batch = 5_000
    while obs_fails / c.num_observables < target and shots < cap:
        det, obs = sampler.sample(batch, separate_observables=True)
        wrong = m.decode_batch(det) != obs
        fails += int(np.sum(np.any(wrong, axis=1)))
        obs_fails += int(wrong.sum())
        shots += batch
        batch = min(batch * 2, 100_000)
    return dict(variant=variant, d=d, T=T, p=p, chunk=chunk, shots=shots, fails=fails, obs_fails=obs_fails,
                patches=c.num_observables, rounds=2 * (d + 1), detectors=c.num_detectors)


def analyze(paths):
    from analyze import fit, load, required_d
    acc = collections.defaultdict(lambda: collections.Counter())
    for line in open(paths[0]):                          # per patch: failures of each observable over shots x patches
        r = json.loads(line)
        a = acc[(r["variant"], r["T"], r["d"])]
        a["shots"] += r["shots"] * r["patches"]
        a["fails"] += r["obs_fails"]
        a["R"] = r["rounds"]
    if len(paths) > 1:                                   # lattice-surgery reference from §7 (either patch fails)
        for (label, T, p, d), r in load(paths[1]).items():
            if label == "MCMit-CNN" and T in TS and d <= 13:
                acc[("latsim_ls", T, d)].update(shots=r["shots"], fails=r["fails"])
                acc[("latsim_ls", T, d)]["R"] = r["R"]
    print("per-patch eps (latsim_ls: per experiment of two patches, as in §7); d@1e-9 from each fit")
    print(f"{'variant':<15}{'T':>6}{'Lambda':>8}{'d@1e-9':>8}   eps per d")
    for variant in ["latsim_ls"] + VARIANTS:
        for T in TS:
            pts = {}
            for (v, TT, d), a in acc.items():
                if (v, TT) == (variant, T) and a["shots"]:
                    P = a["fails"] / a["shots"]
                    eps = (1 - (1 - 2 * P) ** (1 / a["R"])) / 2 if P < 0.5 else float("nan")
                    pts[d] = dict(fails=a["fails"], P=P, eps=eps)
            if not pts:
                continue
            f = fit(pts)
            lam = f"{f[2]:.2f}" if f else "-"
            d9 = required_d(f[0], f[1], 1e-9) if f and f[1] < 0 else None
            print(f"{variant:<15}{T:>6}{lam:>8}{str(d9 or '-'):>8}   "
                  + " ".join(f"d{d}:{pts[d]['eps']:.2e}({pts[d]['fails']})" for d in sorted(pts)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--analyze", nargs="+")
    a = ap.parse_args()
    if a.analyze:
        return analyze(a.analyze)
    jobs = []
    for variant, T, d in itertools.product(VARIANTS, TS, DISTANCES):
        n = CHUNKS[d]
        for ch in range(n):
            jobs.append((variant, d, T, measure_error("MCMit-CNN", T), ch, int(SHOT_CAP[d] / n), -(-TARGET // n)))
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["variant"], r["d"], r["T"], r["chunk"]))
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[4]) not in done]
    jobs.sort(key=lambda j: -j[1])
    print(f"{len(jobs)} jobs, {a.workers} workers", flush=True)
    t0 = time.time()
    with open(a.out, "a") as fo, Pool(a.workers, maxtasksperchild=10) as pool:
        for i, r in enumerate(pool.imap_unordered(run, jobs, chunksize=1)):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            print(f"[{time.time() - t0:6.0f}s] {i + 1}/{len(jobs)} {r['variant']} T={r['T']} d={r['d']}: "
                  f"{r['fails']}/{r['shots']}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
