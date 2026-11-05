"""Logical error of a lattice-surgery operation versus readout length, readout error and distance.

lattice-sim (MCMit/lattice-sim, sim/circuit_4.py) builds two rotated surface-code patches, d+1 rounds
of memory, then d+1 rounds merged (merge=True, rounds_per_op=d+1), with the readout length T as the
measurement latency (idle decoherence on every other qubit) and the readout error p as the flip
probability on every measurement. Noise preset ibm_boston (Heron r3), as compare_readout_artifact.py:
CZ 1.191e-3, CNOT 70 ns, T1/T2 284.95/322.68 us, idle_multiplier 3. A round lasts T + 340 ns.

Two sets of points:
  grid   T in 200..1000 ns (100 ns steps) x p in P_GRID x d
  disc   each discriminator's own (T, p) at the paper's five durations (discriminators.py) x d

Every job samples until TARGET_FAILS (split over chunks) or a shot cap, decoding with PyMatching
(bit-packed), and appends one JSON line to the output: shots, fails, circuit time, detectors and the
decode time per shot (batch, single thread; part C).

Usage (inside the oraqle-latsim image; lattice-sim is the vendored copy in <repo>/qec/lattice-sim/sim,
override with the LATSIM environment variable):
  python system/sweep_ler.py --set grid --out $ORAQLE_WORK/system/ler_grid.jsonl [--workers 160] [--quick]
  (paper runs: docs/reproduce/qec_system.md)
"""
import argparse
import itertools
import json
import os
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np
import pymatching
import stim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discriminators import DURATIONS, FIDELITY, measure_error

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATSIM = os.environ.get("LATSIM", os.path.join(_REPO, "qec", "lattice-sim", "sim"))  # was /work/lattice-sim/sim
PRESETS = {"ibm_boston": dict(cz=1.191e-3, t1=284.95, t2=322.68)}
CNOT_NS, IDLE_MUL = 70, 3
T_GRID = list(range(200, 1001, 100))
P_GRID = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20]
DISTANCES = [5, 7, 9, 11, 13, 15]
TARGET_FAILS = 200
SHOT_CAP = {5: 4e6, 7: 4e6, 9: 3e6, 11: 2e6, 13: 1.5e6, 15: 1e6, 17: 6e5}
CHUNKS = {5: 1, 7: 1, 9: 2, 11: 4, 13: 8, 15: 12, 17: 16}

_cache = {}


def build(d, T, p, preset):
    key = (d, T, p, preset)
    if key not in _cache:
        if LATSIM not in sys.path:
            sys.path.insert(0, LATSIM)
        from circuit_4 import circuit
        n = PRESETS[preset]
        sim = circuit(distance=d, num_patches_x=20, num_patches_y=20, spacing=1, disable_noise=False,
                      fixed_cnot_latency=CNOT_NS, fixed_cnot_noise=n["cz"],
                      fixed_measure_latency=T, fixed_measure_noise=p,
                      fixed_t1=n["t1"], fixed_t2=n["t2"], idle_multiplier=IDLE_MUL,
                      merge=True, basis="Z", ls_basis="X", sync=None,
                      rounds_per_op=d + 1).from_string("qreg q[2];")
        c = sim.ckt if isinstance(sim.ckt, stim.Circuit) else stim.Circuit(str(sim.ckt))
        m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
        _cache.clear()                                 # one circuit per worker at a time
        _cache[key] = (c, m, sim.time)
    return _cache[key]


def run(job):
    d, T, p, preset, label, chunk, cap, target = job
    c, m, t_circ = build(d, T, p, preset)
    sampler = c.compile_detector_sampler(seed=zlib.crc32(repr((d, T, round(p, 6), preset, chunk)).encode()))
    shots = fails = 0
    dec_s = 0.0
    batch = 5_000
    while fails < target and shots < cap:
        det, obs = sampler.sample(batch, separate_observables=True, bit_packed=True)
        t0 = time.perf_counter()
        pred = m.decode_batch(det, bit_packed_shots=True, bit_packed_predictions=True)
        dec_s += time.perf_counter() - t0
        fails += int(np.sum(np.any(pred != obs, axis=1)))
        shots += batch
        if fails > 0.4 * shots:                        # above threshold: one batch is enough
            break
        batch = min(batch * 2, 200_000)
    return dict(d=d, T=T, p=p, preset=preset, label=label, chunk=chunk, shots=shots, fails=fails,
                circuit_ns=t_circ, rounds=2 * (d + 1), detectors=c.num_detectors,
                decode_us_per_shot=dec_s / shots * 1e6)


def jobs_for(points, quick, target=TARGET_FAILS, cap_scale=1.0):
    jobs = []
    for d, T, p, preset, label in points:
        k = CHUNKS[d]
        cap = SHOT_CAP[d] * cap_scale / (1000 if quick else 1)
        for ch in range(k):
            jobs.append((d, T, p, preset, label, ch, int(cap / k), -(-target // k)))
    jobs.sort(key=lambda j: -j[0])                     # large d first
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["grid", "disc"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=160)
    ap.add_argument("--distances", type=int, nargs="*", default=DISTANCES)
    ap.add_argument("--preset", default="ibm_boston")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--target-fails", type=int, default=TARGET_FAILS)
    ap.add_argument("--cap-scale", type=float, default=1.0)
    a = ap.parse_args()

    if a.set == "grid":
        points = [(d, T, p, a.preset, "grid") for d, T, p in itertools.product(a.distances, T_GRID, P_GRID)]
    else:
        points = [(d, T, measure_error(mdl, T), a.preset, mdl)
                  for d, mdl, T in itertools.product(a.distances, FIDELITY, DURATIONS)]
    done = set()
    if os.path.exists(a.out):                          # resume: skip finished chunks
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["d"], r["T"], round(r["p"], 9), r["preset"], r["label"], r["chunk"]))
    jobs = [j for j in jobs_for(points, a.quick, a.target_fails, a.cap_scale) if (j[0], j[1], round(j[2], 9), j[3], j[4], j[5]) not in done]
    print(f"{a.set}: {len(points)} points, {len(jobs)} jobs to run, {a.workers} workers", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0 = time.time()
    with open(a.out, "a") as f, Pool(a.workers, maxtasksperchild=50) as pool:
        for i, r in enumerate(pool.imap_unordered(run, jobs, chunksize=1)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"[{time.time() - t0:7.0f}s] {i + 1}/{len(jobs)}  d={r['d']} T={r['T']} p={r['p']:.4f} "
                      f"{r['label']}: {r['fails']}/{r['shots']}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
