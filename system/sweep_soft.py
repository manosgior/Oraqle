"""Soft readout as erasures (part E): does a confidence flag buy back what a short readout loses?

Input: data/soft/cnn_soft_<T>ns.npz (soft_readout_cnn.py): the CNN's probability of |1> for every test
shot and qubit. Q1, Q3, Q4, Q5 are pooled (Q2 excluded, as everywhere else). For a flag fraction f, the
least confident fraction f of measurements (smallest |p - 1/2|) is flagged:
  e_u   error rate of the unflagged measurements      -> lattice-sim's readout error
  f     flagged measurements                          -> erasures: a flip with probability 1/2 at a
                                                         location the decoder knows (X_ERROR(0.5) in
                                                         the circuit it is built from)
f = 0 is the hard-decision readout (one bit). f > 0 costs one more bit per measurement (2-bit soft).
Erasure locations are drawn per measurement with probability f, PATTERNS patterns per point (each with
its own decoder), shots split over them; the same idea as CXL-Quantum/plots/qec_lambda.py.

Usage (oraqle-latsim image): python system/sweep_soft.py --soft results/system/soft --out $ORAQLE_WORK/system/ler_soft.jsonl
  (writes soft_params.json next to --out)
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
from sweep_ler import CHUNKS, DISTANCES, SHOT_CAP, build

QUBITS = [0, 2, 3, 4]
F_GRID = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
PATTERNS = 8
TARGET_FAILS = 400


def flag_params(path):
    z = np.load(path)
    prob = z["prob"][:, QUBITS].astype(np.float32).ravel()
    label = z["label"][:, QUBITS].ravel()
    conf = np.abs(prob - 0.5)
    wrong = (prob > 0.5) != (label == 1)
    order = np.argsort(conf)                           # least confident first
    out = {}
    for f in F_GRID:
        k = int(round(f * len(conf)))
        keep = order[k:]
        out[f] = dict(f=k / len(conf), e_u=float(wrong[keep].mean()),
                      e_flagged=float(wrong[order[:k]].mean()) if k else 0.0)
    return int(z["length_ns"]), out


def erased(c, f, seed):
    if f == 0:
        return c
    rng = np.random.default_rng(seed)
    out = stim.Circuit()
    for inst in c.flattened():
        if inst.name in ("M", "MR", "MX", "MRX"):
            er = [t.value for t in inst.targets_copy() if rng.random() < f]
            if er:
                out.append("Z_ERROR" if inst.name in ("MX", "MRX") else "X_ERROR", er, 0.5)
        out.append(inst)
    return out


def run(job):
    d, T, f, e_u, pattern, cap, target = job
    c0, _, t_circ = build(d, T, e_u, "ibm_boston")
    c = erased(c0, f, zlib.crc32(repr((d, T, f, pattern)).encode()))
    m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
    sampler = c.compile_detector_sampler(seed=zlib.crc32(repr((d, T, f, pattern, "s")).encode()))
    shots = fails = 0
    batch = 5_000
    while fails < target and shots < cap:
        det, obs = sampler.sample(batch, separate_observables=True, bit_packed=True)
        pred = m.decode_batch(det, bit_packed_shots=True, bit_packed_predictions=True)
        fails += int(np.sum(np.any(pred != obs, axis=1)))
        shots += batch
        if fails > 0.4 * shots:
            break
        batch = min(batch * 2, 200_000)
    return dict(d=d, T=T, f=f, p=e_u, label=f"soft_f{f:g}", pattern=pattern, chunk=pattern, shots=shots,
                fails=fails, circuit_ns=t_circ, rounds=2 * (d + 1), preset="ibm_boston")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=120)
    a = ap.parse_args()
    params = {}
    for fn in sorted(os.listdir(a.soft)):
        if fn.endswith(".npz"):
            T, p = flag_params(os.path.join(a.soft, fn))
            params[T] = p
    json.dump(params, open(os.path.join(os.path.dirname(a.out), "soft_params.json"), "w"), indent=1)
    for T, p in sorted(params.items()):
        print(T, "  ".join(f"f={v['f']:.2f}: e_u={v['e_u']:.4f} (flagged err {v['e_flagged']:.2f})"
                          for v in p.values()), flush=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["d"], r["T"], r["f"], r["pattern"]))
    jobs = []
    for T, p in params.items():
        for f, d in itertools.product(F_GRID, DISTANCES):
            v = p[f]
            n = PATTERNS if f > 0 else CHUNKS[d]
            for pat in range(n):
                if (d, T, f, pat) not in done:
                    jobs.append((d, T, f, v["e_u"], pat, int(SHOT_CAP[d] / n), -(-TARGET_FAILS // n)))
    jobs.sort(key=lambda j: -j[0])
    print(f"{len(jobs)} jobs, {a.workers} workers", flush=True)
    t0 = time.time()
    with open(a.out, "a") as fo, Pool(a.workers, maxtasksperchild=20) as pool:
        for i, r in enumerate(pool.imap_unordered(run, jobs, chunksize=1)):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            if i % 50 == 0:
                print(f"[{time.time() - t0:6.0f}s] {i + 1}/{len(jobs)} d={r['d']} T={r['T']} f={r['f']}: "
                      f"{r['fails']}/{r['shots']}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
