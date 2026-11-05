"""Direct large-distance runs (I11): do the Lambda fits of §7 (d <= 15) hold where the required distances lie?

Same lattice-sim circuits as sweep_ler.py (two patches, d+1 memory + d+1 merged rounds, ibm_boston) at the plateau
(MCMit-CNN 600/800/1000 ns, to d = 25; 600 ns to d = 29) and near threshold (MCMit-CNN 400 ns to d = 49, QubiCML
1000 ns to d = 33),
decoded with full weights ("w"); the uniform-weight decoder of §7.7 ("u") is checked at the plateau to d = 25.

Each point is split into chunks sized from its current fit (summary.json, disc_hi) and the measured decode time per
shot (decode_timing files): the predicted shots for TARGET failures, times the shot cost, over CHUNK_HOURS per chunk.
A chunk stops at its share of the failures, at CAP_FACTOR x its predicted shots, or after --max-hours (a stopping
rule independent of the outcomes, so it adds no bias). Seeds differ from sweep_ler.py (tag "large").

Usage (oraqle-latsim:dec image): python system/sweep_large.py --out $ORAQLE_WORK/system/ler_large.jsonl --summary .../summary.json
  --decoders .../decoders_summary.json --timing '.../decode_*' [--dmin 41] [--dmax 33] [--workers 150] [--plan]
"""
import argparse
import glob
import json
import math
import os
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discriminators import measure_error
from sweep_ler import build
import decode_timing  # noqa: F401,E402  (bypasses lattice-sim's decoder-latency lookup above d = 17)

POINTS = [("MCMit-CNN", 1000, [17, 21, 25], ["w"]),
          ("MCMit-CNN", 800, [17, 21, 25], ["w"]),
          ("MCMit-CNN", 600, [17, 21, 25, 29], ["w"]),
          ("MCMit-CNN", 400, [17, 21, 25, 29, 33, 41, 49], ["w"]),
          ("QubiCML", 1000, [17, 21, 25, 29, 33], ["w"]),
          ("MCMit-CNN", 1000, [17, 21, 25], ["u"]),
          ("MCMit-CNN", 600, [17, 21, 25], ["u"])]
TARGET = 200                      # failures per point; 100 where that would cost more than BIG_CORE_H
BIG_CORE_H = 200
CHUNK_HOURS = 2.0
MAX_CHUNKS = 160
CAP_FACTOR = 5                    # a chunk samples at most this multiple of its predicted shots
SAMPLE_OVERHEAD = 1.6             # sampling and bookkeeping on top of the measured decode time
BATCH_BYTES = 1e9                 # bit-packed detector memory per batch

_dec = {}


def shot_cost_model(paths):
    """Seconds per shot (full-weight decode) as c d^k per readout length, from the decode-timing files."""
    rows = [json.loads(l) for p in paths for l in open(p)]
    fits = {}
    for T in {r["T"] for r in rows}:
        pts = sorted((r["d"], r["batch_ns_per_round"] * r["rounds"] * 1e-9) for r in rows if r["T"] == T and r["d"] >= 9)
        k, logc = np.polyfit(np.log([d for d, _ in pts]), np.log([s for _, s in pts]), 1)
        fits[T] = (math.exp(logc), k)
    return lambda d, T: SAMPLE_OVERHEAD * fits[T][0] * d ** fits[T][1]


def predicted_pl(s, d):
    eps = math.exp(s["a"] + s["b"] * d)
    return (1 - (1 - 2 * eps) ** (2 * (d + 1))) / 2


def plan(summary, decoders, cost):
    """Fits: disc_hi for full weights, the §7.7 uniform-weight fit (decoders_summary.json) for "u"."""
    fits = {("w", s["label"], s["T"]): s for s in summary["disc_hi"] if "a" in s}
    fits.update({("u", "MCMit-CNN", r["T"]): r for r in decoders if r["decoder"] == "mwpm_u" and r["f"] == 0 and "a" in r})
    jobs, rows = [], []
    for label, T, ds, decs in POINTS:
        p = measure_error(label, T)
        for dec in decs:
            s = fits[(dec, label, T)]
            for d in ds:
                pl = predicted_pl(s, d)
                target = TARGET
                core_s = target / pl * cost(d, T) * (1 if dec == "w" else 1.6)
                if core_s / 3600 > BIG_CORE_H:
                    target, core_s = target // 2, core_s / 2
                n = max(1, min(MAX_CHUNKS, math.ceil(core_s / 3600 / CHUNK_HOURS)))
                cap = int(CAP_FACTOR * target / pl / n) + 1000
                rows.append((label, T, dec, d, pl, core_s / 3600, n))
                for ch in range(n):
                    jobs.append((label, T, p, d, dec, ch, cap, -(-target // n)))
    return jobs, rows


def run(job, max_s):
    label, T, p, d, dec, chunk, cap, target = job
    c, m, t_circ = build(d, T, p, "ibm_boston")
    if dec == "u":
        if (d, T, p) not in _dec:
            from sweep_decoders import quantized
            _dec.clear()
            _dec[(d, T, p)] = quantized(m, None)
        m = _dec[(d, T, p)]
    sampler = c.compile_detector_sampler(seed=zlib.crc32(repr((label, d, T, dec, chunk, "large")).encode()))
    bmax = max(500, min(200_000, int(BATCH_BYTES / (c.num_detectors / 8))))
    shots = fails = 0
    batch, t0, dec_s = min(2_000, bmax), time.time(), 0.0
    while fails < target and shots < cap and time.time() - t0 < max_s:
        n = min(batch, cap - shots)
        det, obs = sampler.sample(n, separate_observables=True, bit_packed=True)
        t1 = time.perf_counter()
        pred = m.decode_batch(det, bit_packed_shots=True, bit_packed_predictions=True)
        dec_s += time.perf_counter() - t1
        fails += int(np.sum(np.any(pred != obs, axis=1)))
        shots += n
        batch = min(batch * 2, bmax)
    return dict(label=label, T=T, p=p, d=d, decoder=dec, chunk=chunk, shots=shots, fails=fails, rounds=2 * (d + 1),
                circuit_ns=t_circ, detectors=c.num_detectors, wall_s=time.time() - t0,
                decode_us_per_shot=dec_s / shots * 1e6 if shots else None, hit_time=time.time() - t0 >= max_s)


def _run(args):
    return run(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--decoders", required=True, help="decoders_summary.json (uniform-weight fit)")
    ap.add_argument("--timing", required=True, nargs="+", help="globs of decode-timing jsonl files")
    ap.add_argument("--workers", type=int, default=150)
    ap.add_argument("--dmin", type=int, default=0)
    ap.add_argument("--dmax", type=int, default=999)
    ap.add_argument("--max-hours", type=float, default=4.0)
    ap.add_argument("--plan", action="store_true", help="print the job plan and its cost, run nothing")
    a = ap.parse_args()
    jobs, rows = plan(json.load(open(a.summary)), json.load(open(a.decoders)), shot_cost_model([f for g in a.timing for f in glob.glob(g)]))
    jobs = [j for j in jobs if a.dmin <= j[3] <= a.dmax]
    if a.plan:
        print(f"{'point':<18}{'dec':>4}{'d':>4}{'pred P_L':>11}{'core-h':>9}{'chunks':>8}")
        for label, T, dec, d, pl, ch, n in rows:
            if a.dmin <= d <= a.dmax:
                print(f"{label + '/' + str(T):<18}{dec:>4}{d:>4}{pl:>11.2e}{ch:>9.1f}{n:>8}")
        print(f"total {sum(r[5] for r in rows if a.dmin <= r[3] <= a.dmax):.0f} core-h, {len(jobs)} jobs")
        return
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["label"], r["T"], r["d"], r["decoder"], r["chunk"]))
    jobs = [j for j in jobs if (j[0], j[1], j[3], j[4], j[5]) not in done]
    jobs.sort(key=lambda j: -j[3])
    print(f"{len(jobs)} jobs, {a.workers} workers, d in [{a.dmin}, {a.dmax}]", flush=True)
    t0 = time.time()
    with open(a.out, "a") as fo, Pool(a.workers, maxtasksperchild=4) as pool:
        for i, r in enumerate(pool.imap_unordered(_run, [(j, a.max_hours * 3600) for j in jobs], chunksize=1)):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            print(f"[{time.time() - t0:7.0f}s] {i + 1}/{len(jobs)} {r['label']}/{r['T']} {r['decoder']} d={r['d']}: "
                  f"{r['fails']}/{r['shots']}{' (time limit)' if r['hit_time'] else ''}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
