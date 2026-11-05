"""Logical error per decoder type at each readout operating point (I6): do deployable decoders keep the readout gains?

Same lattice-sim circuits as sweep_ler.py (two patches, d+1 memory + d+1 merged rounds, ibm_boston, MCMit-CNN readout
error at each readout length T), plus two soft-readout points from sweep_soft.py (flagged measurements as erasures).
Every sampled shot is decoded by every decoder, so the comparison is paired:
  mwpm_w    PyMatching, weights from the detector error model (what §7 uses)
  mwpm_c    PyMatching with correlated matching (enable_correlations)
  mwpm_q16  PyMatching, weights linearly quantized to 16 integer levels (Helios's weighted mode, w_max = 16)
  mwpm_q4   PyMatching, weights quantized to 4 levels
  mwpm_u    PyMatching, every edge weight equal (Helios unweighted, its default and the mode its latencies are
            reported for; qshell RTL: "edge weights uniformly two")
  uf_w      union-find (ldpc, peeling), with per-edge LLRs
  uf_u      union-find (ldpc, peeling), no soft information: the closest software proxy of Helios
Erasure edges (probability 1/2, weight 0) keep weight 0 under quantization and in mwpm_u (uniform weights plus a 1-bit
erasure input); union-find without LLRs cannot see them.
Union-find runs per shot in Python and gets its own shot cap (UF_CAP_FRAC of the job's shots).

Usage (oraqle-latsim:dec image): python system/sweep_decoders.py --out $ORAQLE_WORK/system/ler_decoders.jsonl [--workers 150] [--quick]
  (--soft defaults to $ORAQLE_RESULTS/system/soft_params.json)
  [--d 13 --soft-only --uf-frac 0]   (subset rerun; samples match the full run, since seeds depend only on the point)
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
import scipy.sparse as sp
import stim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from discriminators import measure_error
from sweep_ler import build
from oraqle_env import RESULTS

DISTANCES = [5, 7, 9, 11, 13]
HARD_T = [400, 600, 800, 1000]
SOFT = [(600, 0.05), (400, 0.10)]          # (readout length, flagged fraction): the best flags of §7.6
PATTERNS = 4
TARGET_FAILS = 300                          # on mwpm_w, the most accurate decoder that sets the stopping rule
SHOT_CAP = {5: 2e6, 7: 2e6, 9: 2e6, 11: 1.5e6, 13: 1.2e6}
CHUNKS = {5: 2, 7: 2, 9: 4, 11: 6, 13: 10}
UF_CAP_FRAC = 0.25
MATCHING = ["mwpm_w", "mwpm_c", "mwpm_q16", "mwpm_q4", "mwpm_u"]


def quantized(m, levels):
    """Same graph as m, weights in fixed point: w / step rounded and clipped to 1..levels, with the step set so the
    99th-percentile weight maps to the top level (rare outlier edges saturate). Erasure edges (weight 0) stay 0."""
    edges = m.edges()
    ws = np.array([e[2]["weight"] for e in edges if e[2]["weight"] > 0])
    step = np.percentile(ws, 99) / levels if levels else None
    q = pymatching.Matching()
    for u, v, dat in edges:
        w = dat["weight"]
        if w <= 0:
            qw = 0.0
        elif levels is None:
            qw = 1.0                                          # uniform
        else:
            qw = float(min(levels, max(1, round(w / step))))
        kw = dict(fault_ids=dat["fault_ids"], weight=qw, error_probability=dat["error_probability"],
                  merge_strategy="smallest-weight")
        if v is None:
            q.add_boundary_edge(u, **kw)
        else:
            q.add_edge(u, v, **kw)
    q.ensure_num_fault_ids(m.num_fault_ids)
    return q


def uf_setup(m, n_det, n_obs):
    """Graph-like check matrix, observable matrix and LLRs from the matching graph, for ldpc's union-find."""
    from ldpc import UnionFindDecoder
    rows, cols, orow, ocol, llr = [], [], [], [], []
    for j, (u, v, dat) in enumerate(m.edges()):
        rows.append(u); cols.append(j)
        if v is not None:
            rows.append(v); cols.append(j)
        for f in dat["fault_ids"]:
            orow.append(f); ocol.append(j)
        p = min(max(dat["error_probability"], 1e-12), 0.5)
        llr.append(np.log((1 - p) / p) if p < 0.5 else 1e-6)
    n = len(m.edges())
    H = sp.csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(n_det, n))
    L = sp.csr_matrix((np.ones(len(orow), dtype=np.uint8), (orow, ocol)), shape=(n_obs, n))
    return UnionFindDecoder(H, uf_method="peeling"), L, np.array(llr, dtype=np.float64)


def circuit_for(d, T, p, f, pattern):
    c, _, t_circ = build(d, T, p, "ibm_boston")
    if f > 0:
        from sweep_soft import erased
        c = erased(c, f, zlib.crc32(repr((d, T, f, pattern)).encode()))
    return c, t_circ


def run(job):
    d, T, f, p, pattern, cap, target, uf_frac = job
    c, t_circ = circuit_for(d, T, p, f, pattern)
    dem = c.detector_error_model(decompose_errors=True)
    m_w = pymatching.Matching.from_detector_error_model(dem)
    m_c = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    dec = {"mwpm_w": m_w, "mwpm_c": m_c, "mwpm_q16": quantized(m_w, 16), "mwpm_q4": quantized(m_w, 4),
           "mwpm_u": quantized(m_w, None)}
    uf, L, llr = uf_setup(m_w, c.num_detectors, c.num_observables)
    sampler = c.compile_detector_sampler(seed=zlib.crc32(repr((d, T, f, pattern, "dec")).encode()))
    fails = {k: 0 for k in list(dec) + ["uf_w", "uf_u"]}
    secs = {k: 0.0 for k in fails}
    shots = uf_shots = 0
    uf_cap = int(cap * uf_frac)
    batch = 5_000
    while fails["mwpm_w"] < target and shots < cap:
        det, obs = sampler.sample(batch, separate_observables=True)
        for k, mm in dec.items():
            t0 = time.perf_counter()
            pred = mm.decode_batch(det, enable_correlations=(k == "mwpm_c"))
            secs[k] += time.perf_counter() - t0
            fails[k] += int(np.any(pred != obs, axis=1).sum())
        if uf_shots < uf_cap:
            n = min(batch, uf_cap - uf_shots)
            for k, ll in (("uf_w", llr), ("uf_u", None)):
                t0 = time.perf_counter()
                for i in range(n):
                    e = uf.decode(det[i].astype(np.uint8), llrs=ll) if ll is not None else uf.decode(det[i].astype(np.uint8))
                    fails[k] += int(np.any((L @ e) % 2 != obs[i]))
                secs[k] += time.perf_counter() - t0
            uf_shots += n
        shots += batch
        if fails["mwpm_w"] > 0.4 * shots:
            break
        batch = min(batch * 2, 40_000)
    return dict(d=d, T=T, f=f, p=p, pattern=pattern, shots=shots, uf_shots=uf_shots, fails=fails,
                us_per_shot={k: secs[k] / (uf_shots if k.startswith("uf") else shots) * 1e6 if (uf_shots if k.startswith("uf") else shots) else None
                             for k in secs},
                rounds=2 * (d + 1), circuit_ns=t_circ, detectors=c.num_detectors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--soft", default=str(RESULTS / "system" / "soft_params.json"))  # was /work/data/soft_params.json
    ap.add_argument("--workers", type=int, default=150)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--d", type=int, nargs="+", default=DISTANCES)
    ap.add_argument("--soft-only", action="store_true")
    ap.add_argument("--uf-frac", type=float, default=UF_CAP_FRAC, help="0 skips union-find (Python, slow at large d)")
    ap.add_argument("--done-from", nargs="*", default=[], help="other result files whose jobs count as done")
    a = ap.parse_args()
    soft = json.load(open(a.soft)) if os.path.exists(a.soft) else {}
    points = [] if a.soft_only else [(T, 0.0, measure_error("MCMit-CNN", T)) for T in HARD_T]
    for T, f in SOFT:
        if str(T) in soft:
            points.append((T, f, soft[str(T)][str(f)]["e_u"]))
    ds = [5, 7] if a.quick else a.d
    jobs = []
    for (T, f, p), d in itertools.product(points, ds):
        n = PATTERNS if f > 0 else CHUNKS[d]
        cap = SHOT_CAP[d] / (200 if a.quick else 1)
        for k in range(n):
            jobs.append((d, T, f, p, k, int(cap / n), -(-TARGET_FAILS // n), a.uf_frac))
    done = set()
    for path in [a.out] + a.done_from:
        if os.path.exists(path):
            for line in open(path):
                r = json.loads(line)
                done.add((r["d"], r["T"], r["f"], r["pattern"]))
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[4]) not in done]
    jobs.sort(key=lambda j: -j[0])
    print(f"{len(points)} readout points x {len(ds)} distances: {len(jobs)} jobs, {a.workers} workers", flush=True)
    t0 = time.time()
    with open(a.out, "a") as fo, Pool(a.workers, maxtasksperchild=10) as pool:
        for i, r in enumerate(pool.imap_unordered(run, jobs, chunksize=1)):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            print(f"[{time.time() - t0:6.0f}s] {i + 1}/{len(jobs)} d={r['d']} T={r['T']} f={r['f']}: "
                  + " ".join(f"{k}={v}" for k, v in r["fails"].items()) + f" / {r['shots']} ({r['uf_shots']} UF)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
