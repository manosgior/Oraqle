"""Decode cost versus distance and readout (part C): is the readout or the decoder the bottleneck?

For every distance and every MCMit-CNN operating point (T, p) of discriminators.py, build the same
lattice-sim circuit as sweep_ler.py (two patches, d+1 memory + d+1 merged rounds), sample shots, and
time PyMatching on one pinned core:
  single     decode() one shot at a time: median and p99 latency per shot (reaction-time view)
  batch      decode_batch() over all shots: mean time per shot (throughput view)
Both are also reported per round (a shot holds 2(d+1) rounds of both patches). The detection events
per round give the decoder's work. Run one process per core, pinned with taskset, on an idle machine.

Beyond d = 17 lattice-sim's decoder-latency lookup (used only for synchronization, off here) has no
entry, so it is bypassed for those distances.

Usage: python system/decode_timing.py --d 9 --out $ORAQLE_WORK/system/decode_timing_d9.jsonl [--shots 20000] [--T 400 600 1000]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discriminators import DURATIONS, measure_error
from sweep_ler import LATSIM, build

sys.path.insert(0, LATSIM)
import circuit_4  # noqa: E402

_lookup = circuit_4.circuit.assign_decoder_latency


def _assign(self):
    if self.d <= 17:
        return _lookup(self)
    self.decoder_latency = 0          # sync=None: never read


circuit_4.circuit.assign_decoder_latency = _assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shots", type=int, default=20_000)
    ap.add_argument("--model", default="MCMit-CNN")
    ap.add_argument("--preset", default="ibm_boston")
    ap.add_argument("--T", type=int, nargs="*", default=DURATIONS)
    a = ap.parse_args()
    for T in a.T:
        p = measure_error(a.model, T)
        c, m, t_circ = build(a.d, T, p, a.preset)
        det, obs = c.compile_detector_sampler(seed=a.d * 1000 + T).sample(a.shots, separate_observables=True)
        events = det.sum(axis=1)
        lat = np.empty(a.shots)
        for i in range(a.shots):
            t0 = time.perf_counter_ns()
            m.decode(det[i])
            lat[i] = time.perf_counter_ns() - t0
        packed = np.packbits(det, axis=1, bitorder="little")
        t0 = time.perf_counter()
        m.decode_batch(packed, bit_packed_shots=True)
        batch_ns = (time.perf_counter() - t0) / a.shots * 1e9
        rounds = 2 * (a.d + 1)
        r = dict(d=a.d, T=T, p=p, model=a.model, preset=a.preset, shots=a.shots, rounds=rounds,
                 circuit_ns=t_circ, detectors=c.num_detectors,
                 events_per_round=float(events.mean() / rounds),
                 single_median_ns=float(np.median(lat)), single_p99_ns=float(np.percentile(lat, 99)),
                 single_mean_ns=float(lat.mean()), batch_ns_per_shot=batch_ns,
                 batch_ns_per_round=batch_ns / rounds, single_median_ns_per_round=float(np.median(lat)) / rounds)
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
