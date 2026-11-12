"""Reaction wait and runtime when the decoder is not on the readout FPGA (sensitivity for §7.3, tab:sys_reaction).

tab:sys_reaction assumes no transport between the readout electronics and the decoder. This adds a measured
round trip to the reaction time R of system/system_cost.py and recomputes the waiting rounds w = ceil(R / t_r),
the required distance, the runtime and the share of rounds spent waiting, for every program of system/workloads.py
with MCMit-CNN readout (full-weight fits of results/system/summary.json).

Round trips (published, read on 2026-11-12):
  3.96 us  NVQLink, FPGA controller to GPU host over Ethernet, maximum (Caldwell et al., arXiv:2510.25213, abstract)
  2.944 us THQLink, FPGA to CPU host over an HPC interconnect, average (Lao et al., arXiv:2608.03948, abstract)
Both are loopback/transport figures without a decoder in the path, so the decode latency is added on top.

Usage: python extras/transport/transport_wait.py [--data results/system]   (numpy only; writes transport_wait.json here)
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(REPO, "system")]

import system_cost as sc  # noqa: E402
from workloads import WORKLOADS  # noqa: E402

CASES = [("GPU", 0), ("GPU", 3960), ("Micro Blossom", 0), ("Micro Blossom", 2944)]
LENGTHS = (400, 600, 800, 1000)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.path.join(REPO, "results", "system"))
    a = ap.parse_args()
    sc.DATA = a.data
    S = json.load(open(os.path.join(a.data, "summary.json")))
    fit = {s["T"]: s for s in S["disc_hi"] if s["label"] == "MCMit-CNN" and "a" in s and s.get("Lambda", 0) > 1}
    helios = sc.helios_latency()
    out = {}
    print(f"{'decoder':<15}{'transport':>10}" + "".join(f"{T:>22}" for T in LENGTHS))
    for dec, transport in CASES:
        cells, rec = [], {}
        for T in LENGTHS:
            R0, _ = sc.reaction_wait(dec, "MCMit-CNN", T, helios)
            R = R0 + transport
            w = max(1, math.ceil(R / (T + sc.GATES_NS)))
            share, rel = [], []
            for wl in WORKLOADS.values():
                r = sc.program_cost(fit[T]["a"], fit[T]["b"], wl, T, R, w, dec)
                r0 = sc.program_cost(fit[T]["a"], fit[T]["b"], wl, T, *sc.reaction_wait(dec, "MCMit-CNN", T, helios), dec)
                share.append(wl["steps"] * w / r["rounds"])
                rel.append(r["runtime_s"] / r0["runtime_s"])
            rec[T] = dict(R_ns=R, w=w, wait_share=[min(share), max(share)], runtime_vs_no_transport=[min(rel), max(rel)])
            cells.append(f"{R / 1e3:.2f}us w={w} {100 * min(share):.0f}-{100 * max(share):.0f}%")
        out[f"{dec}/{transport}"] = rec
        print(f"{dec:<15}{transport / 1e3:>8.2f}us" + "".join(f"{c:>22}" for c in cells))
    print("cells: reaction time, waiting rounds per feed-forward step, share of all rounds spent waiting (six programs)")
    for key, rec in out.items():
        if not key.endswith("/0"):
            lo = min(v["runtime_vs_no_transport"][0] for v in rec.values())
            hi = max(v["runtime_vs_no_transport"][1] for v in rec.values())
            print(f"{key}: runtime {lo:.3f}-{hi:.3f}x of the same decoder without transport")
    json.dump(out, open(os.path.join(HERE, "transport_wait.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
