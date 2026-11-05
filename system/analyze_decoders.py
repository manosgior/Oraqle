"""Summary of sweep_decoders.py: Lambda, distance for 1e-9 per operation and spacetime per decoder and readout point.

Per decoder, the failures over all chunks/patterns of a (T, f, d) point give the per-round error eps (union-find over
its own, smaller shot count); ln eps = a + b d is fitted over distances with >= MIN_FAILS failures and P_L < 0.3, as in
analyze.py. Spacetime is relative to weighted matching (mwpm_w) at 1000 ns, hard readout.

Several files are merged; a (d, T, f, pattern) job present in more than one counts once (the first file wins).

Usage: python system/analyze_decoders.py data/ler_decoders.jsonl [more.jsonl ...] > data/decoders_summary.txt
  (also writes decoders_summary.json next to the first file)
  The published summary merges THREE files, in this order (system/merge_decoders.sh):
  ler_decoders.jsonl ler_decoders_soft13.jsonl ler_decoders_soft11.jsonl
"""
import collections
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import required_d  # noqa: E402

MIN_FAILS = 30
GATES_NS = 340
DECODERS = ["mwpm_w", "mwpm_c", "mwpm_q16", "mwpm_q4", "mwpm_u", "uf_w", "uf_u"]


def main():
    path = sys.argv[1]
    acc = collections.defaultdict(lambda: collections.Counter())
    seen = set()
    for line in (l for p in sys.argv[1:] for l in open(p)):
        r = json.loads(line)
        if (r["d"], r["T"], r["f"], r["pattern"]) in seen:
            continue
        seen.add((r["d"], r["T"], r["f"], r["pattern"]))
        k = (r["T"], r["f"], r["d"])
        acc[k]["shots"] += r["shots"]
        acc[k]["uf_shots"] += r["uf_shots"]
        acc[k]["rounds"] = r["rounds"]
        for dec, n in r["fails"].items():
            acc[k][dec] += n
    points = sorted({(T, f) for T, f, _ in acc})
    out = []
    for T, f in points:
        ds = sorted(d for TT, ff, d in acc if (TT, ff) == (T, f))
        for dec in DECODERS:
            eps, fails = {}, {}
            for d in ds:
                a = acc[(T, f, d)]
                n = a["uf_shots"] if dec.startswith("uf") else a["shots"]
                P = a[dec] / n if n else float("nan")
                eps[d] = (1 - (1 - 2 * P) ** (1 / a["rounds"])) / 2 if 0 < P < 0.5 else float("nan")
                fails[d] = a[dec]
            ok = [(d, eps[d]) for d in ds if fails[d] >= MIN_FAILS and acc[(T, f, d)][dec] /
                  (acc[(T, f, d)]["uf_shots"] if dec.startswith("uf") else acc[(T, f, d)]["shots"]) < 0.3]
            row = dict(T=T, f=f, decoder=dec, eps=eps, fails=fails)
            if len(ok) >= 2:
                b, a0 = np.polyfit([d for d, _ in ok], np.log([e for _, e in ok]), 1)
                row.update(Lambda=math.exp(-2 * b), a=a0, b=b, fit_ds=[d for d, _ in ok])
                d9 = required_d(a0, b, 1e-9)
                if d9:
                    t_r = (T + GATES_NS) / 1000
                    row.update(d=d9, qubits=2 * d9 * d9 - 1, spacetime=(2 * d9 * d9 - 1) * (d9 + 1) * t_r)
            out.append(row)
    base = next(r for r in out if r["T"] == 1000 and r["f"] == 0 and r["decoder"] == "mwpm_w").get("spacetime")
    for r in out:
        r["spacetime_rel"] = r["spacetime"] / base if r.get("spacetime") and base else None
    json.dump(out, open(os.path.join(os.path.dirname(path), "decoders_summary.json"), "w"), indent=1, default=float)
    print(f"{'T':>5}{'flag':>6} {'decoder':<9}{'Lambda':>8}{'d@1e-9':>8}{'ST rel':>8}   failures per d")
    for r in out:
        lam = f"{r['Lambda']:.2f}" if "Lambda" in r else "  -"
        st = f"{r['spacetime_rel']:.2f}" if r.get("spacetime_rel") else "  -"
        print(f"{r['T']:>5}{r['f']:>6.2f} {r['decoder']:<9}{lam:>8}{str(r.get('d', '-')):>8}{st:>8}   "
              + " ".join(f"d{d}:{n}" for d, n in r["fails"].items()))


if __name__ == "__main__":
    main()
