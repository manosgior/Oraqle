"""Parts B and D: what a readout operating point costs a whole program and its classical plane.

Inputs: data/sims/summary.json (analyze.py): the per-round error fit ln eps(d) = a + b d of every
(discriminator, readout length) point, and the decode timings (decode_timing.py); data/sims/decoders_summary.json
(analyze_decoders.py): the same fit per decoder type for MCMit-CNN readout (I6).

Hardware decoders (surface code, AMD Alveo V80 unless noted), each paired with the simulated decoder whose
accuracy it can reach (FIT):
  Micro Blossom  exact MWPM on the weighted graph -> mwpm_w, the Lambda of disc_hi and of every main result.
                 772 ns latency, 86 ns per round at d = 9 (V80 RTL, ours); 0.37 us streaming (reported).
                 No noise dependence is available, so its latency is a lower bound here.
  Helios         distributed union-find, unweighted (its default; every latency below) -> mwpm_u, an optimistic
                 proxy (union-find approximates matching). Decode stage 184 ns median / 308 ns p99 at d = 9,
                 p = 1e-3 (cycle-accurate RTL, ours; CXL-Quantum Table tab:req-decoders), 168 ns per round. Its
                 p99 grows with the noise: qshell figure8_ler.json gives the native p99 of the same RTL (edge
                 weights uniformly two) at p = 1e-4..1e-2 (SI1000); we add that growth over p = 1e-3 to the 308 ns.
                 Each readout length is placed on that curve by its detection-event density (d = 9 timing runs)
                 matched to uniform circuit noise (UNIFORM_FRACTION, Stim rotated memory, d = 9); beyond 1e-2 the
                 growth is extrapolated in log p (flagged).
  Helios (weighted)  weights 2..w_max ~ -log p, w_max = 16 -> mwpm_q16. The Helios paper (Fig. 7c) reports the
                 mean time per round at d = 13 rising from 15 to 38 ns with w_max = 16; we scale the
                 unweighted p99 by that factor (an estimate: no weighted tail is reported).
  GPU            CUDA-Q QEC on NVQLink, ~2.1 us at d = 13 (reported) -> mwpm_w.
  Footprint      V80 LUTs of one decoder instance at d = 9 (qshell capacity_inputs.json): Helios 116,575, Micro
                 Blossom 155,399, scaled as d^K_LUT with K_LUT fitted to Helios's own synthesis at d = 9..21
                 (Helios paper Table I, VCU129; vertices and edges grow as d^3, PE bit-widths add the rest).
                 d = 3 is left out: there, fixed wrapper logic dominates (it gave the earlier k = 2.47). Micro
                 Blossom also instantiates a unit per vertex and edge, so the same law is assumed. An instance
                 decodes one patch; above d ~ 21 it no longer fits one card and the count assumes a perfectly
                 partitioned decoder (a lower bound). V80 budget 2,574,208 LUT minus the 63,989-LUT shell.

B  Time-to-solution. A program of N_L patches runs QEC for its whole duration:
     reaction R = t_disc + t_dec + FRAME_NS (no transport: decoder on the readout FPGA, the best case)
     wait     w = max(1, ceil(R / t_r)) rounds per feed-forward step
     rounds   = steps (d + 1 + w) + cliff (d + 1);  N_L * rounds * eps(d) <= BUDGET (smallest odd d)
     runtime  = rounds * t_r;  physical qubits = N_L (2 d^2 - 1)
D  Classical plane at the required distance:
     syndrome bandwidth  = (d^2 - 1) b / t_r per logical qubit   b = 1 (hard) or 2 / 8 (soft) bits
     decoder FPGAs       = LUT(d) / usable V80 LUT per logical qubit (Helios, Micro Blossom)
     CPU cores           = tau_dec(d, T) / t_r, PyMatching per patch-round on one core (software reference)
I6 Decoder choice: B for every hardware decoder, with its own reaction wait and the Lambda fit of its FIT decoder
     (paired samples, CNN readout, hard and soft points), relative to Micro Blossom at 1000 ns hard.

Usage: python system/system_cost.py [--data DIR]   (reads DIR/summary.json, decoders_summary.json, decode_timing*;
       prints tables, writes DIR/system_cost.json; DIR defaults to $ORAQLE_WORK/system)
External inputs (qshell, CXL-Quantum, Helios paper) and the derivation of UNIFORM_FRACTION
(system/derive_uniform_fraction.py): results/external/README.md.
"""
import argparse
import collections
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discriminators import LATENCY_NS
from workloads import WORKLOADS

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oraqle_env import WORK  # noqa: E402

DATA = str(WORK / "system")       # was <oraqle-ics>/data/sims; override with --data
BUDGET = 0.01                     # total program failure probability
GATES_NS = 340
FRAME_NS = 100                    # frame update and branch selection
TARGET = "1e-09"

HELIOS_P99_NS = 308               # d = 9, p = 1e-3
QSHELL_P = [1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2]
QSHELL_P99_NS = [1680, 1720, 1804, 1980, 2132, 2360, 2592]
UNIFORM_P = [1e-3, 3e-3, 5e-3, 1e-2, 2e-2]
UNIFORM_FRACTION = [0.0166, 0.0481, 0.0774, 0.1428, 0.2448]   # producer: system/derive_uniform_fraction.py
HELIOS_WEIGHTED = 38 / 15         # Helios Fig. 7c: mean ns per round at d = 13, w_max = 16 vs unweighted
DECODERS = {"Micro Blossom": dict(lat_ns=772, fit="mwpm_w"),
            "Helios (weighted)": dict(lat_ns=None, scale=HELIOS_WEIGHTED, fit="mwpm_q16"),
            "Helios": dict(lat_ns=None, scale=1.0, fit="mwpm_u"),
            "GPU": dict(lat_ns=2100, fit="mwpm_w")}
REFERENCE = "Micro Blossom"       # decodes the graph the main results simulate
LUT_D9 = {"Helios": 116575, "Micro Blossom": 155399}
HELIOS_TABLE1 = {3: 970, 5: 6425, 9: 52111, 13: 165718, 17: 448314, 21: 898715}
K_LUT = float(np.polyfit(np.log([d for d in HELIOS_TABLE1 if d >= 9]),
                         np.log([v for d, v in HELIOS_TABLE1.items() if d >= 9]), 1)[0])
V80_LUT = 2574208 - 63989


def interp_log(x, xs, ys):
    """Piecewise-linear in log x, extrapolated from the last segment."""
    lx, lxs = math.log(x), [math.log(v) for v in xs]
    i = max(1, min(len(xs) - 1, next((j for j, v in enumerate(lxs) if v >= lx), len(xs) - 1)))
    t = (lx - lxs[i - 1]) / (lxs[i] - lxs[i - 1])
    return ys[i - 1] + t * (ys[i] - ys[i - 1])


def helios_latency():
    """Helios p99 per readout length from its detection-event density (d = 9 timing runs)."""
    rows = [json.loads(l) for l in open(os.path.join(DATA, "decode_timing_d9.jsonl"))]
    out = {}
    for r in rows:
        frac = r["events_per_round"] * r["rounds"] / r["detectors"]
        lp = interp_log(frac, UNIFORM_FRACTION, [math.log(p) for p in UNIFORM_P])
        p = math.exp(lp)
        growth = interp_log(p, QSHELL_P, QSHELL_P99_NS) - QSHELL_P99_NS[2]
        out[r["T"]] = dict(detection_fraction=frac, p_equiv=p, p99_ns=HELIOS_P99_NS + growth,
                           extrapolated=p > QSHELL_P[-1])
    return out


def reaction_wait(decoder, model, T, helios):
    spec = DECODERS[decoder]
    t_dec = helios[T]["p99_ns"] * spec["scale"] if spec["lat_ns"] is None else spec["lat_ns"]
    R = (LATENCY_NS.get(model) or 74) + t_dec + FRAME_NS
    return R, max(1, math.ceil(R / (T + GATES_NS)))


def required_d(a, b, n_l, steps, cliff, w, dmax=601):
    for d in range(3, dmax, 2):
        rounds = steps * (d + 1 + w) + cliff * (d + 1)
        if n_l * rounds * math.exp(a + b * d) <= BUDGET:
            return d, rounds
    return None, None


def v80_per_patch(decoder, d):
    return LUT_D9[decoder] * (d / 9) ** K_LUT / V80_LUT


def v80_dmax(decoder):
    """Largest odd d whose decoder instance fits one V80."""
    return max(d for d in range(3, 101, 2) if v80_per_patch(decoder, d) <= 1)


def cpu_model():
    """PyMatching tau per patch-round (ns) as c d^k per readout length, from every timing file."""
    rows = [json.loads(l) for f in glob.glob(os.path.join(DATA, "decode_timing_d*.jsonl")) +
            glob.glob(os.path.join(DATA, "decode_large", "*.jsonl")) for l in open(f)]
    by_T = collections.defaultdict(list)
    for r in rows:
        by_T[r["T"]].append((r["d"], r["batch_ns_per_round"] / 2))      # two patches per experiment
    fits = {}
    for T, pts in by_T.items():
        pts = sorted(p for p in pts if p[0] >= 9)
        k, logc = np.polyfit(np.log([d for d, _ in pts]), np.log([t for _, t in pts]), 1)
        fits[T] = dict(k=k, c=math.exp(logc), dmax=max(d for d, _ in pts), points=pts)
    return fits


def cpu_cores(fits, d, T):
    f = fits.get(T)
    if not f:
        return None, True
    return f["c"] * d ** f["k"] / (T + GATES_NS), d > f["dmax"]


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    DATA = ap.parse_args().data
    S = json.load(open(os.path.join(DATA, "summary.json")))
    disc = {(s["label"], s["T"]): s for s in S["disc_hi"] if "a" in s and s.get("Lambda", 0) > 1}
    helios = helios_latency()
    fits = cpu_model()
    out = dict(budget=BUDGET, frame_ns=FRAME_NS, helios=helios, cpu_fits=fits, reaction={}, per_lq={}, workloads={})

    print("== Helios p99 per readout length (detection density -> equivalent uniform p -> qshell growth)")
    for T, h in sorted(helios.items()):
        print(f"T={T:5}: detections {h['detection_fraction']:.3f}  p_equiv {h['p_equiv']:.4f}  "
              f"p99 {h['p99_ns']:.0f} ns{'  (extrapolated)' if h['extrapolated'] else ''}")
    print("CPU fits tau = c d^k:", "  ".join(f"T={T}: k={f['k']:.2f} (d<={f['dmax']})" for T, f in sorted(fits.items())))
    out.update(k_lut=K_LUT, v80_dmax={dec: v80_dmax(dec) for dec in LUT_D9})
    print(f"Decoder LUT ~ d^{K_LUT:.2f} (Helios Table I, d >= 9); one V80 holds "
          + ", ".join(f"{dec} up to d={v80_dmax(dec)}" for dec in LUT_D9))

    print(f"\n== reaction R and wait w (rounds per feed-forward step), MCMit-CNN")
    print(f"{'decoder':<19}" + "".join(f"{T:>14}" for T in (400, 600, 800, 1000)))
    for dec in DECODERS:
        cells = []
        for T in (400, 600, 800, 1000):
            R, w = reaction_wait(dec, "MCMit-CNN", T, helios)
            out["reaction"][f"{dec}/{T}"] = dict(R_ns=R, w=w, t_r_ns=T + GATES_NS)
            cells.append(f"{R / 1e3:.2f}us w={w}")
        print(f"{dec:<19}" + "".join(f"{c:>14}" for c in cells))

    print(f"\n== D per logical qubit at {TARGET} per op")
    print(f"{'model':<12}{'T':>5}{'d':>5}{'Gb/s 1b':>9}{'2b':>8}{'8b':>8}{'V80 Hel':>9}{'V80 MB':>8}{'CPU':>8}")
    for (label, T), s in sorted(disc.items()):
        d = s.get(f"d@{TARGET}")
        if not d:
            continue
        bw = (d * d - 1) / ((T + GATES_NS) * 1e-9) / 1e9
        cores, extra = cpu_cores(fits, d, T)
        r = dict(d=d, gbps_hard=bw, gbps_2bit=2 * bw, gbps_8bit=8 * bw, v80_helios=v80_per_patch("Helios", d),
                 v80_microblossom=v80_per_patch("Micro Blossom", d), cpu_cores=cores, cpu_extrapolated=extra)
        out["per_lq"][f"{label}/{T}"] = r
        print(f"{label:<12}{T:>5}{d:>5}{bw:>9.2f}{2 * bw:>8.1f}{8 * bw:>8.1f}{r['v80_helios']:>9.2f}"
              f"{r['v80_microblossom']:>8.2f}{(f'{cores:7.0f}' + ('*' if extra else ' ')) if cores else '       -'}")

    configs = [("MCMit-CNN", 1000), ("MCMit-CNN", 800), ("MCMit-CNN", 600), ("MCMit-CNN", 400),
               ("HERQULES", 600), ("QubiCML", 600), ("QubiCML", 1000)]
    for name, wl in WORKLOADS.items():
        print(f"\n== {name}: N_L={wl['lq']} steps={wl['steps']} cliff={wl['cliff']}  (failure budget {BUDGET:g})")
        print(f"{'config':<16}{'decoder':<19}{'w':>3}{'d':>5}{'phys qubits':>13}{'runtime':>11}{'rel':>6}"
              f"{'Gb/s':>8}{'V80s':>7}")
        rows = {}
        for label, T in configs:
            s = disc.get((label, T))
            if not s:
                continue
            for dec, spec in DECODERS.items():
                if spec["fit"] != "mwpm_w":           # disc_hi is weighted matching; other decoders: I6 below
                    continue
                r = program_cost(s["a"], s["b"], wl, T, *reaction_wait(dec, label, T, helios), dec)
                if r:
                    rows[f"{label}/{T}/{dec}"] = r
        print_rows(rows, rows[f"MCMit-CNN/1000/{REFERENCE}"]["runtime_s"])
        out["workloads"][name] = dict(wl, configs=rows)

    fitted = {(r["T"], r["f"], r["decoder"]): r for r in json.load(open(os.path.join(DATA, "decoders_summary.json")))
              if "a" in r and r.get("Lambda", 0) > 1}
    points = sorted({(T, f) for T, f, _ in fitted})
    out["decoder_choice"] = {}
    for name, wl in WORKLOADS.items():
        print(f"\n== decoder choice, {name} (CNN readout; Lambda of each decoder's FIT; f = flagged fraction)")
        print(f"{'config':<16}{'decoder':<19}{'w':>3}{'d':>5}{'phys qubits':>13}{'runtime':>11}{'rel':>6}"
              f"{'Gb/s':>8}{'V80s':>7}")
        rows = {}
        for T, f in points:
            for dec, spec in DECODERS.items():
                s = fitted.get((T, f, spec["fit"]))
                if not s:
                    continue
                r = program_cost(s["a"], s["b"], wl, T, *reaction_wait(dec, "MCMit-CNN", T, helios), dec)
                if r:
                    rows[f"{T}/{f}/{dec}"] = dict(r, Lambda=s["Lambda"], fit=spec["fit"])
        print_rows(rows, rows[f"1000/0.0/{REFERENCE}"]["runtime_s"])
        out["decoder_choice"][name] = rows
    json.dump(out, open(os.path.join(DATA, "system_cost.json"), "w"), indent=1, default=float)
    print("\n* CPU cores extrapolated beyond the measured distances")


def program_cost(a, b, wl, T, R, w, dec):
    d, rounds = required_d(a, b, wl["lq"], wl["steps"], wl["cliff"], w)
    if d is None:
        return None
    t_r = (T + GATES_NS) * 1e-9
    base = dec.split(" (")[0]
    return dict(R_ns=R, w=w, d=d, rounds=rounds, qubits=wl["lq"] * (2 * d * d - 1), runtime_s=rounds * t_r,
                gbps_hard=wl["lq"] * (d * d - 1) / t_r / 1e9,
                v80=wl["lq"] * v80_per_patch(base, d) if base in LUT_D9 else None)


def print_rows(rows, base):
    for key, r in rows.items():
        label_T, dec = key.rsplit("/", 1)
        rt = r["runtime_s"]
        rts = f"{rt * 1e3:.1f} ms" if rt < 1 else f"{rt:.2f} s"
        v = f"{r['v80']:7.0f}" if r["v80"] else "      -"
        print(f"{label_T:<16}{dec:<19}{r['w']:>3}{r['d']:>5}{r['qubits']:>13,}{rts:>11}{rt / base:>6.2f}"
              f"{r['gbps_hard']:>8.1f}{v}")


if __name__ == "__main__":
    main()
