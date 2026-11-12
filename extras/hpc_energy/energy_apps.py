"""Classical power per logical qubit and energy-to-solution of chemistry and materials programs (draft, not in the paper).

Extends parts B and D of system/system_cost.py with two things an HPC facility asks about a readout operating
point: how much classical power it draws and how much energy a real application costs.

Everything upstream is reused unchanged: the per-round error fits of results/system/summary.json, the required
distance, reaction wait, runtime and decoder footprint of system_cost.py, and the Vitis HLS reports of fpga/.

Power model (per logical qubit, at distance d):
  decoder, FPGA   v80_per_patch(Micro Blossom, d) x card power      (perfectly partitioned LUTs, as in §7.5)
  decoder, CPU    cpu_cores(d, T) x package power / cores           (PyMatching keeping pace with the round)
  discriminator   (d^2 - 1) ancillas / qubits per readout line x LUTs of one instance x power per LUT
Only ancilla readout is charged (the same count as the syndrome bandwidth). Power is proportional to occupied
logic and uses vendor ceilings (constants.json), so the totals are estimates for comparing operating points,
not predictions of a build. Qubits, cryogenics, control electronics, links and cooling are not included.

Program energy = N_L x (decoder + discriminator power) x runtime, with Micro Blossom's reaction wait.

Usage: python extras/hpc_energy/energy_apps.py [--data results/system] [--out extras/hpc_energy/results] [--cnn m8|m16]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(REPO, "system"), os.path.join(REPO, "fpga")]

import system_cost as sc  # noqa: E402
from apps import APPS  # noqa: E402
from parse_reports import table as fpga_table  # noqa: E402
from workloads import WORKLOADS  # noqa: E402

CONST = {k: v["value"] for k, v in json.load(open(os.path.join(HERE, "constants.json"))).items() if k != "note"}
W_V80 = CONST["v80_card_w"]
W_CORE = CONST["cpu_package_w"] / CONST["cpu_cores"]
W_LUT = CONST["discriminator_w_per_lut"] or W_V80 / sc.V80_LUT
DECODER = sc.REFERENCE                                  # Micro Blossom: the decoder whose Lambda the fits assume
CONFIGS = [("MCMit-CNN", 1000), ("MCMit-CNN", 800), ("MCMit-CNN", 600), ("MCMit-CNN", 400),
           ("HERQULES", 600), ("QubiCML", 600), ("QubiCML", 1000)]
BASE = ("MCMit-CNN", 1000)
ANCHOR = "Ising-100"                                    # the paper's largest program, for comparison

_lines = []


def say(s=""):
    print(s)
    _lines.append(s)


def disc_luts(cnn):
    """LUTs of one discriminator instance (five-qubit readout) per system label, from the HLS reports."""
    _, rows, alts = fpga_table()
    cnn_row = rows["MCMit CNN"] if cnn == "m8" else alts["MCMit CNN (m16, 3061 params = Table 2 architecture)"]
    return {"MCMit-CNN": cnn_row["total"]["LUT"], "HERQULES": rows["HERQULES"]["total"]["LUT"],
            "QubiCML": rows["QubiCML"]["total"]["LUT"], "KLiNQ": rows["KLiNQ"]["total"]["LUT"]}


def power_per_lq(label, T, d, fits, luts):
    """Watts per logical qubit: FPGA decoder, CPU decoder, discriminator."""
    cores, extrapolated = sc.cpu_cores(fits, d, T)
    lut = luts.get(label)
    return dict(decoder_fpga_w=sc.v80_per_patch(DECODER, d) * W_V80,
                decoder_cpu_w=cores * W_CORE if cores else None, cpu_extrapolated=extrapolated,
                disc_w=(d * d - 1) / CONST["qubits_per_readout_line"] * lut * W_LUT if lut else None)


def duration(s):
    for unit, n in (("d", 86400), ("h", 3600), ("s", 1)):
        if s >= n:
            return f"{s / n:.1f} {unit}"
    return f"{s * 1e3:.1f} ms"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.path.join(REPO, "results", "system"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--cnn", choices=["m8", "m16"], default="m8",
                    help="MCMit-CNN build: m8 = the paper's FPGA row (957 params), m16 = the Table 2 model (3061)")
    a = ap.parse_args()
    sc.DATA = a.data
    S = json.load(open(os.path.join(a.data, "summary.json")))
    disc = {(s["label"], s["T"]): s for s in S["disc_hi"] if "a" in s and s.get("Lambda", 0) > 1}
    helios, fits, luts = sc.helios_latency(), sc.cpu_model(), disc_luts(a.cnn)
    out = dict(constants=CONST, w_per_lut=W_LUT, w_per_core=W_CORE, decoder=DECODER, cnn_build=a.cnn,
               disc_luts=luts, per_lq={}, apps={})

    say(f"Constants: V80 {W_V80} W/card, CPU {W_CORE:.2f} W/core, discriminator {W_LUT * 1e6:.1f} uW/LUT; "
        f"MCMit-CNN build {a.cnn}")
    say("Discriminator LUTs per five-qubit instance: " + ", ".join(f"{k} {v:,}" for k, v in luts.items()))

    say(f"\n== Classical power per logical qubit at {sc.TARGET} per operation (W)")
    say(f"{'model':<12}{'T':>5}{'d':>5}{'dec FPGA':>10}{'dec CPU':>10}{'discrim':>9}{'FPGA total':>12}{'rel':>6}")
    base_total = None
    rows = []
    for (label, T), s in sorted(disc.items(), key=lambda kv: (kv[0] != BASE, kv[0])):
        d = s.get(f"d@{sc.TARGET}")
        if not d or label not in luts:
            continue
        p = power_per_lq(label, T, d, fits, luts)
        p.update(d=d, total_fpga_w=p["decoder_fpga_w"] + p["disc_w"])
        base_total = base_total or p["total_fpga_w"]
        rows.append((label, T, p))
    for label, T, p in sorted(rows, key=lambda r: (r[0], -r[1])):
        p["rel"] = p["total_fpga_w"] / base_total
        out["per_lq"][f"{label}/{T}"] = p
        cpu = f"{p['decoder_cpu_w']:9.0f}" + ("*" if p["cpu_extrapolated"] else " ") if p["decoder_cpu_w"] else "         -"
        say(f"{label:<12}{T:>5}{p['d']:>5}{p['decoder_fpga_w']:>10.0f}{cpu}{p['disc_w']:>9.0f}"
            f"{p['total_fpga_w']:>12.0f}{p['rel']:>6.2f}")

    programs = {ANCHOR: WORKLOADS[ANCHOR], **APPS}
    for name, wl in programs.items():
        say(f"\n== {name}: N_L={wl['lq']:,} steps={wl['steps']:.3g}  ({wl.get('source', 'system/workloads.py')})")
        say(f"{'config':<16}{'d':>5}{'phys qubits':>14}{'runtime':>10}{'rel':>6}{'V80s':>11}{'dec kW':>9}"
            f"{'disc kW':>9}{'CPU kW':>10}{'MWh':>10}{'rel':>6}")
        res = {}
        for label, T in CONFIGS:
            s = disc.get((label, T))
            if not s:
                continue
            r = sc.program_cost(s["a"], s["b"], wl, T, *sc.reaction_wait(DECODER, label, T, helios), DECODER)
            if not r:
                continue
            p = power_per_lq(label, T, r["d"], fits, luts)
            r.update(dec_kw=wl["lq"] * p["decoder_fpga_w"] / 1e3, disc_kw=wl["lq"] * p["disc_w"] / 1e3,
                     cpu_kw=wl["lq"] * p["decoder_cpu_w"] / 1e3 if p["decoder_cpu_w"] else None,
                     cpu_extrapolated=p["cpu_extrapolated"])
            r["energy_mwh"] = (r["dec_kw"] + r["disc_kw"]) * r["runtime_s"] / 3600 / 1e3
            res[f"{label}/{T}"] = r
        b = res[f"{BASE[0]}/{BASE[1]}"]
        for key, r in res.items():
            r.update(runtime_rel=r["runtime_s"] / b["runtime_s"], energy_rel=r["energy_mwh"] / b["energy_mwh"])
            cpu = f"{r['cpu_kw']:9.0f}" + ("*" if r["cpu_extrapolated"] else " ") if r["cpu_kw"] else "         -"
            say(f"{key:<16}{r['d']:>5}{r['qubits']:>14,}{duration(r['runtime_s']):>10}{r['runtime_rel']:>6.2f}"
                f"{r['v80']:>11,.0f}{r['dec_kw']:>9,.0f}{r['disc_kw']:>9,.0f}{cpu}{r['energy_mwh']:>10.3g}"
                f"{r['energy_rel']:>6.2f}")
        out["apps"][name] = dict({k: v for k, v in wl.items()}, configs=res)

    say("\n* CPU cores extrapolated beyond the measured distances (d <= 57)")
    os.makedirs(a.out, exist_ok=True)
    tag = "" if a.cnn == "m8" else f"_{a.cnn}"
    json.dump(out, open(os.path.join(a.out, f"energy_apps{tag}.json"), "w"), indent=1, default=float)
    open(os.path.join(a.out, f"energy_apps{tag}.txt"), "w").write("\n".join(_lines) + "\n")


if __name__ == "__main__":
    main()
