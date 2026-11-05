#!/usr/bin/env python3
"""Rebuild tab:fpga_resources (ml_discriminators.tex:275-293) from the Vitis HLS reports.
[unify, new file]

Reads the synthesis reports in fpga/hls4ml/reports/ (Vitis HLS 2023.2, part
xczu49dr-ffvf1760-2-e, 2 ns clock = 500 MHz; produced by Felix Gust's hls4ml scripts,
github.com/gustifix/mcmit-model @56cfcc0) and prints, per report, the "Total" row of
the utilization summary, the device availability and the latency (cycles x 2 ns).

The paper's rows are compositions of these reports, as the table footnotes say:
  QubiCML   = 5 x qubic.txt                       (per-qubit model, five instances)
  HERQULES  = herqules.txt                        (one five-qubit model)
  MCMit CNN = cnn-m8-parallelizationfactors.txt   (m_param=8, 957 parameters, ds40 input)
  KLiNQ     = 3 x klinq.txt + 2 x klinq-larger-reuse4.txt   (small student on Q1/Q4/Q5,
              larger student on Q2/Q3; latency = slowest instance, the instances run in parallel)
Percentages are of the device resources listed in each report, rounded to integers as
in the paper. These are HLS C-synthesis estimates, not post-place-and-route numbers.

    python fpga/parse_reports.py            # table + all reports
    python fpga/parse_reports.py --json     # machine-readable
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "hls4ml", "reports")

# paper row -> list of (report, multiplicity)
PAPER_ROWS = {
    "QubiCML": [("qubic.txt", 5)],
    "HERQULES": [("herqules.txt", 1)],
    "MCMit CNN": [("cnn-m8-parallelizationfactors.txt", 1)],
    "KLiNQ": [("klinq.txt", 3), ("klinq-larger-reuse4.txt", 2)],
}
# alternatives worth showing next to the paper rows
ALTERNATIVES = {
    "MCMit CNN (m16, 3061 params = Table 2 architecture)": [("cnn-m16.txt", 1)],
    "MCMit CNN (m8, no ParallelizationFactor)": [("cnn-m8.txt", 1)],
    "KLiNQ with klinq-larger (no reuse)": [("klinq.txt", 3), ("klinq-larger.txt", 2)],
    "QubiCML single instance": [("qubic.txt", 1)],
}


def parse(path):
    s = open(path).read()
    tot = re.search(r"^\|Total\s*\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|", s, re.M)
    av = re.search(r"^\|Available\s*\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|", s, re.M)
    clk = re.search(r"\|ap_clk\s*\|\s*([\d.]+) ns\|\s*([\d.]+) ns\|", s)
    lat = re.search(r"\+ Latency:.*?\n(?:.*\n){5}\s*\|\s*(\d+)\|\s*(\d+)\|\s*([\d.]+ \w+)\|\s*([\d.]+ \w+)\|\s*(\d+)\|\s*(\d+)\|\s*(\w+)\|", s)
    part = re.search(r"Target device:\s*(\S+)", s).group(1)
    top = re.search(r"Vitis HLS Report for '([^']+)'", s).group(1)
    names = ["BRAM_18K", "DSP", "FF", "LUT", "URAM"]
    return {
        "report": os.path.basename(path), "top": top, "part": part,
        "total": dict(zip(names, map(int, tot.groups()))), "available": dict(zip(names, map(int, av.groups()))),
        "clock_target_ns": float(clk.group(1)), "clock_estimated_ns": float(clk.group(2)),
        "latency_cycles": [int(lat.group(1)), int(lat.group(2))], "interval": [int(lat.group(5)), int(lat.group(6))],
        "pipeline": lat.group(7),
    }


def compose(parts, reps):
    tot = {k: 0 for k in ("LUT", "FF", "DSP", "BRAM_18K")}
    lat_ns = 0.0
    for name, mult in parts:
        r = reps[name]
        for k in tot:
            tot[k] += mult * r["total"][k]
        lat_ns = max(lat_ns, r["latency_cycles"][1] * r["clock_target_ns"])
    av = reps[parts[0][0]]["available"]
    pct = {k: 100.0 * tot[k] / av[k] for k in ("LUT", "FF", "DSP")}
    return {"parts": parts, "total": tot, "pct": pct, "pct_rounded": {k: round(v) for k, v in pct.items()},
            "latency_ns": lat_ns}


def all_reports():
    return {f: parse(os.path.join(REPORTS, f)) for f in sorted(os.listdir(REPORTS)) if f.endswith(".txt")}


def table():
    reps = all_reports()
    return reps, {k: compose(v, reps) for k, v in PAPER_ROWS.items()}, {k: compose(v, reps) for k, v in ALTERNATIVES.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    reps, rows, alts = table()
    if a.json:
        print(json.dumps({"reports": reps, "paper_rows": rows, "alternatives": alts}, indent=1))
        return
    print("Per report (Total row; % of xczu49dr: 425,280 LUT / 850,560 FF / 4,272 DSP):")
    print(f"  {'report':<36}{'LUT':>8}{'FF':>8}{'DSP':>6}  {'LUT%':>6}{'FF%':>6}{'DSP%':>6}  latency")
    for f, r in reps.items():
        t, v = r["total"], r["available"]
        c0, c1 = r["latency_cycles"]
        lat = f"{c0}" if c0 == c1 else f"{c0}-{c1}"
        print(f"  {f:<36}{t['LUT']:>8}{t['FF']:>8}{t['DSP']:>6}  {100*t['LUT']/v['LUT']:>6.2f}{100*t['FF']/v['FF']:>6.2f}"
              f"{100*t['DSP']/v['DSP']:>6.2f}  {lat} cyc x {r['clock_target_ns']:g} ns = {c1*r['clock_target_ns']:g} ns "
              f"(II {r['interval'][0]}, {r['pipeline']})")
    print("\ntab:fpga_resources as composed from the reports:")
    print(f"  {'row':<12}{'LUT':>6}{'FF':>6}{'DSP':>6}{'Latency':>10}   composition")
    for k, r in rows.items():
        p = r["pct_rounded"]
        comp = " + ".join(f"{m} x {n}" for n, m in r["parts"])
        print(f"  {k:<12}{p['LUT']:>5}%{p['FF']:>5}%{p['DSP']:>5}%{r['latency_ns']:>8g} ns   {comp}  "
              f"(exact {r['pct']['LUT']:.2f}/{r['pct']['FF']:.2f}/{r['pct']['DSP']:.2f}%)")
    print("\nAlternatives (not in the paper):")
    for k, r in alts.items():
        p = r["pct_rounded"]
        print(f"  {k:<52}{p['LUT']:>4}%{p['FF']:>4}%{p['DSP']:>4}%{r['latency_ns']:>7g} ns")


if __name__ == "__main__":
    main()
