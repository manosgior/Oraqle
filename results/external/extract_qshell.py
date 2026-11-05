"""Extract and verify the qShell numbers that system_cost.py hard-codes, from the verbatim qShell files in qshell/.

Sources (qShell Overleaf git, commit f53859d, 2026-09-09; copied byte for byte into results/external/qshell/):
  figure8_ler.json       Helios d=9, nine-round SI1000 memory shots, native RTL occupancy per shot (50,000 per p)
  figure8_core_ler.json  same 350,000 shots, Helios kernel counter only (x 4 ns), no interface/serialization
  capacity_inputs.json   V80 resource model (LUT budget, common shell, per-application footprints)

Checks (exit 1 on any mismatch):
  QSHELL_P / QSHELL_P99_NS   = nearest-rank p99 of evidence[p].native_service_histogram_ns (figure8_ler.json)
  native - kernel offset     = constant over p (figure8_ler vs figure8_core_ler p99, median, max)
  LUT_D9, V80 budget, shell  = capacity_inputs.json
Writes qshell_summary.json next to this file. Pure standard library.
Usage: python results/external/extract_qshell.py [--no-write]
"""
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
Q = HERE / "qshell"

# hard-coded in oraqle-ics/sims/system_cost.py @ d7cd4e9 (lines 66-68, 77, 81)
SYSTEM_COST = dict(QSHELL_P=[1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2],
                   QSHELL_P99_NS=[1680, 1720, 1804, 1980, 2132, 2360, 2592],
                   HELIOS_P99_NS=308, LUT_D9={"Helios": 116575, "Micro Blossom": 155399},
                   V80_BUDGET_LUT=2574208, V80_SHELL_LUT=63989)


def nearest_rank(hist, f):
    h = {int(k): v for k, v in hist.items()}
    n, c = sum(h.values()), 0
    r = math.ceil(f * n)
    for x in sorted(h):
        c += h[x]
        if c >= r:
            return x


def main():
    ok = True
    f8 = json.load(open(Q / "figure8_ler.json"))
    core = json.load(open(Q / "figure8_core_ler.json"))
    cap = json.load(open(Q / "capacity_inputs.json"))

    native, kernel = [], []
    for p in SYSTEM_COST["QSHELL_P"]:
        key = next(k for k in f8["evidence"] if float(k) == p)
        h = f8["evidence"][key]["native_service_histogram_ns"]
        native.append(dict(p=p, n=sum(h.values()), p50=nearest_rank(h, 0.5), p99=nearest_rank(h, 0.99),
                           max=max(int(k) for k in h), min=min(int(k) for k in h)))
        row = next(r for r in core["top"] if r["p"] == p and r["policy"] == "Static")
        s = row["offered_native_service"]
        kernel.append(dict(p=p, p50=s["median_ns"], p99=s["p99_ns"], max=s["max_ns"], min=s["min_ns"]))
    p99 = [r["p99"] for r in native]
    match = p99 == SYSTEM_COST["QSHELL_P99_NS"]
    ok &= match
    offsets = sorted({(n["p99"] - k["p99"], n["p50"] - k["p50"], n["max"] - k["max"]) for n, k in zip(native, kernel)})
    const = len(offsets) == 1 and len(set(offsets[0])) == 1
    ok &= const
    offset = offsets[0][0] if const else None
    i3 = SYSTEM_COST["QSHELL_P"].index(1e-3)
    timing = f8["evidence"][next(k for k in f8["evidence"] if float(k) == 1e-3)]["timing"]

    apps = cap["applications"]
    lut = dict(Helios=apps["Helios d9"][0], **{"Micro Blossom": apps["MicroBlossom d9"][0]})
    lut_ok = lut == SYSTEM_COST["LUT_D9"]
    budget_ok = cap["budget"][0] == SYSTEM_COST["V80_BUDGET_LUT"] and cap["common_static"][0] == SYSTEM_COST["V80_SHELL_LUT"]
    ok &= lut_ok and budget_ok

    out = dict(
        source="qShell paper repository (Overleaf git), figures/*.json", repo="qshell (local: ~/Documents/GitHub/qshell)",
        commit="f53859dfc7693a14f507bd5a3814fc2af2c0fa94", commit_date="2026-09-09",
        producer_sha256=f8["producer_sha256"], noise="SI1000 (qLDPC 0.3.3 SI1000NoiseModel), d=9, 9 rounds, memory-Z",
        decoder="Helios finite-memory RTL (Verilator), edge weights uniformly two",
        native_occupancy_definition=f8["service_statistics"] + "; " + f8["timing_method"],
        fixed_offset_cycles=dict(occupancy=timing["provenance"]["occupancy_offset_cycles"],
                                 response=timing["provenance"]["response_offset_cycles"],
                                 clock_ns=timing["clock_period_ns"]),
        native_p99_per_p=native, kernel_only_per_p=kernel,
        native_minus_kernel_ns=offset, native_minus_kernel_constant=const,
        system_cost=dict(QSHELL_P=SYSTEM_COST["QSHELL_P"], QSHELL_P99_NS=SYSTEM_COST["QSHELL_P99_NS"],
                         matches_histograms=match,
                         growth_over_1e3_ns=[x - p99[i3] for x in p99],
                         kernel_growth_over_1e3_ns=[k["p99"] - kernel[i3]["p99"] for k in kernel],
                         helios_p99_used_ns=[SYSTEM_COST["HELIOS_P99_NS"] + x - p99[i3] for x in p99],
                         helios_p99_minus_kernel_p99_ns=[SYSTEM_COST["HELIOS_P99_NS"] + x - p99[i3] - k["p99"]
                                                         for x, k in zip(p99, kernel)]),
        capacity=dict(V80_budget_LUT=cap["budget"][0], common_static_shell_LUT=cap["common_static"][0],
                      LUT_D9=lut, native_helios_d9_LUT=cap["native_helios_d9"][0],
                      classification=cap["classification"], matches_system_cost=lut_ok and budget_ok),
        all_match=bool(ok))
    if "--no-write" not in sys.argv[1:]:                # verify/qec_system.py checks without rewriting
        json.dump(out, open(HERE / "qshell_summary.json", "w"), indent=1)

    print("p      native p50/p99/max   kernel p50/p99/max   system_cost p99")
    for n, k, s in zip(native, kernel, SYSTEM_COST["QSHELL_P99_NS"]):
        print(f"{n['p']:<7g}{n['p50']:6}{n['p99']:6}{n['max']:6}   {k['p50']:6}{k['p99']:6}{k['max']:6}   {s:6}")
    print(f"native - kernel = {offset} ns at every p and quantile: {const}")
    print(f"QSHELL_P99_NS from histograms: {'OK' if match else 'MISMATCH'}; LUT_D9 {'OK' if lut_ok else 'MISMATCH'}; "
          f"V80 budget/shell {'OK' if budget_ok else 'MISMATCH'}")
    print(f"Helios p99 used by system_cost (308 + growth) - qShell kernel p99 = "
          f"{out['system_cost']['helios_p99_minus_kernel_p99_ns']} ns")
    if "--no-write" not in sys.argv[1:]:
        print("wrote", HERE / "qshell_summary.json")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
