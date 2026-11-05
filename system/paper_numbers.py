"""Every number of §7 (system_bottleneck.tex) from the result files, and the check against results/system/paper_values.json.

Inputs (all in --data, default results/system): summary.json, summary_soft.json, soft_params.json,
decoders_summary.json, system_cost.json (the outputs of analyze.py, summarize_soft.py, merge_decoders.sh and
system_cost.py) and the raw decode-timing files. Constants of the sweep scripts are read with `ast`, so stim and
PyMatching are not needed: numpy only.

compute(data) returns {key: value}; a value is a number, a [min, max] pair, or a list. The definitions below are the
ones that reproduce the paper's numbers (each key names its chapter line in paper_values.json):

  rel. to CNN@1000   spacetime/qubits divided by those of MCMit-CNN at 1000 ns (summary.json disc_hi, 1e-9 per op)
  cores/patch        PyMatching batch time per patch-round / round (decode_timing, two patches per experiment)
  program ratios     system_cost.json workloads: runtime/qubits relative to MCMit-CNN 1000 ns with Micro Blossom
  decoder choice     system_cost.json decoder_choice: relative to Micro Blossom at 1000 ns hard
  best soft flag     the flag fraction with the smallest spacetime at 1e-9 (ties: smallest f)
  bandwidth          (d^2 - 1) b / t_r per logical qubit, b = 1 (hard) or 2 (flagged) bits

Usage: python system/paper_numbers.py [--data results/system] [--check [results/system/paper_values.json]]
       (with --check: print every paper value against the measured one; exit 1 on any mismatch)
"""
import argparse
import ast
import glob
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
from oraqle_env import RESULTS  # noqa: E402

GATES_NS = 340
DISC4 = ["MCMit-CNN", "HERQULES", "KLiNQ", "QubiCML"]          # the discriminators plotted in Fig. sys_spacetime
WORKLOAD_LQ_REF = "MCMit-CNN/1000/Micro Blossom"


def consts(fn, names):
    """Module-level constants of a script, without importing it (sweep scripts import stim)."""
    tree = ast.parse(open(os.path.join(HERE, fn)).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in names:
                try:
                    out[node.targets[0].id] = ast.literal_eval(node.value)
                except ValueError:                     # e.g. PRESETS = {"ibm_boston": dict(cz=...)}
                    out[node.targets[0].id] = eval(compile(ast.Expression(node.value), fn, "eval"),
                                                   {"__builtins__": {}, "dict": dict})
    return out


def rng(xs):
    xs = [x for x in xs if x is not None]
    return [min(xs), max(xs)]


def compute(data=None):
    data = data or str(RESULTS / "system")
    J = lambda f: json.load(open(os.path.join(data, f)))  # noqa: E731
    S, SS, P, DS, C = J("summary.json"), J("summary_soft.json"), J("soft_params.json"), J("decoders_summary.json"), \
        J("system_cost.json")
    import system_cost as SC   # numpy only
    import workloads as WL
    from discriminators import FIDELITY, measure_error
    v = {}

    # ---------------- §7.1 methodology (sys:9, 11, 14, 16)
    disc = {(s["label"], s["T"]): s for s in S["disc_hi"]}
    grid = {(s["T"], round(s["p"], 4)): s for s in S["grid"]}
    lam = lambda s: s.get("Lambda", float("nan"))  # noqa: E731
    v["lambda_cnn_600_1000"] = [lam(disc[("MCMit-CNN", T)]) for T in (600, 800, 1000)]
    v["gates_ns"] = consts("analyze.py", {"GATES_NS"})["GATES_NS"]
    v["readout_share_400_1000_pct"] = rng([100 * T / (T + GATES_NS) for T in (400, 1000)])
    v["distances_disc_hi"] = sorted({int(d) for s in S["disc_hi"] for d in s["fails"]})
    v["distances_grid"] = sorted({int(d) for s in S["grid"] for d in s["fails"]})
    v["disc_hi_min_fails"] = min(f for s in S["disc_hi"] for f in s["fails"].values())
    v["grid_median_fails"] = statistics.median(f for s in S["grid"] for f in s["fails"].values())
    v["grid_T_range"] = rng([s["T"] for s in S["grid"]])
    v["grid_p_range_pct"] = rng([100 * s["p"] for s in S["grid"]])
    v["cnn_readout_error_pct"] = {T: 100 * measure_error("MCMit-CNN", T) for T in (400, 600, 800, 1000)}
    v["models"] = sorted(FIDELITY)
    v["mb_latency_ns"] = SC.DECODERS["Micro Blossom"]["lat_ns"]
    v["helios_p99_ns"] = SC.HELIOS_P99_NS
    v["helios_weighted_factor"] = SC.HELIOS_WEIGHTED
    v["gpu_latency_us"] = SC.DECODERS["GPU"]["lat_ns"] / 1e3
    v["k_lut"] = C["k_lut"]
    v["helios_table_fit_ds"] = rng([d for d in SC.HELIOS_TABLE1 if d >= 9])
    big = [json.loads(l) for f in glob.glob(os.path.join(data, "decode_large", "*.jsonl")) for l in open(f)]
    v["decode_timing_dmax"] = max(r["d"] for r in big + S["decode"])
    v["workload_lq_range"] = rng([w["lq"] for w in WL.WORKLOADS.values()])
    v["rotation_eps"] = WL.EPS
    v["program_budget"] = SC.BUDGET

    # ---------------- §7.2 error suppression and spacetime (sys:33, 39, 42, 44)
    v["grid_lambda_max_p07"] = max(lam(grid[(T, 0.07)]) for T in range(200, 1001, 100))
    v["grid_lambda_max_p06"] = max(lam(grid[(T, 0.06)]) for T in range(200, 1001, 100))
    v["disc4_p400_min_pct"] = min(100 * disc[(m, 400)]["p"] for m in DISC4)
    v["disc4_p600_max_pct"] = max(100 * disc[(m, 600)]["p"] for m in DISC4)
    v["disc4_lambda600"] = {m: lam(disc[(m, 600)]) for m in DISC4}
    v["disc3_p600_max_pct"] = max(100 * disc[(m, 600)]["p"] for m in DISC4 if m != "QubiCML")    # 'every ... but QubiCML'
    v["disc3_lambda600_min"] = min(lam(disc[(m, 600)]) for m in DISC4 if m != "QubiCML")
    base = disc[("MCMit-CNN", 1000)]
    st = lambda m, T: disc[(m, T)]["spacetime@1e-09"] / base["spacetime@1e-09"]  # noqa: E731
    qb = lambda m, T: disc[(m, T)]["qubits@1e-09"] / base["qubits@1e-09"]  # noqa: E731
    v["st400_rel_disc4"] = rng([st(m, 400) for m in DISC4])
    v["st400_rel"] = {m: st(m, 400) for m in DISC4}
    v["grid_lambda_p04_T200"] = lam(grid[(200, 0.04)])
    v["grid_lambda_p04_T1000"] = lam(grid[(1000, 0.04)])
    v["grid_lambda_p05_T600"] = lam(grid[(600, 0.05)])
    v["grid_lambda_p08_T600"] = lam(grid[(600, 0.08)])
    for T in (1000, 800, 600, 400):
        v[f"cnn_lambda_{T}"] = lam(disc[("MCMit-CNN", T)])
        v[f"cnn_d_{T}"] = disc[("MCMit-CNN", T)]["d@1e-09"]
    v["n_below_threshold_200"] = sum(1 for (m, T), s in disc.items() if T == 200 and lam(s) > 1)
    op = lambda m, T: (disc[(m, T)]["d@1e-09"] + 1) * (T + GATES_NS)  # noqa: E731
    v["op_speedup_600_800"] = rng([op("MCMit-CNN", 1000) / op("MCMit-CNN", T) for T in (600, 800)])
    v["st_rel_600_800"] = rng([st("MCMit-CNN", T) for T in (600, 800)])
    v["qubicml_p1000_pct"] = 100 * disc[("QubiCML", 1000)]["p"]
    v["cnn_p1000_pct"] = 100 * disc[("MCMit-CNN", 1000)]["p"]
    v["qubicml_lambda_1000"] = lam(disc[("QubiCML", 1000)])
    v["qubicml_qubits_rel_1000"] = qb("QubiCML", 1000)
    v["qubicml_st_rel_1000"] = st("QubiCML", 1000)
    v["herqules_st_rel_cnn1000_600_1000"] = rng([st("HERQULES", T) for T in (600, 800, 1000)])
    v["herqules_st_rel_same_T_600_1000"] = rng([disc[("HERQULES", T)]["spacetime@1e-09"] /
                                                disc[("MCMit-CNN", T)]["spacetime@1e-09"] for T in (600, 800, 1000)])

    # ---------------- §7.3 decoder load (sys:60, 65-82, 85, 87, 89)
    dec = {(r["d"], r["T"]): r for r in S["decode"]}
    cores = lambda d, T: dec[(d, T)]["batch_ns_per_round"] / 2 / (T + GATES_NS)  # noqa: E731
    rel = lambda d, T: dec[(d, T)]["batch_ns_per_round"] / dec[(d, 1000)]["batch_ns_per_round"]  # noqa: E731
    ds = sorted({d for d, _ in dec})
    for T in (1000, 600, 400, 200):
        v[f"cores_d17_{T}"] = cores(17, T)
    for T in (600, 400, 200):
        v[f"decode_rel_{T}"] = rng([rel(d, T) for d in ds])
    for T in (1000, 600, 400, 200):
        v[f"detection_pct_d9_{T}"] = 100 * C["helios"][str(T)]["detection_fraction"]
    v["p_equiv_pct_400_1000"] = rng([100 * C["helios"][str(T)]["p_equiv"] for T in (400, 600, 800, 1000)])
    v["helios_p99_ns_600_1000"] = rng([C["helios"][str(T)]["p99_ns"] for T in (600, 800, 1000)])
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "paper_figures", "system"))
    import tab_sys_reaction as TAB
    v["tab_sys_reaction_rows"] = [l for l in TAB.latex(*TAB.cells(data)).splitlines() if l != "\\midrule"][1:]
    cnn_cfg = [(name, k, c, wl) for name, wl in C["workloads"].items() for k, c in wl["configs"].items()
               if k.startswith("MCMit-CNN")]
    v["program_d_range_cnn"] = rng([c["d"] for _, _, c, _ in cnn_cfg])
    v["waiting_pct_range_cnn"] = rng([100 * w["steps"] * c["w"] / c["rounds"] for _, _, c, w in cnn_cfg])
    v["sw_decoding_cost_600_400"] = [cores(17, 600) / cores(17, 1000), cores(17, 400) / cores(17, 1000)]

    # ---------------- §7.4 programs (sys:105, 111, 113, 115)
    def ratios(cfg_key):
        out_rt, out_q = [], []
        for wl in C["workloads"].values():
            b, c = wl["configs"][WORKLOAD_LQ_REF], wl["configs"].get(cfg_key)
            if c:
                out_rt.append(c["runtime_s"] / b["runtime_s"])
                out_q.append(c["qubits"] / b["qubits"])
        return rng(out_rt), rng(out_q)
    v["prog_runtime_600"], v["prog_qubits_600"] = ratios("MCMit-CNN/600/Micro Blossom")
    v["prog_runtime_400"], v["prog_qubits_400"] = ratios("MCMit-CNN/400/Micro Blossom")
    v["prog_runtime_qubicml1000"], v["prog_qubits_qubicml1000"] = ratios("QubiCML/1000/Micro Blossom")
    I = C["workloads"]["Ising-100"]["configs"]
    v["ising100_runtime_1000_s"] = I["MCMit-CNN/1000/Micro Blossom"]["runtime_s"]
    v["ising100_runtime_600_s"] = I["MCMit-CNN/600/Micro Blossom"]["runtime_s"]
    v["ising100_qubits_600_rel"] = I["MCMit-CNN/600/Micro Blossom"]["qubits"] / I["MCMit-CNN/1000/Micro Blossom"]["qubits"]
    v["log10_phys_qubits_range_cnn"] = [math.log10(x) for x in rng([c["qubits"] for _, _, c, _ in cnn_cfg])]
    v["workload_lq"] = {n: w["lq"] for n, w in WL.WORKLOADS.items()}

    # ---------------- §7.5 classical plane (sys:127, 129, 131)
    L = C["per_lq"]
    v["gbps_cnn_600_1000"] = rng([L[f"MCMit-CNN/{T}"]["gbps_hard"] for T in (600, 800, 1000)])
    v["gbps_cnn_400"] = L["MCMit-CNN/400"]["gbps_hard"]
    v["v80_dmax_mb"], v["v80_dmax_helios"] = C["v80_dmax"]["Micro Blossom"], C["v80_dmax"]["Helios"]
    v["mb_cards_cnn_600_1000"] = rng([L[f"MCMit-CNN/{T}"]["v80_microblossom"] for T in (600, 800, 1000)])
    v["mb_cards_cnn_400"] = L["MCMit-CNN/400"]["v80_microblossom"]
    v["helios_smaller_pct"] = 100 * (1 - SC.LUT_D9["Helios"] / SC.LUT_D9["Micro Blossom"])
    Q = C["workloads"]["QFT-30"]["configs"]
    v["qft30_cards_600_1000"] = rng([Q[f"MCMit-CNN/{T}/Micro Blossom"]["v80"] for T in (600, 800, 1000)])
    v["qft30_cards_400"] = Q["MCMit-CNN/400/Micro Blossom"]["v80"]
    v["gbps_ratio_600_vs_1000"] = L["MCMit-CNN/600"]["gbps_hard"] / L["MCMit-CNN/1000"]["gbps_hard"]
    g = [L[f"MCMit-CNN/{T}"]["gbps_hard"] for T in (600, 800, 1000)]
    c = [L[f"MCMit-CNN/{T}"]["v80_microblossom"] for T in (600, 800, 1000)]
    v["gbps_mult_400"] = rng([L["MCMit-CNN/400"]["gbps_hard"] / x for x in g])
    v["cards_mult_400"] = rng([L["MCMit-CNN/400"]["v80_microblossom"] / x for x in c])
    v["qubicml_gbps_rel_1000"] = L["QubiCML/1000"]["gbps_hard"] / L["MCMit-CNN/1000"]["gbps_hard"]
    v["qubicml_cards_rel_1000"] = L["QubiCML/1000"]["v80_microblossom"] / L["MCMit-CNN/1000"]["v80_microblossom"]
    v["per_lq_d_range_disc4"] = rng([L[f"{m}/{T}"]["d"] for m in DISC4 for T in (400, 600, 800, 1000)
                                     if f"{m}/{T}" in L])

    # ---------------- §7.6 soft readout (sys:145, 151, 157, 159, 161)
    sw = consts("sweep_soft.py", {"F_GRID", "PATTERNS", "QUBITS"})
    v["soft_f_grid_pct"] = [100 * f for f in sw["F_GRID"]]
    v["soft_patterns"] = sw["PATTERNS"]
    import numpy as np
    npz = sorted(glob.glob(os.path.join(data, "soft", "cnn_soft_*ns.npz")))
    v["soft_test_shots"] = sorted({int(np.load(f)["label"].shape[0]) for f in npz})
    soft = {(s["T"], float(s["label"][6:])): s for s in SS}
    sbase = soft[(1000, 0.0)]["spacetime@1e-09"]
    sst = lambda T, f: soft[(T, f)].get("spacetime@1e-09", float("nan")) / sbase  # noqa: E731
    bw = lambda T, f: (soft[(T, f)]["d@1e-09"] ** 2 - 1) / ((T + GATES_NS) * 1e-9) / 1e9 * (1 if f == 0 else 2)  # noqa: E731
    best = {T: min((f for (TT, f) in soft if TT == T and f > 0 and "spacetime@1e-09" in soft[(T, f)]),
                   key=lambda f: (soft[(T, f)]["spacetime@1e-09"], f)) for T in (400, 600, 800, 1000)}
    v["soft_best_flag"] = best
    v["soft_flagged_err_400_f5_pct"] = 100 * P["400"]["0.05"]["e_flagged"]
    v["soft_eu_400_f0_pct"], v["soft_eu_400_f10_pct"] = 100 * P["400"]["0.0"]["e_u"], 100 * P["400"]["0.1"]["e_u"]
    v["soft_eu_600_f0_pct"], v["soft_eu_600_f5_pct"] = 100 * P["600"]["0.0"]["e_u"], 100 * P["600"]["0.05"]["e_u"]
    for T, f in ((400, 0.0), (400, 0.1), (600, 0.0), (600, 0.05), (1000, 0.0), (1000, 0.02)):
        v[f"soft_lambda_{T}_f{f:g}"] = lam(soft[(T, f)])
    v["soft_lambda_f20_600_1000"] = rng([lam(soft[(T, 0.2)]) for T in (600, 800, 1000)])
    v["soft_lambda_f20_400"] = lam(soft[(400, 0.2)])
    v["soft_n_below_threshold_200"] = sum(1 for (T, f), s in soft.items() if T == 200 and lam(s) > 1)
    v["soft_flag_raises_lambda_all_T"] = min(max(lam(soft[(T, f)]) for f in (0.02, 0.05, 0.1)) - lam(soft[(T, 0.0)])
                                            for T in (400, 600, 800, 1000))
    v["soft_d_400"], v["soft_d_400_f10"] = soft[(400, 0.0)]["d@1e-09"], soft[(400, 0.1)]["d@1e-09"]
    v["soft_d_600"], v["soft_d_600_f5"] = soft[(600, 0.0)]["d@1e-09"], soft[(600, 0.05)]["d@1e-09"]
    v["soft_st_400"], v["soft_st_400_f10"] = sst(400, 0.0), sst(400, 0.1)
    v["soft_st_cut_400"] = sst(400, 0.0) / sst(400, 0.1)
    v["soft_st_600_f5"] = sst(600, 0.05)
    v["soft_gbps_600"], v["soft_gbps_600_f5"] = bw(600, 0.0), bw(600, 0.05)
    v["soft_gbps_400"], v["soft_gbps_400_f10"] = bw(400, 0.0), bw(400, 0.1)
    v["soft_gbps_cost_best_600_1000"] = rng([bw(T, best[T]) / bw(T, 0.0) for T in (600, 800, 1000)])
    v["soft_gbps_ratio_best_400"] = bw(400, best[400]) / bw(400, 0.0)
    v["soft_flagged_err_f5_pct_400_1000"] = rng([100 * P[str(T)]["0.05"]["e_flagged"] for T in (400, 600, 800, 1000)])

    # ---------------- §7.7 decoders (sys:171, 173, 180, 186, 188, 190)
    v["cz_error_pct"] = 100 * consts("sweep_ler.py", {"PRESETS"})["PRESETS"]["ibm_boston"]["cz"]
    v["flip_vs_cz_ratio"] = rng([v["cnn_readout_error_pct"][T] / v["cz_error_pct"] for T in (400, 1000)])
    sd = consts("sweep_decoders.py", {"MATCHING", "TARGET_FAILS", "DISTANCES", "SOFT", "PATTERNS"})
    v["dec_n_matching_variants"] = len(sd["MATCHING"])
    v["dec_target_fails"] = sd["TARGET_FAILS"]
    v["dec_distances"] = sd["DISTANCES"]
    v["dec_soft_points"] = [list(x) for x in sd["SOFT"]]
    t = {(r["T"], r["f"], r["decoder"]): r for r in DS}
    hard = [(T, 0.0) for T in (600, 800, 1000)]
    pts = sorted({(r["T"], r["f"]) for r in DS})
    hard_all = [p for p in pts if p[1] == 0]
    v["dec_lambda_w_600_1000"] = rng([lam(t[(T, f, "mwpm_w")]) for T, f in hard])
    v["dec_lambda_u_600_1000"] = rng([lam(t[(T, f, "mwpm_u")]) for T, f in hard])
    v["dec_lambda_u_400"] = lam(t[(400, 0.0, "mwpm_u")])
    v["dec_d_u_600_1000"] = rng([t[(T, f, "mwpm_u")].get("d") for T, f in hard])
    v["dec_d_w_600_1000"] = rng([t[(T, f, "mwpm_w")].get("d") for T, f in hard])
    v["dec_st_u_600_1000"] = rng([t[(T, f, "mwpm_u")]["spacetime_rel"] for T, f in hard])
    v["dec_st_u_600"], v["dec_st_u_1000"] = t[(600, 0.0, "mwpm_u")]["spacetime_rel"], t[(1000, 0.0, "mwpm_u")]["spacetime_rel"]
    v["dec_q16_max_abs_dlambda_hard"] = max(abs(lam(t[(T, f, "mwpm_q16")]) - lam(t[(T, f, "mwpm_w")])) for T, f in hard_all)
    v["dec_q16_max_abs_dlambda_all"] = max(abs(lam(t[(T, f, "mwpm_q16")]) - lam(t[(T, f, "mwpm_w")])) for T, f in pts)
    v["dec_q16_same_d_all"] = int(all(t[(T, f, "mwpm_q16")].get("d") == t[(T, f, "mwpm_w")].get("d") for T, f in pts))
    v["dec_q4_dlambda_all"] = rng([lam(t[(T, f, "mwpm_w")]) - lam(t[(T, f, "mwpm_q4")]) for T, f in pts])
    v["dec_q4_st_ratio_hard"] = rng([t[(T, f, "mwpm_q4")]["spacetime_rel"] / t[(T, f, "mwpm_w")]["spacetime_rel"]
                                     for T, f in hard_all])
    v["dec_q4_st_ratio_all"] = rng([t[(T, f, "mwpm_q4")]["spacetime_rel"] / t[(T, f, "mwpm_w")]["spacetime_rel"]
                                    for T, f in pts])
    v["dec_corr_dlambda_600_1000"] = rng([lam(t[(T, f, "mwpm_c")]) - lam(t[(T, f, "mwpm_w")]) for T, f in hard])
    v["dec_corr_st_600"] = t[(600, 0.0, "mwpm_c")]["spacetime_rel"]
    v["dec_corr_dlambda_400"] = lam(t[(400, 0.0, "mwpm_c")]) - lam(t[(400, 0.0, "mwpm_w")])
    v["dec_flag600_lambda_u"], v["dec_flag600_lambda_w"], v["dec_flag600_lambda_c"] = \
        lam(t[(600, 0.05, "mwpm_u")]), lam(t[(600, 0.05, "mwpm_w")]), lam(t[(600, 0.05, "mwpm_c")])
    v["dec_flag600_st_c"] = t[(600, 0.05, "mwpm_c")]["spacetime_rel"]
    v["dec_lambda_u_600"] = lam(t[(600, 0.0, "mwpm_u")])

    DC = C["decoder_choice"]
    def dc_ratio(prog_filter, key, what="runtime_s"):
        out = []
        for name, rows in DC.items():
            if prog_filter(name) and key in rows:
                out.append(rows[key][what] / rows["1000/0.0/Micro Blossom"][what])
        return out
    allp = lambda n: True  # noqa: E731
    ising = lambda n: n == "Ising-100"  # noqa: E731
    v["whel_vs_mb_same_point_all"] = rng([rows[k]["runtime_s"] / rows[k.replace("Helios (weighted)", "Micro Blossom")]["runtime_s"]
                                          for rows in DC.values() for k in rows if k.endswith("Helios (weighted)")])
    v["whel_vs_mb_same_point_ising100"] = rng([DC["Ising-100"][k]["runtime_s"] /
                                               DC["Ising-100"][k.replace("Helios (weighted)", "Micro Blossom")]["runtime_s"]
                                               for k in DC["Ising-100"] if k.endswith("Helios (weighted)")])
    v["whel_max_wait"] = max(r["w"] for rows in DC.values() for k, r in rows.items() if k.endswith("Helios (weighted)"))
    v["whel_slowdown"] = SC.HELIOS_WEIGHTED
    v["uhel_rt_1000_all"] = rng(dc_ratio(allp, "1000/0.0/Helios"))
    v["uhel_q_1000_all"] = rng(dc_ratio(allp, "1000/0.0/Helios", "qubits"))
    v["uhel_rt_600_all"] = rng(dc_ratio(allp, "600/0.0/Helios"))
    v["uhel_q_600_all"] = rng(dc_ratio(allp, "600/0.0/Helios", "qubits"))
    v["uhel_runs_at_400_hard"] = int(any("400/0.0/Helios" in rows for rows in DC.values()))
    v["uhel_rt_600f5_all"] = rng(dc_ratio(allp, "600/0.05/Helios"))
    v["uhel_q_600f5_all"] = rng(dc_ratio(allp, "600/0.05/Helios", "qubits"))
    v["mb_rt_600f5_all"] = rng(dc_ratio(allp, "600/0.05/Micro Blossom"))
    v["mb_q_600f5_all"] = rng(dc_ratio(allp, "600/0.05/Micro Blossom", "qubits"))
    v["uhel_rt_hard_ising100"] = rng([x for T in (600, 800, 1000) for x in dc_ratio(ising, f"{T}/0.0/Helios")])
    v["uhel_q_hard_ising100"] = rng([x for T in (600, 800, 1000) for x in dc_ratio(ising, f"{T}/0.0/Helios", "qubits")])
    v["uhel_rt_hard_all"] = rng([x for T in (600, 800, 1000) for x in dc_ratio(allp, f"{T}/0.0/Helios")])
    v["uhel_q_hard_all"] = rng([x for T in (600, 800, 1000) for x in dc_ratio(allp, f"{T}/0.0/Helios", "qubits")])
    return v


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def check(values, spec):
    """Print one line per paper value; return the number of mismatches (known issues are listed, not counted)."""
    bad = 0
    for key, s in spec["values"].items():
        meas = values[s["key"]]
        if "index" in s:
            meas = meas[s["index"]] if isinstance(meas, list) else meas[str(s["index"])] if str(s["index"]) in meas \
                else meas[s["index"]]
        exp, tol, kind = s["expected"], s.get("tol", 0), s.get("kind", "abs")
        if kind == "abs":
            ok = _num(meas) and abs(meas - exp) <= tol
        elif kind == "range":
            ok = isinstance(meas, list) and len(meas) == 2 and abs(meas[0] - exp[0]) <= tol and abs(meas[1] - exp[1]) <= tol
        elif kind == "within":
            ok = all(exp[0] - tol <= m <= exp[1] + tol for m in meas)
        elif kind == "min":
            ok = meas >= exp
        elif kind == "max":
            ok = meas <= exp
        elif kind == "lt":
            ok = meas < exp
        else:                                          # exact (numbers, lists, strings)
            ok = meas == exp
        bad += not ok
        shown = [float(f"{m:.5g}") if _num(m) else m for m in meas] if isinstance(meas, list) and len(meas) < 8 else \
            (float(f"{meas:.5g}") if _num(meas) else meas)
        print(f"{'ok ' if ok else 'BAD'} {key:<36} {s['ref']:<30} paper {str(exp):<14.60} measured {str(shown):.90}")
    for key, s in spec.get("known_issues", {}).items():
        meas = values.get(s["key"]) if s.get("key") else None
        shown = [float(f"{m:.5g}") for m in meas] if isinstance(meas, list) and all(_num(m) for m in meas) else \
            (float(f"{meas:.5g}") if _num(meas) else meas)
        print(f"!!  {key:<36} {s['ref']:<30} known content issue (not counted): {s['issue']}"
              + (f" [measured {shown}]" if meas is not None else ""))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(RESULTS / "system"))
    ap.add_argument("--check", nargs="?", const=str(RESULTS / "system" / "paper_values.json"))
    a = ap.parse_args()
    values = compute(a.data)
    if not a.check:
        print(json.dumps(values, indent=1, default=str))
        return
    bad = check(values, json.load(open(a.check)))
    print(f"\n{bad} mismatch(es)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
