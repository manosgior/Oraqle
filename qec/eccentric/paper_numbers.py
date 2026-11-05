"""Every number of §6 (qec_impact.tex) and the checks behind the QEC-appendix prose, from the ECCentric CSVs.

Inputs: results/qec/experiment_results (the CSV subset of TUM-DSE/eccentric @ da7164c, see results/qec/PROVENANCE.md).
Only the standard library and numpy are needed.

Definitions (the ones that reproduce the paper's numbers):
  CNN line        MCMit-CNN LER. d = 5 (and Gross d = 12, Steane d = 9): the heron_cnn rows of
                  noise_models/ler_noise_models.csv (same noise model; run_ler_discriminators.py skips the CNN);
                  d = 12 surface/Bacon/heavy-hex: the MCMit-CNN rows of ler_discriminators_d12.csv (LER_INCLUDE_CNN=1).
                  Gross at d = 12 comes from the d = 5 sweep (gross is always d = 12), as in plot_discriminators.py.
  duration swing  max_T LER / min_T LER - 1 along the CNN line, in %                  (qec_impact:29, 35, 38)
  disc. spread    max_disc LER / min_disc LER - 1 at one (code, T) over the 7 lines, in % (qec_impact:29, 35, 38)
  4 codes         d = 5: surface, Bacon-Shor, color, heavy-hex; d = 12: surface, Bacon-Shor, heavy-hex, Gross
  low coherence   1 - LER(200 ns) / LER(1000 ns), max over codes and distances of the boston_lowt_backlog sweep
                  (fixed readout, so shorter is strictly better)                           (qec_impact:104, 109, 112)

Usage: python paper_numbers.py [--results DIR] [--check results/qec/paper_values.json]
       (with --check: exit 1 if any paper value is outside its tolerance)
"""
import argparse
import ast
import collections
import csv
import json
import os
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from oraqle_env import RESULTS  # noqa: E402

CODES4_D5 = ["surface", "bacon", "color", "hh"]
CODES4_D12 = ["surface", "bacon", "hh", "gross"]
RESPONSIVE3 = ["surface", "color", "hh"]
TS = [200, 400, 600, 800, 1000]
MCMIT = {"MCMit-CNN", "MCMit-T"}


def _ler(v):
    return float(v) if v not in ("", "None") else float("nan")


def _rows(path):
    return list(csv.DictReader(open(path)))


def load_lines(er):
    """{D: {(code, d): {T: {line: LER}}}} for the two discriminator sweeps of Fig. ler_discriminators."""
    cnn = {(r["code"], int(r["distance"]), int(r["readout_ns"])): _ler(r["logical_error_rate"])
           for r in _rows(os.path.join(er, "noise_models", "ler_noise_models.csv")) if r["noise_model"] == "heron_cnn"}
    d5 = _rows(os.path.join(er, "discriminators", "ler_discriminators_d5.csv"))
    out = {}
    for D in (5, 12):
        own = d5 if D == 5 else _rows(os.path.join(er, "discriminators", "ler_discriminators_d12.csv"))
        have = {r["code"] for r in own}
        rows = own + [r for r in d5 if r["code"] in ("gross", "steane") and r["code"] not in have]
        L = collections.defaultdict(lambda: collections.defaultdict(dict))
        for r in rows:
            L[(r["code"], int(r["distance"]))][int(r["readout_ns"])][r["discriminator"]] = _ler(r["logical_error_rate"])
        for (code, d), byT in L.items():
            for T in byT:
                if (code, d, T) in cnn:              # prefer heron_cnn, as plot_discriminators.cnn_series does
                    byT[T]["MCMit-CNN"] = cnn[(code, d, T)]
        out[D] = L
    return out


def swing(byT):
    c = [byT[T]["MCMit-CNN"] for T in TS]
    return 100 * (max(c) / min(c) - 1)


def spread(lines):
    v = list(lines.values())
    return 100 * (max(v) / min(v) - 1)


def compute(results_dir=None):
    er = str(results_dir or (RESULTS / "qec" / "experiment_results"))
    L = load_lines(er)
    v = {}
    by_code5 = {code: L[5][(code, 12 if code == "gross" else 5)] for code in CODES4_D5}
    by_code12 = {code: L[12][(code, 12)] for code in CODES4_D12}

    # §6.1, d = 5 (qec_impact:29, 35, 41)
    for code in CODES4_D5:
        v[f"d5_swing_{code}_pct"] = swing(by_code5[code])
    v["d5_swing_mean3_pct"] = statistics.mean(v[f"d5_swing_{c}_pct"] for c in RESPONSIVE3)
    v["d5_surface_cnn_ler_200"] = by_code5["surface"][200]["MCMit-CNN"]
    v["d5_surface_cnn_ler_1000"] = by_code5["surface"][1000]["MCMit-CNN"]
    v["d5_surface_qubicml_ler_1000"] = by_code5["surface"][1000]["QubiCML"]
    sp5 = {(c, T): spread(by_code5[c][T]) for c in CODES4_D5 for T in TS}
    for T in TS:
        v[f"d5_spread_mean4_{T}_pct"] = statistics.mean(sp5[(c, T)] for c in CODES4_D5)
    v["d5_spread_mean_pct"] = statistics.mean(sp5.values())
    v["d5_spread_max_pct"] = max(sp5.values())
    v["d5_spread_max_is_surface_1000"] = float(max(sp5, key=sp5.get) == ("surface", 1000))
    bacon = [x for T in TS for x in by_code5["bacon"][T].values()]
    v["d5_bacon_ler_min"], v["d5_bacon_ler_max"] = min(bacon), max(bacon)

    # §6.1, d = 12 (qec_impact:29, 38, 41)
    for code in CODES4_D12:
        v[f"d12_swing_{code}_pct"] = swing(by_code12[code])
    v["d12_swing_max_pct"] = max(v[f"d12_swing_{c}_pct"] for c in CODES4_D12)
    sp12 = [spread(by_code12[c][T]) for c in CODES4_D12 for T in TS]
    v["d12_spread_mean_pct"] = statistics.mean(sp12)
    v["d12_spread_max_pct"] = max(sp12)
    sat = [x for c in ("surface", "bacon", "hh") for T in TS for x in by_code12[c][T].values()]
    v["d12_sat_ler_min"], v["d12_sat_ler_max"] = min(sat), max(sat)
    v["d12_gross_ler_min"] = min(x for T in TS for x in by_code12["gross"][T].values())

    # Caveat checks for the qec_impact:29/35 ordering sentence (1 = the claim holds at that T, 0 = it does not)
    # over the three responsive codes (Bacon-Shor is saturated, qec_impact:35) and over all four codes
    for tag, codes in (("resp3", RESPONSIVE3), ("all4", CODES4_D5)):
        for T in TS:
            best = [min(by_code5[c][T], key=by_code5[c][T].get) for c in codes]
            worst = [max(by_code5[c][T], key=by_code5[c][T].get) for c in codes]
            v[f"claim_best_is_mcmit_{tag}_{T}"] = float(all(b in MCMIT for b in best))
            v[f"claim_worst_is_qubicml_or_linear_{tag}_{T}"] = float(all(w in ("QubiCML", "Linear Threshold") for w in worst))
    v["claim_best_is_mcmit_resp3_allT"] = float(all(v[f"claim_best_is_mcmit_resp3_{T}"] for T in TS))
    v["claim_worst_is_qubicml_or_linear_resp3_allT"] = float(all(v[f"claim_worst_is_qubicml_or_linear_resp3_{T}"] for T in TS))

    # §6.2 low-coherence + backlog (qec_impact:104, 109, 112)
    red = {}
    for r in _rows(os.path.join(er, "noise_models", "ler_noise_models_boston_lowt_backlog.csv")):
        red.setdefault((r["code"], int(r["distance"])), {})[int(r["readout_ns"])] = _ler(r["logical_error_rate"])
    cut = {k: 100 * (1 - x[200] / x[1000]) for k, x in red.items() if x[1000] > 0}
    v["lowcoh_max_cut_pct"] = max(cut.values())
    v["lowcoh_max_cut_is_surface5"] = float(max(cut, key=cut.get) == ("surface", 5))

    v["claim_best_is_mcmit_resp3_from600"] = float(all(v[f"claim_best_is_mcmit_resp3_{T}"] for T in TS if T >= 600))
    v["claim_worst_is_qubicml_or_linear_resp3_from600"] = float(
        all(v[f"claim_worst_is_qubicml_or_linear_resp3_{T}"] for T in TS if T >= 600))

    # App. sim-comparison (appendix_herqules:419, 428, 439; qec_impact:21), MCMit-CNN readout as plotted in
    # fig:ler_current / fig:ler_futuristic (plots/readout_ler_cnn_*.pdf): ECCentric half
    # readout_compare/eccentric_readout_ler_cnn.csv, lattice-sim half results/qec/artifact_readout_ler_cnn_complete.csv
    # (49 points of synchronization-artifact/artifact_readout_ler_cnn.csv + the d=13/200 ns/current point)
    rc = {}
    for tool, path in (("eccentric", os.path.join(er, "readout_compare", "eccentric_readout_ler_cnn.csv")),
                       ("artifact", str(RESULTS / "qec" / "artifact_readout_ler_cnn_complete.csv"))):
        for r in _rows(path):
            rc.setdefault((tool, r["noise_model"], int(r["distance"])), {})[int(r["readout_ns"])] = float(r["logical_error_rate"])
    ds = sorted({k[2] for k in rc})
    lat = {d: rc[("artifact", "current", d)] for d in ds}
    ecc = {d: rc[("eccentric", "current", d)] for d in ds}
    v["app_latsim_current_argmin_all_600_800"] = float(all(min(lat[d], key=lat[d].get) in (600, 800) for d in ds))
    v["app_latsim_current_below_threshold_from400"] = float(all(lat[a][T] > lat[b][T] for T in TS if T >= 400
                                                                for a, b in zip(ds, ds[1:])))
    v["app_latsim_current_above_threshold_200"] = float(all(lat[a][200] < lat[b][200] for a, b in zip(ds, ds[1:])))
    v["app_ecc_over_latsim_600_min_ratio"] = min(ecc[d][600] / lat[d][600] for d in ds)
    v["app_ecc_current_ler_min"] = min(x for d in ds for x in ecc[d].values())
    v["app_ecc_current_ler_max"] = max(x for d in ds for x in ecc[d].values())
    ecc7 = [x for d in ds if d >= 7 for x in ecc[d].values()]
    v["app_ecc_current_d7plus_ler_min"], v["app_ecc_current_d7plus_ler_max"] = min(ecc7), max(ecc7)
    v["app_ecc_current_d7plus_spread"] = max(ecc7) - min(ecc7)
    mean_d = {d: statistics.mean(ecc[d].values()) for d in ds}          # 'rising with d to ~0.75': d=5 lowest,
    v["app_ecc_current_d5_lowest"] = float(all(mean_d[5] < mean_d[d] for d in ds if d > 5))   # d>=7 saturated
    # superseded HERQULES-readout version (plots/readout_ler_herqules_*.pdf, in the paper until 2026-11-05)
    rh = {}
    for tool in ("eccentric", "artifact"):
        for r in _rows(os.path.join(er, "readout_compare", f"{tool}_readout_ler_herqules.csv")):
            rh.setdefault((tool, r["noise_model"], int(r["distance"])), {})[int(r["readout_ns"])] = float(r["logical_error_rate"])
    v["herq_app_latsim_current_argmin_all600"] = float(all(min(rh[("artifact", "current", d)], key=rh[("artifact", "current", d)].get) == 600 for d in ds))
    # App. decoherence-only (appendix_herqules:447, 452): Bacon-Shor ~0, Gross fails
    dec = [r for r in _rows(os.path.join(er, "noise_models", "ler_noise_models.csv")) if r["noise_model"] == "decoherence_only"]
    v["app_decoh_bacon_ler_max"] = max(_ler(r["logical_error_rate"]) for r in dec if r["code"] == "bacon")
    v["app_decoh_gross_ler_min"] = min(_ler(r["logical_error_rate"]) for r in dec if r["code"] == "gross")
    return v


def driver_constants():
    """Noise constants of the footnote (qec_impact:21) as written in the ECCentric drivers.

    Reads overlay/run_ler_discriminators.py (vendored here; 'EXACTLY the heron_cnn model') and, if an ECCentric
    checkout is given in $ECCENTRIC_DIR (fetch_eccentric.sh), also run_ler_noise_models.py and
    compare_readout_eccentric.py there."""
    files = [HERE / "overlay" / "run_ler_discriminators.py"]
    if os.environ.get("ECCENTRIC_DIR"):
        files += [pathlib.Path(os.environ["ECCENTRIC_DIR"]) / f for f in ("run_ler_noise_models.py", "compare_readout_eccentric.py")]
    out = {}
    for f in files:
        consts = {}
        for node in ast.parse(f.read_text()).body:
            if isinstance(node, ast.Assign):
                try:
                    val = ast.literal_eval(node.value)
                except Exception:
                    continue
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        consts[t.id] = val
                    elif isinstance(t, ast.Tuple) and isinstance(val, tuple):
                        consts.update({e.id: x for e, x in zip(t.elts, val) if isinstance(e, ast.Name)})
        out[f.name] = dict(cz=consts.get("HERON_TQ", consts.get("CZ_ERROR")), sx=consts.get("HERON_SQ", consts.get("SX_ERROR")),
                           t1_us=(consts.get("T1_S") or 0) * 1e6 or None, t2_us=(consts.get("T2_S") or 0) * 1e6 or None)
    return out


def check(values, spec, consts):
    bad = 0
    for key, s in spec["values"].items():
        if s.get("source") == "driver":
            got = [c.get(s["field"]) for c in consts.values() if c.get(s["field"]) is not None]
            ok = bool(got) and all(abs(g - s["expected"]) <= s["tol"] for g in got)
            meas = got
        else:
            meas = values[s["key"]]
            kind = s.get("kind", "abs")
            if kind == "min":
                ok = meas >= s["expected"]
            elif kind == "max":
                ok = meas <= s["expected"]
            else:
                ok = abs(meas - s["expected"]) <= s["tol"]
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {key:<34} {s['ref']:<26} paper {s['expected']!s:<9} measured {meas}")
    for key, s in spec.get("known_issues", {}).items():
        meas = values.get(s.get("key"), None) if s.get("key") else None
        print(f"!!  {key:<34} {s['ref']:<26} known content issue (not counted): {s['issue']}"
              + (f" [measured {meas}]" if meas is not None else ""))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS / "qec" / "experiment_results"))
    ap.add_argument("--check", nargs="?", const=str(RESULTS / "qec" / "paper_values.json"))
    a = ap.parse_args()
    values = compute(a.results)
    consts = driver_constants()
    if not a.check:
        print(json.dumps(dict(values=values, driver_constants=consts), indent=1))
        return
    bad = check(values, json.load(open(a.check)), consts)
    print(f"\n{bad} mismatch(es)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
