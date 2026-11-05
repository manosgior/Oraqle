#!/usr/bin/env python3
"""Verify every discriminator/readout number of the paper from results/.

Covers abstract..ml_discriminators (except tab:error_modes / fig:iq_taxonomy, which
belong to the leakage part) and the HERQULES appendix tables. Expected values are
in results/discriminators/paper_values.json (made by
discriminators/tables/extract_paper_values.py from the paper source, with
chapter:line references).

For every entry this script
  1. recomputes the metric from the committed result CSVs (Table 2 is first rebuilt
     from the per-model CSVs with discriminators/tables/build_table2.py and must be
     byte-identical to results/discriminators/master_fidelity.csv),
  2. compares it with `expected` (the data value; a mismatch means results/ or the
     computation changed -> FAIL),
  3. re-checks the paper value against the data and compares that with the recorded
     `status` ("match" or "known_discrepancy" with a note). A paper value that no
     longer agrees with the data and is not a recorded known discrepancy -> FAIL.

tab:fpga_resources is recomputed from the Vitis HLS reports in fpga/hls4ml/reports
(fpga/parse_reports.py). No dataset, GPU or network needed. Runs with the plain stack:
    Discriminators/.venv/bin/python verify/discriminators.py [-v] [--paper-root ../oraqle-ics]

--paper-root additionally checks that each printed value still appears on the
referenced line of the paper source. Exit code 0 = all checks pass.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "discriminators"))
sys.path.insert(0, os.path.join(REPO, "discriminators", "tables"))
from disc_paths import RESULTS_DISC  # noqa: E402
import build_table2 as BT  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("fpga_parse_reports", os.path.join(REPO, "fpga", "parse_reports.py"))
FP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FP)

RD = str(RESULTS_DISC)
NEURAL = ["Baseline FNN", "HERQULES", "KLiNQ", "MCMit-T", "MCMit-CNN"]   # QubiCML excluded (integrating model)
ALL6 = BT.TABLE2_MODELS


def gmean(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))


# ======================================================================= data
class Data:
    def __init__(self):
        self.tables = {m: BT.LOADERS[m]() for m in BT.MASTER_ORDER}
        self.master_text = BT.master_csv_text(self.tables)
        self.cnn_dep_text = BT.cnn_deployment_text()
        self.master = {}
        for r in csv.DictReader(self.master_text.splitlines()):
            self.master[(r["model"], int(r["duration_ns"]))] = r
        self.herq = {(r["variant"], int(r["length_ns"])): r
                     for r in csv.DictReader(open(os.path.join(RD, "herqules_length_faithful_vs_demux.csv")))}
        self.freeze = {(r["policy"], int(r["length_ns"])): r
                       for r in csv.DictReader(open(os.path.join(RD, "herqules_truncation_pertrace_deployment.csv")))}
        self.seed = {r["seed"]: r for r in csv.DictReader(open(os.path.join(RD, "herqules_full_dataset.csv")))}
        self.cnn_dep = {int(r["length_ns"]): r for r in csv.DictReader(self.cnn_dep_text.splitlines())}
        self.meta = json.load(open(os.path.join(RD, "dataset_metadata.json")))
        self.fpga_reports, self.fpga_rows, _ = FP.table()      # fpga/hls4ml/reports (Vitis HLS)
        self._readout()

    def _readout(self):
        rows = []
        for f in glob.glob(os.path.join(RD, "Baseline_Threshold_len*_*.csv")):
            m = re.search(r"Baseline_Threshold_len(\d+)_", os.path.basename(f))
            r = next(csv.DictReader(open(f)))
            rows.append([int(m.group(1))] + [float(r[f"qubit_{q}_accuracy"]) / 100 for q in range(5)])
        rows.sort()
        self.trunc_t = [round(r[0] * 2e-3, 3) for r in rows]            # us
        self.trunc_f = [r[1:] for r in rows]
        self.trunc_g = [math.exp(sum(math.log(max(x, 1e-6)) for x in f) / 5) for f in self.trunc_f]
        d = sorted([[int(r["trace_length"])] + [float(r[f"qubit_{q}_dFdT"]) for q in range(5)]
                    for r in csv.DictReader(open(os.path.join(RD, "Baseline_Threshold_dFdT_analysis_20260514_131943.csv")))])
        self.dfdt_t = [round(r[0] * 2e-3, 3) for r in d]
        self.dfdt = [r[1:] for r in d]
        self.win = {}
        for w, fn in [(250, "Baseline_Threshold_sliding_window_250ns_20260514_204419.csv"),
                      (500, "Baseline_Threshold_sliding_window_500ns.csv")]:
            a = sorted([[float(r["window_center_us"])] + [float(r[f"qubit_{q}_fidelity"]) for q in range(5)]
                        for r in csv.DictReader(open(os.path.join(RD, fn)))])
            c = [x[0] for x in a]
            f = [x[1:] for x in a]
            g = [math.exp(sum(math.log(min(max(v, 1e-6), 1.0)) for v in row) / 5) for row in f]
            self.win[w] = (c, f, g)

    # ------------------------------------------------------------------ metric evaluation
    def mval(self, model, ns, col):
        """Table 2 value at full available precision (the loader values behind
        master_fidelity.csv; master itself stores them rounded to 4 decimals)."""
        d = self.tables[model][ns // 2]
        if col == "params":
            return d["params"]
        if col == "F5Q":
            return BT.f5q_of(d)
        if col == "F4Q":
            return gmean([d["q"][i] for i in (0, 2, 3, 4)])
        return d["q"][int(col[1]) - 1]

    def metric(self, spec):
        k, *a = spec.split(":")
        if k == "master":
            return self.mval(a[0], int(a[1]), a[2])
        if k == "cnn_dep":
            return float(self.cnn_dep[int(a[0])][a[1]])
        if k == "herq":
            return float(self.herq[(a[0], int(a[1]))][a[2]])
        if k == "freeze":
            return float(self.freeze[(a[0], int(a[1]))][a[2]])
        if k == "herq_seed":
            return float(self.seed[a[0]][a[1]])
        if k == "herq_errred":                      # (released - pertrace) / (1 - pertrace), F5Q
            ns = int(a[0])
            b, w = float(self.herq[("demux_subsample", ns)]["gmean5"]), float(self.herq[("deployment_pertrace", ns)]["gmean5"])
            return (b - w) / (1 - w)
        if k == "herq_errred_range":
            v = [self.metric(f"herq_errred:{n}") for n in a[0].split(",")]
            return [min(v), max(v)]
        if k == "range":                            # range:<col>:<ns list>:<model set>
            col, nss, ms = a[0], [int(x) for x in a[1].split(",")], {"neural": NEURAL, "all6": ALL6,
                                                                       "noherq": [m for m in ALL6 if m != "HERQULES"]}[a[2]]
            v = [self.mval(m, n, col) for m in ms for n in nss]
            return [min(v), max(v)]
        if k == "argmax_model":                     # argmax_model:<col>:<ns>
            return max(ALL6, key=lambda m: self.mval(m, int(a[1]), a[0]))
        if k == "err_increase":                     # (1-F5Q@600)/(1-F5Q@1000) - 1, per model or mean
            ms = ALL6 if a[0] == "mean_all6" else NEURAL if a[0] == "mean_neural" else [a[0]]
            v = [(1 - self.mval(m, int(a[2]), "F5Q")) / (1 - self.mval(m, int(a[1]), "F5Q")) - 1 for m in ms]
            return sum(v) / len(v)
        if k == "err_increase_range":
            v = [(1 - self.mval(m, int(a[1]), "F5Q")) / (1 - self.mval(m, int(a[0]), "F5Q")) - 1 for m in ALL6]
            return [min(v), max(v)]
        if k == "errred_vs":                        # (F(m1) - F(m2)) / (1 - F(m2)) at ns
            f1, f2 = self.mval(a[0], int(a[2]), "F5Q"), self.mval(a[1], int(a[2]), "F5Q")
            return (f1 - f2) / (1 - f2)
        if k == "orders_of_magnitude":
            ps = [self.mval(m, n, "params") for m in BT.MASTER_ORDER for n in (200, 400, 600, 800, 1000)]
            return math.log10(max(ps) / min(ps))
        if k == "param_ratio":                      # param_ratio:<m1>:<m2>:<ns>
            return self.mval(a[0], int(a[2]), "params") / self.mval(a[1], int(a[2]), "params")
        if k == "params_div":                       # params_div:<model>:<ns>:<k> (per-qubit count)
            return self.mval(a[0], int(a[1]), "params") / int(a[2])
        if k == "param_ratios_giants":              # Baseline FNN and MCMit-T over MCMit-CNN at 1000 ns
            return [self.metric("param_ratio:Baseline FNN:MCMit-CNN:1000"),
                    self.metric("param_ratio:MCMit-T:MCMit-CNN:1000")]
        if k == "params_range":
            v = [self.mval(a[0], n, "params") for n in (200, 400, 600, 800, 1000)]
            return [min(v), max(v)]
        if k == "params_argmin_ns":
            return min((200, 400, 600, 800, 1000), key=lambda n: self.mval(a[0], n, "params"))
        # ---- section 4 (readout baseline)
        if k == "trunc_g":
            return self.trunc_g[self.trunc_t.index(float(a[0]))]
        if k == "trunc_q":
            return self.trunc_f[self.trunc_t.index(float(a[1]))][int(a[0]) - 1]
        if k == "trunc_peak":                       # trunc_peak:<q>:val|t
            q = int(a[0]) - 1
            i = max(range(len(self.trunc_t)), key=lambda j: self.trunc_f[j][q])
            return self.trunc_f[i][q] if a[1] == "val" else self.trunc_t[i]
        if k == "trunc_last_t":
            return self.trunc_t[-1]
        if k == "trunc_plateau":
            return max(self.trunc_g)
        if k == "trunc_reldrop_max":                # max over qubits of (peak - last)/(1 - last)
            out = []
            for q in [int(x) - 1 for x in a[0].split(",")]:
                col = [f[q] for f in self.trunc_f]
                out.append((max(col) - col[-1]) / (1 - col[-1]))
            return max(out)
        if k == "dfdt_peak_t_all":                  # common peak position if all qubits agree, else list
            ts = sorted({self.dfdt_t[max(range(len(self.dfdt_t)), key=lambda j: self.dfdt[j][q])] for q in range(5)})
            return ts[0] if len(ts) == 1 else ts
        if k == "dfdt_zero_range":                  # first negative dF/dT per qubit -> [min, max]
            z = []
            for q in range(5):
                z.append(next(self.dfdt_t[j] for j in range(len(self.dfdt_t)) if self.dfdt[j][q] < 0))
            return [min(z), max(z)]
        if k == "win_peak":                         # win_peak:<w>:val|center
            c, f, g = self.win[int(a[0])]
            i = max(range(len(c)), key=lambda j: g[j])
            return g[i] if a[1] == "val" else c[i]
        if k == "win_at":
            c, f, g = self.win[int(a[0])]
            i = min(range(len(c)), key=lambda j: abs(c[j] - float(a[1])))
            return g[i]
        if k == "win_reldrop":                      # (peak - value at center) / (1 - value at center)
            p, v = self.metric(f"win_peak:{a[0]}:val"), self.metric(f"win_at:{a[0]}:{a[1]}")
            return (p - v) / (1 - v)
        if k == "win_gap_vs_plateau":               # (trunc plateau - window peak) / (1 - window peak)
            p, w = max(self.trunc_g), self.metric(f"win_peak:{a[0]}:val")
            return (p - w) / (1 - w)
        if k == "win_below_plateau":
            return max(self.win[int(a[0])][2]) <= max(self.trunc_g)
        if k == "win_q2_weakest_all":
            c, f, g = self.win[int(a[0])]
            return all(min(range(5), key=lambda q: row[q]) == 1 for row in f)
        if k == "win_q1q5_top2_frac":
            c, f, g = self.win[int(a[0])]
            return sum(set(sorted(range(5), key=lambda q: -row[q])[:2]) == {0, 4} for row in f) / len(f)
        # ---- tab:fpga_resources (Vitis HLS reports in fpga/hls4ml/reports)
        if k == "fpga":                             # fpga:<row>:LUT|FF|DSP|latency_ns
            r = self.fpga_rows[a[0]]
            return r["latency_ns"] if a[1] == "latency_ns" else r["pct"][a[1]]
        if k == "fpga_max_latency":
            return max(r["latency_ns"] for r in self.fpga_rows.values())
        if k == "fpga_pct_pair":                    # fpga_pct_pair:<row>:<col1>,<col2>
            return [self.fpga_rows[a[0]]["pct"][c] for c in a[1].split(",")]
        # ---- dataset layout (overview Stage 1)
        if k == "dataset":
            fs = self.meta["files"]
            tr, te = fs["DRaw_C_Tr_v0-001"], fs["DRaw_C_Te_v0-002"]
            if a[0] == "shots_total":
                return tr["datasets"]["X_train"]["shape"][0] + te["datasets"]["X_test"]["shape"][0]
            if a[0] == "shots_per_state":
                return tr["shots_per_state"] + te["shots_per_state"]
            if a[0] == "n_states":
                return tr["labels"][1] - tr["labels"][0] + 1
            if a[0] == "window_us":
                return tr["datasets"]["X_train"]["shape"][1] * self.meta["sample_period_ns"] / 1000
            if a[0] == "rate_MSps":
                return self.meta["sampling_rate_MSps"]
        raise KeyError(f"unknown metric {spec}")


# ======================================================================= comparison
def _num_eq(x, y, tol):
    if isinstance(x, list):
        return isinstance(y, list) and len(x) == len(y) and all(_num_eq(a, b, tol) for a, b in zip(x, y))
    if isinstance(x, (bool, str)) or isinstance(y, (bool, str)):
        return x == y
    return abs(float(x) - float(y)) <= tol


def _fmt(x, d):
    return [_fmt(v, d) for v in x] if isinstance(x, list) else f"{x:.{d}f}"


def paper_consistent(entry, value):
    c = entry["compare"]
    pv = entry["paper_value"]
    if c["mode"] == "fmt":                      # printed with c["decimals"] decimals
        # half a printed unit, plus half a unit of the 4-decimal precision some sources
        # (HERQULES/CNN/KLiNQ CSVs) are stored with, so 0.7745 may print as 0.775
        return _num_eq(value, pv, 0.5 * 10 ** -c["decimals"] + 0.5e-4 + 1e-12)
    if c["mode"] == "tol":
        return _num_eq(value, pv, c["tol"])
    if c["mode"] == "eq":
        return value == pv
    if c["mode"] == "within":                   # value (or each element) inside the printed [lo, hi]
        vals = value if isinstance(value, list) else [value]
        return all(pv[0] - 1e-9 <= v <= pv[1] + 1e-9 for v in vals)
    if c["mode"] == "orders":                   # same order of magnitude (rounded log10), elementwise
        vals = value if isinstance(value, list) else [value]
        pvs = pv if isinstance(pv, list) else [pv]
        if len(pvs) == 2 and len(vals) == 2:
            return all(round(math.log10(v)) == round(math.log10(p)) for v, p in zip(vals, pvs))
        return all(round(math.log10(v)) == round(math.log10(pvs[0])) for v in vals)
    raise ValueError(c)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--paper-root", help="oraqle-ics checkout; re-check the printed values on the referenced lines")
    ap.add_argument("--values", default=os.path.join(RD, "paper_values.json"))
    a = ap.parse_args()

    entries = json.load(open(a.values))["entries"]
    D = Data()
    fails, known, resolved = [], [], []

    pub = open(os.path.join(RD, "master_fidelity.csv"), newline="").read()
    if pub != D.master_text:
        fails.append("master_fidelity.csv rebuilt from the per-model CSVs differs from the committed file")
    if open(os.path.join(RD, "cnn_length_deployment.csv"), newline="").read() != D.cnn_dep_text:
        fails.append("cnn_length_deployment.csv rebuilt from cnn_length_*ns.csv differs from the committed file")

    unverifiable = [e for e in entries if e["status"] == "unverifiable"]
    for e in entries:
        if e["status"] == "unverifiable":
            continue
        try:
            val = D.metric(e["metric"])
        except Exception as ex:  # noqa: BLE001
            fails.append(f"{e['id']} ({e['ref']}): cannot compute {e['metric']}: {ex!r}")
            continue
        tol = e.get("expected_tol", 1e-6)
        if not _num_eq(val, e["expected"], tol):
            fails.append(f"{e['id']} ({e['ref']}): data {val!r} != expected {e['expected']!r}")
            continue
        cons = paper_consistent(e, val)
        if e["status"] == "match" and not cons:
            fails.append(f"{e['id']} ({e['ref']}): paper {e['paper_text']!r} no longer agrees with data {val!r}")
        elif e["status"] == "known_discrepancy":
            (resolved if cons else known).append(e)
        if a.verbose:
            print(f"  {'ok ' if cons else 'KD '} {e['id']:<55} paper={e['paper_text']!r:<14} data={val!r}")

    if a.paper_root:
        for e in entries:
            path, line = e["ref"].rsplit(":", 1)
            lines = open(os.path.join(a.paper_root, path)).read().splitlines()
            if e["paper_text"] not in lines[int(line) - 1]:
                fails.append(f"{e['id']}: {e['paper_text']!r} not found on {e['ref']} (paper edited?)")

    n = len(entries) - len(unverifiable)
    print(f"verify/discriminators.py: {n} paper values checked against results/discriminators/")
    print("  master_fidelity.csv and cnn_length_deployment.csv rebuilt from the per-model CSVs: "
          f"{'byte-identical' if not any('rebuilt' in f for f in fails) else 'DIFFERENT'}")
    print(f"  {n - len(known) - len(resolved)} agree with the paper, {len(known)} known discrepancies (paper text vs data):")
    for e in known:
        print(f"    - {e['ref']}  {e['id']}: paper {e['paper_text']!r}, data {e['expected']!r}. {e['note']}")
    print(f"  {len(unverifiable)} not verifiable (no artifact): " + "; ".join(f"{e['ref']} {e['paper_text']}" for e in unverifiable))
    for e in resolved:
        print(f"  NOTE: known discrepancy now agrees with the paper (update paper_values.json): {e['id']}")
    if fails:
        print(f"\nFAIL ({len(fails)}):")
        for f in fails:
            print("   ", f)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
