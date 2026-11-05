#!/usr/bin/env python3
"""Rebuild the paper's master_fidelity.csv and Table 2 (tab:fidelity) from the
per-model result CSVs.  [unify, new file]

Why this exists
---------------
The published results/discriminators/master_fidelity.csv (2026-06-22) is the
source of tab:fidelity, tab:fidelity_linear, fig:pareto and the ECCentric
discriminator runs. It was first written by build_master_and_table.py
(2026-06-01), but its HERQULES and MCMit-CNN rows were later replaced by hand:

  * HERQULES  <- herqules_length_faithful_vs_demux.csv, variant=deployment_pertrace
                 (producer: runners/_herqules_length_both.py)
  * MCMit-CNN <- cnn_length_deployment.csv, i.e. the per-length files
                 cnn_length_{200,400,600,800,1000}ns.csv
                 (producer: runners/_cnn_length_sweep.py, one length per process)

For those two rows the F5Q column was copied from the source CSV (gmean5 / F5Q,
computed from unrounded per-qubit fidelities) instead of being recomputed from
the rounded Q1..Q5; for MCMit-CNN at 600 and 1000 ns the two differ by 1e-4.
All other rows use build_master_and_table.py's loaders unchanged (gmean of the
loaded per-qubit values). This script encodes exactly that merge, so the
published file is reproduced byte for byte (CRLF line ends, as csv.writer wrote
them).

It also rebuilds cnn_length_deployment.csv (no aggregator script survived; the
file is cnn_length_*ns.csv plus gmean4_no_q2 = geometric mean of the rounded
Q1, Q3, Q4, Q5), emits tab:fidelity and tab:fidelity_linear as LaTeX, and, with
--paper-tex / --paper-appendix, diffs every cell (value, bold, shading) against the paper
source; with --check, any value or highlighting difference is an error (the highlighting
rule is documented in flags()).

Usage
-----
    python discriminators/tables/build_table2.py                        # writes $ORAQLE_OUT/discriminators/tables/
    python discriminators/tables/build_table2.py --check                # exit 1 unless byte-identical to results/
    python discriminators/tables/build_table2.py --paper-tex ../oraqle-ics/chapters/ml_discriminators.tex \
        --paper-appendix ../oraqle-ics/chapters/appendix_herqules.tex

No dataset needed. Inputs: results/discriminators/ (override: ORAQLE_REPORTS_IN).
"""
import argparse
import csv
import glob
import io
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from disc_paths import OUT, RESULTS_DISC, ensure_dir  # noqa: E402

IN_DIR = os.environ.get("ORAQLE_REPORTS_IN", str(RESULTS_DISC))
SAMPLE_LENS = [100, 200, 300, 400, 500]          # trace samples (2 ns each)
TABLE_NS = [1000, 800, 600, 400, 200]            # display order in the paper
TABLE2_MODELS = ["Baseline FNN", "HERQULES", "QubiCML", "KLiNQ", "MCMit-T", "MCMit-CNN"]
MASTER_ORDER = ["Linear Threshold"] + TABLE2_MODELS


def ns(s):
    return 2 * s


def gmean(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def first_row(path):
    with open(path) as f:
        return next(csv.DictReader(f))


def lenfile(pattern, s):
    hits = sorted(f for f in glob.glob(os.path.join(IN_DIR, pattern)) if re.search(rf"_len0*{s}(?:_|\.)", f))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one file for {pattern} at len {s}, found {hits}")
    return hits[0]


# ---- loaders: dict s -> {'params': int, 'q': [5 floats], 'f5q': float|None, 'src': [files]}
# (verbatim logic of build_master_and_table.py unless noted)
def load_pct(pattern, params_fixed=None):
    out = {}
    for s in SAMPLE_LENS:
        f = lenfile(pattern, s)
        r = first_row(f)
        q = [float(r[f"qubit_{i}_accuracy"]) / 100 for i in range(5)]
        p = params_fixed if params_fixed is not None else int(r["total_parameters"])
        out[s] = {"params": p, "q": q, "f5q": None, "src": [os.path.basename(f)]}
    return out


def load_arxiv():
    out = {}
    for s in SAMPLE_LENS:
        q, src = [], []
        for i in range(5):
            f = lenfile(f"Arxiv240618807FNN_q{i}_len*.csv", s)
            q.append(float(first_row(f)[f"qubit_{i}_accuracy"]) / 100)
            src.append(os.path.basename(f))
        out[s] = {"params": 65 * 5, "q": q, "f5q": None, "src": src}
    return out


def load_klinq():
    files = sorted(glob.glob(os.path.join(IN_DIR, "KLiNQ_paper_repro_F5Q_*.csv")))
    if len(files) != 1:
        raise SystemExit(f"expected one KLiNQ_paper_repro_F5Q_*.csv, found {files}")
    out = {}
    for r in csv.DictReader(open(files[0])):
        s = int(r["trace_samples"])
        out[s] = {"params": int(r["total_student_params"]), "q": [float(r[f"Q{i + 1}_fidelity"]) for i in range(5)],
                  "f5q": None, "src": [os.path.basename(files[0])]}
    return out


def load_herqules_pertrace():
    """NEW: the hand-patched source of the HERQULES row."""
    f = os.path.join(IN_DIR, "herqules_length_faithful_vs_demux.csv")
    out = {}
    for r in csv.DictReader(open(f)):
        if r["variant"] != "deployment_pertrace":
            continue
        s = int(r["trace_samples"])
        out[s] = {"params": 1002, "q": [float(r[f"Q{i}"]) for i in range(1, 6)], "f5q": float(r["gmean5"]),
                  "src": [os.path.basename(f) + " (variant=deployment_pertrace)"]}
    return out


def load_cnn_lengths(lens=SAMPLE_LENS):
    """NEW: the hand-patched source of the MCMit-CNN row (per-length _cnn_length_sweep.py outputs)."""
    out = {}
    for s in lens:
        f = os.path.join(IN_DIR, f"cnn_length_{ns(s)}ns.csv")
        r = first_row(f)
        assert int(r["trace_samples"]) == s
        out[s] = {"params": 3061, "q": [float(r[f"Q{i}"]) for i in range(1, 6)], "f5q": float(r["F5Q"]),
                  "src": [os.path.basename(f)], "raw": r}
    return out


LOADERS = {
    "Linear Threshold": lambda: load_pct("Baseline_Threshold_len*.csv", params_fixed=5),
    "Baseline FNN": lambda: load_pct("FNN_len*.csv"),
    "HERQULES": load_herqules_pertrace,
    "QubiCML": load_arxiv,
    "KLiNQ": load_klinq,
    "MCMit-T": lambda: load_pct("Transformer_len*.csv"),
    "MCMit-CNN": load_cnn_lengths,
}


def f5q_of(d):
    return d["f5q"] if d["f5q"] is not None else gmean(d["q"])


def master_csv_text(data):
    buf = io.StringIO()
    w = csv.writer(buf)                      # default lineterminator \r\n, as in the original
    w.writerow(["model", "duration_ns", "trace_samples", "parameters",
                "Q1_fidelity", "Q2_fidelity", "Q3_fidelity", "Q4_fidelity", "Q5_fidelity", "F5Q_gmean"])
    for name in MASTER_ORDER:
        for s in SAMPLE_LENS:
            d = data[name][s]
            w.writerow([name, ns(s), s, d["params"], *[f"{v:.4f}" for v in d["q"]], f"{f5q_of(d):.4f}"])
    return buf.getvalue()


def cnn_deployment_text():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["variant", "length_ns", "trace_samples", "Q1", "Q2", "Q3", "Q4", "Q5", "gmean5", "gmean4_no_q2"])
    lens = sorted(int(m.group(1)) // 2 for f in glob.glob(os.path.join(IN_DIR, "cnn_length_*ns.csv"))
                  if (m := re.search(r"cnn_length_(\d+)ns\.csv$", f)))
    for s, d in sorted(load_cnn_lengths(lens).items()):
        q = d["q"]
        w.writerow(["MCMit-CNN", ns(s), s, *[f"{v:.4f}" for v in q], f"{d['f5q']:.4f}",
                    f"{gmean([q[0], q[2], q[3], q[4]]):.4f}"])
    return buf.getvalue()


# ---------------------------------------------------------------- LaTeX
COLS = ["Q1", "Q2", "Q3", "Q4", "Q5", "F5Q"]


def cell_values(data, models, with_f4q=False):
    rows = {}
    for m in models:
        for t in TABLE_NS:
            d = data[m][t // 2]
            vals = {f"Q{i + 1}": d["q"][i] for i in range(5)}
            vals["F5Q"] = f5q_of(d)
            if with_f4q:
                vals["F4Q"] = gmean([d["q"][0], d["q"][2], d["q"][3], d["q"][4]])
            rows[(m, t)] = vals
    return rows


def flags(rows, models, cols, eps=1e-9, rule="paper", tie_model="MCMit-CNN"):
    """Bold = best duration within a model, shading = best over all rows.

    rule="exact": compare full-precision values (ties at full precision all marked).
    rule="paper": the convention the paper's highlighting follows (tab:fidelity and
    tab:fidelity_linear, oraqle-ics 2026-11-05):
      bold    compare the printed 3-decimal values; among durations tied at that
              precision, bold only the shortest (e.g. tab:fidelity_linear F4Q: 800 ns,
              0.941, over 600 ns, 0.938);
      shading the highest printed 3-decimal value of the column over all rows; a tie at
              that precision goes to `tie_model` (MCMit-CNN), on its bold cell (Table 2
              Q5: MCMit-CNN 600 ns 0.9700, tied at 0.970 with MCMit-T 800 ns 0.9704 and
              MCMit-CNN 1000 ns); without `tie_model` among the tied cells, the
              full-precision maximum is shaded.
    """
    out = {}
    for m in models:
        for c in cols:
            if rule == "exact":
                best = max(rows[(m, t)][c] for t in TABLE_NS)
                bold_t = {t for t in TABLE_NS if abs(rows[(m, t)][c] - best) < eps}
            else:
                best3 = max(round(rows[(m, t)][c], 3) for t in TABLE_NS)
                bold_t = {min(t for t in TABLE_NS if abs(round(rows[(m, t)][c], 3) - best3) < eps)}
            for t in TABLE_NS:
                out[(m, t, c)] = [t in bold_t, False]
    for c in cols:
        if rule == "exact":
            best = max(v[c] for v in rows.values())
            shade = {k for k, v in rows.items() if abs(v[c] - best) < eps}
        else:
            best3 = max(round(v[c], 3) for v in rows.values())
            tied = [k for k, v in rows.items() if abs(round(v[c], 3) - best3) < eps]
            pref = [k for k in tied if k[0] == tie_model and out[(k[0], k[1], c)][0]]
            if pref:
                shade = set(pref)
            else:
                best = max(rows[k][c] for k in tied)
                shade = {k for k in tied if abs(rows[k][c] - best) < eps}
        for k in shade:
            out[(k[0], k[1], c)][1] = True
    return {k: tuple(v) for k, v in out.items()}


def fmt_cell(v, bold, shade):
    s = f"{v:.3f}"
    if bold:
        s = r"\textbf{" + s + "}"
    if shade:
        s = r"\cellcolor{blue!12}" + s
    return s


def table2_tex(data):
    rows = cell_values(data, TABLE2_MODELS)
    fl = flags(rows, TABLE2_MODELS, COLS)
    L = [r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}ll rrrrr r@{}}", r"\toprule",
         r"Model & $T$ (ns) & Q1 & Q2 & Q3 & Q4 & Q5 & $F_{5Q}$ \\", r"\midrule"]
    for k, m in enumerate(TABLE2_MODELS):
        for j, t in enumerate(TABLE_NS):
            label = m + (r"\footnotemark" if m == "HERQULES" else "")
            head = r"\multirow{5}{*}{%s}" % label if j == 0 else ""
            cells = " & ".join(fmt_cell(rows[(m, t)][c], *fl[(m, t, c)]) for c in COLS)
            L.append(f"{head} & {t} & {cells} \\\\")
        L.append(r"\midrule" if k < len(TABLE2_MODELS) - 1 else r"\bottomrule")
    L.append(r"\end{tabular*}")
    return "\n".join(L) + "\n", rows, fl


def linear_tex(data):
    cols = COLS + ["F4Q"]
    rows = cell_values(data, ["Linear Threshold"], with_f4q=True)
    fl = flags(rows, ["Linear Threshold"], cols)
    L = [r"\begin{tabular}{l rrrrr rr}", r"\toprule",
         r"Duration & Qubit 1 & Qubit 2 & Qubit 3 & Qubit 4 & Qubit 5 & $F_{5Q}$ & $F_{4Q}$ \\", r"\midrule"]
    for t in TABLE_NS:
        cells = " & ".join(fmt_cell(rows[("Linear Threshold", t)][c], fl[("Linear Threshold", t, c)][0], False)
                           for c in cols)
        L.append(f"{t}\\,ns & {cells} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n", rows, fl


# ---------------------------------------------------------------- paper diff
CELL_RE = re.compile(r"^(\\cellcolor\{blue!12\})?(\\textbf\{)?([0-9.]+)\}?$")


def parse_cell(c):
    m = CELL_RE.match(c.strip())
    if not m:
        raise ValueError(f"cannot parse cell {c!r}")
    return float(m.group(3)), bool(m.group(2)), bool(m.group(1)), m.group(3)


def strip_comments(lines):
    return [(i, l) for i, l in lines if not l.lstrip().startswith("%")]


def parse_paper_table(path, label, models_in_order, cols, row_re):
    lines = list(enumerate(open(path).read().splitlines(), start=1))
    start = next(i for i, l in lines if rf"\label{{{label}}}" in l and not l.lstrip().startswith("%"))
    body = []
    for i, l in strip_comments(lines[start:]):
        if r"\end{tabular" in l:
            break
        body.append((i, l))
    cells, cur = {}, None
    for i, l in body:
        m = re.match(row_re, l.strip())
        if not m:
            continue
        model, t, rest = m.group("model"), int(m.group("t")), m.group("rest")
        if model:
            cur = model
        parts = [p.strip() for p in rest.rstrip("\\").rstrip().rstrip("\\").split("&")]
        parts = [p for p in parts if p != ""]
        if len(parts) != len(cols):
            raise SystemExit(f"{path}:{i}: expected {len(cols)} cells, got {parts}")
        for c, p in zip(cols, parts):
            cells[(cur, t, c)] = parse_cell(p) + (i,)
    return cells


def diff_against_paper(rows, fl, paper, cols, chapter, models):
    num, fmt = [], []
    for m in models:
        for t in TABLE_NS:
            for c in cols:
                key = (m, t, c)
                if key not in paper:
                    num.append(f"{m} {t} {c}: missing in paper")
                    continue
                pv, pb, ps, ptxt, line = paper[key]
                ours = f"{rows[(m, t)][c]:.3f}"
                if ours != ptxt:
                    num.append(f"{chapter}:{line} {m} {t}ns {c}: paper {ptxt} vs data {ours} "
                               f"(full {rows[(m, t)][c]:.4f})")
                b, s = fl[(m, t, c)]
                if c in ("Q1", "Q2", "Q3", "Q4", "Q5", "F5Q", "F4Q") and (b != pb or (s != ps)):
                    fmt.append(f"{chapter}:{line} {m} {t}ns {c}: paper bold={pb} shade={ps} vs rule bold={b} shade={s}"
                               f" (value {rows[(m, t)][c]:.4f})")
    return num, fmt


def paper_diffs(data, ml_tex=None, app_tex=None):
    """{table label: (value mismatches, bold/shading differences, cells parsed)} of the paper's
    tab:fidelity (ml_tex) and tab:fidelity_linear (app_tex) against the data and the rule of flags()."""
    out = {}
    if ml_tex:
        _, rows, fl = table2_tex(data)
        row_re = r"^(?:\\multirow\{5\}\{\*\}\{(?P<model>[^}]*?)(?:\\footnotemark)?\})?\s*&\s*(?P<t>\d+)\s*&(?P<rest>.*)$"
        paper = parse_paper_table(ml_tex, "tab:fidelity", TABLE2_MODELS, COLS, row_re)
        out["tab:fidelity"] = diff_against_paper(rows, fl, paper, COLS, os.path.basename(ml_tex), TABLE2_MODELS) + (len(paper),)
    if app_tex:
        _, lrows, lfl = linear_tex(data)
        row_re = r"^(?P<model>)(?P<t>\d+)\\,ns\s*&(?P<rest>.*)$"
        paper = parse_paper_table(app_tex, "tab:fidelity_linear", ["Linear Threshold"], COLS + ["F4Q"], row_re)
        paper = {("Linear Threshold", t, c): v for (_, t, c), v in paper.items()}
        lfl2 = {k: (b, False) for k, (b, s) in lfl.items()}          # the appendix table has no shading
        out["tab:fidelity_linear"] = diff_against_paper(lrows, lfl2, paper, COLS + ["F4Q"], os.path.basename(app_tex),
                                                        ["Linear Threshold"]) + (len(paper),)
    return out


def main():
    ap = argparse.ArgumentParser(description="Rebuild master_fidelity.csv and tab:fidelity")
    ap.add_argument("--out", default=str(OUT / "discriminators" / "tables"))
    ap.add_argument("--check", action="store_true", help="exit 1 unless outputs match results/ byte for byte")
    ap.add_argument("--paper-tex", help="path to oraqle-ics/chapters/ml_discriminators.tex")
    ap.add_argument("--paper-appendix", help="path to oraqle-ics/chapters/appendix_herqules.tex")
    a = ap.parse_args()

    data = {m: LOADERS[m]() for m in MASTER_ORDER}
    out = ensure_dir(a.out)
    master = master_csv_text(data)
    dep = cnn_deployment_text()
    t2, rows, fl = table2_tex(data)
    tl, lrows, lfl = linear_tex(data)
    open(os.path.join(out, "master_fidelity.csv"), "w", newline="").write(master)
    open(os.path.join(out, "cnn_length_deployment.csv"), "w", newline="").write(dep)
    open(os.path.join(out, "tab_fidelity.tex"), "w").write(t2)
    open(os.path.join(out, "tab_fidelity_linear.tex"), "w").write(tl)
    with open(os.path.join(out, "table2_sources.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model", "duration_ns", "source_files"])
        for m in MASTER_ORDER:
            for s in SAMPLE_LENS:
                w.writerow([m, ns(s), "; ".join(data[m][s]["src"])])
    print(f"wrote {out}/master_fidelity.csv, cnn_length_deployment.csv, tab_fidelity.tex, "
          f"tab_fidelity_linear.tex, table2_sources.csv")

    ok = True
    for name, text in [("master_fidelity.csv", master), ("cnn_length_deployment.csv", dep)]:
        pub = open(os.path.join(IN_DIR, name), newline="").read()
        same = pub == text
        ok &= same
        n_cells = sum(len(l.split(",")) for l in text.strip().splitlines()[1:])
        print(f"[{'OK' if same else 'DIFF'}] {name}: {'byte-identical to' if same else 'differs from'} "
              f"{IN_DIR}/{name} ({n_cells} cells)")
        if not same:
            for i, (x, y) in enumerate(zip(pub.splitlines(), text.splitlines())):
                if x != y:
                    print(f"    line {i + 1}: published {x!r}\n             rebuilt   {y!r}")

    if a.paper_tex:
        num, fmt, n = paper_diffs(data, ml_tex=a.paper_tex)["tab:fidelity"]
        print(f"\ntab:fidelity vs {a.paper_tex}: {n} cells parsed, {len(num)} value mismatches, "
              f"{len(fmt)} bold/shading differences from the stated rule")
        for x in num + fmt:
            print("   ", x)
        ok &= not num and not fmt
    if a.paper_appendix:
        num, fmt, n = paper_diffs(data, app_tex=a.paper_appendix)["tab:fidelity_linear"]
        print(f"\ntab:fidelity_linear vs {a.paper_appendix}: {n} cells parsed, {len(num)} value mismatches, "
              f"{len(fmt)} bold differences")
        for x in num + fmt:
            print("   ", x)
        ok &= not num and not fmt
    if a.check and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
