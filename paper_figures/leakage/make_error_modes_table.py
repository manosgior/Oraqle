"""Regenerate Table tab:error_modes (chapters/ml_discriminators.tex) from results/leakage.

Source: results/leakage/em_e8fix_final_numbers.json, key table_error_modes_1us (the final, reviewed numbers;
its entries point back to em_e4recon_final.json). The error-rate column is recomputed from the error and shot
counts (n_err / n_shots of em_e4recon_final.json) instead of re-rounding the stored 3-decimal value: Q4 is
5.31455 %, stored as 5.315, which would round a second time to 5.32 (the paper prints the correct 5.31).
Output: $ORAQLE_OUT/leakage/tab_error_modes.tex, the full table environment as it appears in the paper
(caption and footnotes copied verbatim from the paper). Verified identical to ml_discriminators.tex l.392-412.

--exact recomputes every share from the counts of em_e4recon_final.json (no re-rounding of stored values).
It differs from the paper in one cell: Q5 overlap = 2901.5 / 33162 = 8.7495 % -> 8.7 (the paper prints 8.8,
a double rounding of the stored 8.75). See docs/reproduce/leakage.md (known discrepancies).

No dataset needed. Usage:
    python paper_figures/leakage/make_error_modes_table.py [--exact] [--out FILE] [--rows-only]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "leakage"))
import lkpaths as LP  # noqa: E402

CAPTION = (r"MCMit-CNN error modes at $1\,\mu$s: error rate (\% of shots), then \% of each qubit's \emph{errors}. "
           r"Leak.: off-axis (mirror-corrected) population, not necessarily $|2\rangle$. Last col.: \% of shots "
           r"leaked (\% read correctly). n/m: not measurable.")
HEADER = r"Qubit & Err.\ rate & Leak. & State ch.\ (${\geq}$) & Overlap (${\leq}$) & Leaked shots \\"
FOOT = r"\multicolumn{6}{l}{\textsuperscript{\dag}States $0.5\sigma$ apart in the unused tail. \textsuperscript{\ddag}Multimodal $|1\rangle$; split indicative.} \\"
LABELS = {"Q1": "Q1", "Q2": r"Q2 (hardware)\textsuperscript{\dag}", "Q3": "Q3", "Q4": r"Q4\textsuperscript{\ddag}",
          "Q5": "Q5", "Q1Q3Q5": "Q1, Q3, Q5"}


def _v(entry):
    return entry["value"] if isinstance(entry, dict) else entry


def _cell(x, d, bold=False):
    if x == "n/m":
        return "n/m"
    s = f"{x:.{d}f}"
    return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"


def err_rate_pct(e4, q):
    t = e4["table_1us_cnn_canonical"]
    r = t["pooled"][q] if q == "Q1Q3Q5" else t["per_qubit"][q]
    return 100.0 * r["n_err"] / r["n_shots"]


def exact_shares(e4, q, stored):
    """Shares recomputed from counts; 'n/m' cells are kept as decided in em_e8fix_final_numbers.json."""
    t = e4["table_1us_cnn_canonical"]
    r = t["pooled"][q] if q == "Q1Q3Q5" else t["per_qubit"][q]
    out = {}
    for col, key in (("leakage", "leakage"), ("state_change", "state_change_total_T_plus_S"),
                     ("overlap", "overlap_near_boundary_upper")):
        out[col] = "n/m" if _v(stored[col]) == "n/m" else 100.0 * r[key]["count"] / r["n_err"]
    if q != "Q1Q3Q5" and _v(stored["leaked_shots_pct_of_shots"]) != "n/m":
        lk = e4["leakage_E3_by_window"]["1us"][q]
        h = e4["text_numbers"]["hidden_leakage_1us"][q]
        out["leaked_shots_pct_of_shots"] = (lk["prep0"]["E3_pct"] + lk["prep1"]["E3_pct"]) / 2   # 560k shots per bit
        out["leaked_read_correctly_pct"] = 100.0 * (1 - h["leaked_errors_pct_of_shots"]["value"]
                                                    / h["leaked_pct_of_shots"]["value"])
    return out


def table_rows(e8, e4, exact=False):
    """Return {row key: LaTeX row} for the five qubits and the Q1/Q3/Q5 pool."""
    t = e8["table_error_modes_1us"]
    rows = {}
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q1Q3Q5"):
        r = dict(t[q])
        assert r["n_err"] == (e4["table_1us_cnn_canonical"]["pooled"][q] if q == "Q1Q3Q5"
                              else e4["table_1us_cnn_canonical"]["per_qubit"][q])["n_err"], q
        if exact:
            r.update(exact_shares(e4, q, r))
        cells = [LABELS[q], _cell(err_rate_pct(e4, q), 2), _cell(_v(r["leakage"]), 1),
                 _cell(_v(r["state_change"]), 1, bold=(q == "Q1Q3Q5")), _cell(_v(r["overlap"]), 1)]
        if q == "Q1Q3Q5":
            cells.append("---")
        elif _v(r["leaked_shots_pct_of_shots"]) == "n/m":
            cells.append("n/m")
        else:
            cells.append(f"{_cell(_v(r['leaked_shots_pct_of_shots']), 1)} ({_cell(_v(r['leaked_read_correctly_pct']), 1)})")
        rows[q] = " & ".join(cells) + r" \\"
    return rows


def render(e8, e4, exact=False):
    r = table_rows(e8, e4, exact)
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{" + CAPTION + "}", r"\label{tab:error_modes}", r"\begin{tabular}{l r r r r r}", r"\toprule",
             HEADER, r"\midrule", r["Q1"], r["Q2"], r["Q3"], r["Q4"], r["Q5"], r"\midrule", r["Q1Q3Q5"], r"\bottomrule",
             FOOT, r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(LP.RES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--rows-only", action="store_true")
    ap.add_argument("--exact", action="store_true", help="recompute every share from counts")
    a = ap.parse_args()
    e8 = json.load(open(os.path.join(a.results, "em_e8fix_final_numbers.json")))
    e4 = json.load(open(os.path.join(a.results, "em_e4recon_final.json")))
    txt = "\n".join(table_rows(e8, e4, a.exact).values()) + "\n" if a.rows_only else render(e8, e4, a.exact)
    out = a.out or os.path.join(LP.outdir(LP.FIGS), "tab_error_modes_exact.tex" if a.exact else "tab_error_modes.tex")
    open(out, "w").write(txt)
    print(txt, end="")
    print("wrote", out)


if __name__ == "__main__":
    main()
