"""Table tab:sys_reaction (system_bottleneck.tex:61-78): reaction time R and waiting rounds w per hardware decoder.

The paper's table was hand-transcribed from system_cost.py (stdout and system_cost.json["reaction"]); this script
prints the same cells from results/system/system_cost.json (or --data) and writes the LaTeX body to
$ORAQLE_OUT/paper_figures/system/tab_sys_reaction.tex. Needs only the standard library.

Usage: python paper_figures/system/tab_sys_reaction.py [--data results/system] [--out DIR]
"""
import argparse
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from oraqle_env import OUT, RESULTS  # noqa: E402

TS = (400, 600, 800, 1000)
ROWS = [("Micro Blossom", "Micro Blossom (V80, measured)"),
        ("Helios", "Helios, unweighted$^\\ast$ (V80, measured)"),
        ("Helios (weighted)", "Helios, $16$ weight levels$^\\dagger$"),
        ("GPU", "GPU (reported)")]


def cells(data):
    R = json.load(open(os.path.join(data, "system_cost.json")))["reaction"]
    t_r = {T: R[f"Micro Blossom/{T}"]["t_r_ns"] / 1e3 for T in TS}
    rows = {dec: {T: (R[f"{dec}/{T}"]["R_ns"] / 1e3, R[f"{dec}/{T}"]["w"]) for T in TS} for dec, _ in ROWS}
    return t_r, rows


def latex(t_r, rows):
    out = ["Readout length & " + " & ".join(f"${T}$\\,ns" for T in TS) + " \\\\",
           "Round $t_r$ & " + " & ".join(f"${t_r[T]:.2f}$\\,\\textmu s" for T in TS) + " \\\\", "\\midrule"]
    for dec, label in ROWS:
        out.append(label + " & " + " & ".join(f"${rows[dec][T][0]:.2f}$ / ${rows[dec][T][1]}$" for T in TS) + " \\\\")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(RESULTS / "system"))
    ap.add_argument("--out", default=str(OUT / "paper_figures" / "system"))
    a = ap.parse_args()
    body = latex(*cells(a.data))
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "tab_sys_reaction.tex"), "w").write(body + "\n")
    print(body)


if __name__ == "__main__":
    main()
