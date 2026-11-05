"""tab:fpga_resources (ml_discriminators.tex:277-296) from the Vitis HLS reports.  [unify, new file]

Composes the rows from fpga/hls4ml/reports with fpga/parse_reports.py (QubiCML = 5 x qubic,
HERQULES, MCMit CNN = m8 with ParallelizationFactor, KLiNQ = 3 x small + 2 x larger reuse-4)
and writes them as LaTeX rows. No dataset and no Vitis installation needed.
"""
import importlib.util
import os

from _common import FIG_OUT, REPO

spec = importlib.util.spec_from_file_location("parse_reports", os.path.join(REPO, "fpga", "parse_reports.py"))
FP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FP)
_, rows, _ = FP.table()
mark = {"QubiCML": r"\textsuperscript{\dag}", "KLiNQ": r"\textsuperscript{\ddag}"}
L = []
for name, r in rows.items():
    p = r["pct_rounded"]
    L.append(f"{name}{mark.get(name, '')} & {p['LUT']}\\% & {p['FF']}\\% & {p['DSP']}\\% & ${r['latency_ns']:g}$\\,ns \\\\")
out = os.path.join(FIG_OUT, "tab_fpga_resources.tex")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("wrote", out)
