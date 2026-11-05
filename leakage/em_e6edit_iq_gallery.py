"""em_e6edit_iq_gallery.py (tag e6edit): re-render Fig. iq_taxonomy with an honest leakage label.

The original generator (Oraqle/Discriminators/runners/iq_error_gallery.py, paper layout:
column width, legend bottom-centre) labels the off-axis cloud "leakage |2>". The leakage audit
(data/leakage/LEAKAGE_REPORT.md) shows the state is non-computational but NOT known to be |2>
(geometry disfavours |2>). This script runs the ORIGINAL plot() unchanged except for that one
legend label, from the original cached npz (re-plot only, no re-extraction), and writes a NEW file.

Verified: with the label untouched, this pipeline reproduces plots/iq_error_gallery_q5.pdf
pixel for pixel (mean |diff| 6e-5 grey levels at 300 dpi).

Usage: python em_e6edit_iq_gallery.py <workdir>   (workdir receives a copy of the npz and the output)
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import os
import shutil
import sys

GEN = str(LP.IQ_GALLERY_GEN)
NPZ = str(LP.REPORTS / "iq_error_gallery_q5.npz")
OUT = _os.path.join(LP.outdir(LP.FIGS), "iq_error_gallery_q5_v2.pdf")
OLD_LABEL = 'label=r"leakage $|2\\rangle$"'
NEW_LABEL = 'label="leakage"'

work = sys.argv[1]
os.makedirs(work, exist_ok=True)
shutil.copy(NPZ, os.path.join(work, os.path.basename(NPZ)))
src = open(GEN).read()
assert src.count(OLD_LABEL) == 1, "generator changed: leakage label not found exactly once"
src = src.replace(OLD_LABEL, NEW_LABEL)
os.environ["IQ_REPORTS"] = work
os.environ["IQ_REPLOT"] = "1"
os.environ.setdefault("IQ_QUBIT", "5")
ns = {"__name__": "em_e6edit_iq_gallery_gen", "__file__": GEN}
exec(compile(src, GEN, "exec"), ns)
ns["plot"]()
pdf = os.path.join(work, "iq_error_gallery_q5.pdf")
assert not os.path.exists(OUT), f"{OUT} exists; refusing to overwrite"
shutil.copy(pdf, OUT)
print("wrote", OUT)
