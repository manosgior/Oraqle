"""Regenerate Fig. fig:iq_taxonomy (plots/iq_error_gallery_q5_v2.pdf, chapters/ml_discriminators.tex l.342-353).

Pipeline (same as leakage/em_e6edit_iq_gallery.py, which made the paper PDF on 2026-11-05):
  1. extract (needs the raw dataset; run once, already done): discriminators/runners/iq_error_gallery.py
     extract() demodulates Q5 with the HERQULES front-end (trainers.HERQULES_original.get_data) and stores
     integrated IQ points of real shots (4000 per computational cloud, up to 2500 off-axis "leak" shots), the
     two centroids and the example points in iq_error_gallery_q5.npz (83 kB). The committed copy is results/discriminators/iq_error_gallery_q5.npz
     (= graham ~/oraqle_reports/iq_error_gallery_q5.npz, 2026-06-02).
  2. plot (no dataset): run the generator's plot() on that npz with ONE change, the legend label
     'leakage |2>' -> 'leakage' (the leakage audit shows the off-axis state is not known to be |2>).

This script performs step 2 into $ORAQLE_OUT/leakage/iq_error_gallery_q5_v2.pdf (overwrites). It accepts a
generator that already carries the new label. Usage:
    python paper_figures/leakage/make_iq_taxonomy.py [--npz FILE] [--gen FILE] [--out FILE]
To redo step 1 from the raw data:  IQ_QUBIT=5 IQ_REPORTS=<dir> HERQULES_TRACE_LENGTH=500 \\
    python discriminators/runners/iq_error_gallery.py   (needs ORAQLE_DATA; see docs/reproduce/leakage.md)
"""
import argparse
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "leakage"))
import lkpaths as LP  # noqa: E402

OLD_LABEL = 'label=r"leakage $|2\\rangle$"'
NEW_LABEL = 'label="leakage"'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(LP.REPORTS / "iq_error_gallery_q5.npz"))
    ap.add_argument("--gen", default=str(LP.IQ_GALLERY_GEN))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(LP.outdir(LP.FIGS), "iq_error_gallery_q5_v2.pdf")
    src = open(a.gen).read()
    n_old, n_new = src.count(OLD_LABEL), src.count(NEW_LABEL)
    assert (n_old, n_new) in ((1, 0), (0, 1)), f"unexpected generator: {n_old} old / {n_new} new leakage labels"
    src = src.replace(OLD_LABEL, NEW_LABEL)
    with tempfile.TemporaryDirectory() as work:
        shutil.copy(a.npz, os.path.join(work, "iq_error_gallery_q5.npz"))
        os.environ["IQ_REPORTS"] = work
        os.environ["IQ_REPLOT"] = "1"
        os.environ["IQ_QUBIT"] = "5"
        ns = {"__name__": "make_iq_taxonomy_gen", "__file__": a.gen}
        exec(compile(src, a.gen, "exec"), ns)
        ns["plot"]()
        shutil.copy(os.path.join(work, "iq_error_gallery_q5.pdf"), out)
        shutil.copy(os.path.join(work, "iq_error_gallery_q5.png"), out.replace(".pdf", ".png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
