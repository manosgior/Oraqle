"""
[Oraqle artifact, audit 2026-11-05] fig:ler_current / fig:ler_futuristic with the MCMit-CNN readout.

The paper's original figures (plots/readout_ler_herqules_{current,futuristic}.pdf) plotted the old mixed-shot
HERQULES readout, while qec_impact.tex states the MCMit-CNN readout. Since the 2026-11-05 [audit fix] the paper
includes plots/readout_ler_cnn_{current,futuristic}.pdf (appendix_herqules.tex:427, :438), made by this script
(pixel- and drawing-stream-identical; results/qec/reference_figures). It redraws both figures with the CNN-readout reruns of both
simulators, using the paper's own plotting function (qec/eccentric/plots/plot_readout_compare.py @7858c2c, imported
unchanged: same size, fonts, colours, markers, legend and annotation); only its input data and output file names
differ.

Inputs (MCMit-CNN readout: flip error 1 - gmean(Q1,Q3,Q4,Q5) of the tab:fidelity MCMit-CNN row, CNN_ACC of the
drivers; 200-1000 ns; d = 5..13; current and futuristic ibm_boston noise):
  (a) ECCentric   results/qec/experiment_results/readout_compare/eccentric_readout_ler_cnn.csv  (eccentric @da7164c,
                  MEMORY_ROUNDS=dplus1, the same round setting as the plotted HERQULES CSV; 50 points)
  (b) lattice-sim results/qec/artifact_readout_ler_cnn_complete.csv = the 49 points of
                  results/qec/experiment_results/synchronization-artifact/artifact_readout_ler_cnn.csv
                  + results/qec/latsim_cnn_d13_200ns_current.csv (the missing d=13 / 200 ns / current point,
                  computed 2026-11-05 by extras/qec/run_latsim_cnn_missing_point.py; 500k shots)

  python paper_figures/qec/make_readout_ler_cnn.py --merge   # (re)write artifact_readout_ler_cnn_complete.csv
  python paper_figures/qec/make_readout_ler_cnn.py [OUTDIR]  # -> OUTDIR/readout_ler_cnn_{current,futuristic}.pdf
OUTDIR defaults to $ORAQLE_OUT/paper_figures/qec. The paper copies are oraqle-ics/plots/readout_ler_cnn_*.pdf
(= results/qec/reference_figures/readout_ler_cnn_*.pdf). paper_figures/qec/make_figure.py fig:ler_current /
fig:ler_futuristic call main() of this file.
"""
import csv
import os
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "qec" / "eccentric" / "plots"))
from _paths import res, FIG_OUT  # noqa: E402

QEC = REPO / "results" / "qec"
ECC_CSV = res("readout_compare", "eccentric_readout_ler_cnn.csv")
LAT_49 = res("synchronization-artifact", "artifact_readout_ler_cnn.csv")
LAT_NEW = QEC / "latsim_cnn_d13_200ns_current.csv"
LAT_ALL = QEC / "artifact_readout_ler_cnn_complete.csv"
COLS = ["tool", "distance", "readout_ns", "noise_model", "measure_error", "logical_error_rate"]


def merge():
    rows = list(csv.DictReader(open(LAT_49)))
    new = list(csv.DictReader(open(LAT_NEW)))
    keys = {(r["distance"], r["readout_ns"], r["noise_model"]) for r in rows}
    for r in new:
        assert (r["distance"], r["readout_ns"], r["noise_model"]) not in keys
        rows.append({c: r[c] for c in COLS})
    assert len(rows) == 50
    rows.sort(key=lambda r: (int(r["distance"]), r["noise_model"], int(r["readout_ns"])))  # the driver's final sort
    with open(LAT_ALL, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {LAT_ALL} ({len(rows)} points)")


def main(outdir, noise_models=None):
    import plot_readout_compare as M  # the paper's plotting code, unchanged
    saved = M.DATA, M.OUTDIR
    M.DATA = {"eccentric": M.load(ECC_CSV), "artifact": M.load(str(LAT_ALL))}
    assert len(M.DATA["eccentric"]) == 50 and len(M.DATA["artifact"]) == 50
    os.makedirs(outdir, exist_ok=True)
    # render into a scratch directory: make_figure() names its output readout_ler_herqules_*, which must not
    # overwrite the HERQULES-readout figures (paper_figures/qec extras) in outdir
    with tempfile.TemporaryDirectory(dir=outdir) as tmp:
        M.OUTDIR = tmp
        try:
            for nm in noise_models or M.NOISE_MODELS:
                M.make_figure(nm)
                for ext in (".pdf", ".png"):
                    src = os.path.join(tmp, f"readout_ler_herqules_{nm}{ext}")
                    dst = os.path.join(outdir, f"readout_ler_cnn_{nm}{ext}")
                    os.replace(src, dst)
                    print(f"  -> {dst}")
        finally:
            M.DATA, M.OUTDIR = saved


if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge()
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else FIG_OUT)
