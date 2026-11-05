"""Regenerate one §6 / QEC-appendix figure from results/qec into $ORAQLE_OUT/paper_figures/qec.

Each figure is made by the plotting code that made the paper's PDF (vendored in qec/eccentric/plots, path plumbing
only), called exactly as that script's __main__ block calls it:

  label (paper)          chapter:line                 script (origin)                                output
  fig:ler_discriminators qec_impact.tex:23-30         plot_paper_pairs.discriminators()  (recovered)  ler_pair_discriminators.pdf
  fig:six_noise          qec_impact.tex:97-104        plot_paper_pairs.six_pair(current | lowt)       ler_pair_noise.pdf
  fig:six_future         qec_impact.tex:111-118       plot_paper_pairs.six_pair(highfid | futuristic) ler_pair_future.pdf
  fig:ler_current        appendix_herqules.tex:413-418 plot_readout_compare.make_figure (@7858c2c)    readout_ler_herqules_current.pdf
  fig:ler_futuristic     appendix_herqules.tex:422-427 plot_readout_compare.make_figure (@7858c2c)    readout_ler_herqules_futuristic.pdf
  fig:six_decoh          appendix_herqules.tex:435-440 plot_noise_models_paper (cluster-only)         noise_models_ler_decoherence_only.pdf
  fig:six_fut_fixed      appendix_herqules.tex:458-463 plot_single_model_paper (cluster-only)         noise_models_ler_futuristic_fixedreadout.pdf

Usage: python make_figure.py <label> [<label> ...] | all      (needs numpy + matplotlib)
"""
import os
import pathlib
import sys

PLOTS = pathlib.Path(__file__).resolve().parents[2] / "qec" / "eccentric" / "plots"
sys.path.insert(0, str(PLOTS))
LABELS = ["fig:ler_discriminators", "fig:six_noise", "fig:six_future", "fig:ler_current", "fig:ler_futuristic",
          "fig:six_decoh", "fig:six_fut_fixed"]


def make(label):
    if label in ("fig:ler_discriminators", "fig:six_noise", "fig:six_future"):
        argv, sys.argv = sys.argv, sys.argv[:1]            # plot_paper_pairs reads OUT_DIR from argv[1]
        import plot_paper_pairs as P
        sys.argv = argv
        NM = P.NM
        if label == "fig:ler_discriminators":
            P.discriminators()
        elif label == "fig:six_noise":
            cur = P.filter_csv(f"{NM}/ler_noise_models.csv", "heron_cnn", os.path.join(P.OUT_DIR, "_heron_cnn.csv"))
            P.six_pair(cur, "Current ibm_boston noise", f"{NM}/ler_noise_models_boston_lowt_backlog.csv",
                       "Low coherence + decoder backlog", "ler_pair_noise")
        else:
            P.six_pair(f"{NM}/ler_noise_models_highfid.csv", "High-fidelity readout",
                       f"{NM}/ler_noise_models_futuristic.csv", "Futuristic noise", "ler_pair_future")
    elif label in ("fig:ler_current", "fig:ler_futuristic"):
        import plot_readout_compare as M
        M.make_figure("current" if label == "fig:ler_current" else "futuristic")
    elif label == "fig:six_decoh":
        import plot_noise_models_paper as M
        M.make_figure(*next(m for m in M.MODELS if m[0] == "decoherence_only"))
    elif label == "fig:six_fut_fixed":
        import plot_single_model_paper as M
        M.plot(*next(j for j in M.JOBS if j[0].endswith("ler_noise_models_futuristic_fixedreadout.csv")))
    else:
        sys.exit(f"unknown label {label}; known: {', '.join(LABELS)}")


if __name__ == "__main__":
    todo = LABELS if sys.argv[1:] in (["all"], []) else sys.argv[1:]
    for lab in todo:
        make(lab)
