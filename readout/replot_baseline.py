"""Re-plot baseline (no-ML threshold) results purely from the saved
Baseline_Threshold_*.csv files in oraqle_reports -- no recomputation.
Produces the two 2-panel figures (figsize 16x4):
  1) Readout-length effect: (a) fidelity vs duration, (b) info-gain rate
  2) Sliding-window effect:  (a) 250 ns window, (b) 500 ns window
"""
import os, glob, re, csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gmean

import sys as _sys  # [unify] repo layout: reuse discriminators/disc_paths.py
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'discriminators'))
from disc_paths import RAW_TRAIN, RAW_TEST, RESULTS_DISC, REPORTS_OUT, ensure_dir, require_dataset  # [unify]
R = os.environ.get("ORAQLE_READOUT_RES", str(RESULTS_DISC))  # [unify] was "/r" (~/oraqle_reports mount)
R_OUT = ensure_dir(REPORTS_OUT)  # [unify] figures go to $ORAQLE_OUT, not into results/
DT = 2e-9
N_Q = 5
COLORS = sns.color_palette("deep")          # Q0 blue, Q1 orange, Q2 green, Q3 red, Q4 purple
MARKERS = ['o', 's', '^', 'D', 'P']
BLUE = "#1f4e9c"


def _load_sweep():
    rows = {}
    for f in glob.glob(os.path.join(R, "Baseline_Threshold_len*.csv")):
        m = re.search(r"_len(\d+)_", os.path.basename(f))
        if not m:
            continue
        L = int(m.group(1))
        r = list(csv.DictReader(open(f)))[0]
        rows[L] = [float(r[f"qubit_{q}_accuracy"]) / 100.0 for q in range(N_Q)]
    Ls = sorted(rows)
    fid = np.array([rows[L] for L in Ls]).T          # (n_q, n_len)
    T_us = np.array(Ls) * DT * 1e6
    return T_us, fid


def _dFdT(fid, T_s, sigma=2.0):
    out = np.zeros_like(fid)
    for q in range(fid.shape[0]):
        out[q] = np.gradient(gaussian_filter1d(fid[q], sigma), T_s)
    return out


def _load_sliding(pattern):
    f = sorted(glob.glob(os.path.join(R, pattern)))[-1]
    rr = list(csv.DictReader(open(f)))
    centers = np.array([float(r["window_center_us"]) for r in rr])
    fid = np.array([[float(r[f"qubit_{q}_fidelity"]) for q in range(N_Q)] for r in rr]).T
    return centers, fid


def _qline(ax, x, fid, gmean_line=True):
    for q in range(N_Q):
        ax.plot(x, fid[q], color=COLORS[q], marker=MARKERS[q], ms=5, lw=2, label=f"Q{q+1}")
    if gmean_line:
        gm = gmean(fid, axis=0)
        ax.plot(x, gm, color="black", marker="o", ms=4, lw=2, label="G. Mean")


def _suptitle(fig):
    # sits right above the panels (no reserved band / empty space)
    fig.text(0.5, 1.0, "Higher is better $\\uparrow$", ha="center", va="bottom",
             color=BLUE, fontweight="bold", fontsize=14)


# ---------- Figure 1: readout-length effect ----------
T_us, fid = _load_sweep()
dF = _dFdT(fid, T_us * 1e-6)
sns.set_theme(style="whitegrid")
fig, (a, b) = plt.subplots(1, 2, figsize=(16, 4))
_qline(a, T_us, fid, gmean_line=True)
a.set_title("(a) Readout Length Effect on Fidelity", fontweight="bold", fontsize=14)
a.set_xlabel("Readout Duration (µs)", fontsize=12); a.set_ylabel("Assignment Fidelity", fontsize=12)
a.set_xlim(0, 2); a.legend(frameon=True, loc="lower right", ncol=2, fontsize=9)
_qline(b, T_us, dF, gmean_line=False)
for q in range(N_Q):
    b.axvline(T_us[np.nanargmax(dF[q])], color=COLORS[q], lw=1.0, ls=":", alpha=0.6)
b.set_title("(b) Readout Length Effect on Information Gain Rate", fontweight="bold", fontsize=14)
b.set_xlabel("Readout Duration (µs)", fontsize=12); b.set_ylabel("Information Gain Rate", fontsize=12)
b.set_xlim(0, 2); b.legend(frameon=True, loc="upper right", fontsize=9)
fig.tight_layout(); _suptitle(fig)
fig.savefig(os.path.join(R_OUT, "baseline_readout_length_effect.pdf"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(R_OUT, "baseline_readout_length_effect.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------- Figure 2: sliding-window effect ----------
c250, f250 = _load_sliding("Baseline_Threshold_sliding_window_250ns*.csv")
c500, f500 = _load_sliding("Baseline_Threshold_sliding_window_500ns*.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(16, 4))
_qline(a, c250, f250, gmean_line=True)
a.set_title("(a) Readout Window Effect on Fidelity - 250 ns Window", fontweight="bold", fontsize=14)
a.set_xlabel("Window Center Time (µs)", fontsize=12); a.set_ylabel("Assignment Fidelity", fontsize=12)
a.set_xlim(0, 2); a.legend(frameon=True, loc="lower right", ncol=2, fontsize=9)
_qline(b, c500, f500, gmean_line=True)
b.set_title("(b) Readout Window Effect on Fidelity - 500 ns Window", fontweight="bold", fontsize=14)
b.set_xlabel("Window Center Time (µs)", fontsize=12); b.set_ylabel("Assignment Fidelity", fontsize=12)
b.set_xlim(0, 2); b.legend(frameon=True, loc="lower right", ncol=2, fontsize=9)
fig.tight_layout(); _suptitle(fig)
fig.savefig(os.path.join(R_OUT, "baseline_sliding_window_effect.pdf"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(R_OUT, "baseline_sliding_window_effect.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved baseline_readout_length_effect.{pdf,png} and baseline_sliding_window_effect.{pdf,png}")
