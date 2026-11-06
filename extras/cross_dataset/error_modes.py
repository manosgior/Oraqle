"""How much of the error is Gaussian overlap, and how much is something else? Across datasets.

The paper's split (tab:error_modes) comes from the HMM and segment pipeline in leakage/, which needs five
channels and a long chain of steps. This is a simpler estimator that runs on any single-qubit dataset:

  1. Bin the traces in 100 ns steps and fit LDA on the training set, giving one score per shot.
  2. For each state, fit a Gaussian to the far side of its score distribution, the side away from the
     other state. State changes, which pull a shot toward the other state, do not contaminate that side.
  3. Overlap error = the mass of those two Gaussians beyond the decision threshold. It is an upper bound
     on what overlap alone can explain, because tails on the near side are counted as non-overlap.
     Everything above it is called "excess" (state change, preparation error, leakage or drift).

Run the same code on the Lienhard channels first. It is only a fair proxy if it comes close to the
paper's split there (Q1, Q3, Q5 pooled at 1 us: 81% state change, 17% overlap).

    python -I error_modes.py qick lienhard_q1 lienhard_q3 lienhard_q5 cao
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402

from common import RUNS, XD, Dataset, fidelity  # noqa: E402


def bins_100ns(X, dt):
    k = max(1, int(round(100 / dt)))
    n = X.shape[1] // k
    if n == 0:
        return X.mean(1)
    return X[:, : n * k].reshape(len(X), n, k, 2).mean(2).reshape(len(X), -1)


def far_side_gaussian(s, side):
    """Gaussian from the half of the distribution on `side` of its mode (+1 or -1)."""
    h, e = np.histogram(s, bins=400)
    c = 0.5 * (e[1:] + e[:-1])
    k = max(5, len(h) // 80)
    mode = c[np.argmax(np.convolve(h, np.ones(k) / k, mode="same"))]
    far = s[s > mode] - mode if side > 0 else mode - s[s < mode]
    sigma = float(np.sqrt(np.mean(far ** 2)))
    return float(mode), sigma


def analyse(ds, T_ns, cnn_seed=1):
    L = ds.samples(T_ns)
    Xtr, ytr, Xte, yte = ds.split(L)
    F, G = bins_100ns(Xtr, ds.dt), bins_100ns(Xte, ds.dt)
    lda = LinearDiscriminantAnalysis().fit(F, ytr)
    s = lda.decision_function(G)  # > 0 means |1>
    Flda, p01, p10 = fidelity(yte, s > 0)
    m0, s0 = far_side_gaussian(s[yte == 0], -1)
    m1, s1 = far_side_gaussian(s[yte == 1], +1)
    ov10 = float(norm.sf((0 - m0) / s0))  # |0> beyond the threshold
    ov01 = float(norm.cdf((0 - m1) / s1))  # |1> beyond the threshold
    overlap = 0.5 * (ov10 + ov01)
    err = 1 - Flda
    rec = dict(dataset=ds.name, T_ns=T_ns, F_lda=Flda, err_lda=err, p01=p01, p10=p10, sep_sigma=abs(m1 - m0) / (0.5 * (s0 + s1)),
               overlap_err=overlap, overlap_share_of_lda=overlap / err, excess_err=max(0.0, err - overlap))
    cnn = RUNS / ds.name / f"cnn_T{int(T_ns)}_s{cnn_seed}.json"
    if cnn.exists():
        c = json.loads(cnn.read_text())
        ec = 1 - c["F"]
        rec.update(F_cnn=c["F"], err_cnn=ec, cnn_p01=c["p01"], cnn_p10=c["p10"], overlap_share_of_cnn=min(1.0, overlap / ec))
    return rec


if __name__ == "__main__":
    out = []
    for name in sys.argv[1:]:
        ds = Dataset(name)
        maxT = ds.Xtr.shape[1] * ds.dt
        grid = [t for t in ([400, 600, 800, 1000, 1200, 1500, 1800, 2400]) if t <= maxT]
        for T in grid:
            r = analyse(ds, T)
            out.append(r)
            print(name, T, "F_lda %.4f  sep %.2f sigma  overlap %.4f  share of LDA error %.2f  p01 %.3f p10 %.3f" % (
                r["F_lda"], r["sep_sigma"], r["overlap_err"], r["overlap_share_of_lda"], r["p01"], r["p10"]),
                ("| CNN err %.4f, overlap share %.2f" % (r["err_cnn"], r["overlap_share_of_cnn"])) if "err_cnn" in r else "", flush=True)
    (XD / "error_modes.json").write_text(json.dumps(out, indent=1))
