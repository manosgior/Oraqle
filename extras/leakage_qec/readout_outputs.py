#!/usr/bin/env python3
"""readout_outputs.py -- what a readout can tell the decoder, per measurement: bit, confidence, leakage flag.

For each readout length T this measures, on the Lienhard et al. TEST shots:

  leakage rates    P(qubit leaves the readout in L | prepared b), from the frozen 3-state HMM of
                   leakage/m3time_hmm.py run on the full 2 us trace (labels: Viterbi state in the first 100 ns
                   bin after the window, which no detector below sees)
  flag detectors   per qubit, fitted on half of TEST and scored on the other half:
                     own_mean   logistic regression on the qubit's own window-mean I/Q (no training of the CNN)
                     cnn_probe  logistic regression on the frozen CNN's 16 pooled features (report section 5.3 a)
                     input      logistic regression on the 10-channel CNN input
                   ROC operating points at fixed false-flag rates
  label-free check for flagged shots, the mirror-corrected off-axis excess in the window after T (never seen by the
                   detector, no HMM), compared with unflagged shots
  confidence       table of error rate versus |logit| of the soft CNN, for k-bit soft readout

The dataset is needed. Output: --out/readout_outputs.json. No per-shot data is written.
"""
import argparse, json, os, pathlib, sys, time
import numpy as np
import torch as T
from sklearn.linear_model import LogisticRegression

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "discriminators" / "runners"), str(ROOT / "discriminators" / "networks")]
import _colleague_prep as P  # noqa: E402
from CNN import CNN  # noqa: E402
from oraqle_env import RAW_TEST, RAW_TRAIN, RESULTS, WORK, require_dataset  # noqa: E402

NAMES = ["Q1", "Q2", "Q3", "Q4", "Q5"]
LENGTHS = {400: 200, 600: 300, 800: 400, 1000: 500}  # ns -> raw samples
FPR_TARGETS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
DS = 40


def gap_features(model, X, bs=8192):
    """Pooled features and bit logits of the frozen CNN."""
    feats, logits = [], []
    model.eval()
    with T.no_grad():
        for s in range(0, len(X), bs):
            x = T.tensor(X[s:s + bs])
            x = T.relu(model.bn1(model.conv1(x)))
            x = T.relu(model.bn2(model.conv2(x)))
            x = model.res_block(x).mean(dim=2)
            feats.append(x.numpy())
            logits.append(T.cat([h(x) for h in model.output_layers], 1).numpy())
    return np.concatenate(feats), np.concatenate(logits)


def operating_points(score_fit, y_fit, score_ev, y_ev):
    """Thresholds set for false-flag targets on the fit half, scored on the held-out half."""
    out = []
    neg = np.sort(score_fit[y_fit == 0])
    for t in FPR_TARGETS:
        thr = neg[int((1 - t) * (len(neg) - 1))]
        fl = score_ev > thr
        tp, fp = int((fl & (y_ev == 1)).sum()), int((fl & (y_ev == 0)).sum())
        out.append(dict(fpr_target=t, thr=float(thr), tpr=tp / max(1, int((y_ev == 1).sum())),
                        fpr=fp / max(1, int((y_ev == 0).sum())), precision=tp / max(1, tp + fp),
                        flagged_frac=float(fl.mean())))
    return out


def fit_lr(Z, y, fit):
    mu, sd = Z[fit].mean(0), Z[fit].std(0) + 1e-9
    m = LogisticRegression(C=1.0, max_iter=300).fit((Z[fit] - mu) / sd, y[fit])
    return m.decision_function((Z - mu) / sd)


def mirror_excess(X_future, bits_q, q, flags):
    """Mirror-corrected off-axis excess after the window, per flag state. Label-free: no HMM, no window data."""
    w = X_future[:, [q, 5 + q]].mean(2)              # (N, 2) own-channel I/Q; channels are I0..I4, Q0..Q4
    c0, c1 = np.median(w[bits_q == 0], 0), np.median(w[bits_q == 1], 0)
    u = (c1 - c0) / np.linalg.norm(c1 - c0)
    v = np.array([-u[1], u[0]])
    perp = (w - c0) @ v
    sig = 1.4826 * np.median(np.abs(perp[bits_q == 0] - np.median(perp[bits_q == 0])))
    z = (perp - np.median(perp[bits_q == 0])) / sig
    if np.median(z[flags]) < 0:                      # the L side is whichever side flagged shots sit on
        z = -z
    res = {}
    for name, m in (("flagged", flags), ("unflagged", ~flags)):
        res[name] = float(((z[m] > 3).sum() - (z[m] < -3).sum()) / max(1, m.sum()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WORK / "leakage_qec"))
    ap.add_argument("--labels", default=str(WORK / "leakage" / "m3time" / "out_re" / "featbox_xt1"),
                    help="directory of m3time_re_q*_arrays.npz (frozen HMM on TEST)")
    ap.add_argument("--checkpoints", default=str(WORK / "leakage_qec" / "models"), help="output of train_cnn.py")
    ap.add_argument("--soft", default=str(RESULTS / "system" / "soft"))
    a = ap.parse_args()
    require_dataset()
    os.makedirs(a.out, exist_ok=True)
    T.set_num_threads(int(os.environ.get("THREADS", "16")))
    t0 = time.time()
    path = {q: np.load(f"{a.labels}/m3time_re_q{q}_arrays.npz")["path"] for q in range(5)}
    prm = P.calibrate(str(RAW_TRAIN), "X_train", "y_train", 15000, (0, 500))
    res = {"note": __doc__, "per_T": {}}
    for Tns, L in LENGTHS.items():
        k = Tns // 100                               # first 100 ns bin after the window
        Xraw, y = P.preprocess(str(RAW_TEST), "X_test", "y_test", prm, DS, (0, L))
        X = P.to_torch_layout(Xraw).astype(np.float32)
        Xfut_raw, _ = P.preprocess(str(RAW_TEST), "X_test", "y_test", prm, DS, (L + 50, min(L + 300, 1000)))
        Xfut = P.to_torch_layout(Xfut_raw).astype(np.float32)
        bits = P.bits(y).astype(np.int8)
        ck = f"{a.checkpoints}/cnn_len{L}.pth"
        model = CNN(in_channels=10, m_param=16, num_qubits=5)
        model.load_state_dict(T.load(ck, map_location="cpu", weights_only=True))
        feats, logits = gap_features(model, X)
        N = len(y)
        fit = np.arange(N) % 2 == 0
        row = {"checkpoint": os.path.basename(ck), "n_ds_samples": int(X.shape[2]), "per_qubit": {}}
        for q in range(5):
            b = bits[:, q]
            lab = (path[q][:, k] == 2).astype(np.int8)
            start = (path[q][:, 1] == 2) & (path[q][:, 2] == 2)
            bit_pred = (logits[:, q] > 0).astype(np.int8)
            r = {"bit_err_pct_ckpt": 100 * float((bit_pred != b).mean()),
                 "P_L_end": {"prep0": float(lab[b == 0].mean()), "prep1": float(lab[b == 1].mean())},
                 "P_L_start": {"prep0": float(start[b == 0].mean()), "prep1": float(start[b == 1].mean())},
                 "P_read1_given_L_end": {"prep0": float(bit_pred[(b == 0) & (lab == 1)].mean()) if (lab[b == 0]).any() else None,
                                         "prep1": float(bit_pred[(b == 1) & (lab == 1)].mean()) if (lab[b == 1]).any() else None},
                 "detectors": {}}
            own = X[:, [q, 5 + q]].mean(2)
            for name, Z in (("own_mean", own), ("cnn_probe", feats), ("input", X.reshape(N, -1))):
                s = fit_lr(Z, lab, fit)
                ops = operating_points(s[fit], lab[fit], s[~fit], lab[~fit])
                fl = s[~fit] > ops[2]["thr"]            # the 1% false-flag point
                ops[2]["future_offaxis"] = mirror_excess(Xfut[~fit], b[~fit], q, fl) if fl.sum() > 100 else None
                r["detectors"][name] = ops
            row["per_qubit"][NAMES[q]] = r
            print(f"T={Tns} {NAMES[q]} err={r['bit_err_pct_ckpt']:.2f} P(L_end|1)={r['P_L_end']['prep1']:.4f} "
                  + " ".join(f"{n}:tpr@1%={v[2]['tpr']:.3f}" for n, v in r["detectors"].items())
                  + f" ({time.time()-t0:.0f}s)", flush=True)
        # confidence table of the soft CNN of system/soft_readout_cnn.py, Q1,Q3,Q4,Q5 pooled as in sweep_soft.py
        z = np.load(f"{a.soft}/cnn_soft_{Tns}ns.npz")
        prob = np.clip(z["prob"][:, [0, 2, 3, 4]].astype(np.float64).ravel(), 1e-5, 1 - 1e-5)
        wrong = (prob > 0.5) != (z["label"][:, [0, 2, 3, 4]].ravel() == 1)
        al = np.abs(np.log(prob / (1 - prob)))
        edges = np.arange(0, 12.01, 0.25)
        idx = np.minimum(np.digitize(al, edges) - 1, len(edges) - 2)
        row["confidence_table"] = dict(abs_logit_edges=edges.tolist(),
                                       frac=(np.bincount(idx, minlength=len(edges) - 1) / len(al)).tolist(),
                                       err=[float(wrong[idx == i].mean()) if (idx == i).any() else 0.0 for i in range(len(edges) - 1)],
                                       hard_err=float(wrong.mean()))
        res["per_T"][str(Tns)] = row
        json.dump(res, open(f"{a.out}/readout_outputs.json", "w"), indent=1)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
