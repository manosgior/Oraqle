"""m2mix_prep.py -- window means + neighbour (crosstalk) shift correction for the
semi-supervised mixture-model leakage analysis (agent tag m2mix).

Reads the shared feature dataset (x agent) and writes, for TEST and TRAIN:
  windows: early = feat bins 0-1 (0-200 ns), 1us = feat1us bins 0-9 (strictly causal
           samples 0-499, = CNN "1 us" input), 2us = feat bins 0-19 (whole trace).
  x[w]          (N,5,2) float32 window mean (along, perp), raw ADC units
  shift[w]      (32,5,2) per-prepared-state core shift (mean-shift mode of the state's
                own core minus the average over the 16 states with the same own bit)
  core[w]       (32,5,2) per-state core mode
  sig0[w]       (5,)    robust noise sigma (MAD of prepared-0 along around the |0> mode)
Correction applied downstream: x_corr = x - shift[y].
Usage: python m2mix_prep.py /feat /out
"""
import sys, json, numpy as np

FEAT, OUT = sys.argv[1], sys.argv[2]
WINDOWS = {"early": ("feat", 0, 2), "1us": ("feat1us", 0, 10), "2us": ("feat", 0, 20)}


def mad_sigma(v):
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def mean_shift(x, c, h, iters=40):
    c = np.asarray(c, float).copy()
    for _ in range(iters):
        d2 = ((x - c) ** 2).sum(1) / h ** 2
        w = np.exp(-0.5 * d2)
        cn = (w[:, None] * x).sum(0) / w.sum()
        if np.abs(cn - c).max() < 1e-5 * h:
            c = cn
            break
        c = cn
    return c


res = {}
for split in ("test", "train"):
    y = np.load(f"{FEAT}/y_{split}.npy")
    bits = (y[:, None] >> np.arange(5)) & 1
    arrs = {}
    for name in ("feat", "feat1us"):
        arrs[name] = np.load(f"{FEAT}/{name}_{split}.npy", mmap_mode="r")
    out = {"y": y}
    meta = {}
    for w, (name, a, b) in WINDOWS.items():
        x = np.asarray(arrs[name][:, :, a:b]).mean(2).astype(np.float32)  # (N,5,2)
        shift = np.zeros((32, 5, 2), np.float32)
        core = np.zeros((32, 5, 2), np.float32)
        sig0 = np.zeros(5)
        for q in range(5):
            xq = x[:, q, :].astype(np.float64)
            m0 = bits[:, q] == 0
            med0 = np.median(xq[m0], 0)
            s0 = mad_sigma(xq[m0, 0])
            c0 = mean_shift(xq[m0][::4], med0, s0)
            s0 = mad_sigma(xq[m0, 0][np.abs(xq[m0, 0] - c0[0]) < 6 * s0])
            sig0[q] = s0
            pooled = {}
            for bb in (0, 1):
                mb = bits[:, q] == bb
                pooled[bb] = mean_shift(xq[mb][::4], np.median(xq[mb], 0), s0)
            for s in range(32):
                bb = (s >> q) & 1
                ms = y == s
                core[s, q] = mean_shift(xq[ms], pooled[bb], s0)
            for bb in (0, 1):
                sel = [s for s in range(32) if (s >> q) & 1 == bb]
                avg = core[sel, q].mean(0)
                shift[sel, q] = core[sel, q] - avg
        out[f"x_{w}"] = x
        out[f"shift_{w}"] = shift
        out[f"core_{w}"] = core
        out[f"sig0_{w}"] = sig0.astype(np.float32)
        meta[w] = {
            "sig0": sig0.tolist(),
            "max_abs_shift_over_sig0": [float(np.abs(shift[:, q]).max() / sig0[q]) for q in range(5)],
            "rms_shift_over_sig0_along_perp": [
                (np.sqrt((shift[:, q] ** 2).mean(0)) / sig0[q]).tolist() for q in range(5)],
        }
        print(split, w, json.dumps(meta[w]), flush=True)
    np.savez(f"{OUT}/m2mix_winmeans_{split}.npz", **out)
    res[split] = meta
json.dump(res, open(f"{OUT}/m2mix_prep_meta.json", "w"), indent=1)
print("done")
