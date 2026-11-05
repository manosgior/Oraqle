"""m4conf_rawprobe.py -- screen (f): raw-trace probe of a few hundred candidates per qubit (agent tag m4conf).

For TEST shots: candidates of set A (|z_perp|>5, 1 us) per prepared bit, B-only candidates (prior |2> label but
not in A), and matched controls (same prepared state, non-candidate in A and B for that qubit, within +-2000 rows).
For each shot the residual r = x - meantrace[state] (per-state mean of all 35000 TEST shots) is
  * demodulated on every tone j (mix f_j, 101-tap 5 MHz FIR 'same', rotate theta_j, mean over samples 0..499)
    -> R_j (along, perp), in units of the controls' robust sigma for that tone;
  * expressed relative to the state-mean signal of tone j: rel_j = R_j / M_j (complex). A common-mode drive
    amplitude glitch gives the same real rel_j on all tones, a common phase glitch the same imaginary rel_j;
  * out-of-band residual power (FFT of r over 0..499, readout bands +-8 MHz, images and DC excluded).
Usage: python m4conf_rawprobe.py /data /feat /work/out
"""
import json, os, sys
import numpy as np
import h5py
from scipy.signal import firwin, fftconvolve

DATA, FEAT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FS = 500e6; NQ = 5; NPER = 35000
CAL = json.load(open(f"{FEAT}/calib.json"))["used_for_features"]
FQ = np.array([c["f_Hz"] for c in CAL]); TH = np.array([c["theta_rad"] for c in CAL])
TAPS = firwin(101, 5e6, fs=FS)
t = np.arange(1000) / FS
rng = np.random.default_rng(1)
NC = 200

cand = np.load(f"{OUT}/m4conf_cand_rows_test.npz")
mt = np.load(f"{OUT}/m4conf_rawstats_test.npz")["meantrace"].astype(np.complex128)      # (32,1000)
y = np.load(f"{FEAT}/y_test.npy")
h = h5py.File(f"{DATA}/DRaw_C_Te_v0-002", "r")
X = h["X_test"]

f = np.fft.fftfreq(500, 1 / FS)
keep = np.abs(f) >= 3e6
for fq in FQ:
    keep &= (np.abs(f - fq) >= 8e6) & (np.abs(f + fq) >= 8e6)


def demod(z, fq, th):
    m = z * np.exp(-1j * 2 * np.pi * fq * t[: z.shape[1]])[None]
    return fftconvolve(m, TAPS[None], mode="same", axes=1) * np.exp(1j * th)


# state-mean demodulated window means M[s, j] (complex)
M = np.stack([demod(mt[:, :500], FQ[j], TH[j]).mean(1) for j in range(NQ)], 1)      # (32,5)


def probe(rows):
    rows = np.sort(np.asarray(rows))
    if len(rows) == 0:
        return None
    x = X[rows]
    z = x[:, :, 0] + 1j * x[:, :, 1]
    s = y[rows]
    r = z - mt[s]
    Rj = np.stack([demod(r[:, :500], FQ[j], TH[j]).mean(1) for j in range(NQ)], 1)  # (n,5) complex
    Rimg = np.stack([demod(r[:, :500], -FQ[j], 0.0).mean(1) for j in range(NQ)], 1)
    Fr = np.fft.fft(r[:, :500], axis=1)
    oob = (np.abs(Fr[:, keep]) ** 2).sum(1) / 500 ** 2
    rel = Rj / M[s]
    return {"rows": rows, "state": s, "R": Rj, "Rimg": Rimg, "oob": oob, "rel": rel}


res = {}
summary = {}
for q in range(NQ):
    groups = {}
    for b in (0, 1):
        rr = cand[f"A_q{q}_p{b}"]
        groups[f"A_p{b}"] = rng.choice(rr, min(NC, len(rr)), replace=False) if len(rr) else np.array([], int)
    rr = cand[f"Bonly_q{q}"]
    groups["Bonly"] = rng.choice(rr, min(NC, len(rr)), replace=False) if len(rr) else np.array([], int)
    # controls matched in state and position: one per candidate of the A/B groups (up to 400)
    excl = set(np.concatenate([cand[f"A_q{q}_p0"], cand[f"A_q{q}_p1"], cand[f"B_q{q}_p0"], cand[f"B_q{q}_p1"]]).tolist())
    src = np.concatenate([groups["A_p0"], groups["A_p1"], groups["Bonly"]])
    ctl = []
    for r0 in src:
        s0 = r0 // NPER
        for _ in range(50):
            c = int(np.clip(r0 + rng.integers(-2000, 2001), s0 * NPER, s0 * NPER + NPER - 1))
            if c not in excl:
                ctl.append(c); break
    extra = rng.choice(np.setdiff1d(np.arange(len(y)), list(excl)), 400, replace=False)
    groups["ctl"] = np.unique(np.concatenate([ctl, extra]))
    P = {g: probe(v) for g, v in groups.items()}
    c = P["ctl"]
    sig = {}
    for j in range(NQ):
        sig[j] = (1.4826 * np.median(np.abs(c["R"][:, j].real - np.median(c["R"][:, j].real))),
                  1.4826 * np.median(np.abs(c["R"][:, j].imag - np.median(c["R"][:, j].imag))))
    oob_ref = np.median(c["oob"])
    sq = {}
    for g, p in P.items():
        if p is None:
            sq[g] = {"n": 0}; continue
        e = {"n": int(len(p["rows"]))}
        for j in range(NQ):
            za = p["R"][:, j].real / sig[j][0]; zp = p["R"][:, j].imag / sig[j][1]
            amp = np.hypot(za, zp)
            e[f"tone_q{j}"] = {"median_abs_z_along": float(np.median(np.abs(za))), "median_abs_z_perp": float(np.median(np.abs(zp))),
                               "frac_2d_gt4": float((amp > 4).mean()), "median_2d": float(np.median(amp))}
        others = [j for j in range(NQ) if j != q]
        relo = p["rel"][:, others]
        e["other_tones_rel_residual"] = {"median_of_median_real": float(np.median(np.median(relo.real, 1))),
                                         "median_of_median_imag": float(np.median(np.median(relo.imag, 1))),
                                         "frac_all4_same_sign_real": float(((relo.real > 0).all(1) | (relo.real < 0).all(1)).mean()),
                                         "frac_all4_same_sign_imag": float(((relo.imag > 0).all(1) | (relo.imag < 0).all(1)).mean())}
        e["oob_over_ctl_median"] = {"median": float(np.median(p["oob"] / oob_ref)), "p99": float(np.percentile(p["oob"] / oob_ref, 99)),
                                    "frac_gt_1.2": float((p["oob"] / oob_ref > 1.2).mean())}
        e["image_residual_abs_median"] = [float(np.median(np.abs(p["Rimg"][:, j]))) for j in range(NQ)]
        sq[g] = e
        res[f"q{q}_{g}"] = np.concatenate([p["rows"][:, None].astype(float), np.abs(p["R"]) / np.array([np.hypot(*sig[j]) / np.sqrt(2) for j in range(NQ)]),
                                           (p["oob"] / oob_ref)[:, None]], 1)
    summary[f"q{q}"] = sq
    print(f"Q{q + 1}/q{q}", {g: (v.get("n"), [round(v[f'tone_q{j}']['frac_2d_gt4'], 3) for j in range(NQ)] if v.get('n') else None,
                                   v.get("oob_over_ctl_median", {}).get("median")) for g, v in sq.items()}, flush=True)
json.dump(summary, open(f"{OUT}/m4conf_rawprobe.json", "w"), indent=1)
np.savez(f"{OUT}/m4conf_rawprobe.npz", **res)
print("done")
