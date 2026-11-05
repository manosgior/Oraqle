"""m4conf_dccheck.py -- is the raw-DC shift of Q3/Q2 candidates a glitch or leakage of their own tone? (agent tag m4conf)

Observation (m4conf_results.json, b_simul.A.per_qubit): A candidates of Q3 (q2) have median robust z of the raw
1 us DC level dcI1 = +2.75 (prep 0) / +2.18 (prep 1) and essentially all 128 dcI1 'glitch' flags (|rz| >= 6) of the
whole data set fall on Q3 candidates. Two readings:
  (G) a baseband / DC disturbance that also perturbs the lowest-|IF| tones (Q2 +25.4 MHz, Q3 -24.8 MHz) -> confounder;
  (L) the 500-sample boxcar DC mean picks up a residual of each tone, dc = sum_q (1/N) sum_n A_q(n) e^{i w_q n}, so a
      large Q3 amplitude change shifts dc deterministically -> benign, and S_raw then removes real Q3 events.
Test: complex least squares, per TEST state, of dc = dcI1 + i dcQ1 (state-centred) on the state-centred complex
feat1us bins of all 5 tones (10 bins x 5 tones = 50 complex regressors, + their conjugates for IQ imbalance), fitted on
NON-candidate shots only (no shot that is an A candidate on any qubit). Then compare candidates' dc robust z before and
after subtracting the prediction. Under (L) the candidates' residual dc is normal; under (G) it stays shifted.
Also reports the theoretical boxcar leakage |K_q| = |(1/500) sum_n exp(i 2 pi f_q n / fs)| per tone.
Usage: python m4conf_dccheck.py /feat /work/out
"""
import json, sys
import numpy as np

FEAT, OUT = sys.argv[1], sys.argv[2]
NQ = 5; NPER = 35000; FS = 500e6
cal = json.load(open(f"{FEAT}/calib.json"))["used_for_features"]
FQ = np.array([c["f_Hz"] for c in cal]); TH = np.array([c["theta_rad"] for c in cal])
n = np.arange(500)
K = np.array([np.exp(1j * 2 * np.pi * f * n / FS).mean() for f in FQ])

y = np.load(f"{FEAT}/y_test.npy")
F1 = np.load(f"{FEAT}/feat1us_test.npy", mmap_mode="r")
Fc = np.asarray(F1[..., 0], dtype=np.float64) + 1j * np.asarray(F1[..., 1], dtype=np.float64)   # (N,5,10)
rs = np.load(f"{OUT}/m4conf_rawstats_test.npz")
dc = rs["dcI1"].astype(np.float64) + 1j * rs["dcQ1"].astype(np.float64)
D = np.load(f"{OUT}/m4conf_pershot.npz")
spi = D["spi"]; sel = spi == 1
A = D["A"][sel]; zA = D["zA"][sel]
assert len(A) == len(y)
bits = (y[:, None] >> np.arange(NQ)) & 1
anyc = A.any(1)


def robust(x):
    m = np.median(x); return m, 1.4826 * np.median(np.abs(x - m))


Xc = Fc.reshape(len(y), -1)
res = np.empty_like(dc); pred = np.empty_like(dc)
r2 = []
for s in range(32):
    m = np.flatnonzero(y == s)
    X = Xc[m] - np.median(Xc[m].real, 0) - 1j * np.median(Xc[m].imag, 0)
    d = dc[m] - (np.median(dc[m].real) + 1j * np.median(dc[m].imag))
    Z = np.concatenate([X, X.conj()], 1)
    fit = ~anyc[m]
    beta, *_ = np.linalg.lstsq(Z[fit], d[fit], rcond=None)
    p = Z @ beta
    pred[m] = p; res[m] = d - p
    v0 = np.var(d[fit].real) + np.var(d[fit].imag); v1 = np.var((d - p)[fit].real) + np.var((d - p)[fit].imag)
    r2.append(1 - v1 / v0)
    dc[m] = d
out = {"theoretical_abs_K_boxcar_500": np.abs(K).tolist(), "R2_noncand_per_state_median": float(np.median(r2)),
       "R2_noncand_per_state_min_max": [float(np.min(r2)), float(np.max(r2))]}
for comp, f in (("dcI", np.real), ("dcQ", np.imag)):
    raw = np.empty(len(y)); rr = np.empty(len(y))
    for s in range(32):
        m = y == s
        c, sg = robust(f(dc[m])); raw[m] = (f(dc[m]) - c) / sg
        c, sg = robust(f(res[m])); rr[m] = (f(res[m]) - c) / sg
    e = {"all_shots_frac_absz_ge6_before": float((np.abs(raw) >= 6).mean()), "all_shots_frac_absz_ge6_after": float((np.abs(rr) >= 6).mean()),
         "noncand_sigma_raw_units_median_state": float(np.median([robust(f(dc[(y == s) & ~anyc]))[1] for s in range(32)])),
         "noncand_sigma_resid_units_median_state": float(np.median([robust(f(res[(y == s) & ~anyc]))[1] for s in range(32)]))}
    for q in range(NQ):
        for b in (0, 1):
            m = A[:, q] & (bits[:, q] == b)
            if m.sum() == 0:
                continue
            e[f"q{q}_prep{b}"] = {"n": int(m.sum()), "median_z_before": float(np.median(raw[m])), "median_z_after": float(np.median(rr[m])),
                                  "frac_absz_ge4_before": float((np.abs(raw[m]) >= 4).mean()), "frac_absz_ge4_after": float((np.abs(rr[m]) >= 4).mean()),
                                  "frac_absz_ge6_before": float((np.abs(raw[m]) >= 6).mean()), "frac_absz_ge6_after": float((np.abs(rr[m]) >= 6).mean())}
    out[comp] = e
json.dump(out, open(f"{OUT}/m4conf_dccheck.json", "w"), indent=1)
print(json.dumps(out, indent=1))
