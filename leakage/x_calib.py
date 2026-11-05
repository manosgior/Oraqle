"""x_calib.py -- step 1: per-qubit IF frequency f_q and rotation theta_q.

* Runs the VERBATIM calibrate() of MCMit _colleague_prep.py (copied unchanged to
  x_colleague_prep_verbatim.py) on the TRAIN file for time_slice (0,1000) and (0,500).
  calibrate() uses 500 shots per prepared state (rows s*15000+5000 .. +500), i.e.
  16000 shots, 8000 with bit q = 0 and 8000 with bit q = 1 for every qubit.
* Diagnoses the colleague theta: calibrate() computes theta = -angle(m1 - m0) from
  the mean of the RAW (not demodulated) complex trace, so it is NOT the angle of
  the |0>->|1> axis of the demodulated signal. We measure the angle the
  |1>-|0> centroid difference actually has after demodulation + FIR + rotation.
* Frequency refinement (used for the features): calibrate() returns the FFT-bin
  frequency (0.5 MHz grid for 1000 samples, 1 MHz for 500). All five IFs lie
  0.1-0.26 MHz off that grid, so the demodulated |1>-|0> axis rotates by up to
  ~190 deg over the 2 us trace. Starting from the calibrate(0,1000) value we
  iterate f += slope/(2 pi), slope = |d|-weighted linear fit of the unwrapped
  phase of the mean difference trajectory d(t) = <z|1>(t) - <z|0>(t), samples
  100..899 (z = mixed + FIR). Converges in 1-2 iterations, identical from the
  0-500 start; residual on independent TEST shots < 1 kHz.
* Corrected theta (used for the features): theta_q = -angle(mean_t d(t)) over
  samples 0..999 at the refined f, i.e. the window-averaged |1>-|0> centroid
  difference lies on +real.
Writes calib.json to the output dir.
"""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json, os, sys, time
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x_colleague_prep_verbatim as P
from x_common import FS, NUM_QUBITS, TAPS, NUMTAPS, CUTOFF, BIN, FILES, fir_same

OUT = sys.argv[1] if len(sys.argv) > 1 else str(LP.FEAT)
path, kx, ky, sps = FILES["train"]


def load_cal_shots():
    """Exactly the calibration rows used by calibrate(): 500 per state from offset 5000."""
    Xs, ys, idx = [], [], []
    with h5py.File(path, "r") as hf:
        for s in range(32):
            a = s * sps + 5000
            Xs.append(hf[kx][a:a + 500]); ys.append(hf[ky][a:a + 500]); idx.append(np.arange(a, a + 500))
    return np.concatenate(Xs), np.concatenate(ys).astype(int), np.concatenate(idx)


def wrap_deg(a):
    return float((np.degrees(a) + 180) % 360 - 180)


def main():
    t0 = time.time()
    res = {"fs_assumed_Hz": FS, "fir": {"numtaps": NUMTAPS, "cutoff_Hz": CUTOFF, "window": "hamming (firwin default)",
                                         "taps": TAPS.tolist()}}
    cal = {}
    for name, sl in [("full_0_1000", (0, 1000)), ("first_0_500", (0, 500))]:
        prm = P.calibrate(path, kx, ky, sps, sl)
        cal[name] = [{"f_Hz": float(p["f"]), "theta_colleague_rad": float(p["theta"])} for p in prm]
        print(f"[calib] colleague calibrate{sl}: " +
              ", ".join(f"q{q}: f={p['f']/1e6:+.3f} MHz th={p['theta']:+.4f}" for q, p in enumerate(prm)), flush=True)
    res["colleague_calibrate"] = cal
    res["fft_bin_resolution_Hz"] = {"full_0_1000": FS / 1000, "first_0_500": FS / 500}

    X, y, idx = load_cal_shots()
    Xc = X[:, :, 0] + 1j * X[:, :, 1]
    del X
    # independent validation shots: TEST file, rows s*35000 .. s*35000+499 (never used for fitting)
    pte, kxte, kyte, spste = FILES["test"]
    with h5py.File(pte, "r") as hf:
        Xv = np.concatenate([hf[kxte][s * spste: s * spste + 500] for s in range(32)])
        yv = np.concatenate([hf[kyte][s * spste: s * spste + 500] for s in range(32)]).astype(int)
    Xvc = Xv[:, :, 0] + 1j * Xv[:, :, 1]
    del Xv
    t = np.arange(Xc.shape[1]) / FS
    SS = slice(100, 900)          # steady-state samples for the phase-slope fit (skip ring-up + FIR edges)

    def traj(Z, b, f):
        filt = fir_same(Z * np.exp(-1j * 2 * np.pi * f * t)[None, :])
        return filt[b == 0].mean(0), filt[b == 1].mean(0)

    def slope_hz(v):
        ph = np.unwrap(np.angle(v[SS])); w = np.abs(v[SS]); tt = t[SS]
        A = np.vstack([tt, np.ones_like(tt)]).T * w[:, None]
        return float(np.linalg.lstsq(A, ph * w, rcond=None)[0][0] / (2 * np.pi))

    def zp_fft_peak(Z, b, f):
        nfft = 16384
        F0 = np.abs(np.fft.fft(Z[b == 0], n=nfft, axis=1)).mean(0)
        F1 = np.abs(np.fft.fft(Z[b == 1], n=nfft, axis=1)).mean(0)
        fr = np.fft.fftfreq(nfft, 1 / FS); win = np.abs(fr - f) < 1.5e6
        return float(fr[win][np.argmax(np.abs(F1 - F0)[win])])

    def describe(Z, b, f, theta):
        m0, m1 = traj(Z, b, f); d = m1 - m0
        r = np.exp(1j * theta)
        dbin = d.reshape(20, BIN).mean(1) * r
        return {"residual_df_from_d_Hz": slope_hz(d), "residual_df_from_mean0_Hz": slope_hz(m0),
                "residual_df_from_mean1_Hz": slope_hz(m1),
                "angle_c1_minus_c0_fullwindow_deg": wrap_deg(np.angle(d.mean() * r)),
                "angle_c1_minus_c0_first500_deg": wrap_deg(np.angle(d[:500].mean() * r)),
                "angle_d_per_bin_deg": [wrap_deg(np.angle(v)) for v in dbin],
                "abs_d_per_bin": [float(abs(v)) for v in dbin]}

    diag = {}
    used = []
    for q in range(NUM_QUBITS):
        b = (y >> q) & 1; bv = (yv >> q) & 1
        dq = {}
        # (A) exact colleague calibrate() output, as is
        for name in ["full_0_1000", "first_0_500"]:
            f = cal[name][q]["f_Hz"]; th_c = cal[name][q]["theta_colleague_rad"]
            dq[f"colleague_{name}_as_is"] = describe(Xc, b, f, th_c)
            dq[f"colleague_{name}_as_is"]["phase_drift_deg_over_2us"] = \
                dq[f"colleague_{name}_as_is"]["residual_df_from_d_Hz"] * 360 * 2e-6
        # (B) refinement: start from calibrate(0,1000) f, iterate phase-slope correction of d(t)
        refine = {}
        for start in ["full_0_1000", "first_0_500"]:
            f = cal[start][q]["f_Hz"]; hist = [f]
            for it in range(4):
                m0, m1 = traj(Xc, b, f)
                f = f + slope_hz(m1 - m0); hist.append(f)
            refine[start] = hist
        f_ref = refine["full_0_1000"][-1]
        m0, m1 = traj(Xc, b, f_ref); d = m1 - m0
        th_full = float(-np.angle(d.mean())); th_1us = float(-np.angle(d[:500].mean()))
        dq["refined"] = {"f_start_Hz": cal["full_0_1000"][q]["f_Hz"], "f_iterates_from_full_0_1000_Hz": refine["full_0_1000"],
                         "f_iterates_from_first_0_500_Hz": refine["first_0_500"],
                         "f_refined_Hz": f_ref, "theta_fullwindow_rad": th_full, "theta_first500_rad": th_1us,
                         "theta_first500_minus_full_deg": wrap_deg(th_1us - th_full),
                         "zero_padded_fft_peak_Hz": zp_fft_peak(Xc, b, cal["full_0_1000"][q]["f_Hz"]),
                         "train_cal_shots": describe(Xc, b, f_ref, th_full),
                         "test_validation_shots": describe(Xvc, bv, f_ref, th_full)}
        R = dq["refined"]
        print(f"[calib] q{q} (paper Q{q+1}): calibrate(0,1000) f={cal['full_0_1000'][q]['f_Hz']/1e6:+.3f} MHz -> "
              f"drift {dq['colleague_full_0_1000_as_is']['phase_drift_deg_over_2us']:+.1f} deg/2us, colleague theta leaves axis at "
              f"{dq['colleague_full_0_1000_as_is']['angle_c1_minus_c0_fullwindow_deg']:+.1f} deg", flush=True)
        print(f"        refined f={f_ref/1e6:+.4f} MHz (from 0-500 start: {refine['first_0_500'][-1]/1e6:+.4f}; zp-FFT {R['zero_padded_fft_peak_Hz']/1e6:+.4f})"
              f"  theta={th_full:+.4f} rad (first500 differs by {R['theta_first500_minus_full_deg']:+.1f} deg)", flush=True)
        for nm in ["train_cal_shots", "test_validation_shots"]:
            D = R[nm]
            print(f"        {nm}: residual df d/m0/m1 = {D['residual_df_from_d_Hz']/1e3:+.2f}/{D['residual_df_from_mean0_Hz']/1e3:+.2f}/"
                  f"{D['residual_df_from_mean1_Hz']/1e3:+.2f} kHz; axis angle full={D['angle_c1_minus_c0_fullwindow_deg']:+.2f} "
                  f"first500={D['angle_c1_minus_c0_first500_deg']:+.2f} deg", flush=True)
            print(f"          per-bin angle: {np.round(D['angle_d_per_bin_deg'], 1).tolist()}", flush=True)
        print(f"          per-bin |d|  : {np.round(R['train_cal_shots']['abs_d_per_bin'], 2).tolist()}", flush=True)
        diag[f"q{q}"] = dq
        used.append({"qubit_code": q, "qubit_paper": f"Q{q+1}", "f_Hz": f_ref, "theta_rad": th_full,
                     "f_calibrate_0_1000_Hz": cal["full_0_1000"][q]["f_Hz"],
                     "f_calibrate_0_500_Hz": cal["first_0_500"][q]["f_Hz"],
                     "theta_colleague_0_1000_rad": cal["full_0_1000"][q]["theta_colleague_rad"]})
    res["diagnostics"] = diag
    res["used_for_features"] = used
    res["used_for_features_note"] = (
        "f_Hz = colleague calibrate() on TRAIN with time_slice (0,1000) (quantised to the 0.5 MHz FFT grid), then "
        "refined by 4 iterations of f += slope/(2 pi), slope = |d|-weighted linear fit of the unwrapped phase of the "
        "mean difference trajectory d(t) = <z|bit=1>(t) - <z|bit=0>(t) over samples 100..899, z = mixed + 101-tap FIR. "
        "theta_rad = -angle(mean over samples 0..999 of d(t)) at the refined f. Both are fitted on the 16000 calibration "
        "shots of calibrate() and validated on 16000 independent TEST shots. The unrefined calibrate() f leaves a "
        "linear phase drift of the |0>-|1> axis of up to ~190 deg over 2 us, and the colleague theta (computed from the "
        "undemodulated trace mean) does not align the axis; both are kept here for reference only.")
    res["calibration_rows"] = "train rows s*15000+5000 .. s*15000+5499 for s=0..31 (16000 shots)"
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "calib.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[calib] wrote {os.path.join(OUT, 'calib.json')} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
