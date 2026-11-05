"""vphysics_ringup.py -- linear-resonator (dispersive) model of the class-mean readout trajectories (tag vphysics).

Input: vphysics_classmeans.npz / vphysics_raw_classes.json from vphysics_raw.py (graham).
For one 5-bit configuration and qubit q:
  data z_c(t): class-mean raw trace, DC removed, mixed with the HERQULES IF (verified flat to ~1 kHz for |0>),
  rotated by the shared theta, filtered with box11*box13 (nulls the other tones to <= -28 dB; 48 ns span).
  model  S_c(t) = A eps(t) - B alpha_c(t),  d alpha/dt = -(kappa/2 + i delta_c) alpha + eps(t)
         eps(t) = cosine-edged step (t0, tau_e); same filter applied to the model.
Step 1: fit A, B (complex), kappa, delta_0, delta_1, t0, tau_e to core0 + core1 (0-900 ns).
Step 2: third-state class (Lstart: in L at 200-500 ns and 800-1000 ns): for a grid of delta_L, NNLS over
        entry times tau in {0 (= in L at pulse start), 20, ..., 500 ns} of |1> -> L trajectories; best delta_L.
        Reports: implied level pull r = (delta_L - delta_1)/(delta_1 - delta_0) (transmon |2>: ~0.76, |3>: ~1.35
        from Lienhard Tab. III/IV), the steady-state miss of the measured L point from the dispersive circle,
        and the weight at tau = 0 (present at pulse start) vs later entries.
"""
import json, sys
import numpy as np
from scipy.optimize import least_squares, nnls

D = sys.argv[1] if len(sys.argv) > 1 else '.'
OUTJ = sys.argv[2] if len(sys.argv) > 2 else 'vphysics_ringup.json'
d = np.load(f'{D}/vphysics_classmeans.npz'); R = json.load(open(f'{D}/vphysics_raw_classes.json'))
FH = np.array(R['F_HERQ']); TH = np.array(R['TH_OURS'])
DT = 2e-9; N = 1000; t = np.arange(N) * DT
K = np.convolve(np.ones(11) / 11, np.ones(13) / 13)
filt = lambda x: np.convolve(x, K, mode='same')
NFIT = 450


def data(key, cl, q):
    z = d[f'{key}__{cl}']; z = z - z.mean()
    return filt(z * np.exp(-1j * 2 * np.pi * FH[q] * t) * np.exp(1j * TH[q]))


def envelope(t0, te):
    tt = (t - t0)
    e = np.where(tt < 0, 0.0, np.where(tt < te, 0.5 * (1 - np.cos(np.pi * np.clip(tt, 0, None) / max(te, 1e-12))), 1.0))
    return e


def alpha_traj(eps, kap, deltas, switch=None):
    """deltas: scalar or (delta_before, delta_after) with switch index"""
    a = np.zeros(N, complex)
    for n in range(1, N):
        dl = deltas if switch is None else (deltas[0] if n <= switch else deltas[1])
        lam = kap / 2 + 1j * dl
        e = np.exp(-lam * DT)
        a[n] = a[n - 1] * e + eps[n - 1] * (1 - e) / lam
    return a


def model(p, dels, switch=None):
    t0, te, kap, Ar, Ai, Br, Bi = p
    eps = envelope(t0, te)
    a = alpha_traj(eps, kap, dels, switch)
    return filt((Ar + 1j * Ai) * eps - (Br + 1j * Bi) * a * 1e6)  # alpha in units of 1/(1e6 s^-1)


def fit_core(z0, z1, kap0):
    def res(v):
        p = v[:7]; d0, d1 = v[7], v[8]
        r0 = model(p, d0)[:NFIT] - z0[:NFIT]; r1 = model(p, d1)[:NFIT] - z1[:NFIT]
        return np.concatenate([r0.real, r0.imag, r1.real, r1.imag])
    best = None
    for d0 in (-6e6, -2e6, 2e6, 6e6):
        for d1s in (-1, 1):
            v0 = [5e-9, 20e-9, kap0, 10, 5, 1, 1, d0, d0 + d1s * 5e6]
            try:
                s = least_squares(res, v0, x_scale='jac', max_nfev=3000)
            except Exception:
                continue
            if best is None or s.cost < best.cost:
                best = s
    return best


out = {}
JOBS = [('q2_y0_4', 2, 'Lstart', 'Llate', 'L0'), ('q4_y4_20', 4, 'Lstart', 'Llate', None), ('q0_y4_5', 0, 'Lstart', 'Llate', None)]
KAP_L = {0: 4.29e6, 1: 4.25e6, 2: 4.41e6, 3: 3.33e6, 4: 6.90e6}
CHI_L = {0: 0.83e6, 1: 0.51e6, 2: 0.77e6, 3: 0.49e6, 4: 0.80e6}
R2 = {0: 0.734, 1: 0.793, 2: 0.761, 3: 0.807, 4: 0.759}; R3 = {0: 1.289, 1: 1.432, 2: 1.353, 3: 1.467, 4: 1.348}
for key, q, Lcl, Llate, L0 in JOBS:
    z0 = data(key, 'core0', q); z1 = data(key, 'core1', q)
    s = fit_core(z0, z1, 2 * np.pi * KAP_L[q])
    p = s.x[:7]; d0, d1 = s.x[7], s.x[8]
    kap = p[2]
    m0 = model(p, d0); m1 = model(p, d1)
    rms = np.sqrt(np.mean(np.abs(np.concatenate([m0[:NFIT] - z0[:NFIT], m1[:NFIT] - z1[:NFIT]])) ** 2))
    sep = abs(z1[150:450].mean() - z0[150:450].mean())
    o = dict(t0_ns=p[0] * 1e9, tau_edge_ns=p[1] * 1e9, kappa_over_2pi_MHz=kap / 2 / np.pi / 1e6,
             delta0_over_2pi_MHz=d0 / 2 / np.pi / 1e6, delta1_over_2pi_MHz=d1 / 2 / np.pi / 1e6,
             pull_1_minus_0_MHz=(d1 - d0) / 2 / np.pi / 1e6, x0=2 * d0 / kap, x1=2 * d1 / kap,
             s_data=2 * (d1 - d0) / kap, s_lienhard=4 * CHI_L[q] / KAP_L[q],
             kappa_lienhard_MHz=KAP_L[q] / 1e6, rms_resid_core=rms, sep_ss=sep,
             P_inf=[float(p[3]), float(p[4])])
    print(f'{key}: kappa/2pi={o["kappa_over_2pi_MHz"]:.2f} MHz (Lienhard {KAP_L[q]/1e6}), pull(1-0)={o["pull_1_minus_0_MHz"]:+.3f} MHz '
          f'(Lienhard 2chi={2*CHI_L[q]/1e6:.2f}), x0={o["x0"]:+.2f} x1={o["x1"]:+.2f}, t0={o["t0_ns"]:.1f} ns edge={o["tau_edge_ns"]:.1f} ns, '
          f'rms={rms:.3f} vs sep {sep:.2f}', flush=True)
    # ---- third state ----
    for cl, taus in ((Lcl, np.arange(0, 520e-9, 20e-9)), (Llate, np.arange(0, 1800e-9, 50e-9)), (L0, np.arange(0, 1800e-9, 50e-9))):
        if cl is None or f'{key}__{cl}' not in d:
            continue
        zL = data(key, cl, q)
        base_from = d1 if cl != 'L0' else d0
        nmax = 500 if cl == Lcl else 900
        grid = np.linspace(d1 - 12 * abs(d1 - d0), d1 + 12 * abs(d1 - d0), 241)
        best = None
        for dL in grid:
            cols = []
            for tau in taus:
                sw = int(round(tau / DT))
                cols.append(model(p, (base_from, dL), switch=sw) if sw > 0 else model(p, dL))
            # allow also shots that never enter (stay in origin state) -- absorbs selection impurity
            cols.append(model(p, base_from))
            Am = np.array(cols).T[:nmax]
            Ar = np.vstack([Am.real, Am.imag]); br = np.concatenate([zL[:nmax].real, zL[:nmax].imag])
            w, rn = nnls(Ar, br)
            if best is None or rn < best[1]:
                best = (dL, rn, w)
        dL, rn, w = best
        cols_ss = model(p, dL)
        resid_rms = rn / np.sqrt(nmax)
        r_impl = (dL - d1) / (d1 - d0)
        # steady-state geometry: measured L late mean vs model circle
        o[cl] = dict(deltaL_over_2pi_MHz=dL / 2 / np.pi / 1e6, r_implied=r_impl, r2_transmon=R2[q], r3_transmon=R3[q],
                     xL=2 * dL / kap, rms_resid=resid_rms, weights_sum=float(w.sum()),
                     w_tau0=float(w[0]), w_by_tau={f'{tau*1e9:.0f}': float(wi) for tau, wi in zip(taus, w[:-1]) if wi > 1e-3},
                     w_never=float(w[-1]), n_shots=R['classes'][key]['n'][cl],
                     L_ss_model=[float(cols_ss[300].real), float(cols_ss[300].imag)],
                     L_data_200_500ns=[float(zL[100:250].real.mean()), float(zL[100:250].imag.mean())])
        print(f'   {cl}: n={o[cl]["n_shots"]} best delta_L/2pi={dL/2/np.pi/1e6:+.3f} MHz -> r={r_impl:.2f} (|2> {R2[q]}, |3> {R3[q]}); '
              f'rms resid={resid_rms:.3f}; w(tau=0)={w[0]:.3f}, sum w={w.sum():.3f}, never={w[-1]:.3f}; '
              f'w by tau: {o[cl]["w_by_tau"]}', flush=True)
        # residual of a forced |2> (r = R2) and |3> fit for comparison
        for lab, rr in (('forced_r2', R2[q]), ('forced_r3', R3[q])):
            dLf = d1 + rr * (d1 - d0)
            cols = [model(p, (base_from, dLf), switch=int(round(tau / DT))) if tau > 0 else model(p, dLf) for tau in taus] + [model(p, base_from)]
            Am = np.array(cols).T[:nmax]
            w2, rn2 = nnls(np.vstack([Am.real, Am.imag]), np.concatenate([zL[:nmax].real, zL[:nmax].imag]))
            o[cl][lab] = dict(rms_resid=rn2 / np.sqrt(nmax), w_tau0=float(w2[0]), w_sum=float(w2.sum()))
            print(f'      {lab}: rms resid={rn2/np.sqrt(nmax):.3f} (free {resid_rms:.3f})', flush=True)
    out[key] = o
    # save traces for plotting
    np.savez(OUTJ.replace('.json', f'_{key}.npz'), z0=z0, z1=z1, m0=m0, m1=m1,
             **{f'z_{c}': data(key, c, q) for c in (Lcl, Llate, L0) if c and f'{key}__{c}' in d})
json.dump(out, open(OUTJ, 'w'), indent=1, default=float)
