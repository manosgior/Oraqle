"""vphysics: where can |2> (and |3>) sit in the IQ plane, given Lienhard et al. Tab. III/IV device parameters?

Linear dispersive readout: steady-state output S(delta) = P_inf - K / (1 + i x), x = 2*(omega_r(j) - omega_probe)/kappa.
All qubit-level responses of one resonator lie on ONE circle through the far-detuned point P_inf.
Normalise c0 = 0, c1 = 1 (complex along + i perp, in units of |c1-c0|):
    c(x) = (f(x0) - f(x)) / (f(x0) - f(x1)),   f(x) = 1/(1+ix)
x1 = x0 + s,   s = -2*(2chi)/kappa  (resonator pull |1> vs |0> = 2chi for H = chi a^dag a sigma_z)
x_j = x1 + s * r_j, r_j = (omega_r(j) - omega_r(1)) / (omega_r(1) - omega_r(0)) from the multilevel transmon
dispersive formula with Lienhard's g, omega_r, omega_q, alpha.
The probe offset x0 is unknown -> 1-parameter locus per level; mirror (conj) covers the IQ-sign convention.
Also reports the |0>-|1> chord |f(x1)-f(x0)| (signal size), so implausible probe settings can be spotted.
"""
import json, sys
import numpy as np

# Lienhard et al. PR Applied 17, 014024 (arXiv 2102.12481v2) Tab. III / IV
DEV = {  # code: (omega_r GHz, omega_q GHz, alpha MHz, g MHz, chi MHz, kappa MHz)
    0: (7.06, 5.092, -212, 116.3, 0.83, 4.29),
    1: (7.10, 4.404, -216, 143.3, 0.51, 4.25),
    2: (7.15, 5.000, -204, 125.7, 0.77, 4.41),
    3: (7.20, 4.309, -214, 133.1, 0.49, 3.33),
    4: (7.25, 5.165, -200, 125.4, 0.80, 6.90),
}
# measured third-state steady-state locations (along, perp) in units of the per-bin |0>-|1> separation,
# m3 HMM state means averaged over 100-1000 ns (bins 1..9) -- from m3time_re_feat_xt1_q*.json 'geometry'
MEAS = {0: (1.456, 0.493), 1: (2.225, 2.877), 2: (1.898, 1.513), 3: (1.996, 1.042), 4: (1.020, 0.511)}


def pulls(wr, wq, a, g, nmax=5):
    """resonator frequency pull (MHz) for transmon level j (straightforward 2nd-order multilevel formula)"""
    wr, wq = wr * 1e3, wq * 1e3
    chi = [((k + 1) * g * g) / ((wq + k * a) - wr) for k in range(nmax + 1)]  # chi_{k,k+1}
    d = []
    for j in range(nmax):
        d.append((chi[j - 1] if j > 0 else 0.0) - chi[j])
    return np.array(d)


def f(x):
    return 1.0 / (1.0 + 1j * x)


def locus(s, r, x0):
    x1 = x0 + s
    xj = x1 + s * r
    return (f(x0) - f(xj)) / (f(x0) - f(x1)), np.abs(f(x1) - f(x0))


out = {}
x0g = np.linspace(-8, 8, 32001)
for q, (wr, wq, a, g, chi, kap) in DEV.items():
    d = pulls(wr, wq, a, g)
    r2 = (d[2] - d[1]) / (d[1] - d[0])
    r3 = (d[3] - d[1]) / (d[1] - d[0])
    r4 = (d[4] - d[1]) / (d[1] - d[0])
    res = {"pull_formula_MHz": d.tolist(), "2chi_formula_MHz": float(d[1] - d[0]), "2chi_table_MHz": 2 * chi,
           "r2": float(r2), "r3": float(r3), "r4": float(r4)}
    meas = MEAS[q][0] + 1j * MEAS[q][1]
    for conv, s in (("2chi=table(H=chi a+a sz)", -4 * chi / kap), ("chi_total=table", -2 * chi / kap)):
        for lev, r in (("|2>", r2), ("|3>", r3), ("|4>", r4)):
            for mir in (1, -1):
                c, chord = locus(s, r, x0g)
                if mir == -1:
                    c = np.conj(c)
                dist1 = np.abs(c - 1)
                miss = np.abs(c - meas)
                i = int(np.argmin(miss))
                imid = int(np.argmin(np.abs(x0g - (-s / 2))))  # probe midway between |0> and |1>
                res[f"{conv}|{lev}|mirror{mir}"] = {
                    "s": s, "max_dist_from_c1_sep": float(dist1.max()),
                    "x0_at_max": float(x0g[int(np.argmax(dist1))]),
                    "chord_rel_to_midpoint_at_max": float(chord[int(np.argmax(dist1))] / chord[imid]),
                    "best_x0": float(x0g[i]), "best_pred": [float(c[i].real), float(c[i].imag)],
                    "best_miss_sep": float(miss[i]), "chord_rel_to_midpoint_at_best": float(chord[i] / chord[imid]),
                    "pred_if_probe_midpoint": [float(c[imid].real), float(c[imid].imag)],
                }
    res["measured_L"] = MEAS[q]
    res["measured_dist_from_c1_sep"] = float(abs(meas - 1))
    out[f"q{q}"] = res

json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "vphysics_dispersive_locus.json", "w"), indent=1)
for q in range(5):
    o = out[f"q{q}"]
    print(f"q{q} (Q{q+1}) r2={o['r2']:.3f} r3={o['r3']:.3f} 2chi formula={o['2chi_formula_MHz']:.2f} table={o['2chi_table_MHz']:.2f}; "
          f"measured L {o['measured_L']} dist_c1={o['measured_dist_from_c1_sep']:.2f}")
    for k, v in o.items():
        if "|mirror" in k:
            print(f"   {k:40s} maxdist={v['max_dist_from_c1_sep']:.2f} (x0={v['x0_at_max']:.2f}, chord {v['chord_rel_to_midpoint_at_max']:.2f}x mid) "
                  f"best miss={v['best_miss_sep']:.2f} at x0={v['best_x0']:.2f} pred={np.round(v['best_pred'],2)} chord {v['chord_rel_to_midpoint_at_best']:.2f}x; mid-probe pred={np.round(v['pred_if_probe_midpoint'],2)}")
