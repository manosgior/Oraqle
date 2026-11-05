"""vphysics: for each measured third-state location, solve the linear-dispersive circle model for the
probe offset x0 and the implied level pull r = (omega_r(L)-omega_r(1))/(omega_r(1)-omega_r(0)), given s=2(2chi)/kappa.
Prints all exact (miss<0.02 sep) solutions and the |0>-|1> signal chord relative to a midpoint probe."""
import numpy as np, json, sys
DEV = {0: (0.83, 4.29), 1: (0.51, 4.25), 2: (0.77, 4.41), 3: (0.49, 3.33), 4: (0.80, 6.90)}
MEAS = {"q0_HMM": (0, 1.456, 0.493), "q0_m2": (0, 1.44, 0.53), "q0_m1": (0, 1.60, 0.86),
        "q1_HMM": (1, 2.225, 2.877),
        "q2_HMM": (2, 1.898, 1.513), "q2_m2far": (2, 1.73, 1.08), "q2_endcloudADC": (2, 2.38, 1.32),
        "q3_HMM_L": (3, 1.996, 1.042), "q3_blob": (3, 0.41, -0.22),
        "q4_HMM": (4, 1.020, 0.511), "q4_m2": (4, 1.03, 0.48)}
f = lambda x: 1 / (1 + 1j * x)
xg = np.linspace(-15, 15, 6001)
out = {}
for name, (q, a, p) in MEAS.items():
    chi, kap = DEV[q]
    meas = a + 1j * p
    for conv, s in (("2chi", -4 * chi / kap), ("chi", -2 * chi / kap), ("2chi_x2", -8 * chi / kap)):
        sols = []
        for mir in (1, -1):
            m = meas if mir == 1 else np.conj(meas)
            x0 = xg[:, None]; x1 = x0 + s; xL = xg[None, :]
            c = (f(x0) - f(xL)) / (f(x0) - f(x1))
            miss = np.abs(c - m)
            j = miss.argmin(1)
            mm = miss[np.arange(len(xg)), j]
            # local minima along x0
            for i in range(1, len(xg) - 1):
                if mm[i] <= mm[i - 1] and mm[i] <= mm[i + 1] and mm[i] < 0.05:
                    x0v = xg[i]; xLv = xg[j[i]]
                    chord = abs(f(x0v + s) - f(x0v)) / abs(f(s / 2) - f(-s / 2))
                    sols.append(dict(mirror=mir, x0=float(x0v), xL=float(xLv), r=float((xLv - (x0v + s)) / s),
                                     miss=float(mm[i]), chord_vs_mid=float(chord)))
        out[f"{name}|{conv}"] = sols
        print(name, conv, f"s={s:.3f}", [(d['mirror'], round(d['x0'], 2), 'r=%.2f' % d['r'], 'chord %.2f' % d['chord_vs_mid']) for d in sols])
json.dump(out, open(sys.argv[1], "w"), indent=1)
