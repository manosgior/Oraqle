"""vphysics_circle.py -- model-light resonance-circle geometry of the class-mean responses (tag vphysics).
Steady states from class-mean raw traces (HERQULES IF, DC removed, box11*box13 filter):
  c0 = core0, c1 = core1 over 300-900 ns; L = Lstart over 500-800 ns (bins 5-7, not used for its selection) and
  Llate over 1.6-2.0 us. Early direction theta_E = angle of the common trajectory at 8-24 ns (direct path, P_inf ray).
Circle through c0, c1, L; P_inf = far intersection of the theta_E ray with it; resonance point = antipode of P_inf.
Each point's Moebius coordinate x = tan(-psi/2), psi = angle from the resonance point (x = 2 delta/kappa).
Gives data-driven s = x1 - x0 (Lienhard 4chi/kappa), level pull r = (xL - x1)/s and the Fano angle phi implied."""
import json, sys
import numpy as np
D = sys.argv[1]; OUTJ = sys.argv[2]
d = np.load(f'{D}/vphysics_classmeans.npz'); R = json.load(open(f'{D}/vphysics_raw_classes.json'))
FH = np.array(R['F_HERQ']); TH = np.array(R['TH_OURS'])
t = np.arange(1000) * 2e-9
K = np.convolve(np.ones(11) / 11, np.ones(13) / 13)
S_L = {0: 4 * 0.83 / 4.29, 1: 4 * 0.51 / 4.25, 2: 4 * 0.77 / 4.41, 3: 4 * 0.49 / 3.33, 4: 4 * 0.80 / 6.90}
R2 = {0: 0.734, 1: 0.793, 2: 0.761, 3: 0.807, 4: 0.759}; R3 = {0: 1.289, 1: 1.432, 2: 1.353, 3: 1.467, 4: 1.348}

def z(key, cl, q):
    x = d[f'{key}__{cl}']; x = x - x.mean()
    return np.convolve(x * np.exp(-1j * 2 * np.pi * FH[q] * t) * np.exp(1j * TH[q]), K, mode='same')

def circ(a, b, c):
    ax, ay, bx, by, cx, cy = a.real, a.imag, b.real, b.imag, c.real, c.imag
    dd = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / dd
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / dd
    C = ux + 1j * uy
    return C, abs(a - C)

out = {}
for key, q, cls in [('q2_y0_4', 2, [('Lstart', 250, 400), ('Llate', 800, 1000), ('L0', 500, 950)]),
                    ('q4_y4_20', 4, [('Lstart', 250, 400), ('Llate', 800, 1000)]),
                    ('q4_y0_16', 4, [('Lstart', 250, 400), ('Llate', 800, 1000)]),
                    ('q0_y4_5', 0, [('Llate', 800, 1000)])]:
    z0 = z(key, 'core0', q); z1 = z(key, 'core1', q)
    c0 = z0[150:450].mean(); c1 = z1[150:450].mean()
    early = {n: float(np.degrees(np.angle((z0[n] + z1[n]) / 2))) for n in (4, 6, 8, 10, 12)}
    o = {'c0': [c0.real, c0.imag], 'c1': [c1.real, c1.imag], 'early_angle_deg_by_sample': early, 'third': {}}
    for cl, a, b in cls:
        if f'{key}__{cl}' not in d:
            continue
        zl = z(key, cl, q); L = zl[a:b].mean()
        C, Rr = circ(c0, c1, L)
        res = {'L': [L.real, L.imag], 'L_rel_sep': [((L - c0) / (c1 - c0)).real, ((L - c0) / (c1 - c0)).imag],
               'centre': [C.real, C.imag], 'radius': Rr, 'by_thetaE': {}}
        for thE in np.arange(10, 61, 5):
            u = np.exp(1j * np.radians(thE))
            # intersections of ray s*u (s>0) with circle |p - C| = R
            bq = -2 * (np.conj(u) * C).real; cq = abs(C) ** 2 - Rr ** 2
            disc = bq ** 2 - 4 * cq
            if disc < 0:
                res['by_thetaE'][int(thE)] = None; continue
            ss = np.array([(-bq + np.sqrt(disc)) / 2, (-bq - np.sqrt(disc)) / 2])
            ss = ss[ss > 0]
            if len(ss) == 0:
                res['by_thetaE'][int(thE)] = None; continue
            P = ss.max() * u
            Cres = 2 * C - P
            ang = lambda S: np.angle((S - C) / (Cres - C))
            xs = {nm: float(np.tan(-ang(S) / 2)) for nm, S in (('x0', c0), ('x1', c1), ('xL', L))}
            s = xs['x1'] - xs['x0']; r = (xs['xL'] - xs['x1']) / s
            phi = float(np.degrees(np.angle((C - P) / (-P))))
            res['by_thetaE'][int(thE)] = dict(P=[P.real, P.imag], **xs, s=s, r=r, phi_deg=phi, Q=float(2 * Rr / abs(P)))
        o['third'][cl] = res
        print(f"{key} {cl}: L rel sep ({res['L_rel_sep'][0]:.2f},{res['L_rel_sep'][1]:.2f}); early angle {early}; Lienhard s={S_L[q]:.2f} r2={R2[q]} r3={R3[q]}")
        for th, v in res['by_thetaE'].items():
            if v is None:
                print(f'    thetaE {th}: ray misses circle'); continue
            print(f"    thetaE {th:2d}: x0={v['x0']:+.2f} x1={v['x1']:+.2f} xL={v['xL']:+.2f}  s={v['s']:+.2f}  r={v['r']:+.2f}  phi={v['phi_deg']:+.0f} deg  Q={v['Q']:.2f}")
    out[key] = o
json.dump(out, open(OUTJ, 'w'), indent=1, default=float)
