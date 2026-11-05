"""vphysics_circle2.py -- resonance-circle test for every qubit/class (tag vphysics); generalises vphysics_circle.py.
Per qubit: qubit-tailored boxcar cascade nulling every other tone and the raw DC (all <= -40 dB except noted),
HERQULES IF, DC removed. theta_E = angle of the common (core0+core1)/2 trajectory, averaged over samples 6-12
(first 12-24 ns, before the resonator response dominates); scanned +-15 deg around it.
For each third-state class: circle through c0, c1, L; P_inf on the theta_E ray; implied s, r, Fano phi, Q."""
import json, sys
import numpy as np
D = sys.argv[1]; OUTJ = sys.argv[2]
SETS = [('vphysics_classmeans2.npz', 'vphysics_raw_classes2.json'), ('vphysics_classmeans.npz', 'vphysics_raw_classes.json')]  # rerun (noise-scaled cores) first
JOBS = [('q2_y0_4', 2, ['Lstart', 'Llate', 'L0']), ('q2_y16_20', 2, ['Lstart', 'Llate', 'L0']),
        ('q4_y4_20', 4, ['Lstart', 'Llate']), ('q4_y0_16', 4, ['Lstart', 'Llate']),
        ('q0_y4_5', 0, ['Lstart', 'Llate']), ('q1_y0_2', 1, ['Lstart', 'Llate']), ('q1_y4_6', 1, ['Lstart', 'Llate']),
        ('q3_y0_8', 3, ['Q4k0', 'Q4k1', 'Q4k2'])]
R2 = {0: 0.734, 1: 0.793, 2: 0.761, 3: 0.807, 4: 0.759}; R3 = {0: 1.289, 1: 1.432, 2: 1.353, 3: 1.467, 4: 1.348}
RBARE = {}
DEV = {0: (7.06, 5.092, -212, 116.3), 1: (7.10, 4.404, -216, 143.3), 2: (7.15, 5.000, -204, 125.7), 3: (7.20, 4.309, -214, 133.1), 4: (7.25, 5.165, -200, 125.4)}
for q, (wr, wq, a, g) in DEV.items():
    wr *= 1e3; wq *= 1e3
    chi = [((k + 1) * g * g) / ((wq + k * a) - wr) for k in range(3)]
    d0 = -chi[0]; d1 = chi[0] - chi[1]
    RBARE[q] = (0 - d1) / (d1 - d0)
t = np.arange(1000) * 2e-9

def load(key):
    for npz, js in SETS:
        d = np.load(f'{D}/{npz}'); R = json.load(open(f'{D}/{js}'))
        if f'{key}__core0' in d:
            return d, R
    raise KeyError(key)

def kern(FH, q):
    offs = [FH[j] - FH[q] for j in range(5) if j != q] + [-FH[q]]
    k = np.array([1.0])
    for o in offs:
        L = int(round(500e6 / abs(o)))
        while L > 40:
            L //= 2
        k = np.convolve(k, np.ones(L) / L)
    resp = [float(20 * np.log10(abs(np.sum(k * np.exp(-1j * 2 * np.pi * o * np.arange(len(k)) * 2e-9))) + 1e-15)) for o in offs]
    return k, resp

def circ(a, b, c):
    ax, ay, bx, by, cx, cy = a.real, a.imag, b.real, b.imag, c.real, c.imag
    dd = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / dd
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / dd
    C = ux + 1j * uy
    return C, abs(a - C)

def geom(C, Rr, c0, c1, L, th):
    u = np.exp(1j * np.radians(th)); bq = -2 * (np.conj(u) * C).real; cq = abs(C) ** 2 - Rr ** 2; disc = bq ** 2 - 4 * cq
    if disc < 0:
        return None
    ss = [s for s in ((-bq + np.sqrt(disc)) / 2, (-bq - np.sqrt(disc)) / 2) if s > 0]
    if not ss:
        return None
    P = max(ss) * u; Cres = 2 * C - P
    ang = lambda S: np.angle((S - C) / (Cres - C))
    x0, x1, xL = [float(np.tan(-ang(S) / 2)) for S in (c0, c1, L)]
    return dict(x0=x0, x1=x1, xL=xL, s=x1 - x0, r=(xL - x1) / (x1 - x0), phi_deg=float(np.degrees(np.angle((C - P) / (-P)))),
                Q=float(2 * Rr / abs(P)), P=[float(P.real), float(P.imag)])

out = {}
for key, q, cls in JOBS:
    d, R = load(key)
    FH = np.array(R['F_HERQ']); TH = np.array(R['TH_OURS'])
    k, resp = kern(FH, q)
    z = lambda cl: np.convolve((d[f'{key}__{cl}'] - d[f'{key}__{cl}'].mean()) * np.exp(-1j * 2 * np.pi * FH[q] * t) * np.exp(1j * TH[q]), k, mode='same')
    z0, z1 = z('core0'), z('core1')
    avg = (z0 + z1) / 2
    thE = float(np.degrees(np.angle(avg[6:13].mean())))
    c0 = z0[150:450].mean(); c1 = z1[150:450].mean()
    o = dict(kernel_len=len(k), kernel_resp_dB=resp, thetaE_deg=thE, early_angles={int(n): float(np.degrees(np.angle(avg[n]))) for n in range(0, 25, 3)},
             c0=[c0.real, c0.imag], c1=[c1.real, c1.imag], r2=R2[q], r3=R3[q], r_bare=RBARE[q], n=R['classes'][key]['n'], third={})
    print(f'{key}: thetaE={thE:.1f} deg (early {list(o["early_angles"].values())[:6]}), r2={R2[q]} r3={R3[q]} r_bare={RBARE[q]:.2f}; kernel resp {np.round(resp,1)}')
    for cl in cls:
        if f'{key}__{cl}' not in d:
            continue
        zl = z(cl)
        win = (250, 400) if cl in ('Lstart',) else ((800, 1000) if cl in ('Llate',) else (500, 950) if cl == 'L0' else (150, 450))
        L = zl[win[0]:win[1]].mean()
        rel = (L - c0) / (c1 - c0)
        C, Rr = circ(c0, c1, L)
        rows = {}
        for th in np.arange(round(thE) - 15, round(thE) + 16, 5):
            rows[int(th)] = geom(C, Rr, c0, c1, L, th)
        o['third'][cl] = dict(L=[L.real, L.imag], L_rel_sep=[rel.real, rel.imag], window=win, n=R['classes'][key]['n'][cl], by_thetaE=rows)
        txt = ' | '.join(f"{th}:{('r=%.2f phi=%+.0f Q=%.2f s=%.2f' % (v['r'], v['phi_deg'], v['Q'], v['s'])) if v else 'miss'}" for th, v in rows.items())
        print(f'   {cl:6s} n={o["third"][cl]["n"]:5d} L rel sep ({rel.real:+.2f},{rel.imag:+.2f}) :: {txt}')
    out[key] = o
json.dump(out, open(OUTJ, 'w'), indent=1, default=float)
