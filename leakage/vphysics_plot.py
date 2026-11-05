"""vphysics_plot.py -- figure for the resonance-circle test (tag vphysics). Local, matplotlib."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
D = sys.argv[1]; OUT = sys.argv[2]
o = json.load(open(f'{D}/vphysics_circle2.json'))
t = np.arange(1000) * 2e-9

def load(key):
    for npz, js in (('vphysics_classmeans2.npz', 'vphysics_raw_classes2.json'), ('vphysics_classmeans.npz', 'vphysics_raw_classes.json')):
        d = np.load(f'{D}/{npz}'); R = json.load(open(f'{D}/{js}'))
        if f'{key}__core0' in d:
            return d, R

def kern(FH, q):
    offs = [FH[j] - FH[q] for j in range(5) if j != q] + [-FH[q]]
    k = np.array([1.0])
    for of in offs:
        L = int(round(500e6 / abs(of)))
        while L > 40:
            L //= 2
        k = np.convolve(k, np.ones(L) / L)
    return k

def circ(a, b, c):
    ax, ay, bx, by, cx, cy = a.real, a.imag, b.real, b.imag, c.real, c.imag
    dd = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / dd
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / dd
    C = ux + 1j * uy
    return C, abs(a - C)

fig, axs = plt.subplots(1, 3, figsize=(17, 5.6))
for ax, key, q, lcl, title in ((axs[0], 'q2_y0_4', 2, 'Lstart', 'Q3 (q2), prepared state y=4 (others |0>)'),
                               (axs[1], 'q4_y4_20', 4, 'Lstart', 'Q5 (q4), y=20 (Q3 and Q5 prepared |1>)')):
    d, R = load(key); FH = np.array(R['F_HERQ']); TH = np.array(R['TH_OURS']); k = kern(FH, q)
    z = lambda cl: np.convolve((d[f'{key}__{cl}'] - d[f'{key}__{cl}'].mean()) * np.exp(-1j * 2 * np.pi * FH[q] * t) * np.exp(1j * TH[q]), k, mode='same')
    cols = {'core0': 'tab:blue', 'core1': 'tab:orange', 'Lstart': 'tab:red', 'Llate': 'tab:purple'}
    for cl, c in cols.items():
        zz = z(cl)[:900]
        ax.plot(zz.real, zz.imag, '-', color=c, lw=0.8, alpha=0.8, label=f'{cl} class-mean trajectory (n={R["classes"][key]["n"][cl]})')
    c0 = complex(*o[key]['c0']); c1 = complex(*o[key]['c1']); L = complex(*o[key]['third'][lcl]['L'])
    C, Rr = circ(c0, c1, L)
    th = np.linspace(0, 2 * np.pi, 400); ax.plot((C + Rr * np.exp(1j * th)).real, (C + Rr * np.exp(1j * th)).imag, 'k:', lw=1, label='circle through c0, c1, L')
    thE = o[key]['thetaE_deg']; g = o[key]['third'][lcl]['by_thetaE']
    best = g[min(g, key=lambda s: abs(int(s) - thE) if g[s] else 999)]
    P = complex(*best['P']); Cres = 2 * C - P
    ray = np.linspace(0, abs(P) * 1.1, 10) * np.exp(1j * np.radians(thE))
    ax.plot(ray.real, ray.imag, 'g--', lw=1, label=f'early direction (direct path) {thE:.0f} deg')
    ax.plot(P.real, P.imag, 'g*', ms=13, label='P_inf (far-detuned point)')
    pt = lambda x: C + (Cres - C) * np.exp(-2j * np.arctan(x))
    for lab, rr, mk in (('|2> (r=%.2f)' % o[key]['r2'], o[key]['r2'], 'D'), ('|3> (r=%.2f)' % o[key]['r3'], o[key]['r3'], 's'), ('bare resonator (r=%.1f)' % o[key]['r_bare'], o[key]['r_bare'], 'P')):
        S = pt(best['x1'] + rr * best['s']); ax.plot(S.real, S.imag, mk, mfc='none', mec='k', ms=10, mew=1.5, label='predicted ' + lab)
    for nm, S, c in (('c0', c0, 'tab:blue'), ('c1', c1, 'tab:orange'), ('L', L, 'tab:red')):
        ax.plot(S.real, S.imag, 'o', color=c, ms=9, mec='k')
    ax.set_title(f"{title}\nfit at theta_E={thE:.0f} deg: r={best['r']:.2f}, Fano phi={best['phi_deg']:+.0f} deg, Q={best['Q']:.2f}", fontsize=9)
    ax.set_xlabel('I (rotated frame, ADC units)'); ax.set_ylabel('Q (rotated frame, ADC units)'); ax.set_aspect('equal')
    ax.legend(fontsize=6, loc='lower center', bbox_to_anchor=(0.5, -0.62), ncol=2)
ax = axs[2]
for key, cl, c, ls in (('q2_y0_4', 'Lstart', 'tab:red', '-'), ('q2_y0_4', 'Llate', 'tab:purple', '-'), ('q2_y0_4', 'L0', 'tab:brown', '-'),
                       ('q2_y16_20', 'Lstart', 'tab:red', '--'), ('q2_y16_20', 'Llate', 'tab:purple', '--'), ('q2_y16_20', 'L0', 'tab:brown', '--')):
    g = o[key]['third'][cl]['by_thetaE']
    xs = [int(s) for s in g if g[s]]; ys = [g[s]['r'] for s in g if g[s]]
    ax.plot(xs, ys, ls, color=c, marker='o', ms=3, label=f'Q3 {key} {cl}')
ax.axhline(0.761, color='k', lw=1); ax.text(20, 0.85, '|2> (transmon, Lienhard Tab. III/IV)', fontsize=7)
ax.axhline(1.353, color='k', lw=1, ls='--'); ax.text(20, 1.45, '|3>', fontsize=7)
ax.axhline(4.77, color='g', lw=1, ls='-.'); ax.text(20, 5.0, 'bare resonator (qubit decoupled)', fontsize=7, color='g')
ax.axvspan(28, 34, color='0.85', label='measured early direction')
ax.set_yscale('log'); ax.set_ylim(0.3, 60); ax.set_xlabel('assumed P_inf direction theta_E (deg)'); ax.set_ylabel('implied level pull r = (d_L - d_1)/(d_1 - d_0)')
ax.set_title('Q3: implied resonator pull of the third state', fontsize=9)
ax.legend(fontsize=6, loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=2)
plt.tight_layout(); plt.savefig(OUT, dpi=130, bbox_inches='tight')
print('wrote', OUT)
