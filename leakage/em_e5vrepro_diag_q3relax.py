# e5vrepro diagnostic: why does the known-level relaxation test fail its null on Q3 (and Q4) quasi-null shots?
# Compares the per-bin along-axis (a) and off-axis (z) trajectories of flagged vs unflagged quasi-null shots.
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import numpy as np, json, sys
OUT = str(LP.stage('em/e5vrepro') / 'out')
src = open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'em_e5vrepro_analyze.py')).read()
head = src[:src.index("log('loading bins')")]
preds = src[src.index('# ---------------- model predictions'):src.index('# ---------------- frames ----------------')]
exec(head); exec(preds)
exec(src[src.index('def fit_frames'):src.index('res = dict(')])
Bte = np.load(OUT + '/em_e5vrepro_bins2_test.npy', mmap_mode='r')
Btr = np.load(OUT + '/em_e5vrepro_bins2_train.npy', mmap_mode='r')
ps = np.load(OUT + '/em_e5vrepro_pershot_v2.npz')
out = {}
for q in (2, 3, 4, 0):
    btr, bte = bits_tr[:, q], bits_te[:, q]; ctr, cte = cfg_of(Ytr, q), cfg_of(Yte, q)
    Xtr = np.asarray(Btr[:, q, :]); Xte = np.asarray(Bte[:, q, :])
    fr = fit_frames(c2r(Xtr[:, 0:50].mean(1)), btr, ctr)
    flag = ps['1us_q%d_relax_flag' % q]; lc = ps['q%d_late_core_r2' % q]; z1 = ps['q%d_z1us' % q]
    tlev = ps['1us_q%d_relax_t' % q]; lam = ps['1us_q%d_relax_lam' % q]
    err = ERR['1us'][:, q]
    QN = (bte == 1) & lc & (np.abs(z1) <= 3)
    groups = {'qn_flag': QN & flag, 'qn_noflag': QN & ~flag, 'err_flag': (bte == 1) & err & flag & (np.abs(z1) <= 3)}
    res = {}
    # per-bin coordinates in the cfg frame (1-us frame; a in units of window-mean separation, z in window sigma)
    for g, m in groups.items():
        A = np.zeros(50); Zp = np.zeros(50); n = 0
        for c in range(16):
            mm = m & (cte == c); f = fr[c]
            if mm.sum() == 0: continue
            P = Xte[mm][:, :50]
            D = np.stack([P.real - f['c0'][0], P.imag - f['c0'][1]], -1)
            A += ((D @ f['u']) / np.hypot(*(f['c1'] - f['c0']))).sum(0)
            Zp += ((D @ f['n']) / f['sig']).sum(0); n += mm.sum()
        res[g] = dict(n=int(n), a=(A / n)[::5].round(2).tolist(), zperp=(Zp / n)[::5].round(2).tolist())
    # dependence on the other qubits' prepared states (flag rate among quasi-null shots per neighbour bit)
    nb = {}
    for qq in range(5):
        if qq == q: continue
        for v in (0, 1):
            mm = QN & (bits_te[:, qq] == v)
            nb['Q%d=%d' % (qq + 1, v)] = [round(100 * flag[mm].mean(), 3), round(100 * ps['1us_q%d_relax_pfp' % q][mm].mean(), 3)]
    res['flag_vs_expected_pct_by_neighbour'] = nb
    # z1us distribution of flagged quasi-null shots
    res['qn_flag_z1us_quantiles'] = np.quantile(z1[QN & flag], [0.05, 0.25, 0.5, 0.75, 0.95]).round(2).tolist()
    res['qn_noflag_z1us_quantiles'] = np.quantile(z1[QN & ~flag], [0.05, 0.25, 0.5, 0.75, 0.95]).round(2).tolist()
    res['qn_flag_level_t_quantiles'] = np.quantile(tlev[QN & flag], [0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist()
    # excess as a function of z1us band
    zb = {}
    for lo, hi in ((-3, -1), (-1, 1), (1, 2), (2, 3)):
        mm = QN & (z1 > lo) & (z1 <= hi)
        zb['%g..%g' % (lo, hi)] = [int(mm.sum()), round(100 * flag[mm].mean(), 3), round(100 * ps['1us_q%d_relax_pfp' % q][mm].mean(), 3)]
    res['flag_vs_expected_pct_by_z1us'] = zb
    out[q] = res
    print(q, json.dumps(res, default=float), flush=True)
json.dump(out, open(OUT + '/em_e5vrepro_diag_q3relax.json', 'w'), indent=1, default=float)
