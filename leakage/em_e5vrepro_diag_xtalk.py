# e5vrepro diagnostic 3: size of neighbour-tone crosstalk in my 20-ns bins.  For target qubit q, compare TRAIN
# stationary (late-core, |z|<=3) mean trajectories of configurations that differ only in neighbour j's prepared bit,
# projected on the target's per-bin |0>->|1> direction, in units of the target's |delta_k| (k>=10, i.e. after ring-up).
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import numpy as np, json
OUT = str(LP.stage('em/e5vrepro') / 'out')
src = open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'em_e5vrepro_analyze.py')).read()
exec(src[:src.index("log('loading bins')")])
exec(src[src.index('# ---------------- model predictions'):src.index('# ---------------- frames ----------------')])
exec(src[src.index('def fit_frames'):src.index('res = dict(')])
Btr = np.load(OUT + '/em_e5vrepro_bins2_train.npy', mmap_mode='r')
out = {}
for q in (2, 4, 0):
    btr = bits_tr[:, q]; ctr = cfg_of(Ytr, q)
    Xtr = np.asarray(Btr[:, q, :])
    P = {w: c2r(Xtr[:, a:b].mean(1)) for w, (a, b) in WIN.items()}
    fr1 = fit_frames(P['1us'], btr, ctr); z1 = apply_frame(P['1us'], ctr, fr1)[0]
    aend = apply_frame(c2r(Xtr[:, 40:50].mean(1)), ctr, fr1)[1]
    frL = fit_frames(P['late'], btr, ctr, sel0=(aend < 0.4), sel1=(aend > 0.6) & (np.abs(z1) <= 3))
    _, aL, d0L, d1L = apply_frame(P['late'], ctr, frL)
    LC = np.where(btr == 1, (d1L < 2) & (aL > 0.5), (d0L < 2) & (aL < 0.5))
    stat = LC & (np.abs(z1) <= 3)
    M = {}
    for c in range(16):
        for b in (0, 1):
            M[(c, b)] = Xtr[stat & (btr == b) & (ctr == c), :50].mean(0)
    others = [k for k in range(5) if k != q]
    res = {}
    for i, j in enumerate(others):
        rel = []; relperp = []
        for c in range(16):
            if (c >> i) & 1: continue
            c2 = c | (1 << i)
            for b in (0, 1):
                dl = M[(c, 1)] - M[(c, 0)]; e = dl / np.abs(dl)
                diff = M[(c2, b)] - M[(c, b)]
                rel.append(np.abs((diff * np.conj(e)).real[10:]) / np.abs(dl[10:]))
        rel = np.array(rel)
        res['Q%d' % (j + 1)] = dict(median_abs_rel=float(np.median(rel)), p95_abs_rel=float(np.quantile(rel, 0.95)),
                                    max_abs_rel=float(rel.max()))
    out['Q%d' % (q + 1)] = res
    print('target Q%d' % (q + 1), json.dumps(res), flush=True)
json.dump(out, open(OUT + '/em_e5vrepro_diag_xtalk.json', 'w'), indent=1)
