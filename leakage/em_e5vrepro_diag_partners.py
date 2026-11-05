# e5vrepro diagnostic 2: Q3 (q2) relaxation-test excess on quasi-null shots vs the prepared state of Q1/Q4/Q5,
# and the same for Q4 (q3) and Q5 (q4).  Uses only my own per-shot outputs (pershot_v2) and y.
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import numpy as np, json
OUT = str(LP.stage('em/e5vrepro') / 'out')
y = np.load(OUT + '/em_e5vrepro_y_test.npy').astype(np.int64)
ps = np.load(OUT + '/em_e5vrepro_pershot_v2.npz')
bits = np.stack([(y >> q) & 1 for q in range(5)], 1)
out = {}
for q, w in ((2, '1us'), (2, '400ns'), (3, '1us'), (4, '1us'), (0, '1us')):
    flag = ps['%s_q%d_relax_flag' % (w, q)]; pfp = ps['%s_q%d_relax_pfp' % (w, q)]
    z = ps['q%d_z1us' % q] if w == '1us' else ps['q%d_z400' % q]
    QN = (bits[:, q] == 1) & ps['q%d_late_core_r2' % q] & (np.abs(z) <= 3)
    others = [k for k in range(5) if k != q]
    res = {}
    for combo in range(16):
        m = QN.copy(); lab = []
        for i, k in enumerate(others):
            v = (combo >> i) & 1; m &= bits[:, k] == v; lab.append('Q%d=%d' % (k + 1, v))
        res[','.join(lab)] = dict(n=int(m.sum()), obs=int(flag[m].sum()), exp=round(float(pfp[m].sum()), 1),
                                  ratio=round(float(flag[m].sum() / pfp[m].sum()), 3))
    out['%s_q%d' % (w, q)] = res
    print(w, q, json.dumps(res), flush=True)
json.dump(out, open(OUT + '/em_e5vrepro_diag_partners.json', 'w'), indent=1)
