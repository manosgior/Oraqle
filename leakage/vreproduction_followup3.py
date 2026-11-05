#!/usr/bin/env python3
"""vreproduction follow-up 3: model-light segment transition matrices for Q3 (q2) and Q5 (q4):
nearest-reference classification (confident: gap > 1 sigma) of segment means, for several
segment pairs; P(state at later segment | state at earlier segment), separately for prepared 0/1.
References per segment: c0 = mean of clean prepared-0 shots (2 us window on-axis at |0>), c1 = mean of
clean prepared-1 shots (2 us window on-axis at |1>), cL = median of deep-L shots (2 us z > 8)."""
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json
import numpy as np

INP = str(LP.stage('vreproduction') / 'out')
S = {sp: dict(np.load(f'{INP}/vreproduction_sub_{sp}.npz')) for sp in ('train', 'test')}
PS = np.load(f'{INP}/vreproduction_pershot.npz')
y = np.concatenate([S['train']['y'], S['test']['y']])
bits = (y[:, None] >> np.arange(5)) & 1
BOX = np.concatenate([S['train']['box'], S['test']['box']])
PAIRS = {'A: 0.5-1.0us -> 1.2-2.0us': ((5, 10), (12, 20)),
         'B: 0.3-0.7us -> 0.8-1.2us': ((3, 7), (8, 12)),
         'C: 0.8-1.2us -> 1.6-2.0us': ((8, 12), (16, 20)),
         'D: 0.3-0.7us -> 1.6-2.0us': ((3, 7), (16, 20))}
R = {}
for q in (2, 4):
    z2 = PS[f'z_2us_q{q}']; a2 = PS[f'a_2us_q{q}']
    clean0 = (bits[:, q] == 0) & (np.abs(z2) < 2) & (np.abs(a2) < 0.3)
    clean1 = (bits[:, q] == 1) & (np.abs(z2) < 2) & (np.abs(a2 - 1) < 0.3)
    deepL = z2 > (8 if q == 2 else 5)

    def seg(b0, b1):
        Xs = BOX[:, q, b0:b1].mean(1)
        c0 = Xs[clean0].mean(); c1 = Xs[clean1].mean()
        cL = np.median(Xs[deepL].real) + 1j * np.median(Xs[deepL].imag)
        u = (c1 - c0) / abs(c1 - c0)
        sig = 1.4826 * np.median(np.abs(((Xs[clean0] - c0) * np.conj(u)).imag))
        dist = np.stack([np.abs(Xs - c0), np.abs(Xs - c1), np.abs(Xs - cL)], 1) / sig
        lab = dist.argmin(1); srt = np.sort(dist, 1)
        return lab, (srt[:, 1] - srt[:, 0]) > 1.0, [abs(c1 - c0) / sig, abs(cL - c1) / sig]

    for pname, (s_a, s_b) in PAIRS.items():
        la, ca, ia = seg(*s_a); lb, cb, ib = seg(*s_b)
        out = {'sep_sigma_first [01, 1L]': ia, 'sep_sigma_second [01, 1L]': ib,
               'dt_us (centre to centre)': (np.mean(s_b) - np.mean(s_a)) * 0.1}
        for b in (1, 0):
            rows = bits[:, q] == b
            M = np.zeros((3, 3), int)
            for i in range(3):
                for j in range(3):
                    M[i, j] = int(np.sum(rows & ca & cb & (la == i) & (lb == j)))
            out[f'p{b} counts [from 0,1,L][to 0,1,L]'] = M.tolist()
            with np.errstate(invalid='ignore'):
                out[f'p{b} P(to|from)'] = (M / M.sum(1, keepdims=True)).round(4).tolist()
        R[f'Q{q+1} | {pname}'] = out
json.dump(R, open(f'{INP}/vreproduction_followup3.json', 'w'), indent=0, default=float)
print('done', flush=True)
