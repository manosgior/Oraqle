# e5vrepro: assemble the L -> T -> O -> S attribution from my own raw run (em_e5vrepro_raw_v2.json) and compare with
# the reconciled numbers (em_e4recon_final.json, read only as the comparison target).
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import json, numpy as np
D = str(LP.RES) + '/'
raw = json.load(open(D + 'em_e5vrepro_raw_v2.json'))
rec = json.load(open(D + 'em_e4recon_final.json'))
partners = json.load(open(D + 'em_e5vrepro_diag_partners.json'))
xtalk = json.load(open(D + 'em_e5vrepro_diag_xtalk.json'))
NSH = 1120000
QN = {0: 'Q1', 1: 'Q2', 2: 'Q3', 3: 'Q4', 4: 'Q5'}
A = 'alpha_0.01'

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 3), round(100 * (c + h), 3)]

def pct(val, var, n, nd=2):
    s = 1.96 * np.sqrt(max(var, 0.0))
    return dict(value=round(100 * val / n, nd), ci95=[round(100 * (val - s) / n, nd), round(100 * (val + s) / n, nd)],
                count=round(float(val), 1))

def attribute(q, w, t_mode='gauss'):
    """per-bit and per-qubit L,T,O,S counts with variances. t_mode: 'gauss' (primary), 'emp', 'ungated'"""
    Q = raw['qubits'][str(q)][w]
    bits = {}
    for b in (0, 1):
        pb = Q['per_bit'][str(b)]
        N = pb['N']
        if q == 3:   # Q4: leakage n/m (multimodal |1>)
            L, varL = 0.0, 0.0
        else:
            L, varL = float(pb['L']), float(pb['Lp'] + pb['Lm'])
        st = Q['steps']['relax' if b == 1 else 'excite'][A]
        gated_in = st['gate_pass'] and q != 3
        if t_mode == 'ungated':
            gated_in = q != 3 and q != 1
        Tg, varT = (st['T'], st['varT']) if gated_in else (0.0, 0.0)
        ov = pb['overlap']['2.0']
        Oexp, varO = ov['Oexp'], ov['varO']
        R = N - L - Tg
        T = Tg
        if gated_in and t_mode == 'emp':
            qn = st['quasi_null']
            g = qn['exp_err'] / qn['n_err']; f = qn['obs_err'] / qn['n_err']
            Ouse = min(Oexp, R)
            T = Tg + Ouse * (g - f)
            varT = varT + Ouse ** 2 * f * (1 - f) / qn['n_err']
            R = N - L - T
        capped = Oexp > R
        O = min(R, Oexp)
        if capped:
            varO = varL + varT
        S = R - O
        bits[b] = dict(N=N, L=L, varL=varL, T=T, varT=varT, T_status='measured' if gated_in else 'n/m',
                       O=O, varO=varO, O_capped=bool(capped), S=S, varS=varL + varT + varO,
                       TS=N - L - O, varTS=varL + varO, n_shots=pb['n_shots'])
    tot = {k: bits[0][k] + bits[1][k] for k in ('N', 'L', 'varL', 'T', 'varT', 'O', 'varO', 'S', 'varS', 'TS', 'varTS')}
    tot['T_status'] = {b: bits[b]['T_status'] for b in bits}
    return bits, tot

def fmt(tot, nshots=NSH):
    N = tot['N']
    return dict(n_err=int(N), err_rate=round(100 * N / nshots, 3), err_rate_ci=wilson(N, nshots),
                leakage=pct(tot['L'], tot['varL'], N), transition_T=pct(tot['T'], tot['varT'], N),
                overlap_O_upper=pct(tot['O'], tot['varO'], N), unresolved_S_lower=pct(tot['S'], tot['varS'], N),
                state_change_TS_lower=pct(tot['TS'], tot['varTS'], N),
                per_shot_pct=dict(state_change=pct(tot['TS'], tot['varTS'], nshots, 3),
                                  overlap=pct(tot['O'], tot['varO'], nshots, 3)))

out = dict(tag='e5vrepro', date='2026-11-05',
           what='Independent reproduction from raw TEST HDF5 (full 1.12M shots, all 32 states): own demodulation (IF from '
                'stationarity of the prepared-|0> mean phase, DC removed), 20-ns boxcar bins, own per-configuration '
                'core-line frames (TRAIN->TEST), own mirror-corrected leakage flag, own known-level change-point step '
                'test with an exact level-conditioned Gaussian null, own late-core overlap reference (2-sigma disc, '
                'late window 1-2 us). Model predictions are the only shared inputs (canonical preds_MCMitCNN_pb.npy at 1 us, '
                'cnn_soft_400ns.npz at 400 ns; both verified to align with y_test).',
           ifs_MHz={QN[q]: round(raw['meta']['if_json'][str(q)]['f_if'] / 1e6, 5) for q in range(5)},
           alignment_checks=raw['meta']['checks'])

# ---- per qubit tables ----
tables = {}
for w in ('1us', '400ns'):
    tables[w] = {}
    for mode in ('gauss', 'emp', 'ungated'):
        tables[w][mode] = {}
        pool = {'Q1,Q3-Q5': [0, 2, 3, 4], 'all5': [0, 1, 2, 3, 4], 'Q1,Q2,Q3,Q5': [0, 1, 2, 4]}
        tots = {}
        for q in range(5):
            bits, tot = attribute(q, w, mode)
            tots[q] = tot
            d = fmt(tot)
            d['per_bit'] = {('prep%d' % b): dict(n_err=bits[b]['N'], leakage=pct(bits[b]['L'], bits[b]['varL'], bits[b]['N']),
                                                 transition_T=pct(bits[b]['T'], bits[b]['varT'], bits[b]['N']),
                                                 T_status=bits[b]['T_status'],
                                                 overlap_O_upper=pct(bits[b]['O'], bits[b]['varO'], bits[b]['N']),
                                                 O_capped=bits[b]['O_capped'],
                                                 state_change_TS_lower=pct(bits[b]['TS'], bits[b]['varTS'], bits[b]['N']))
                            for b in (0, 1)}
            if q == 3: d['leakage'] = 'n/m (multimodal |1>)'
            tables[w][mode][QN[q]] = d
        for pn, qs in pool.items():
            t = {k: sum(tots[q][k] for q in qs) for k in ('N', 'L', 'varL', 'T', 'varT', 'O', 'varO', 'S', 'varS', 'TS', 'varTS')}
            tables[w][mode]['pooled_' + pn] = fmt(t, NSH * len(qs))
out['attribution'] = tables

# ---- leakage E3 cross-check and step-test diagnostics ----
out['leak_E3_prep1_prep0_all'] = {QN[q]: {w: dict(prep1=dict(v=round(raw['leak_E3'][str(q)][w]['test_prep1']['E3'], 3),
                                                             ci=[round(x, 3) for x in raw['leak_E3'][str(q)][w]['test_prep1']['ci']]),
                                                   prep0=dict(v=round(raw['leak_E3'][str(q)][w]['test_prep0']['E3'], 3),
                                                              ci=[round(x, 3) for x in raw['leak_E3'][str(q)][w]['test_prep0']['ci']]),
                                                   all=round(raw['leak_E3'][str(q)][w]['test_all']['E3'], 3),
                                                   train_prep1=round(raw['leak_E3'][str(q)][w]['train_prep1']['E3'], 3))
                                         for w in ('1us', '400ns')} for q in range(5)}
steps = {}
for w in ('1us', '400ns'):
    for q in range(5):
        for dname in ('relax', 'excite'):
            s = raw['qubits'][str(q)][w]['steps'][dname]
            steps['%s_%s_%s' % (w, QN[q], dname)] = {
                a: dict(T=round(s[a]['T'], 1), flags=s[a]['flags_in_errors'], fp_sum=round(s[a]['fp_sum_errors'], 1),
                        quasi_null_obs_over_exp=round(s[a]['quasi_null']['ratio'], 3),
                        quasi_null_ERRORS_obs_over_exp=round(s[a]['quasi_null']['ratio_err'], 3),
                        eff_given_mf_misread=round(s[a]['eff_given_mf_misread'], 3),
                        pflag_ge_0p5_window_ns=s[a]['pflag_ge_0p5_ns'], pflag_max=round(s[a]['pflag_max'], 3),
                        gate_pass=s[a]['gate_pass']) for a in ('alpha_0.05', 'alpha_0.01', 'alpha_0.002')}
            steps['%s_%s_%s' % (w, QN[q], dname)]['noise'] = s['noise']
out['step_test'] = steps
out['q3_relax_null_by_partner_state_1us'] = partners['1us_q2']
out['q3_relax_null_by_partner_state_400ns'] = partners['400ns_q2']
out['q4_relax_null_by_partner_state_1us'] = partners['1us_q3']
out['crosstalk_per_bin_rel_to_delta'] = xtalk

# late-window model-free check
lw = {}
for q in range(5):
    for b in (0, 1):
        x = raw['qubits'][str(q)]['1us']['per_bit'][str(b)]['latewin']
        lw['%s_prep%d' % (QN[q], b)] = dict(errors_in_other_late_core=round(x['err_in_other_core'], 1),
                                            ci=[round(v, 1) for v in x['err_in_other_core_ci']],
                                            correct_in_other_late_core=round(x['ok_in_other_core'], 1))
out['late_window_check_1us_r2'] = lw

# leaked share of errors pooled Q1/Q2/Q3/Q5 at 1us (L of Q4 excluded)
Lp = sum(raw['qubits'][str(q)]['1us']['per_bit'][str(b)]['Lp'] for q in (0, 1, 2, 4) for b in (0, 1))
Lm = sum(raw['qubits'][str(q)]['1us']['per_bit'][str(b)]['Lm'] for q in (0, 1, 2, 4) for b in (0, 1))
Ne = sum(raw['qubits'][str(q)]['1us']['per_bit'][str(b)]['N'] for q in (0, 1, 2, 4) for b in (0, 1))
out['leaked_share_pooled_Q1Q2Q3Q5_1us'] = pct(Lp - Lm, Lp + Lm, Ne)
out['overlap_radius_sensitivity_pct_err'] = {
    w: {QN[q]: {r: round(100 * sum(raw['qubits'][str(q)][w]['per_bit'][str(b)]['overlap'][r]['Oexp'] for b in (0, 1)) /
                         sum(raw['qubits'][str(q)][w]['per_bit'][str(b)]['N'] for b in (0, 1)), 2)
                for r in ('1.5', '2.0', '2.5')} for q in range(5)} for w in ('1us', '400ns')}

# ---- comparison with reconciled ----
tn = rec['text_numbers']
cmp = {}
keymap = {'leakage': 'leakage', 'transition_T': 'transition_resolved', 'overlap_O_upper': 'overlap_near_boundary_upper',
          'state_change_TS_lower': 'state_change_total_T_plus_S'}
for label, w, mine_key, rkey in (('Q3_1us', '1us', 'Q3', 'Q3_1us_cnn'), ('Q3_400ns', '400ns', 'Q3', 'Q3_400ns_cnn'),
                                 ('Q5_1us', '1us', 'Q5', 'Q5_1us_cnn'), ('Q5_400ns', '400ns', 'Q5', 'Q5_400ns_cnn'),
                                 ('pooled_Q1Q3Q4Q5_1us', '1us', 'pooled_Q1,Q3-Q5', 'pooled_Q1Q3Q4Q5_1us_cnn'),
                                 ('pooled_all5_1us', '1us', 'pooled_all5', 'pooled_all5_1us_cnn')):
    m = tables[w]['gauss'][mine_key]; me = tables[w]['emp'][mine_key]; r = tn[rkey]
    row = {}
    for mk, rk in keymap.items():
        rv = r.get(rk, {})
        mv = m[mk]
        if isinstance(mv, str):
            row[mk] = dict(mine=mv, rec=rv.get('value')); continue
        rval = rv.get('value'); rci = rv.get('ci95')
        e = dict(mine=mv['value'], mine_ci=mv['ci95'], rec=rval, rec_ci=rci)
        if rval is not None and rci is not None:
            hw_m = (mv['ci95'][1] - mv['ci95'][0]) / 2; hw_r = (rci[1] - rci[0]) / 2
            e['diff_pp'] = round(mv['value'] - rval, 2)
            e['combined_hw_pp_indep'] = round(np.hypot(hw_m, hw_r), 2)
            e['agree_within_combined_ci'] = bool(abs(mv['value'] - rval) <= np.hypot(hw_m, hw_r))
        if mk == 'transition_T':
            e['mine_emp_null'] = me[mk]['value']; e['mine_emp_null_ci'] = me[mk]['ci95']
        row[mk] = e
    if 'err_rate_pct' in r:
        row['err_rate'] = dict(mine=m['err_rate'], rec=r['err_rate_pct']['value'])
    cmp[label] = row
out['comparison_with_e4recon'] = cmp
json.dump(out, open(D + 'em_e5vrepro_results.json', 'w'), indent=1, default=float)

# ---- console summary ----
for label, row in cmp.items():
    print(label)
    for k, e in row.items():
        print('   %-24s %s' % (k, json.dumps(e)))
for w in ('1us', '400ns'):
    print('==', w)
    for k, d in tables[w]['gauss'].items():
        if isinstance(d.get('leakage'), str):
            Ls = 'n/m'
        else:
            Ls = '%.2f %s' % (d['leakage']['value'], d['leakage']['ci95'])
        print('%-18s err %.3f  L %s  T %.2f %s [emp %.2f, ungated %.2f]  O %.2f %s  TS %.2f %s  S %.2f | shots TS %.3f O %.3f' % (
            k, d['err_rate'], Ls, d['transition_T']['value'], d['transition_T']['ci95'],
            tables[w]['emp'][k]['transition_T']['value'], tables[w]['ungated'][k]['transition_T']['value'],
            d['overlap_O_upper']['value'], d['overlap_O_upper']['ci95'], d['state_change_TS_lower']['value'],
            d['state_change_TS_lower']['ci95'], d['unresolved_S_lower']['value'],
            d['per_shot_pct']['state_change']['value'], d['per_shot_pct']['overlap']['value']))
print('leaked share pooled Q1/Q2/Q3/Q5 1us', out['leaked_share_pooled_Q1Q2Q3Q5_1us'])
print('O radius sensitivity', json.dumps(out['overlap_radius_sensitivity_pct_err']))
print('late window', json.dumps(lw))
