#!/usr/bin/env python3
"""Build a master CSV (per model x trace length: params, per-qubit fidelity, F5Q gmean)
and a LaTeX long table with two-level best-fidelity highlighting.

Units: all per-qubit fidelities stored as FRACTIONS (0-1).
Trace length: samples s  <->  duration = 2*s ns.
"""
import csv, glob, re, math
import os, sys  # [unify]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from disc_paths import RESULTS_DISC, OUT, ensure_dir  # [unify]
# [unify] The script ran with cwd = ~/oraqle_reports. It now globs the published
# CSVs in results/discriminators/ and writes into $ORAQLE_OUT/discriminators/tables/,
# so the committed master_fidelity.csv is never overwritten.
IN_DIR = os.environ.get('ORAQLE_REPORTS_IN', str(RESULTS_DISC))
OUT_DIR = ensure_dir(os.environ.get('ORAQLE_TABLES_OUT', str(OUT / 'discriminators' / 'tables_original_builder')))

SAMPLE_LENS = [100, 200, 300, 400, 500]          # master CSV: all 5
TABLE_NS    = [200, 400, 600, 800, 1000]         # latex table: 5 durations
TABLE_NS_DISPLAY = sorted(TABLE_NS, reverse=True)  # show 1000 ns first, descending
def ns(s): return 2 * s

def first_row(path):
    with open(path) as f:
        return next(csv.DictReader(f))

def lenfile(pattern, s):
    hits = sorted(f for f in glob.glob(os.path.join(IN_DIR, pattern)) if re.search(rf'_len0*{s}(?:_|\.)', f))  # [unify]
    return hits[0] if hits else None

def gmean(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals))

# ---- per-model loaders: return dict s -> {'params':int, 'q':[5 fractions]} ----
def load_pct(pattern, params_fixed=None):
    out = {}
    for s in SAMPLE_LENS:
        f = lenfile(pattern, s)
        if not f: continue
        r = first_row(f)
        q = [float(r[f'qubit_{i}_accuracy'])/100 for i in range(5)]
        p = params_fixed if params_fixed is not None else int(r['total_parameters'])
        out[s] = {'params': p, 'q': q}
    return out

def load_arxiv():
    out = {}
    for s in SAMPLE_LENS:
        q = []
        for i in range(5):
            f = lenfile(f'Arxiv240618807FNN_q{i}_len*.csv', s)
            r = first_row(f)
            q.append(float(r[f'qubit_{i}_accuracy'])/100)
        out[s] = {'params': 65*5, 'q': q}
    return out

def load_klinq():
    f = sorted(glob.glob(os.path.join(IN_DIR, 'KLiNQ_paper_repro_F5Q_*.csv')))[0]  # [unify]
    out = {}
    for r in csv.DictReader(open(f)):
        s = int(r['trace_samples'])
        q = [float(r[f'Q{i+1}_fidelity']) for i in range(5)]
        out[s] = {'params': int(r['total_student_params']), 'q': q}
    return out

MODELS = [   # display name, loader
    ('Linear Threshold', lambda: load_pct('Baseline_Threshold_len*.csv', params_fixed=5)),
    ('Baseline FNN',     lambda: load_pct('FNN_len*.csv')),
    ('HERQULES',         lambda: load_pct('HERQULES_len*_pertrace_*.csv', params_fixed=1002)),
    ('QubiCML',          load_arxiv),                       # arXiv:2406.18807 FNN
    ('KLiNQ',            load_klinq),
    ('MCMit-T',          lambda: load_pct('Transformer_len*.csv')),
    ('MCMit-CNN',        lambda: load_pct('CNN_m16_len*.csv', params_fixed=3061)),
]

data = {name: loader() for name, loader in MODELS}

# ---------------- master CSV ----------------
with open(os.path.join(OUT_DIR, 'master_fidelity.csv'), 'w', newline='') as fh:  # [unify]
    w = csv.writer(fh)
    w.writerow(['model','duration_ns','trace_samples','parameters',
                'Q1_fidelity','Q2_fidelity','Q3_fidelity','Q4_fidelity','Q5_fidelity',
                'F5Q_gmean'])
    for name, _ in MODELS:
        for s in SAMPLE_LENS:
            d = data[name].get(s)
            if not d: continue
            q = d['q']
            w.writerow([name, ns(s), s, d['params'],
                        *[f'{v:.4f}' for v in q], f'{gmean(q):.4f}'])
print('wrote master_fidelity.csv')

# ---------------- LaTeX table ----------------
# rows for table: (model, ns, [q1..q5], f5q)  for the 4 table durations
rows = []
for name, _ in MODELS:
    for d_ns in TABLE_NS_DISPLAY:
        s = d_ns // 2
        d = data[name].get(s)
        q = d['q']
        rows.append({'model': name, 'ns': d_ns, 'q': q, 'f5q': gmean(q)})

# (a) global best per column (5 qubits + F5Q) across ALL rows
cols = ['q0','q1','q2','q3','q4','f5q']
def cellval(r, c): return r['f5q'] if c == 'f5q' else r['q'][int(c[1])]
global_best = {c: max(cellval(r, c) for r in rows) for c in cols}
# (b) per-model best per column
model_best = {}
for name, _ in MODELS:
    mrows = [r for r in rows if r['model'] == name]
    model_best[name] = {c: max(cellval(r, c) for r in mrows) for c in cols}

EPS = 1e-9
def fmt(r, c):
    v = cellval(r, c)
    s = f'{v:.3f}'
    is_model = abs(v - model_best[r['model']][c]) < EPS      # bold
    is_global = abs(v - global_best[c]) < EPS                # faint background
    if is_model: s = r'\textbf{' + s + '}'
    if is_global: s = r'\cellcolor{blue!12}' + s
    return s

lines = []
lines.append(r'% requires: \usepackage{booktabs,longtable,multirow,xcolor,colortbl}')
lines.append(r'\begin{longtable}{ll rrrrr r}')
lines.append(r'\caption{Per-qubit readout fidelity vs.\ trace duration. '
             r'\textbf{Bold} = best duration for that qubit within a model; '
             r'\colorbox{blue!12}{shaded} = highest fidelity for that qubit across all configurations.}'
             r'\label{tab:fidelity}\\')
lines.append(r'\toprule')
header = (r'Model & Duration & Qubit 1 & Qubit 2 & Qubit 3 & Qubit 4 & Qubit 5 & '
          r'$F_{5Q}$ \\')
lines.append(header)
lines.append(r'\midrule')
lines.append(r'\endfirsthead')
lines.append(r'\toprule')
lines.append(header)
lines.append(r'\midrule')
lines.append(r'\endhead')

for name, _ in MODELS:
    mrows = [r for r in rows if r['model'] == name]
    for j, r in enumerate(mrows):
        head = (r'\multirow{%d}{*}{%s}' % (len(mrows), name)) if j == 0 else ''
        cells = ' & '.join(fmt(r, c) for c in cols)
        lines.append(f'{head} & {r["ns"]}\\,ns & {cells} \\\\')
    lines.append(r'\midrule' if name != MODELS[-1][0] else r'\bottomrule')
lines.append(r'\end{longtable}')

open(os.path.join(OUT_DIR, 'fidelity_table.tex'), 'w').write('\n'.join(lines) + '\n')  # [unify]
print('wrote fidelity_table.tex')

# console preview
print('\n--- master preview ---')
for name, _ in MODELS:
    d = data[name].get(400)
    print(f'{name:<12} 800ns params={d["params"]:>8}  F5Q={gmean(d["q"]):.4f}  '
          f'q=[{", ".join(f"{v:.3f}" for v in d["q"])}]')
