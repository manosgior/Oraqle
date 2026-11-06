import json, csv, sys
import numpy as np
SC=sys.argv[1]; S=json.load(open(SC+'/summary.json'))
M=list(csv.DictReader(open('/home/manosgior/Documents/GitHub/Oraqle/results/discriminators/master_fidelity.csv')))
name={'linear':'Linear Threshold','fnn':'Baseline FNN','herqules':'HERQULES','qubic':'QubiCML','klinq':'KLiNQ','cnn':'MCMit-CNN'}
print('=== control minus Table 2 (pp), single-channel code vs published 5-channel row')
ref={}
for r in M: ref[(r['model'],int(r['duration_ns']))]=[float(r[f'Q{i}_fidelity']) for i in range(1,6)]
rows=[]
for m,n in name.items():
    line=[]
    for T in (200,400,600,800,1000):
        ds=[100*(S['datasets'][f'lienhard_q{q+1}']['table'][m][str(T)]['F']-ref[(n,T)][q]) for q in range(5)]
        rows.append((m,T,ds)); line.append(f"T{T}: "+' '.join(f'{d:+.2f}' for d in ds))
    print(f'{m:9s}',' | '.join(line))
for lo,hi,lab in ((200,400,'T<=400'),(600,1000,'T>=600')):
    a=np.array([d for m,T,ds in rows if lo<=T<=hi for d in ds]); print(lab,'mean diff %+.2f pp, median abs %.2f, max abs %.2f, n=%d'%(a.mean(),np.median(abs(a)),abs(a).max(),len(a)))
# ranking agreement at T>=600 on published vs control, excl. MCMit-T
print('=== paired comparisons, QICK and Cao (pp, 95% CI), + = first model better')
for ds in ('qick','cao','lienhard_q1','lienhard_q5'):
    print('--',ds)
    for c in S['datasets'][ds]['comparisons']:
        if (c['a'],c['b']) in (('cnn','fnn'),('herqules','fnn'),('klinq','fnn'),('cnn','herqules'),('fnn','linear'),('cnn','linear')) and c['T'] in (400,600,1000,1200,1800,2000):
            sig='' if c['lo_pp']<=0<=c['hi_pp'] else '*'
            print(f"  T={c['T']:5d} {c['a']:>8s}-{c['b']:<7s} {c['diff_pp']:+.3f} [{c['lo_pp']:+.3f},{c['hi_pp']:+.3f}]{sig}")
print('=== training noise across seeds (sd in pp)')
for ds in ('qick','cao','lienhard_q1'):
    for m,v in S['datasets'][ds]['table'].items():
        for T,x in v.items():
            if x['n_seeds']>1: print(f"  {ds} {m} T={T}: F={x['F']:.4f} sd={100*x['sd']:.3f} pp (n={x['n_seeds']})")
print('=== params'); 
print({m:S['datasets']['qick']['table'][m]['1000']['params'] for m in S['datasets']['qick']['table'] if '1000' in S['datasets']['qick']['table'][m]})
print('=== truncation cost: error at T / error at best T')
for ds in ('qick','cao','lienhard_q1','lienhard_q3','lienhard_q4','lienhard_q5'):
    print('--',ds)
    for m,r in S['datasets'][ds]['takeaways']['truncation_cost'].items():
        print(f"  {m:9s} best T={r['best_T']:5d} F={r['F_best']:.4f}  "+' '.join(f"{k[10:]}ns x{v:.2f}" for k,v in r.items() if k.startswith('err_ratio')))
print('=== asymmetry p01/p10 at chosen T, CNN')
for ds,T in (('qick','1000'),('qick','1800'),('cao','1000'),('cao','1360'),('lienhard_q1','1000'),('lienhard_q5','1000')):
    x=S['datasets'][ds]['table']['cnn'][T]; print(f"  {ds} T={T} F={x['F']:.4f} p(0|1)={100*x['p01']:.2f}% p(1|0)={100*x['p10']:.2f}%")
print('=== error modes with CNN')
for r in S['error_modes']:
    if r['T_ns'] in (600,1000,1200,1500,1800) and 'err_cnn' in r: print(f"  {r['dataset']:12s} T={r['T_ns']:5d} sep={r['sep_sigma']:.2f}s overlap={100*r['overlap_err']:.2f}% cnn_err={100*r['err_cnn']:.2f}% overlap_share={100*r['overlap_share_of_cnn']:.0f}%")
