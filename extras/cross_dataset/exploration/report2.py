import json, sys, os, glob, csv
import numpy as np
SC=sys.argv[1]; R='/home/manosgior/Documents/GitHub/Oraqle'
sys.path[:0]=[R,R+'/system']
from analyze import load, summarize
S=json.load(open(SC+'/summary.json'))
print('=== QEC published, Lienhard chip, ibm_boston')
rows=summarize(load(R+'/results/system/ler_disc.jsonl'))
for r in rows:
    if r['label'] in ('MCMit-CNN','HERQULES'): print(f"  {r['label']:10s} T={r['T']:5d} p={r['p']:.4f} L={r.get('Lambda') or float('nan'):.3f} d={r.get('d@1e-09')} ST={r.get('spacetime@1e-09')}")
print('=== QEC QICK readout')
Q=json.load(open(SC+'/ler_qick.summary.json'))+json.load(open(SC+'/ler_qick_linear.summary.json'))
for r in Q:
    if r['T']>=600 and 'linear' in r['label']: print(f"  {r['label']:14s} T={r['T']:5d} p={r['p']:.4f} L={r['Lambda'] or float('nan'):.3f} d={r['d@1e-09']} ST={r['spacetime@1e-09']}")
print('=== QICK and Cao tables'); 
for ds in ('qick','cao'):
    t=S['datasets'][ds]['table']; Ts=sorted({int(x) for m in t for x in t[m]})
    print(ds,Ts)
    for m in ('linear','qubic','herqules','klinq','fnn','cnn','transformer'):
        if m in t: print(f"  {m:11s}"+' '.join(f"{t[m][str(T)]['F']:.4f}" if str(T) in t[m] else '  .   ' for T in Ts), ' params',list(t[m].values())[0]['params'])
print('=== linear curves')
for ds in ('qick','cao'):
    r=json.load(open(f'{SC}/linear_{ds}.json')); F=np.array(r['F']); i=int(F.argmax())
    print(ds,'peak F %.4f at %.0f ns; F(end) %.4f at %.0f'%(F[i],r['T_ns'][i],F[-1],r['T_ns'][-1]))
    for w in ('window_250ns','window_500ns'):
        c=np.array(r[w]['center_us']);f=np.array(r[w]['F']);k=int(f.argmax()); print('  ',w,'first %.3f (center %.2f), best %.3f at center %.2f, last %.3f'%(f[0],c[0],f[k],c[k],f[-1]))
    print('  start scan',{T:{s:round(v,4) for s,v in d.items()} for T,d in r['start_scan'].items()})
C=json.load(open(SC+'/linear_filter_check.json'))
print('=== filter check: mean abs diff from published over qubits, by variant and T (pp)')
for v,d in C['variants'].items():
    print(' ',v,{T:round(float(np.mean([abs(d[f'Q{q+1}'][T]-C['published'][T][q]) for q in range(5)]))*100,2) for T in ('200','400','600','1000')})
print('  signed causal-lp mean', {T:round(float(np.mean([C['variants']['causal_lp'][f'Q{q+1}'][T]-C['published'][T][q] for q in range(5)]))*100,2) for T in ('200','400','600','1000')})
print('  scan Lienhard no LP', {q:{T:(max(d,key=d.get),round(100*(max(d.values())-d['0']),2)) for T,d in v.items()} for q,v in C['start_scan'].items()})
for n in ('hassani',):
    r=json.load(open(f'{SC}/linear_{n}.json'))['powers']['neg16']; F=np.array(r['F']); i=int(F.argmax()); print('hassani -16 dBm peak %.3f at %.0f ns; n=%d; F at 996ns %.3f; windows'%(F[i],r['T_ns'][i],r['n'],np.interp(1000,r['T_ns'],F)), [round(x,3) for x in r['window_500ns']['F']][:12])
g=json.load(open(f'{SC}/linear_gunyho.json')); F=np.array(g['F']); i=int(F.argmax()); print('gunyho peak %.3f at %.0f ns, end %.3f at %.0f'%(F[i],g['T_ns'][i],F[-1],g['T_ns'][-1]))
z=json.load(open(f'{SC}/linear_zheng.json'))['devices']
for d,v in z.items(): F=np.array(v['F']); i=int(F.argmax()); print('zheng',d,'peak %.3f at %.0f; within 0.5pp from %.0f ns; end %.3f'%(F[i],v['T_ns'][i],v['T_ns'][int(np.argmax(F>=F[i]-0.005))],F[-1]))
