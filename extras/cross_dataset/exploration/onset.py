import glob, numpy as np, h5py, scipy.io as sio, re, os
print("=== HASSANI -16 dBm, pulse 1")
def load(s):
    p=glob.glob(f"/w/external/hassani_8004359/unz/*/Fig7/fig7b/2pulseQND_{s}_neg16_*.mat")[0]; d=sio.loadmat(p); k={x.split('_')[0]:x for x in d if not x.startswith('__')}
    return d[k['I']]+1j*d[k['Q']], d[k['Timens']].ravel(), d[k['I1']]+1j*d[k['Q1']]
g,t,g2=load('g'); e,_,e2=load('e')
mg,me=g.mean(0),e.mean(0); sd=np.sqrt((g.std(0)**2+e.std(0)**2)/2)
print(" t(ns): |<g>|  |<e>|  |<e-g>|  noise")
for i in range(0,64,3): print(f"{t[i]:7.0f} {abs(mg[i]):8.3g} {abs(me[i]):8.3g} {abs(me[i]-mg[i]):8.3g} {sd[i]:8.3g}")
print("=== GUNYHO run 76")
with h5py.File("/w/external/gunyho_10000511/unz/data/bolometer-characterization-timetrace-76.nc","r") as f:
    Z=f["readout_meta_CH0_re"][:]+1j*f["readout_meta_CH0_im"][:]; t=f["readout_meta_time"][:]*1e6
m0,m1=Z[0].mean(0),Z[1].mean(0); s=np.sqrt((Z[0].std(0)**2+Z[1].std(0)**2)/2)
print(" t(us): |<0>|  |<1>|  |<1-0>|  noise")
for i in list(range(0,120,10))+list(range(120,781,60)): print(f"{t[i]:7.2f} {abs(m0[i]):9.3g} {abs(m1[i]):9.3g} {abs(m1[i]-m0[i]):9.3g} {s[i]:9.3g}")
