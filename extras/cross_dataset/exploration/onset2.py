import sys, glob, re, json, numpy as np, h5py, scipy.io as sio
sys.path[:0]=["/w/cross_dataset/repo","/w/cross_dataset/repo/extras/cross_dataset"]
p="/w/external/gunyho_10000511/unz/data/bolometer-characterization-timetrace-76.nc"
with h5py.File(p,"r") as f:
    keys=list(f.attrs); print("attrs",[k[:40] for k in keys][:12])
    for k in keys:
        v=f.attrs[k]; s=v.decode() if isinstance(v,bytes) else str(v)
        for m in re.finditer(r'"?(?:readout_meta[._])?(heat_pulse_delay|heat_pulse_duration|heat_pulse_amplitude|readout_delay|readout_pulse_duration|readout_duration|pi_pulse_delay|qubit_pi_pulse_delay|integration_time|time_of_flight|probe_duration|probe_delay)"?[^0-9eE-]{0,60}"?(?:value"?: *)?(-?[0-9.]+(?:e-?[0-9]+)?)',s):
            print("  ",k[:18],m.group(1),m.group(2))
print("=== Hassani onset per sample, -16 dBm")
d=sio.loadmat(glob.glob("/w/external/hassani_8004359/unz/*/Fig7/fig7b/2pulseQND_g_neg16_*.mat")[0]); k={x.split('_')[0]:x for x in d if not x.startswith('__')}
z=d[k['I']]+1j*d[k['Q']]; t=d[k['Timens']].ravel(); m=np.abs(z.mean(0)); print(" ".join(f"{t[i]:.0f}:{m[i]*1e3:.2f}" for i in range(0,14)))
print("=== Lienhard Q1..Q5 unfiltered: |mean|, contrast, first 150 ns")
import types
for m_ in ("matplotlib","matplotlib.pyplot","matplotlib.gridspec","seaborn"): sys.modules.setdefault(m_,types.ModuleType(m_))
sys.modules["matplotlib"].pyplot=sys.modules["matplotlib.pyplot"]; sys.modules["matplotlib"].gridspec=sys.modules["matplotlib.gridspec"]
sys.path[:0]=["/w/cross_dataset/repo/readout","/w/cross_dataset/repo/discriminators"]
import baseline_no_ML as B
X=[];Y=[]
with h5py.File(B.RAW_TEST_FILE,"r") as f:
    for s in range(32):
        a=s*35000; X.append(f["X_test"][a:a+800]); Y.append(f["y_test"][a:a+800])
X=np.concatenate(X).astype(np.float64); Y=np.concatenate(Y)
print("raw rms per sample (ns:rms) ", " ".join(f"{2*i}:{np.sqrt((X[:,i]**2).sum(1).mean()):.1f}" for i in (0,2,4,6,8,10,12,15,20,25,30,40,60)))
t=np.arange(X.shape[1])*2e-9
for q in range(5):
    z=(X[:,:,0]+1j*X[:,:,1])*np.exp(-2j*np.pi*B.FREQ_READOUT[q]*t)
    # coarse 10-sample bins, enough to average out the other tones
    y=(Y>>q)&1; a=z[y==0].mean(0); b=z[y==1].mean(0); bin_=lambda v:[abs(v[i:i+10].mean()) for i in range(0,120,10)]
    print(f"Q{q+1} |<x>|", " ".join(f"{v:6.1f}" for v in bin_(0.5*(a+b))), "| contrast", " ".join(f"{v:6.1f}" for v in bin_(b-a)))
