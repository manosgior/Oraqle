import sys, types, numpy as np, h5py, json
sys.path[:0]=["/w/cross_dataset/repo","/w/cross_dataset/repo/readout","/w/cross_dataset/repo/discriminators","/w/cross_dataset/repo/extras/cross_dataset"]
for m in ("matplotlib","matplotlib.pyplot","matplotlib.gridspec","seaborn"): sys.modules.setdefault(m,types.ModuleType(m))
sys.modules["matplotlib"].pyplot=sys.modules["matplotlib.pyplot"]; sys.modules["matplotlib"].gridspec=sys.modules["matplotlib.gridspec"]
import baseline_no_ML as B
from common import Dataset
def rise(t_ns, c, label, bin_ns=50):
    # contrast in fixed-time bins; t10/t50/t90 relative to the maximum of the smoothed curve
    edges=np.arange(0,t_ns[-1]+bin_ns,bin_ns); cen=[];val=[]
    for a,b in zip(edges[:-1],edges[1:]):
        m=(t_ns>=a)&(t_ns<b)
        if m.any(): cen.append(0.5*(a+b)); val.append(abs(c[m].mean()))
    cen,val=np.array(cen),np.array(val); mx=val.max(); t=lambda f: cen[int(np.argmax(val>=f*mx))]
    print(f"{label:12s} max contrast at {cen[int(val.argmax())]:6.0f} ns; reaches 50% at {t(.5):5.0f} ns, 90% at {t(.9):5.0f} ns; value at end {val[-1]/mx:.2f} of max")
X=[];Y=[]
with h5py.File(B.RAW_TEST_FILE,"r") as f:
    for s in range(32):
        a=s*35000; X.append(f["X_test"][a:a+1000]); Y.append(f["y_test"][a:a+1000])
X=np.concatenate(X).astype(np.float64); Y=np.concatenate(Y); t=np.arange(X.shape[1])*2e-9
for q in range(5):
    z=(X[:,:,0]+1j*X[:,:,1])*np.exp(-2j*np.pi*B.FREQ_READOUT[q]*t); y=(Y>>q)&1
    rise(t*1e9,z[y==1].mean(0)-z[y==0].mean(0),f"Lienhard Q{q+1}")
for n in ("qick","cao"):
    d=Dataset(n); Z=np.array(d.Xte[:,:,0])+1j*np.array(d.Xte[:,:,1]); rise(np.arange(Z.shape[1])*d.dt,Z[d.yte==1].mean(0)-Z[d.yte==0].mean(0),n)
