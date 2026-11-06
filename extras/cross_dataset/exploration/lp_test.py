"""Which filter reproduces the published Linear Threshold row? And does skipping early samples help at a fixed end time?"""
import sys, types, csv, glob, numpy as np, h5py
sys.path[:0] = ["/w/cross_dataset/repo", "/w/cross_dataset/repo/readout", "/w/cross_dataset/repo/discriminators", "/w/cross_dataset/repo/extras/cross_dataset"]
for m in ("matplotlib","matplotlib.pyplot","matplotlib.gridspec","seaborn"): sys.modules.setdefault(m, types.ModuleType(m))
sys.modules["matplotlib"].pyplot=sys.modules["matplotlib.pyplot"]; sys.modules["matplotlib"].gridspec=sys.modules["matplotlib.gridspec"]
import baseline_no_ML as B
from scipy.signal import butter, sosfilt, sosfiltfilt
from common import Dataset, lowpass
rng=np.random.default_rng(0)
# 40,000 shots, 1250 per state, from both raw files
X=[];Y=[]
with h5py.File(B.RAW_TEST_FILE,"r") as f:
    for s in range(32):
        a=s*35000; X.append(f["X_test"][a:a+1250]); Y.append(f["y_test"][a:a+1250])
X=np.concatenate(X).astype(np.float64); Y=np.concatenate(Y)
print("shots",X.shape)
def mix(X, fq, dt=2e-9):
    I=X[:,:,0]-X[:,:,0].mean(1,keepdims=True); Q=X[:,:,1]-X[:,:,1].mean(1,keepdims=True)
    Q=Q*I.std(1,keepdims=True)/(Q.std(1,keepdims=True)+1e-10); t=np.arange(X.shape[1])*dt
    lo_i=np.cos(2*np.pi*fq*t); lo_q=-np.sin(2*np.pi*fq*t)
    return I*lo_i-Q*lo_q, I*lo_q+Q*lo_i
sos=butter(3,10e6,btype="low",fs=500e6,output="sos")
pub={}
for T,L in ((200,100),(400,200),(600,300),(1000,500)):
    f=sorted(glob.glob(f"/w/cross_dataset/repo/results/discriminators/Baseline_Threshold_len{L}_*.csv"))[-1]
    r=list(csv.DictReader(open(f)))[0]; pub[T]=[float(r[f"qubit_{q}_accuracy"])/100 for q in range(5)]
Ts=[(200,100),(400,200),(600,300),(1000,500)]
print("variant        "+" ".join(f"T={T:<5d}" for T,_ in Ts))
for q in (0,4):
    y=((Y>>q)&1).astype(int); I,Qd=mix(X,B.FREQ_READOUT[q])
    variants={"causal LP":(sosfilt(sos,I,axis=1),sosfilt(sos,Qd,axis=1)),"zero-phase LP":(sosfiltfilt(sos,I,axis=1),sosfiltfilt(sos,Qd,axis=1)),"no LP":(I,Qd)}
    perm=rng.permutation(len(y)); n=int(.7*len(y)); tr,te=perm[:n],perm[n:]
    print(f"Q{q+1} published   "+" ".join(f"{pub[T][q]:.4f} " for T,_ in Ts))
    for name,(a,b) in variants.items():
        out=[]
        for T,L in Ts:
            F=np.c_[np.trapezoid(a[:,:L+1],axis=1),np.trapezoid(b[:,:L+1],axis=1)]
            c=B.fit_threshold(F[tr],y[tr]); out.append(B.assignment_fidelity(y[te],c.predict(F[te])))
        print(f"Q{q+1} {name:13s}"+" ".join(f"{v:.4f} " for v in out))
print("=== linear integration over [s, T]: does skipping the start help at a fixed end time? (70/30 split, F)")
for name,Tlist in (("qick",(600,1000,1500,2000)),("cao",(600,1000,1300)),("lienhard_q1",(600,1000)),("lienhard_q4",(600,1000)),("lienhard_q5",(600,1000))):
    ds=Dataset(name); idx=np.sort(rng.choice(len(ds.yte),min(100000,len(ds.yte)),replace=False)); Xs=np.array(ds.Xte[idx],dtype=np.float32); ys=ds.yte[idx]
    if not ds.filtered: Xs=lowpass(Xs,ds.dt)
    perm=rng.permutation(len(ys)); n=int(.7*len(ys)); tr,te=perm[:n],perm[n:]
    for T in Tlist:
        row=[]
        for s in (0,50,100,200,300,400,500):
            if s>=T: continue
            a,b=int(round(s/ds.dt)),int(round(T/ds.dt)); F=np.trapezoid(Xs[:,a:b],axis=1)
            c=B.fit_threshold(F[tr],ys[tr]); row.append((s,round(B.assignment_fidelity(ys[te],c.predict(F[te])),4)))
        best=max(row,key=lambda x:x[1]); print(f"  {name:12s} T={T:5d} "+" ".join(f"s{a}:{b}" for a,b in row)+f"  best s={best[0]} gain {100*(best[1]-row[0][1]):+.2f} pp")
