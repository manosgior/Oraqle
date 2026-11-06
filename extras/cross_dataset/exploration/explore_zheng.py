import glob, numpy as np, h5py, json, sys
root="/w/external/zheng_23367046/unz/RED_package/data/transmon"
for r,lab in (("03664","I g"),("03665","I e")):
    p=glob.glob(f"{root}/{r}-*/data.ddh5")[0]
    with h5py.File(p,"r") as f:
        g=f["data"]; print(r,lab,{k:(g[k].shape,g[k].dtype.name) for k in g})
        d=g["data"][:]; G=g["ground_data"][:]; E=g["excited_data"][:]; Fd=g["f_data"][:]; gs=g["gate_seq"][:]; t=g["sweep_param"][:]
    print(" sweep",np.ravel(t)[:3],np.ravel(t)[-2:]); print(" gate_seq[0][:6]",gs[0][:6])
    for k in (0,4,10,25,39):
        print(f"  k={k} |ground mean|={abs(G[k].mean()):.3g} ground std={G[k].std():.3g} excited mean diff={abs(E[k].mean()-G[k].mean()):.3g} f diff={abs(Fd[k].mean()-G[k].mean()):.3g} data[:,0] mean={abs(d[k,:,0].mean()):.3g} std={d[k,:,0].std():.3g}")
pj=glob.glob(f"{root}/03664-*/param_dict.json")[0]; P=json.load(open(pj))
def walk(o,pre=""):
    if isinstance(o,dict):
        for k,v in o.items(): walk(v,pre+"/"+k)
    elif any(w in pre.lower() for w in ("ro_","readout","pulse_len","integ","msmt","demod")): print("  ",pre[-70:],str(o)[:60])
walk(P)
