"""Structure of Gunyho (NetCDF), Hassani Fig7b, Zheng and Cao. Reads data only; prints text of deposited scripts, never runs them."""
import sys, os, re, glob, collections, json, zipfile, numpy as np, h5py, scipy.io as sio
root = sys.argv[1]
def unzip(rec):
    d = f"{root}/{rec}"; z = glob.glob(f"{d}/*.zip"); out = f"{d}/unz"
    if z and not os.path.isdir(out):
        with zipfile.ZipFile(z[0]) as f: f.extractall(out)
    return out
print("=== GUNYHO"); g = unzip("gunyho_10000511")
p = os.path.join(g, "data/bolometer-characterization-timetrace-76.nc")
print("magic", open(p, "rb").read(8))
with h5py.File(p, "r") as f:
    def show(n, o):
        if isinstance(o, h5py.Dataset): print("  ", n, o.shape, o.dtype, {k: (o.attrs[k].decode() if isinstance(o.attrs[k], bytes) else o.attrs[k]) for k in o.attrs if k in ("units", "long_name")})
    f.visititems(show)
    t = f["readout_meta_time"][:]; print("   time", t[:3], t[-1]); print("   pi amp", f["qubit_pi_pulse_amplitude"][:])
print([x for x in os.listdir(g)][:12])
print("=== HASSANI Fig7b"); h = unzip("hassani_8004359")
fb = sorted(glob.glob(f"{h}/*/Fig7/fig7b/*"))
print(len(fb), [os.path.basename(x) for x in fb[:4]], "...")
ge = collections.Counter(); powers = collections.Counter()
for x in fb:
    n = os.path.basename(x); m = re.search(r"_(g|e)_", n); ge[m.group(1) if m else "?"] += 1
    m2 = re.search(r"_(g|e)_(neg[\d.,]+)", n); powers[m2.group(2) if m2 else "?"] += 1
print(ge, powers.most_common(20))
mats = [x for x in fb if x.endswith(".mat")]
if mats:
    m = sio.loadmat(mats[0]); print(os.path.basename(mats[0]), {k: (v.shape, v.dtype.name) for k, v in m.items() if not k.startswith("__")})
print("=== ZHENG"); z = unzip("zheng_23367046")
fl = [os.path.relpath(os.path.join(a, x), z) for a, _, b in os.walk(z) for x in b]
print(len(fl), "files", collections.Counter(os.path.splitext(x)[1] for x in fl).most_common(8))
for x in fl:
    if x.endswith((".ddh5", ".h5", ".hdf5")): print("  ", x, round(os.path.getsize(os.path.join(z, x))/1e6, 1), "MB")
for x in fl:
    if os.path.basename(x).lower().startswith("readme") and "data" in x.lower():
        print("--- ", x); print(open(os.path.join(z, x), errors="replace").read()[:3500])
print("=== CAO"); c = f"{root}/cao_DPMID3"; L = sorted(os.listdir(c)); print(len(L), L[:6], L[-8:])
