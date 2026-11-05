"""Example input for the LARGER KLiNQ student (q2, input_size=201): 100 time-bins
(I/Q) + 1 matched-filter scalar. Mirrors the q0 (31-dim) extraction but for q2.
"""
import os, h5py, numpy as np, torch
import os as _os, sys as _sys  # [unify] repo layout
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'discriminators'))
from disc_paths import RAW_TRAIN, RAW_TEST, CNN_DATA, CHECKPOINTS, OUT as _OUT, ensure_dir, require_dataset  # [unify]
from networks import KLiNQStudentModel

RAW_TR = str(RAW_TRAIN); RAW_TE = str(RAW_TEST)
MODELS = str(CHECKPOINTS); OUT = ensure_dir(_OUT / "fpga" / "example_inputs"); L = 500; N_TR, N_TE = 60000, 512; q = 2  # [unify] was MODELS = "/m"; OUT = "/out"
np.random.seed(0)


def load(path, key, n):
    with h5py.File(path, "r") as f:
        tot = f[f"X_{key}"].shape[0]
        idx = np.sort(np.random.RandomState(0).choice(tot, min(n, tot), replace=False))
        return f[f"X_{key}"][idx, :L, :].astype(np.float32), f[f"y_{key}"][idx].astype(np.int64)


Xtr, ytr = load(RAW_TR, "train", N_TR); Xte, yte = load(RAW_TE, "test", N_TE)
sd = torch.load(f"{MODELS}/KLiNQ_student_paper_q{q}_1000ns.pth", map_location="cpu")
fc1_in = sd["fc1.weight"].shape[1]
nb = (fc1_in - 1) // 2; bs = L // nb; nb = L // bs
print(f"q{q}: fc1_in={fc1_in} -> n_bins={nb}, bin_size={bs}", flush=True)

aI = Xte[:, :nb*bs, 0].reshape(N_TE, nb, bs).mean(2)
aQ = Xte[:, :nb*bs, 1].reshape(N_TE, nb, bs).mean(2)
avg = np.concatenate([aI, aQ], 1)                              # (N, 2*nb)
yq = ((ytr >> q) & 1)
g = Xtr.reshape(N_TR, -1)[yq == 0]; e = Xtr.reshape(N_TR, -1)[yq == 1]
n = min(len(g), len(e)); d = g[:n] - e[:n]; env = d.mean(0) / (d.var(0) + 1e-10)
mf = (Xte.reshape(N_TE, -1) @ env[:L*2]).reshape(-1, 1)
zn = lambda a: (a - a.mean(0)) / (a.std(0) + 1e-10)
inp = np.concatenate([zn(avg), zn(mf)], 1).astype(np.float32)  # (N, 2*nb+1)
assert inp.shape[1] == fc1_in, (inp.shape, fc1_in)

m = KLiNQStudentModel(input_size=fc1_in); m.load_state_dict(sd); m.eval()
with torch.no_grad():
    out = torch.sigmoid(m(torch.tensor(inp))).numpy()
lab = ((yte >> q) & 1).astype(np.int64)
acc = float(((out[:, 0] > 0.5).astype(int) == lab).mean())
np.save(f"{OUT}/KLiNQ_student_LARGER_q2_input.npy", inp)
np.save(f"{OUT}/KLiNQ_student_LARGER_q2_output.npy", out.astype(np.float32))
np.save(f"{OUT}/KLiNQ_student_LARGER_q2_labels.npy", lab)
print(f"saved input {inp.shape} output {out.shape} labels {lab.shape}  acc(q2)={acc:.3f}", flush=True)
