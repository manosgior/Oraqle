"""Smaller hls4ml-friendly CNN: ds40 input (10,13) + m_param=8, full data, her
preprocessing/training. Reports per-qubit fidelity vs the m16/(10,25) baseline,
then emits the single-head (Linear(8,5)) checkpoint + matching example input.
Pool kernel for (10,13) input = 4.
"""
import os, numpy as np, torch as T, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
import os as _os, sys as _sys  # [unify] repo layout
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'discriminators'))
from disc_paths import RAW_TRAIN, RAW_TEST, CNN_DATA, CHECKPOINTS, OUT as _OUT, ensure_dir, require_dataset  # [unify]
from networks.CNN import CNN
import runners._colleague_prep as P

RAW_TR = str(RAW_TRAIN); RAW_TE = str(RAW_TEST)
L, DS, M = 500, 40, 8
DEV = T.device("cuda" if T.cuda.is_available() else "cpu")
CK = ensure_dir(_OUT / "fpga" / "checkpoints"); EX = ensure_dir(_OUT / "fpga" / "example_inputs")  # [unify] was /ckpt, /ex
print(f"[small] device={DEV} m_param={M} ds={DS}", flush=True)


def bits(y): return np.stack([(y >> q) & 1 for q in range(5)], 1).astype(int)


prm = P.calibrate(RAW_TR, "X_train", "y_train", 15000, (0, L))
Xtr, ytr = P.preprocess(RAW_TR, "X_train", "y_train", prm, DS, (0, L))
Xte, yte = P.preprocess(RAW_TE, "X_test", "y_test", prm, DS, (0, L))
Xtr = P.to_torch_layout(Xtr); Xte = P.to_torch_layout(Xte)          # (N,10,13)
Ytr = P.bits(ytr); tb = bits(yte)
print(f"[small] input shape {Xtr.shape[1:]}  train {len(ytr)} test {len(yte)}", flush=True)

T.manual_seed(1); np.random.seed(1)
m = CNN(in_channels=10, m_param=M, num_qubits=5).to(DEV)
crit = nn.BCEWithLogitsLoss(); opt = T.optim.Adam(m.parameters(), lr=5e-4)
sch = T.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.95)
tri, vai = train_test_split(np.arange(len(Ytr)), test_size=0.1, random_state=1)
ld = T.utils.data.DataLoader(T.utils.data.TensorDataset(T.tensor(Xtr[tri]), T.tensor(Ytr[tri])), batch_size=64, shuffle=True)
Xv = T.tensor(Xtr[vai]).to(DEV); Yv = Ytr[vai]; best, bs, bad = -1, None, 0
for ep in range(30):
    m.train()
    for xb, yb in ld:
        loss = crit(m(xb.to(DEV)), yb.to(DEV)); opt.zero_grad(); loss.backward(); opt.step()
    sch.step(); m.eval()
    with T.no_grad():
        va = (((T.sigmoid(m(Xv)).cpu().numpy() > 0.5).astype(int)) == Yv).mean()
    if va >= best: best, bs, bad = va, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, 0
    else: bad += 1
    if bad >= 6: break
m.load_state_dict(bs); m.eval()

with T.no_grad():
    P2 = [(T.sigmoid(m(T.tensor(Xte[s:s+8192]).to(DEV))).cpu().numpy()) for s in range(0, len(Xte), 8192)]
prob = np.concatenate(P2, 0); pb = (prob > 0.5).astype(int)
fq = [float(1 - ((pb[tb[:, q]==1, q]==0).mean() + (pb[tb[:, q]==0, q]==1).mean())/2) for q in range(5)]
g5 = float(np.prod(fq) ** 0.2)
nparams = sum(p.numel() for p in m.parameters())
print(f"[small] m={M} ds40 (10,13)  F5Q={g5:.4f}  Q={[round(a,4) for a in fq]}  params={nparams}", flush=True)

# remap 5 heads -> single Linear(M,5)
sd = m.state_dict()
new = {k: v for k, v in sd.items() if not k.startswith("output_layers")}
new["head.weight"] = T.cat([sd[f"output_layers.{i}.weight"] for i in range(5)], 0)
new["head.bias"] = T.cat([sd[f"output_layers.{i}.bias"] for i in range(5)], 0)
T.save(new, f"{CK}/CNN_small_m8_ds40_1000ns_singlehead.pth")

# example input/output/labels (512 random test shots)
ridx = np.sort(np.random.RandomState(7).choice(len(Xte), 512, replace=False))
np.save(f"{EX}/CNN_small_m8_ds40_input.npy", Xte[ridx].astype(np.float32))
np.save(f"{EX}/CNN_small_m8_ds40_output.npy", prob[ridx].astype(np.float32))
np.save(f"{EX}/CNN_small_m8_ds40_labels.npy", tb[ridx].astype(np.int64))
print(f"[small] saved singlehead ckpt + example input {Xte[ridx].shape} | pool kernel_size=4 | head=Linear({M},5)", flush=True)
print("[small] done", flush=True)
