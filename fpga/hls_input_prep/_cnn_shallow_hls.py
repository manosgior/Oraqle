"""Shallow, low-latency CNN variants for hls4ml (m=8, ds40 input (10,13)).
Drops the residual block to cut pipeline depth. Two depths:
  noRes_2conv : conv(s2)->conv(s2)->AvgPool(4)->Linear(8,5)
  1conv       : conv(s2)->AvgPool(7)->Linear(8,5)
Both are already single-head + AvgPool (hls4ml-ready, no remap). Reports the
per-qubit fidelity cost vs the m8/4-conv baseline (F5Q=0.9084).
"""
import os, numpy as np, torch as T, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
import os as _os, sys as _sys  # [unify] repo layout
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'discriminators'))
from disc_paths import RAW_TRAIN, RAW_TEST, CNN_DATA, CHECKPOINTS, OUT as _OUT, ensure_dir, require_dataset  # [unify]
import runners._colleague_prep as P

RAW_TR = str(RAW_TRAIN); RAW_TE = str(RAW_TEST)
L, DS, M = 500, 40, 8
DEV = T.device("cuda" if T.cuda.is_available() else "cpu")
CK = ensure_dir(_OUT / "fpga" / "checkpoints"); EX = ensure_dir(_OUT / "fpga" / "example_inputs")  # [unify] was /ckpt, /ex
print(f"[shallow] device={DEV} m={M} ds={DS}", flush=True)


class ShallowCNN(nn.Module):
    def __init__(self, m=8, n_conv=2, pool_k=4):
        super().__init__()
        self.n_conv = n_conv
        self.conv1 = nn.Conv1d(10, m, 3, stride=2, padding=1); self.bn1 = nn.BatchNorm1d(m)
        if n_conv >= 2:
            self.conv2 = nn.Conv1d(m, m, 3, stride=2, padding=1); self.bn2 = nn.BatchNorm1d(m)
        self.gap = nn.AvgPool1d(kernel_size=pool_k)
        self.head = nn.Linear(m, 5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        if self.n_conv >= 2:
            x = F.relu(self.bn2(self.conv2(x)))
        x = self.gap(x); x = x.flatten(1)
        return self.head(x)


def bits(y): return np.stack([(y >> q) & 1 for q in range(5)], 1).astype(int)


prm = P.calibrate(RAW_TR, "X_train", "y_train", 15000, (0, L))
Xtr, ytr = P.preprocess(RAW_TR, "X_train", "y_train", prm, DS, (0, L))
Xte, yte = P.preprocess(RAW_TE, "X_test", "y_test", prm, DS, (0, L))
Xtr = P.to_torch_layout(Xtr); Xte = P.to_torch_layout(Xte)         # (N,10,13)
Ytr = P.bits(ytr); tb = bits(yte)
print(f"[shallow] input {Xtr.shape[1:]} train {len(ytr)} test {len(yte)}", flush=True)


def train_eval(name, n_conv, pool_k):
    T.manual_seed(1); np.random.seed(1)
    m = ShallowCNN(M, n_conv, pool_k).to(DEV)
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
        prob = np.concatenate([T.sigmoid(m(T.tensor(Xte[s:s+8192]).to(DEV))).cpu().numpy() for s in range(0, len(Xte), 8192)], 0)
    pb = (prob > 0.5).astype(int)
    fq = [float(1 - ((pb[tb[:, q]==1, q]==0).mean() + (pb[tb[:, q]==0, q]==1).mean())/2) for q in range(5)]
    g5 = float(np.prod(fq) ** 0.2); npar = sum(p.numel() for p in m.parameters())
    print(f"[shallow] {name:11s} F5Q={g5:.4f} Q={[round(a,4) for a in fq]} params={npar} pool_k={pool_k}", flush=True)
    T.save(m.state_dict(), f"{CK}/CNN_{name}_m8_ds40_1000ns.pth")
    ridx = np.sort(np.random.RandomState(7).choice(len(Xte), 512, replace=False))
    np.save(f"{EX}/CNN_{name}_m8_ds40_input.npy", Xte[ridx].astype(np.float32))
    np.save(f"{EX}/CNN_{name}_m8_ds40_output.npy", prob[ridx].astype(np.float32))
    np.save(f"{EX}/CNN_{name}_m8_ds40_labels.npy", tb[ridx].astype(np.int64))


train_eval("noRes_2conv", 2, 4)
train_eval("1conv", 1, 7)
print("[shallow] done", flush=True)
