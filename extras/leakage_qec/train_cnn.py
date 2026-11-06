#!/usr/bin/env python3
"""train_cnn.py -- retrain MCMit-CNN with the recipe of system/soft_readout_cnn.py and keep the weights.

The soft-readout run saved only probabilities, and the CNN_m16_* checkpoints in checkpoints/ come from a different
pipeline (they score about 50% error on this preprocessing), so the L-flag probe needs its own model.
Usage: ORAQLE_DATA=... CNN_LEN=<samples> python train_cnn.py --out DIR     (CPU: about 1 h per length)
"""
import argparse, os, pathlib, sys, time
import numpy as np
import torch as T
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "discriminators" / "runners"), str(ROOT / "discriminators" / "networks")]
import _colleague_prep as P  # noqa: E402
from CNN import CNN  # noqa: E402
from oraqle_env import RAW_TEST, RAW_TRAIN, require_dataset  # noqa: E402

T.manual_seed(1)
np.random.seed(1)
DS, LR, DECAY, STEP, BS, EPOCHS = 40, 5e-4, 0.95, 3, 64, 30

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
require_dataset()
L = int(os.environ["CNN_LEN"])
T.set_num_threads(int(os.environ.get("THREADS", "8")))
prm = P.calibrate(str(RAW_TRAIN), "X_train", "y_train", 15000, (0, 500))
Xtr_raw, ytr = P.preprocess(str(RAW_TRAIN), "X_train", "y_train", prm, DS, (0, L))
Xte_raw, yte = P.preprocess(str(RAW_TEST), "X_test", "y_test", prm, DS, (0, L))
Xtr, Xte = P.to_torch_layout(Xtr_raw), P.to_torch_layout(Xte_raw)
Ytr, Yte = P.bits(ytr), P.bits(yte)
Xtr, Xv, Ytr, Yv = train_test_split(Xtr, Ytr, test_size=0.2, random_state=42)
model = CNN(in_channels=10, m_param=16, num_qubits=5)
crit = T.nn.BCEWithLogitsLoss()
opt = T.optim.Adam(model.parameters(), lr=LR)
sched = T.optim.lr_scheduler.StepLR(opt, step_size=STEP, gamma=DECAY)
loader = T.utils.data.DataLoader(T.utils.data.TensorDataset(T.tensor(Xtr), T.tensor(Ytr)), batch_size=BS, shuffle=True)
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    for xb, yb in loader:
        loss = crit(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    sched.step()
    print(f"L={L} epoch {ep + 1}/{EPOCHS} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
model.eval()
with T.no_grad():
    Xt = T.tensor(Xte)
    pr = np.concatenate([(model(Xt[s:s + 8192]) > 0).numpy() for s in range(0, len(Xt), 8192)])
print(f"L={L} TEST error % per qubit", np.round(100 * (pr != Yte.astype(bool)).mean(0), 3), flush=True)
os.makedirs(a.out, exist_ok=True)
T.save(model.state_dict(), f"{a.out}/cnn_len{L}.pth")
