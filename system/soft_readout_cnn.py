"""Per-shot soft outputs of the MCMit-CNN at one readout length (part E).

Same preprocessing, model and training recipe as the paper's CNN row
(MCMit/qubit_state_discrimination/runners/_cnn_length_sweep.py; CNN.py and _colleague_prep.py are imported
from <repo>/discriminators/{networks,runners}, byte-identical to the qsd/ copies the paper run used):
FFT frequency calibration on a fixed 500-sample window, FIR + phase rotation, downsampling by
40, CNN m=16, 30 epochs, retrained per length. Instead of only the assignment fidelity it saves the
sigmoid probability of |1> for every test shot and qubit, so the confidence of each measurement can be
turned into an erasure flag.

Writes <out>/cnn_soft_<T>ns.npz with prob (N, 5, float16), label (N, 5, uint8) and the per-qubit
fidelities (balanced, as the paper). Needs the raw Lienhard et al. dataset ($ORAQLE_DATA) and, for the
paper's runtime, a GPU (oraqle-torch-gpu image). Usage (env CNN_LEN = raw samples, 2 ns each):
  ORAQLE_DATA=/data CNN_LEN=300 python system/soft_readout_cnn.py --out $ORAQLE_WORK/system/soft
"""
import argparse
import os
import sys

import numpy as np
import torch as T
from sklearn.model_selection import train_test_split

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
# was: sys.path.insert(0, <oraqle-ics>/sims/qsd), a byte copy of these two files
# (CNN.py md5 8b269681cdf5a5cba416d1f71a98a1a6, _colleague_prep.py md5 c0e0a6ffcdf73f9ebd2a293b8b6959b9)
sys.path.insert(0, os.path.join(_REPO, "discriminators", "runners"))
sys.path.insert(0, os.path.join(_REPO, "discriminators", "networks"))
import _colleague_prep as P  # noqa: E402
from CNN import CNN  # noqa: E402
from oraqle_env import RAW_TEST, RAW_TRAIN, require_dataset  # noqa: E402

T.manual_seed(1)
np.random.seed(1)
TR = (str(RAW_TRAIN), "X_train", "y_train", 15000)   # was /data/five_qubit_data/DRaw_C_Tr_v0-001
TE = (str(RAW_TEST), "X_test", "y_test", 35000)      # was /data/five_qubit_data/DRaw_C_Te_v0-002
DS = 40
LR, DECAY, STEP, BS, EPOCHS = 5e-4, 0.95, 3, 64, 30
DEV = T.device("cuda" if T.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    require_dataset()
    L = int(os.environ["CNN_LEN"])
    T.set_num_threads(int(os.environ.get("THREADS", "4")))
    print(f"[cnn-soft] L={L} samples ({2 * L} ns) device={DEV}", flush=True)

    prm = P.calibrate(TR[0], TR[1], TR[2], TR[3], (0, 500))
    Xtr_raw, ytr = P.preprocess(TR[0], TR[1], TR[2], prm, DS, (0, L))
    Xte_raw, yte = P.preprocess(TE[0], TE[1], TE[2], prm, DS, (0, L))
    Xtr, Xte = P.to_torch_layout(Xtr_raw), P.to_torch_layout(Xte_raw)
    Ytr, Yte = P.bits(ytr), P.bits(yte)
    Xtr, Xv, Ytr, Yv = train_test_split(Xtr, Ytr, test_size=0.2, random_state=42)
    print(f"[cnn-soft] train {len(Xtr)} test {len(Xte)}", flush=True)

    model = CNN(in_channels=10, m_param=16, num_qubits=5).to(DEV)
    crit = T.nn.BCEWithLogitsLoss()
    opt = T.optim.Adam(model.parameters(), lr=LR)
    sched = T.optim.lr_scheduler.StepLR(opt, step_size=STEP, gamma=DECAY)
    loader = T.utils.data.DataLoader(T.utils.data.TensorDataset(T.tensor(Xtr), T.tensor(Ytr)),
                                     batch_size=BS, shuffle=True)
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            loss = crit(model(xb.to(DEV)), yb.to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    probs = []
    with T.no_grad():
        Xte_t = T.tensor(Xte)
        for s in range(0, len(Xte_t), 8192):
            probs.append(T.sigmoid(model(Xte_t[s:s + 8192].to(DEV))).cpu().numpy())
    prob = np.concatenate(probs, 0)
    tb = Yte.astype(np.uint8)
    pb = (prob > 0.5).astype(np.uint8)
    fq = [1 - ((pb[tb[:, q] == 1, q] == 0).mean() + (pb[tb[:, q] == 0, q] == 1).mean()) / 2 for q in range(5)]
    print(f"[cnn-soft] {2 * L} ns  Q1-5 {[round(float(f), 3) for f in fq]}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, f"cnn_soft_{2 * L}ns.npz"), prob=prob.astype(np.float16), label=tb,
                        fidelity=np.array(fq), length_ns=2 * L)
    print("[cnn-soft] done", flush=True)


if __name__ == "__main__":
    main()
