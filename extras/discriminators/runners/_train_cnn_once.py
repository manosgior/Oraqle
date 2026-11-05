"""
_train_cnn_once.py
==================
Train the CNN baseline once on a CNN-format synthetic dataset (produced by
runners/synth_to_cnn.py) and dump per-qubit test fidelity as JSON.

Mirrors runners/train_CNN.py's recipe (m_param=8, BCEWithLogits, Adam 5e-4,
StepLR 0.95/3, 30 epochs, batch 64, train/val split 0.35, no external
normalization — the CNN's internal BatchNorm handles scale).

    python -m runners._train_cnn_once <cnn_format.hdf5> <result.json> [raw_len]

`raw_len` is the ORIGINAL (pre-downsample) trace length, default 500, used so
prepare_cnn_data computes total_ds_steps = raw_len // downsample_factor
consistently with how the file was built.
"""

import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.CNN import CNN
from helpers.cnn_helpers import prepare_cnn_data

NUM_QUBITS        = 5
DOWNSAMPLE_FACTOR = 20
M_PARAM           = 8
LR                = 5e-4
DECAY_RATE        = 0.95
DECAY_STEP        = 3
VAL_SPLIT         = 0.35
EPOCHS            = 30
BATCH_SIZE        = 64
RANDOM_STATE      = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _per_qubit_accuracy(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            out = torch.sigmoid(model(X.to(DEVICE))).cpu().numpy()
            preds.append(out)
            targets.append(y.numpy())
    preds = (np.vstack(preds) >= 0.5).astype(int)
    targets = np.vstack(targets).astype(int)
    return [float(np.mean(preds[:, q] == targets[:, q])) for q in range(NUM_QUBITS)]


def main():
    if len(sys.argv) < 3:
        print("usage: python -m runners._train_cnn_once <cnn.hdf5> <result.json> [raw_len]")
        sys.exit(1)
    cnn_file = sys.argv[1]
    out_path = sys.argv[2]
    raw_len = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    common = dict(
        downsample_factor=DOWNSAMPLE_FACTOR,
        original_length=raw_len,
        num_qubits=NUM_QUBITS,
        time_slice=(0, raw_len),
    )
    X_train_raw, y_train_raw = prepare_cnn_data(cnn_file, is_test=False, **common)
    X_test, y_test = prepare_cnn_data(cnn_file, is_test=True, **common)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_raw, y_train_raw, test_size=VAL_SPLIT, random_state=RANDOM_STATE,
    )
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = CNN(in_channels=2 * NUM_QUBITS, m_param=M_PARAM, num_qubits=NUM_QUBITS).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_STEP, gamma=DECAY_RATE)

    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for epoch in range(EPOCHS):
        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                vl += criterion(model(X.to(DEVICE)), y.to(DEVICE)).item() * X.size(0)
        vl /= len(val_loader.dataset)
        if vl <= best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    per_q = _per_qubit_accuracy(model, test_loader)

    result = {
        "per_qubit": per_q,
        "raw_len": raw_len,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "cnn_file": cnn_file,
    }
    with open(out_path, "w") as f:
        json.dump(result, f)
    print("WROTE_RESULT:" + out_path + "  per_qubit=" + str(["%.4f" % a for a in per_q]))


if __name__ == "__main__":
    main()
