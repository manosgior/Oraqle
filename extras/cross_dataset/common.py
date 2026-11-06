"""Shared paths, dataset access and metrics for the cross-dataset check.

Every dataset is reduced to the same form: single-qubit baseband traces of shape
(N, L, 2) float32, a binary prepared-state label, an official train/test split and a
sample period in ns. All time constants in the models are fixed in ns, not in samples.

Run scripts with `python -I`, so that nothing next to the data or in the working
directory can be imported. Downloaded files are read as data only: .npy without pickle,
HDF5 and MATLAB through their parsers.
"""
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "discriminators", ROOT / "readout", pathlib.Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from oraqle_env import RESULTS, WORK  # noqa: E402

XD = WORK / "cross_dataset"
CACHE = XD / "cache"
RUNS = XD / "runs"
EXT = pathlib.Path(os.environ.get("ORAQLE_EXTERNAL", str(WORK / "external")))


class Dataset:
    """A prepared dataset. Arrays are memory-mapped; use `split` to load a length L."""

    def __init__(self, name):
        self.name = name
        self.dir = CACHE / name
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.dt = float(self.meta["dt_ns"])
        self.Xtr = np.load(self.dir / "Xtr.npy", mmap_mode="r", allow_pickle=False)
        self.Xte = np.load(self.dir / "Xte.npy", mmap_mode="r", allow_pickle=False)
        self.ytr = np.load(self.dir / "ytr.npy", allow_pickle=False).astype(np.int64)
        self.yte = np.load(self.dir / "yte.npy", allow_pickle=False).astype(np.int64)
        self.filtered = bool(self.meta.get("already_lowpassed", False))

    def samples(self, T_ns):
        L = int(round(T_ns / self.dt))
        if L > self.Xtr.shape[1]:
            raise ValueError(f"{self.name}: T={T_ns} ns needs {L} samples, have {self.Xtr.shape[1]}")
        return L

    def split(self, L, max_train=None, seed=42):
        """(Xtr[:L], ytr, Xte[:L], yte) as float32 arrays, optionally subsampling train."""
        if max_train is not None and max_train < len(self.ytr):
            idx = np.sort(np.random.RandomState(seed).choice(len(self.ytr), size=max_train, replace=False))
            Xtr, ytr = np.array(self.Xtr[idx, :L, :], dtype=np.float32), self.ytr[idx]
        else:
            Xtr, ytr = np.array(self.Xtr[:, :L, :], dtype=np.float32), self.ytr
        return Xtr, ytr, np.array(self.Xte[:, :L, :], dtype=np.float32), self.yte


def write_dataset(name, Xtr, ytr, Xte, yte, **meta):
    d = CACHE / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "Xtr.npy", Xtr)
    np.save(d / "Xte.npy", Xte)
    np.save(d / "ytr.npy", np.asarray(ytr, dtype=np.int8))
    np.save(d / "yte.npy", np.asarray(yte, dtype=np.int8))
    meta.update(n_train=int(len(ytr)), n_test=int(len(yte)), samples=int(Xtr.shape[1]))
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def fidelity(y, pred):
    """Assignment fidelity as in the paper: F = 1 - [P(0|1) + P(1|0)] / 2."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    p10 = float(np.mean(pred[y == 0] == 1))
    p01 = float(np.mean(pred[y == 1] == 0))
    return 1.0 - 0.5 * (p01 + p10), p01, p10


def lowpass(X, dt_ns, cutoff_hz=10e6):
    """The readout baseline's filter: 3rd-order Butterworth low-pass, causal, along time."""
    from scipy.signal import butter, sosfilt

    fs = 1e9 / dt_ns
    if cutoff_hz >= 0.5 * fs:  # already slower than the filter: nothing to remove
        return X
    sos = butter(3, cutoff_hz, btype="low", fs=fs, output="sos")
    return sosfilt(sos, X, axis=1).astype(np.float32)


def run_path(dataset, model, T_ns, seed, tag=""):
    d = RUNS / dataset
    d.mkdir(parents=True, exist_ok=True)
    t = f"_{tag}" if tag else ""
    return d / f"{model}{t}_T{int(T_ns)}_s{seed}"
