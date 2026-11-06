"""Readout-axis cross-check: F(T), dF/dT and sliding windows, using the paper's own functions.

Calls sweep_fidelity_incremental, evaluate_sliding_window_fidelity and derivative_critical_window
from readout/baseline_no_ML.py unchanged (70/30 split of all shots, logistic regression refit at each T).

    python -I sweep_linear.py qick cao          # prepared datasets (see prepare.py)
    python -I sweep_linear.py hassani gunyho    # small time-resolved deposits, read directly
    python -I sweep_linear.py zheng             # true duration sweep: one integrated point per shot

Writes $ORAQLE_WORK/cross_dataset/linear_<name>.json.
"""
import glob
import json
import os
import re
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from common import EXT, XD, Dataset, lowpass  # noqa: E402

for _m in ("matplotlib", "matplotlib.pyplot", "matplotlib.gridspec", "seaborn"):
    sys.modules.setdefault(_m, types.ModuleType(_m))  # plotting only; never called here
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]
sys.modules["matplotlib"].gridspec = sys.modules["matplotlib.gridspec"]
import baseline_no_ML as B  # noqa: E402


def scan_start(X, y, dt_ns, T_list, starts=(0, 50, 100, 200, 300, 400, 500)):
    """Linear integration over [s, T] at a fixed end time T: does skipping the first samples help?"""
    perm = np.random.default_rng(B.SAMPLE_SEED).permutation(len(y))
    n = int(0.7 * len(y))
    tr, te = perm[:n], perm[n:]
    out = {}
    for T in T_list:
        row = {}
        for s in starts:
            if s >= T:
                continue
            a, b = int(round(s / dt_ns)), int(round(T / dt_ns))
            F = np.trapezoid(X[:, a:b], axis=1)
            c = B.fit_threshold(F[tr], y[tr])
            row[s] = float(B.assignment_fidelity(y[te], c.predict(F[te])))
        out[int(T)] = row
    return out


def sweep(X, y, dt_ns, T_grid_ns, windows_ns=(250, 500), step_ns=100, lp=False):
    """X is (N, L, 2) baseband, y is binary. Returns F(T), dF/dT and sliding-window curves.
    No low-pass by default: the published Linear Threshold row matches no filter (check_linear_filter.py)."""
    if lp:
        X = lowpass(X, dt_ns)
    demod = X[:, :, None, :]  # (N, T, n_qubits=1, 2)
    dt = dt_ns * 1e-9
    perm = np.random.default_rng(B.SAMPLE_SEED).permutation(len(y))
    demod, y = demod[perm], np.asarray(y)[perm]
    idx = np.unique(np.minimum((np.asarray(T_grid_ns) / dt_ns).round().astype(int), X.shape[1] - 1))
    idx = idx[idx >= 1]
    F = B.sweep_fidelity_incremental(demod, y, idx, dt=dt)
    dF = B.derivative_critical_window(F, idx * dt) if len(idx) > 3 else np.full_like(F, np.nan)
    out = dict(T_ns=(idx * dt_ns).tolist(), F=F[0].tolist(), dFdT_per_us=(dF[0] * 1e-6).tolist())
    for w in windows_ns:
        if w / dt_ns < 1 or w >= X.shape[1] * dt_ns:
            continue
        c, f = B.evaluate_sliding_window_fidelity(demod, y, window_ns=w, step_ns=step_ns, dt=dt)
        out[f"window_{w}ns"] = dict(center_us=np.asarray(c).tolist(), F=f[0].tolist())
    return out


def from_cache(name, T_grid_ns):
    ds = Dataset(name)
    X = np.concatenate([np.array(ds.Xtr, dtype=np.float32), np.array(ds.Xte, dtype=np.float32)])
    y = np.r_[ds.ytr, ds.yte]
    ends = [t for t in (600, 1000, 1300, 1500, 2000) if t <= X.shape[1] * ds.dt]
    return dict(meta=ds.meta, start_scan=scan_start(X, y, ds.dt, ends), **sweep(X, y, ds.dt, T_grid_ns))


def hassani():
    """Fig. 7b: 501 two-pulse traces per state and power, 64 samples at 50.79 ns. Pulse 1 only.
    The acquisition starts before the pulse: |<x>| leaves its floor at sample 6 (305 ns), so T = 0 is there."""
    import scipy.io as sio

    t0 = 6

    files = glob.glob(str(EXT / "hassani_8004359/unz/*/Fig7/fig7b/2pulseQND_*_neg*.mat"))
    groups = {}
    for p in files:
        m = re.search(r"2pulseQND_(g|e)_(neg[\d.,]+)_", os.path.basename(p))
        if m:
            groups.setdefault(m.group(2), {})[m.group(1)] = p
    res = {}
    for power, g in sorted(groups.items()):
        if set(g) != {"g", "e"}:
            continue
        X, y = [], []
        for lab, s in enumerate("ge"):
            d = sio.loadmat(g[s])
            key = {k.split("_")[0]: k for k in d if not k.startswith("__")}
            t = d[key["Timens"]].ravel()
            X.append(np.stack([d[key["I"]], d[key["Q"]]], -1).astype(np.float32)[:, t0:])
            y.append(np.full(len(X[-1]), lab))
        X, y = np.concatenate(X), np.concatenate(y)
        dt = float(np.median(np.diff(t)))
        res[power] = dict(dt_ns=dt, n=len(y), **sweep(X, y, dt, np.arange(dt, X.shape[1] * dt, dt), windows_ns=(500,), step_ns=dt))
        print(power, "best F", round(max(res[power]["F"]), 4), flush=True)
    return dict(meta=dict(source="zenodo.org/records/8004359", platform="inductively shunted transmon (ISTA)", dt_ns=dt, t0_ns=t0 * dt), powers=res)


def gunyho():
    """Bolometer run 76: 1,000 shots with and without a pi pulse, 781 samples at 64 ns.
    The file holds no pulse timing, and no signal step is visible in |<x>|. T = 0 is put at 5 us, where the
    state contrast first clears the noise of the mean (about 3 sigma), so it is uncertain by about 1 us."""
    import h5py

    t0 = 78  # samples, 4.99 us

    with h5py.File(EXT / "gunyho_10000511/unz/data/bolometer-characterization-timetrace-76.nc", "r") as f:
        re_, im = f["readout_meta_CH0_re"][:], f["readout_meta_CH0_im"][:]
        t = f["readout_meta_time"][:]
    dt = float(np.median(np.diff(t)) * 1e9)
    X = np.stack([re_, im], -1).astype(np.float32).reshape(-1, re_.shape[2], 2)[:, t0:]
    y = np.repeat([0, 1], re_.shape[1])
    return dict(meta=dict(source="zenodo.org/records/10000511", platform="transmon read by a bolometer (Aalto)", dt_ns=dt, t0_ns=t0 * dt),
                **sweep(X, y, dt, np.arange(dt * 8, X.shape[1] * dt, dt * 8), windows_ns=(2000, 8000), step_ns=1000))


def zheng():
    """True duration sweep: separate experiments at each t_RO, one integrated I/Q point per shot.

    Uses the first readout of each run of the I sequence, with the two branches from the data README's
    table. Not the ground_data/excited_data arrays: on the transmon their noise does not change along
    the sweep, so they were taken at one fixed readout length and say nothing about duration."""
    import h5py
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    root = glob.glob(str(EXT / "zheng_23367046/unz/RED_package/data"))[0]
    out = {}
    # (device, run in |g>, run in |e>, name of the sweep axis)
    for dev, rg, re_, axis in (("transmon", "03664", "03665", "sweep_param"), ("fluxonium1", "32504", "32505", "sweep_param1")):
        d = {}
        for lab, r in ((0, rg), (1, re_)):
            with h5py.File(glob.glob(f"{root}/{dev}/{r}-*/data.ddh5")[0], "r") as f:
                g = f["data"]
                t = np.asarray(g[axis][:], dtype=float)
                t = np.unique(t.ravel())
                d[lab] = g["data"][:, :, 0]  # (n_tRO, n_runs), first readout of each run
        rng = np.random.default_rng(B.SAMPLE_SEED)
        Fs = []
        for k in range(d[0].shape[0]):
            Z = np.r_[d[0][k], d[1][k]]
            X = np.c_[Z.real, Z.imag] * 1e6  # volts are tiny; only matters for numerics
            y = np.r_[np.zeros(d[0].shape[1]), np.ones(d[1].shape[1])]
            perm = rng.permutation(len(y))
            n = int(0.7 * len(y))
            tr, te = perm[:n], perm[n:]
            clf = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=1e6, max_iter=1000))]).fit(X[tr], y[tr])
            Fs.append(float(B.assignment_fidelity(y[te], clf.predict(X[te]))))
        out[dev] = dict(T_ns=(t * 1e9).tolist(), F=Fs, n_per_state=int(d[0].shape[1]))
    return dict(meta=dict(source="zenodo.org/records/23367046", platform="EPFL transmon and fluxonium", kind="true duration sweep"), devices=out)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        if name == "qick":
            r = from_cache("qick", np.arange(100, 2500, 100))
        elif name == "cao":
            r = from_cache("cao", np.arange(40, 1361, 40))
        elif name.startswith("lienhard"):
            r = from_cache(name, np.arange(100, 1000, 100))
        else:
            r = dict(hassani=hassani, gunyho=gunyho, zheng=zheng)[name]()
        (XD / f"linear_{name}.json").write_text(json.dumps(r))
        print(name, "written", flush=True)
