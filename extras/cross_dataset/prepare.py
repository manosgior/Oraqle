"""Reduce each downloaded dataset to the common form of common.py (cache under $ORAQLE_WORK).

    python -I prepare.py qick
    python -I prepare.py cao
    python -I prepare.py lienhard     # control; needs $ORAQLE_DATA, derived data stays on the cluster
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from common import CACHE, EXT, write_dataset  # noqa: E402


def prepare_qick():
    """Di Guglielmo et al., Zenodo 14427490. Interleaved I,Q at 2.5 us / 770 samples.
    The last sample is a glitch in both classes (the state contrast drops by half), so L = 769."""
    d = EXT / "qick_14427490"
    out = CACHE / "qick"
    out.mkdir(parents=True, exist_ok=True)
    for split, tag in (("train", "tr"), ("test", "te")):
        X = np.load(d / f"0528_X_{split}_0_770.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(d / f"0528_y_{split}_0_770.npy", allow_pickle=False).astype(np.int8)
        N = X.shape[0]
        Xo = np.lib.format.open_memmap(out / f"X{tag}.npy", mode="w+", dtype=np.float32, shape=(N, 769, 2))
        for a in range(0, N, 50000):
            b = min(a + 50000, N)
            Xo[a:b] = np.asarray(X[a:b]).reshape(b - a, 770, 2)[:, :769, :].astype(np.float32)
        Xo.flush()
        np.save(out / f"y{tag}.npy", y)
    import json

    (out / "meta.json").write_text(json.dumps(dict(
        dt_ns=2500 / 770, already_lowpassed=False, t0_note="T=0 is the first sample; the state contrast rises from about 0 there",
        source="zenodo.org/records/14427490", qubits=1, platform="transmon, QICK on ZCU216",
        samples=769, n_train=900000, n_test=100000), indent=2))
    print("qick done")


def _cao_centers(z_first, ref):
    from sklearn.mixture import GaussianMixture

    P = np.c_[z_first.real, z_first.imag]
    gm = GaussianMixture(3, covariance_type="tied", means_init=np.c_[ref.real, ref.imag], random_state=0).fit(P[::2])
    return gm.means_[:, 0] + 1j * gm.means_[:, 1]


def prepare_cao(n_delays=4, test_frac=0.2):
    """Cao et al., Harvard Dataverse DPMID3, natural_decay files.

    Rows alternate (herald, measurement). Each file holds 110,000 pairs over a 120-point delay
    sequence with 250 ns steps. A shot keeps its label only at short delay, so the delay indices
    below n_delays (at most 750 ns) are used, and shots are heralded on the g blob of the first
    measurement (nearest centre, the first test of the authors' get_population; they add a
    stricter quadrant cut, which would keep a quarter of the data).

    Sample period: not stated in the deposit. 20 ns is inferred from the 50 MHz readout IF and the
    1.2 us measurement time of the paper, which equals the notebook's 60-sample window (20:80).
    Blobs are refit per file, starting from the centres validated on the first files.
    """
    d = EXT / "cao_DPMID3"
    ROT = np.exp(1j * np.pi / 2)  # the notebook's -90 degree rotation
    ref = np.array([-3.47 + 5.9j, -4.91 + 0.31j, 1.99 - 1.11j])  # g, e, f
    S, W0, W1 = 120, 20, 80
    import h5py

    per_state = {s: {"X": [], "file": [], "j": []} for s in "gef"}
    stats = []
    for s in "gef":
        files = sorted(glob.glob(f"{d}/001_prep_{s}_natural_decay_*"), key=lambda p: int(p.rsplit("_", 1)[1]))
        # One file in the deposit (prep_g 59) is not a valid HDF5 file; skip anything without the signature.
        bad = [p for p in files if open(p, "rb").read(8) != b"\x89HDF\r\n\x1a\n"]
        files = [p for p in files if p not in bad]
        print(s, len(files), "files, skipped", [os.path.basename(p) for p in bad], flush=True)
        for k, p in enumerate(files):
            with h5py.File(p, "r") as f:
                Z = (f["I_data"][:] + 1j * f["Q_data"][:]) * ROT
            w = Z[:, W0:W1].mean(1)
            first, second = w[0::2], w[1::2]
            n = min(len(first), len(second))
            c = _cao_centers(first[:n], ref)
            ok = np.argmin(np.abs(first[:n, None] - c[None]), axis=1) == 0
            j = np.arange(n) % S
            keep = ok & (j < n_delays)
            stats.append(dict(state=s, file=int(p.rsplit("_", 1)[1]), herald=float(ok.mean()), kept=int(keep.sum()),
                              g=complex(c[0]).__repr__(), e=complex(c[1]).__repr__(), f=complex(c[2]).__repr__()))
            T = Z[1::2][:n][keep]
            per_state[s]["X"].append(np.stack([T.real, T.imag], -1).astype(np.float32))
            per_state[s]["file"].append(np.full(len(T), k, dtype=np.int32))
            per_state[s]["j"].append(j[keep].astype(np.int8))
    arr = {s: {k: np.concatenate(v) for k, v in per_state[s].items()} for s in per_state}

    # Split by file, so nearby shots of one run never sit on both sides of the split.
    def split(s):
        a = arr[s]
        nf = int(a["file"].max()) + 1
        is_test = a["file"] >= nf - max(1, int(round(nf * test_frac)))
        return a["X"][~is_test], a["X"][is_test]

    Xg_tr, Xg_te = split("g")
    Xe_tr, Xe_te = split("e")
    Xf_tr, Xf_te = split("f")
    Xtr = np.concatenate([Xg_tr, Xe_tr])
    ytr = np.r_[np.zeros(len(Xg_tr)), np.ones(len(Xe_tr))]
    Xte = np.concatenate([Xg_te, Xe_te])
    yte = np.r_[np.zeros(len(Xg_te)), np.ones(len(Xe_te))]
    # The mean traces show the pulse arriving at sample 12 and switching off near sample 36; after that the
    # state information keeps coming in the ring-down, so this is a pulse-plus-ring-down readout.
    # The ring-down of |<g>| fits the paper's kappa/2pi = 0.761 MHz at about 18 ns per sample.
    t0 = 12
    write_dataset("cao", Xtr[:, t0:], ytr, Xte[:, t0:], yte, dt_ns=20.0, already_lowpassed=False,
                  dt_note="20 ns inferred, not stated in the deposit", t0_sample=t0, herald="nearest blob = g",
                  delays_ns=[250 * j for j in range(n_delays)], source="doi:10.7910/DVN/DPMID3", qubits=1,
                  platform="transmon with SNAIL, Hatridge lab", readout_kappa_MHz=0.761, T1_us_from_paper=19.2)
    np.save(CACHE / "cao" / "Xf_te.npy", Xf_te[:, t0:])
    np.save(CACHE / "cao" / "Xf_tr.npy", Xf_tr[:, t0:])
    import json

    (CACHE / "cao" / "file_stats.json").write_text(json.dumps(stats))
    print("cao", Xtr.shape, Xte.shape, "herald mean", np.mean([s["herald"] for s in stats]))


def prepare_lienhard(keep_samples=500):
    """Control: the paper's dataset, one demodulated channel at a time, through the same code path
    as the external datasets. Uses the demodulation of readout/baseline_no_ML.py (10 MHz Butterworth,
    already low-passed). Derived data stays under $ORAQLE_WORK; nothing is committed."""
    import types

    for m in ("matplotlib", "matplotlib.pyplot", "matplotlib.gridspec", "seaborn"):
        sys.modules.setdefault(m, types.ModuleType(m))  # plotting only; never called here
    sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]
    sys.modules["matplotlib"].gridspec = sys.modules["matplotlib.gridspec"]
    import h5py
    import baseline_no_ML as B

    B.require_dataset()
    for q in range(5):
        (CACHE / f"lienhard_q{q + 1}").mkdir(parents=True, exist_ok=True)
    mm = {}
    for key, path, tag in (("train", B.RAW_TRAIN_FILE, "tr"), ("test", B.RAW_TEST_FILE, "te")):
        with h5py.File(path, "r") as hf:
            X, y = hf[f"X_{key}"], hf[f"y_{key}"][:]
            N = X.shape[0]
            for q in range(5):
                mm[q] = np.lib.format.open_memmap(CACHE / f"lienhard_q{q + 1}" / f"X{tag}.npy", mode="w+",
                                                  dtype=np.float32, shape=(N, keep_samples, 2))
                np.save(CACHE / f"lienhard_q{q + 1}" / f"y{tag}.npy", ((y >> q) & 1).astype(np.int8))
            for a in range(0, N, 40000):
                b = min(a + 40000, N)
                dem = B.demodulate(X[a:b], B.FREQ_READOUT, dt=B.DT)  # (n, 1000, 5, 2)
                for q in range(5):
                    mm[q][a:b] = dem[:, :keep_samples, q, :]
                print(key, b, flush=True)
            for q in range(5):
                mm[q].flush()
    import json

    for q in range(5):
        (CACHE / f"lienhard_q{q + 1}" / "meta.json").write_text(json.dumps(dict(
            dt_ns=2.0, already_lowpassed=True, qubit=f"Q{q + 1}", source="Lienhard et al. (not public)", qubits=1,
            samples=keep_samples, n_train=480000, n_test=1120000), indent=2))
    print("lienhard done")


if __name__ == "__main__":
    {"qick": prepare_qick, "cao": prepare_cao, "lienhard": prepare_lienhard}[sys.argv[1]]()
