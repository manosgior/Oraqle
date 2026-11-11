"""One discriminator, one dataset, one readout duration. Single-qubit versions of the paper's
six discriminators and its linear baseline, following each model's Table 2 recipe.

    python -I run_one.py --dataset qick --model cnn --T 1000 --seed 1 --threads 8

Writes <run>.json (fidelity, errors, settings) and <run>_scores.npy (per-shot test score, higher
means "1") under $ORAQLE_WORK/cross_dataset/runs/<dataset>/. A finished run is skipped.

Departures from the Table 2 recipes, all forced by the data or the hardware:
  * one channel and one binary output instead of five channels and 32 classes or five logits;
  * time constants fixed in ns, so filter lengths, decimation and patches scale with the sample period;
  * no IF mixing on baseband data; the CNN's FFT frequency calibration is skipped there (f = 0),
    because on baseband data it would lock onto noise near DC and rotate the signal away;
  * random search in place of Optuna for the FNN and QubiCML. With 10 trials these are equivalent up
    to pruning, because TPE's first 10 trials are random by default (optuna is not in the image);
  * MCMit-T trains the configuration the paper's 50-trial search selected at that duration instead of
    repeating the search: about 8 GPU-days per dataset, and no GPU is free. See --search for a check;
  * HERQULES keeps its hidden sizes, so Net_rmf becomes 2-10-20-2. The boxcar is 1 sample unless the
    dataset is the Lienhard control, which uses the paper's per-qubit values.
"""
import argparse
import glob
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from common import RESULTS, Dataset, fidelity, lowpass, run_path  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from networks.CNN import CNN  # noqa: E402
from networks.KLiNQ_StudentModel import KLiNQStudentModel  # noqa: E402
from networks.Qubic import Arxiv240618807FNN  # noqa: E402
from networks.SingleQubitFNN import SingleQubitFNN  # noqa: E402  (the class of the published FNN and KLiNQ teacher)
from networks.Transformer import QubitClassifierTransformer  # noqa: E402

EPS = 1e-10


def batches(n, bs, shuffle=True):
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for s in range(0, n, bs):
        idx = order[s:s + bs]
        if shuffle and len(idx) < 2:  # BatchNorm cannot take a single sample
            continue
        yield idx


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # MCMit-T only; the small models stay on CPU


@torch.no_grad()
def predict(model, X, bs=8192):
    model.eval()
    dev = next(model.parameters()).device  # data stays on the host; only batches move to the GPU
    return torch.cat([model(X[s:s + bs].to(dev)).cpu() for s in range(0, len(X), bs)])


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


# ------------------------------------------------------------------ linear baseline
def run_linear(ds, L, seed, a):
    """Sec. 4 baseline: trapezoid-integrate to T, standardize, logistic regression. With --lp, a causal
    10 MHz low-pass first, as in the current baseline_no_ML.demodulate. The published row matches no filter
    (check_linear_filter.py), so that is the comparison; the filter costs up to 4 pp at short T."""
    Xtr, ytr, Xte, yte = ds.split(L)
    if a.lp and not ds.filtered:
        Xtr, Xte = lowpass(Xtr, ds.dt), lowpass(Xte, ds.dt)
    Ftr = np.trapezoid(Xtr, dx=ds.dt, axis=1)
    Fte = np.trapezoid(Xte, dx=ds.dt, axis=1)
    clf = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000))])
    clf.fit(Ftr, ytr)
    return clf.decision_function(Fte), dict(params=5)


# ------------------------------------------------------------------ Baseline FNN
def run_fnn(ds, L, seed, a):
    """hyper_optimize_paper.py: 320k training shots, 10 trials x 25 epochs, best validation loss."""
    Xtr, ytr, Xte, yte = ds.split(L, max_train=320000, seed=42)
    F = Xtr.reshape(len(Xtr), -1)
    mu, sd = F.mean(0), F.std(0) + EPS
    F = (F - mu) / sd
    Ft, Fv, yt, yv = train_test_split(F, ytr, test_size=0.2, random_state=42)
    Ft, Fv = torch.from_numpy(Ft), torch.from_numpy(Fv)
    yt, yv = torch.from_numpy(yt), torch.from_numpy(yv)
    Fe = torch.from_numpy((Xte.reshape(len(Xte), -1) - mu) / sd)
    rng = np.random.default_rng(seed)
    crit = nn.CrossEntropyLoss()
    best = (math.inf, None, None)
    for trial in range(a.trials or 10):
        lr = 10 ** rng.uniform(-5, -2)
        bs = int(rng.choice([128, 256, 512, 1024]))
        torch.manual_seed(seed * 1000 + trial)
        m = SingleQubitFNN(input_size=2 * L, output_size=2)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        for ep in range(a.epochs or 25):
            m.train()
            for idx in batches(len(Ft), bs):
                opt.zero_grad()
                crit(m(Ft[idx]), yt[idx]).backward()
                opt.step()
        v = float(crit(predict(m, Fv), yv))
        print(f"  trial {trial} lr={lr:.2e} bs={bs} val={v:.5f}", flush=True)
        if v < best[0]:
            best = (v, {k: t.clone() for k, t in m.state_dict().items()}, dict(lr=lr, batch_size=bs, trial=trial))
    m.load_state_dict(best[1])
    logit = predict(m, Fe)
    return (logit[:, 1] - logit[:, 0]).numpy(), dict(params=n_params(m), val_loss=best[0], **best[2])


# ------------------------------------------------------------------ QubiCML
def run_qubic(ds, L, seed, a):
    """objective_arxiv: time-averaged (I, Q), min-max scaled, 2-8-4-1, 10 trials x 25 epochs."""
    Xtr, ytr, Xte, yte = ds.split(L, max_train=320000, seed=42)
    Ftr, Fte = Xtr.mean(1), Xte.mean(1)
    lo = Ftr.min(0)
    rg = Ftr.max(0) - lo
    rg[rg == 0] = EPS
    Ftr, Fte = (Ftr - lo) / rg, (Fte - lo) / rg
    Ft, Fv, yt, yv = train_test_split(Ftr, ytr, test_size=0.2, random_state=42, stratify=ytr)
    Ft, Fv, Fe = map(lambda x: torch.from_numpy(x.astype(np.float32)), (Ft, Fv, Fte))
    yt = torch.from_numpy(yt.astype(np.float32))[:, None]
    yv = torch.from_numpy(yv.astype(np.float32))[:, None]
    rng = np.random.default_rng(seed)
    crit = nn.BCELoss()
    best = (math.inf, None, None)
    for trial in range(a.trials or 10):
        lr = 10 ** rng.uniform(-4, -2)
        bs = int(rng.choice([64, 128, 256, 512]))
        torch.manual_seed(seed * 1000 + trial)
        m = Arxiv240618807FNN()
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        for ep in range(a.epochs or 25):
            m.train()
            for idx in batches(len(Ft), bs):
                opt.zero_grad()
                crit(m(Ft[idx]), yt[idx]).backward()
                opt.step()
        v = float(crit(predict(m, Fv), yv))
        print(f"  trial {trial} lr={lr:.2e} bs={bs} val={v:.5f}", flush=True)
        if v < best[0]:
            best = (v, {k: t.clone() for k, t in m.state_dict().items()}, dict(lr=lr, batch_size=bs, trial=trial))
    m.load_state_dict(best[1])
    return predict(m, Fe)[:, 0].numpy() - 0.5, dict(params=n_params(m), val_loss=best[0], **best[2])


# ------------------------------------------------------------------ HERQULES
class NetRmf1Q(nn.Module):
    """Net_rmf with its hidden sizes, for one qubit: 2 features, 2 classes."""

    def __init__(self):
        super().__init__()
        self.hid1, self.hid2, self.oupt = nn.Linear(2, 10), nn.Linear(10, 20), nn.Linear(20, 2)
        for lay in (self.hid1, self.hid2, self.oupt):
            nn.init.xavier_uniform_(lay.weight)
            nn.init.zeros_(lay.bias)

    def forward(self, x):
        return self.oupt(torch.relu(self.hid2(torch.relu(self.hid1(x)))))


def _balanced_mf(a, b, rng):
    """mean(a - b) / var(a - b), balancing the classes by random subsampling, as in HERQULES."""
    n = min(len(a), len(b))
    if len(a) > n:
        a = a[rng.choice(len(a), n, replace=False)]
    if len(b) > n:
        b = b[rng.choice(len(b), n, replace=False)]
    d = a - b
    return d.mean(0) / (d.var(0) + EPS)


def run_herqules(ds, L, seed, a):
    """_herqules_length_both.py, deployment_pertrace variant: per-trace MF + RMF scalars into a small net.
    The RMF comes from the purified clouds at 1 us (or at T, if T is longer), as in rmf.fit + predict."""
    rng = np.random.default_rng(seed)
    L_full = max(L, min(int(round(1000 / ds.dt)), ds.Xtr.shape[1]))
    Xtr, ytr, Xte, yte = ds.split(L_full)
    if not ds.filtered:
        Xtr, Xte = lowpass(Xtr, ds.dt), lowpass(Xte, ds.dt)
    # preclassifier.get_traces: purify on the time-averaged clouds at the full window.
    m = Xtr.mean(1)
    z = m[:, 0] + 1j * m[:, 1]
    c0, c1 = z[ytr == 0].mean(), z[ytr == 1].mean()
    r = abs(c0 - c1) / 2
    flat = Xtr.reshape(len(Xtr), -1)
    zero = flat[(ytr == 0) & (np.abs(z - c0) < r)]
    relax = flat[(ytr == 1) & (np.abs(z - c1) >= r) & (np.abs(z - c0) <= r)]
    n_relax = len(relax)
    rmf = _balanced_mf(relax, zero, rng) if n_relax >= 10 else np.zeros(flat.shape[1], np.float32)
    del zero, relax
    # MF at this length, with the boxcar on the last samples.
    Ftr = Xtr[:, :L].reshape(len(Xtr), -1)
    Fte = Xte[:, :L].reshape(len(Xte), -1)
    bc = {"lienhard_q1": 1, "lienhard_q2": 1, "lienhard_q3": 9, "lienhard_q4": 2, "lienhard_q5": 9}.get(ds.name, 1)
    mf = _balanced_mf(Ftr[ytr == 0], Ftr[ytr == 1], rng)
    mf = mf * np.heaviside(2 * (L - bc) - np.arange(2 * L), 1).astype(np.float32)
    feats = lambda F: np.stack([F @ mf, F @ rmf[: F.shape[1]]], 1).astype(np.float32)  # noqa: E731
    Gtr, Gte = feats(Ftr), feats(Fte)
    del Xtr, Xte, flat, Ftr, Fte
    Gt, Gv, yt, yv = train_test_split(Gtr, ytr, test_size=0.35, random_state=seed, stratify=ytr)
    Gt, Gv, Ge = map(torch.from_numpy, (Gt, Gv, Gte))
    yt, yvn = torch.from_numpy(yt), yv
    torch.manual_seed(seed)
    net = NetRmf1Q()
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    best, state = -1.0, None
    for ep in range(a.epochs or 100):
        lr = 0.01 * (0.1 ** sum(ep >= e for e in (30, 60, 90)))  # adjust_learning_rate
        for g in opt.param_groups:
            g["lr"] = lr
        net.train()
        for idx in batches(len(Gt), 512):
            opt.zero_grad()
            crit(net(Gt[idx]), yt[idx]).backward()
            opt.step()
        va = float((predict(net, Gv).argmax(1).numpy() == yvn).mean())
        if va >= best:
            best, state = va, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(state)
    logit = predict(net, Ge)
    return (logit[:, 1] - logit[:, 0]).numpy(), dict(params=n_params(net), val_acc=best, boxcar=bc, n_relax=n_relax)


# ------------------------------------------------------------------ KLiNQ
def _bins(X, n):
    g = np.array_split(np.arange(X.shape[1]), n)
    return np.stack([X[:, i, 0].mean(1) for i in g] + [X[:, i, 1].mean(1) for i in g], 1).astype(np.float32)


def _teacher(ds, seed, a):
    """One teacher per dataset and window: 1 us as in the paper, or the whole trace for longer T."""
    from common import XD

    L = a.teacher_samples
    path = XD / "runs" / ds.name / f"klinq_teacher_L{L}_s{seed}.pt"
    Xtr, ytr, _, _ = ds.split(L)
    F = Xtr.reshape(len(Xtr), -1)
    mu, sd = F.mean(0), F.std(0) + EPS
    if path.exists():
        return torch.load(path, weights_only=True), mu, sd
    F = (F - mu) / sd
    idx_tr, idx_va = train_test_split(np.arange(len(F)), test_size=0.1, random_state=42, stratify=ytr)
    Ft, Fv = torch.from_numpy(F[idx_tr]), torch.from_numpy(F[idx_va])
    yt = torch.from_numpy(ytr[idx_tr].astype(np.float32))[:, None]
    yv = torch.from_numpy(ytr[idx_va].astype(np.float32))[:, None]
    torch.manual_seed(seed)
    m = SingleQubitFNN(input_size=2 * L, output_size=1)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.75, patience=15, cooldown=2)
    best, state, cnt = None, None, 0
    for ep in range(300):
        m.train()
        for idx in batches(len(Ft), 1024):
            opt.zero_grad()
            crit(m(Ft[idx]), yt[idx]).backward()
            opt.step()
        v = float(crit(predict(m, Fv), yv))
        sched.step(v)
        if best is None or v < best - 1e-4:
            best, state, cnt = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
        else:
            cnt += 1
            if cnt >= 15:
                break
    print(f"  teacher stopped at epoch {ep + 1}, val {best:.5f}, params {n_params(m)}", flush=True)
    torch.save(state, path)
    return state, mu, sd


def run_klinq(ds, L, seed, a):
    """reproduce_klinq_paper.py. Student: `bins` averaged bins plus one MF scalar, KD T=4, soft 0.2, CE 0.8."""
    n_bins = min(a.bins, L)
    L1 = min(int(round(1000 / ds.dt)), ds.Xtr.shape[1])
    a.teacher_samples = L1 if L <= L1 else ds.Xtr.shape[1]
    state, tmu, tsd = _teacher(ds, seed, a)
    Xfull, ytr, _, yte = ds.split(a.teacher_samples)
    Xtr, _, Xte, _ = ds.split(L)
    teacher = SingleQubitFNN(input_size=2 * a.teacher_samples, output_size=1)
    teacher.load_state_dict(state)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    Ftea = (Xfull.reshape(len(Xfull), -1) - tmu) / tsd
    Ftr, Fte = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)
    n = min((ytr == 0).sum(), (ytr == 1).sum())
    d = Ftr[ytr == 0][:n] - Ftr[ytr == 1][:n]
    env = (d.mean(0) / (d.var(0) + EPS)).astype(np.float32)
    atr, ate = _bins(Xtr, n_bins), _bins(Xte, n_bins)
    mtr, mte = (Ftr @ env)[:, None], (Fte @ env)[:, None]
    amu, asd, mmu, msd = atr.mean(0), atr.std(0) + EPS, mtr.mean(0), mtr.std(0) + EPS
    Str = np.concatenate([(atr - amu) / asd, (mtr - mmu) / msd], 1).astype(np.float32)
    Ste = np.concatenate([(ate - amu) / asd, (mte - mmu) / msd], 1).astype(np.float32)
    tr, va = train_test_split(np.arange(len(Str)), test_size=0.1, random_state=42, stratify=ytr)
    St, Sv, Ge = torch.from_numpy(Str[tr]), torch.from_numpy(Str[va]), torch.from_numpy(Ste)
    Tt = torch.from_numpy(Ftea[tr].astype(np.float32))
    yt = torch.from_numpy(ytr[tr].astype(np.float32))[:, None]
    yv = torch.from_numpy(ytr[va].astype(np.float32))[:, None]
    torch.manual_seed(seed)
    m = KLiNQStudentModel(input_size=Str.shape[1])
    bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.75, patience=5, cooldown=2)
    TEMP, SOFT, CE = 4.0, 0.2, 0.8
    best, st, cnt = None, None, 0
    for ep in range(a.epochs or 50):
        m.train()
        for idx in batches(len(St), 1024):
            opt.zero_grad()
            with torch.no_grad():
                pt = torch.sigmoid(teacher(Tt[idx]) / TEMP)
            lo = m(St[idx])
            loss = SOFT * mse(torch.sigmoid(lo / TEMP), pt) * TEMP ** 2 + CE * bce(lo, yt[idx])
            loss.backward()
            opt.step()
        v = float(bce(predict(m, Sv), yv))
        sched.step(v)
        if best is None or v < best - 1e-4:
            best, st, cnt = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
        else:
            cnt += 1
            if cnt >= 5:
                break
    m.load_state_dict(st)
    return predict(m, Ge)[:, 0].numpy(), dict(params=n_params(m), bins=n_bins, val_loss=best, teacher_samples=a.teacher_samples)


# ------------------------------------------------------------------ MCMit-T
def _paper_transformer_config(T_ns):
    """The configuration of the paper's 50-trial search at the nearest published duration."""
    L = int(min(max(round(T_ns / 2 / 100) * 100, 100), 500))
    f = sorted(glob.glob(str(RESULTS / "discriminators" / f"Transformer_len{L}_*.csv")))[-1]
    r = list(csv.DictReader(open(f)))[0]
    return dict(paper_L=L, lr=float(r["learning_rate"]), wd=float(r["weight_decay"]), batch_size=int(r["batch_size"]),
                patch_ns=2 * int(r["patch_size"]), d=int(r["embedding_dim"]), h=int(r["num_heads"]),
                layers=int(r["num_layers"]), dropout=float(r["dropout"]), hard_fraction=float(r["hard_fraction"]))


def run_transformer(ds, L, seed, a):
    """hyper_optimize_transformer.py minus the search: AdamW, warmup + cosine, clipping, 30 epochs, 80/20 split."""
    c = _paper_transformer_config(a.T)
    patch = max(1, int(round(c["patch_ns"] / ds.dt)))
    Lp = max(patch, int(round(L / patch)) * patch)  # the patch size must divide L
    Lp = min(Lp, ds.Xtr.shape[1] - ds.Xtr.shape[1] % patch)
    Xtr, ytr, Xte, yte = ds.split(Lp)
    Xt, Xv, yt, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=42, stratify=ytr)
    mu, sd = Xt.mean((0, 1), keepdims=True), Xt.std((0, 1), keepdims=True) + EPS
    Xt, Xv, Xe = (Xt - mu) / sd, (Xv - mu) / sd, (Xte - mu) / sd
    # the traces stay in host memory and move to the GPU one batch at a time, so a run needs well under 2 GB of GPU memory
    Xt, Xv, Xe = (torch.from_numpy(a) for a in (Xt, Xv, Xe))
    yt, yv = torch.from_numpy(yt), torch.from_numpy(yv)
    torch.manual_seed(seed)
    m = QubitClassifierTransformer(num_classes=2, patch_size=patch, embedding_dim=c["d"], num_heads=c["h"],
                                   num_layers=c["layers"], dropout=c["dropout"]).to(DEV)
    crit = nn.CrossEntropyLoss()
    decay, no_decay = [], []
    for name, p in m.named_parameters():
        (no_decay if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": c["wd"]}, {"params": no_decay, "weight_decay": 0.0}], lr=c["lr"])
    epochs = a.epochs or 30

    def lam(ep):
        if ep < 3:
            return (ep + 1) / 3
        pr = min(max((ep - 3) / max(1, epochs - 3), 0.0), 1.0)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * pr))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    for ep in range(epochs):
        t0 = time.time()
        m.train()
        for idx in batches(len(Xt), c["batch_size"]):
            opt.zero_grad(set_to_none=True)
            crit(m(Xt[idx].to(DEV, non_blocking=True)), yt[idx].to(DEV, non_blocking=True)).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  ep {ep} val {float(crit(predict(m, Xv, 2048), yv)):.5f} ({time.time() - t0:.0f} s)", flush=True)
    val = float(crit(predict(m, Xv, 2048), yv))
    logit = predict(m, Xe, 2048)
    c["T_actual_ns"] = Lp * ds.dt
    c["device"] = str(DEV)
    return (logit[:, 1] - logit[:, 0]).numpy(), dict(params=n_params(m), val_loss=val, patch_samples=patch, **c)


# ------------------------------------------------------------------ MCMit-CNN
def _cnn_features(X, dt, f_hz, theta):
    """_colleague_prep.preprocess at fixed ns: FIR 5 MHz over 200 ns, decimate to 80 ns, rotate.
    Returns (N, 2, T_ds)."""
    from scipy.signal import firwin

    numtaps = int(round(200 / dt)) + 1
    numtaps += (numtaps + 1) % 2  # keep it odd
    ds_factor = max(1, int(round(80 / dt)))
    fs = 1e9 / dt
    taps = firwin(numtaps, 5e6, fs=fs) if 5e6 < 0.5 * fs and numtaps >= 3 else np.ones(1)
    half = len(taps) // 2
    idxs = np.arange(0, X.shape[1], ds_factor)
    out = np.zeros((len(X), 2, len(idxs)), np.float32)
    t = np.arange(X.shape[1]) * dt * 1e-9
    lo = np.exp(-2j * np.pi * f_hz * t)
    for a in range(0, len(X), 20000):
        c = X[a:a + 20000]
        z = (c[:, :, 0] + 1j * c[:, :, 1]) * lo
        f = np.zeros((len(z), len(idxs)), complex)
        for i, k in enumerate(idxs):
            s, e = k - half, k + half + 1
            seg = z[:, max(0, s):min(z.shape[1], e)]
            seg = np.pad(seg, ((0, 0), (max(0, -s), max(0, e - z.shape[1]))))
            f[:, i] = seg @ taps[::-1]
        f = f * np.exp(1j * theta)
        out[a:a + len(z), 0] = f.real
        out[a:a + len(z), 1] = f.imag
    return out


def run_cnn(ds, L, seed, a):
    """_cnn_length_sweep.py: CNN m=16, 30 epochs, lr 5e-4 with step decay, batch 64, 80/20 split, final epoch."""
    np.random.seed(seed)
    Xtr, ytr, Xte, yte = ds.split(L)
    # Calibration on a fixed window of about 1 us, on 16,000 training shots, as in _colleague_prep.calibrate.
    Lcal = min(int(round(1000 / ds.dt)), ds.Xtr.shape[1])
    idx = np.sort(np.random.RandomState(5000).choice(len(ds.ytr), size=min(16000, len(ds.ytr)), replace=False))
    Xc = np.array(ds.Xtr[idx, :Lcal, :], dtype=np.float32)
    yc = ds.ytr[idx]
    z = Xc[:, :, 0] + 1j * Xc[:, :, 1]
    f_cal = 0.0  # every dataset here, including the demodulated control channels, is at baseband
    theta = -np.angle(z[yc == 1].mean() - z[yc == 0].mean())
    Ftr = _cnn_features(Xtr, ds.dt, f_cal, theta)
    Fte = _cnn_features(Xte, ds.dt, f_cal, theta)
    Ft, _, yt, _ = train_test_split(Ftr, ytr, test_size=0.2, random_state=42)
    Ft, Fe = torch.from_numpy(Ft), torch.from_numpy(Fte)
    yt = torch.from_numpy(yt.astype(np.float32))[:, None]
    torch.manual_seed(seed)
    m = CNN(in_channels=2, m_param=16, num_qubits=1)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(m.parameters(), lr=a.lr or 5e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.95)
    for ep in range(a.epochs or 30):
        t0 = time.time()
        m.train()
        tot = 0.0
        for idx in batches(len(Ft), 64):
            opt.zero_grad()
            loss = crit(m(Ft[idx]), yt[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        if ep % 5 == 0 or ep == (a.epochs or 30) - 1:
            print(f"  ep {ep} train {tot / len(Ft):.5f} ({time.time() - t0:.0f} s)", flush=True)
    return predict(m, Fe)[:, 0].numpy(), dict(params=n_params(m), T_ds=Ftr.shape[2], theta=float(theta), f_cal=f_cal)


MODELS = dict(linear=run_linear, fnn=run_fnn, qubic=run_qubic, herqules=run_herqules, klinq=run_klinq,
              transformer=run_transformer, cnn=run_cnn)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--T", type=float, required=True, help="readout duration in ns")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--epochs", type=int, default=0, help="override (smoke tests)")
    p.add_argument("--trials", type=int, default=0, help="override (smoke tests)")
    p.add_argument("--lr", type=float, default=0)
    p.add_argument("--bins", type=int, default=15, help="KLiNQ student bins: 15 = FNN-A, 100 = FNN-B")
    p.add_argument("--tag", default="")
    p.add_argument("--lp", action="store_true", help="linear baseline only: apply the causal 10 MHz low-pass first")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    out = run_path(a.dataset, a.model, a.T, a.seed, a.tag)
    if out.with_suffix(".json").exists():
        print("already done", out)
        return
    np.random.seed(a.seed)
    ds = Dataset(a.dataset)
    L = ds.samples(a.T)
    t0 = time.time()
    scores, info = MODELS[a.model](ds, L, a.seed, a)
    pred = (scores > 0).astype(int)
    F, p01, p10 = fidelity(ds.yte, pred)
    rec = dict(dataset=a.dataset, model=a.model, T_ns=a.T, samples=L, seed=a.seed, F=F, p01=p01, p10=p10,
               n_test=int(len(ds.yte)), seconds=round(time.time() - t0, 1), threads=a.threads, tag=a.tag,
               epochs_override=a.epochs, **info)
    np.save(str(out) + "_scores.npy", scores.astype(np.float32))
    out.with_suffix(".json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("dataset", "model", "T_ns", "F", "p01", "p10", "params", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
