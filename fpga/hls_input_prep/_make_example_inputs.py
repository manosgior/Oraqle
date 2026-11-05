"""Produce the FINAL preprocessed input array that each model ingests, from real
traces, plus the float-model reference output (for hls4ml bit-accuracy checks).
1000 ns window (500 samples). Saves .npy per model to /out.

Models & final input shapes (per the live pipelines in evaluate_truncated.py):
  QubiCML (Arxiv240618807FNN)  (N, 2)    demod+time-average -> (I,Q), min-max[0,1]
  KLiNQ student                (N, 31)   [z(binned I/Q avg, 15 bins) | z(MF scalar)]
  MCMit-CNN (CNN.py)           (N,10,50) per-qubit demod+FIR+ds20, per-channel z-norm
  HERQULES (Net_rmf)           (N, 10)   [MF x5 | RMF x5]
"""
import os, h5py, numpy as np, torch
torch.manual_seed(0); np.random.seed(0)
import os as _os, sys as _sys  # [unify] repo layout
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'discriminators'))
from disc_paths import RAW_TRAIN, RAW_TEST, CNN_DATA, CHECKPOINTS, OUT as _OUT, ensure_dir, require_dataset  # [unify]
from networks import Arxiv240618807FNN, CNN, KLiNQStudentModel
from trainers.HERQULES_original import Net_rmf
from trainers.train_HERQULES import demodulate_all_qubits, compute_all_envelopes, build_features
from helpers.cnn_helpers import prepare_cnn_data

RAW_TR = str(RAW_TRAIN)
RAW_TE = str(RAW_TEST)
CNN_TR = str(CNN_DATA / "Qubit_5Channel_ds20_train.h5")
CNN_TE = str(CNN_DATA / "Qubit_5Channel_ds20_test.h5")
MODELS = str(CHECKPOINTS); OUT = ensure_dir(_OUT / "fpga" / "example_inputs")  # [unify] was /m, /out
L = 500                       # 1000 ns
N_TR, N_TE = 60000, 512       # train subset for stats/envelopes; test examples to dump
FREQ = -np.array([-64.729e6, -25.366e6, 24.79e6, 70.269e6, 127.282e6])


def load_raw(path, key, n, length):
    with h5py.File(path, "r") as hf:
        tot = hf[f"X_{key}"].shape[0]
        idx = np.sort(np.random.RandomState(0).choice(tot, min(n, tot), replace=False))
        return hf[f"X_{key}"][idx, :length, :].astype(np.float32), hf[f"y_{key}"][idx].astype(np.int64)


def bits(y): return np.stack([(y >> q) & 1 for q in range(5)], 1).astype(np.int64)


def demod_avg(tr, dt=2e-9):
    N = tr.shape[0]; out = np.zeros((N, 5, 2), np.float32)
    t = np.arange(tr.shape[1]) * dt; I, Q = tr[:, :, 0], tr[:, :, 1]
    for i, f in enumerate(FREQ):
        li, lq = np.cos(2*np.pi*f*t), np.sin(2*np.pi*f*t)
        out[:, i, 0] = np.mean(I*li - Q*lq, 1); out[:, i, 1] = np.mean(I*lq + Q*li, 1)
    return out


def save(tag, inp, out, lab):
    np.save(f"{OUT}/{tag}_input.npy", inp.astype(np.float32))
    np.save(f"{OUT}/{tag}_output.npy", out.astype(np.float32))
    np.save(f"{OUT}/{tag}_labels.npy", lab.astype(np.int64))
    print(f"[{tag}] input {inp.shape} {inp.dtype}  output {out.shape}  labels {lab.shape}", flush=True)


Xtr, ytr = load_raw(RAW_TR, "train", N_TR, L)
Xte, yte = load_raw(RAW_TE, "test", N_TE, L)
yte_bits = bits(yte)
report = []

# ---- 1. QubiCML (qubit 0) ----
try:
    dtr, dte = demod_avg(Xtr), demod_avg(Xte)
    q = 0; xmin = dtr[:, q, :].min(0); xr = dtr[:, q, :].max(0) - xmin; xr[xr == 0] = 1e-10
    inp = (dte[:, q, :] - xmin) / xr                                   # (N,2) in ~[0,1]
    m = Arxiv240618807FNN(); m.load_state_dict(torch.load(f"{MODELS}/Arxiv_q0_len500_trial3.pth", map_location="cpu")); m.eval()
    with torch.no_grad(): o = m(torch.tensor(inp, dtype=torch.float32)).numpy()   # (N,1) prob
    save("QubiCML_q0", inp, o, yte_bits[:, q]); report.append("QubiCML ok")
except Exception as e:
    report.append(f"QubiCML FAILED: {e}")

# ---- 2. KLiNQ student (qubit 0) ----
try:
    q = 0; sd = torch.load(f"{MODELS}/KLiNQ_student_paper_q0_1000ns.pth", map_location="cpu")
    fc1_in = sd["fc1.weight"].shape[1]; nb = (fc1_in - 1) // 2; bs = L // nb; nb = L // bs
    aI = Xte[:, :nb*bs, 0].reshape(N_TE, nb, bs).mean(2)
    aQ = Xte[:, :nb*bs, 1].reshape(N_TE, nb, bs).mean(2)
    avg = np.concatenate([aI, aQ], 1)
    yq = bits(ytr)[:, q]; g = Xtr.reshape(N_TR, -1)[yq == 0]; e = Xtr.reshape(N_TR, -1)[yq == 1]
    n = min(len(g), len(e)); d = g[:n] - e[:n]; env = d.mean(0) / (d.var(0) + 1e-10)
    mf = (Xte.reshape(N_TE, -1) @ env[:L*2]).reshape(-1, 1)
    zn = lambda a: (a - a.mean(0)) / (a.std(0) + 1e-10)
    inp = np.concatenate([zn(avg), zn(mf)], 1)                          # (N,31)
    m = KLiNQStudentModel(input_size=fc1_in); m.load_state_dict(sd); m.eval()
    with torch.no_grad(): o = torch.sigmoid(m(torch.tensor(inp, dtype=torch.float32))).numpy()
    save("KLiNQ_student_q0", inp, o, yte_bits[:, q]); report.append(f"KLiNQ ok (fc1_in={fc1_in}, n_bins={nb})")
except Exception as e:
    report.append(f"KLiNQ FAILED: {e}")

# ---- 3. MCMit-CNN ----
try:
    Xc_te, yc_te = prepare_cnn_data(CNN_TE, downsample_factor=20, original_length=1000,
                                    num_qubits=5, time_slice=(0, L), is_test=True)
    Xc_tr, _ = prepare_cnn_data(CNN_TR, downsample_factor=20, original_length=1000,
                                num_qubits=5, time_slice=(0, L), is_test=False)
    Xc_te = Xc_te.numpy() if torch.is_tensor(Xc_te) else Xc_te
    Xc_tr = Xc_tr.numpy() if torch.is_tensor(Xc_tr) else Xc_tr
    yc_te = yc_te.numpy() if torch.is_tensor(yc_te) else yc_te
    mu = Xc_tr.mean((0, 2), keepdims=True); sdv = Xc_tr.std((0, 2), keepdims=True) + 1e-10
    inp = ((Xc_te[:N_TE] - mu) / sdv).astype(np.float32)                # (N,10,50)
    sd = torch.load(f"{MODELS}/CNN_m16_len500_trial36.pth", map_location="cpu")
    m = CNN(in_channels=10, m_param=sd["conv1.weight"].shape[0], num_qubits=5); m.load_state_dict(sd); m.eval()
    with torch.no_grad(): o = torch.sigmoid(m(torch.tensor(inp))).numpy()   # (N,5)
    lab = bits(yc_te[:N_TE].astype(np.int64)) if yc_te.ndim == 1 else yc_te[:N_TE]
    save("MCMitCNN", inp, o, lab); report.append("CNN ok")
except Exception as e:
    report.append(f"CNN FAILED: {e}")

# ---- 4. HERQULES (Net_rmf) ----
try:
    dtr = demodulate_all_qubits(Xtr); dte = demodulate_all_qubits(Xte)
    np.random.seed(42); mf_e, rmf_e = compute_all_envelopes(dtr, ytr)
    inp = build_features(dte, mf_e, rmf_e, trace_length=L)              # (N,10)
    m = Net_rmf(); m.load_state_dict(torch.load(f"{MODELS}/HERQULES_len500.pth", map_location="cpu")); m.eval()
    with torch.no_grad(): logit = m(torch.tensor(inp, dtype=torch.float32)).numpy()  # (N,32)
    save("HERQULES", inp, logit, yte_bits); report.append("HERQULES ok")
except Exception as e:
    report.append(f"HERQULES FAILED: {e}")

print("\n=== SUMMARY ===")
for r in report: print(" ", r)
print("files:", sorted(os.listdir(OUT)))
