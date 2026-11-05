"""HERQULES head ablation (no shot scrambling): joint 32-class softmax (original
Net_rmf) vs 5 independent per-qubit binary heads, on the SAME MF/RMF features
and the SAME seeds. Isolates the effect of the head on fidelity + stability.

MF/RMF front-end built ONCE (memory/time-cheap); only the tiny head is retrained
per seed. 1000 ns (500 samples). Reports per-qubit F + F5Q geomean, mean/std and
catastrophic-failure count over N seeds.
"""
import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import os, csv, numpy as np, torch as T
import torch.nn as nn
import trainers.HERQULES_original as H
from trainers.HERQULES_original import (
    preclassifier, relaxation_mf_classifier, mf_demux_data_prep,
    Net_rmf, MFOutputDataset, adjust_learning_rate)
from matched_filter import search_matched_filter_for_all_qubits_demux, matched_filter_preprocess_demux

H.SUBSET_NUM_TRAIN_VAL = 3000; H.SUBSET_NUM_TEST = 7000
_d = [0.35 if (isinstance(x, float) and abs(x-0.53) < 1e-9) else x
      for x in H.get_train_val_and_test_set.__defaults__]
H.get_train_val_and_test_set.__defaults__ = tuple(_d)
REPORTS = ensure_dir(REPORTS_OUT); BOX = [1, 1, 9, 2, 9]
DEV = T.device("cuda" if T.cuda.is_available() else "cpu")
NSEED = int(os.environ.get("NSEED", "7")); FAIL = 0.85
print(f"[heads] device={DEV} seeds=1..{NSEED}", flush=True)


class NetBin(nn.Module):
    """Same trunk as Net_rmf (10->10->20) but 5 independent binary heads."""
    def __init__(self):
        super().__init__()
        self.hid1 = nn.Linear(10, 10); self.hid2 = nn.Linear(10, 20)
        self.heads = nn.ModuleList([nn.Linear(20, 1) for _ in range(5)])
        self.relu = nn.ReLU()
    def forward(self, x):
        z = self.relu(self.hid1(x)); z = self.relu(self.hid2(z))
        return T.cat([h(z) for h in self.heads], 1)        # (N,5) logits


def per_qubit_fid(tb, pb):
    return [float(1 - ((pb[tb[:, q]==1, q]==0).mean() + (pb[tb[:, q]==0, q]==1).mean())/2) for q in range(5)]


# ---- build MF/RMF features ONCE (no scramble) ----
pre = preclassifier(); pre.fit()
rmf = relaxation_mf_classifier(); rmf.fit(pre.get_traces(), boxcars=BOX)
dtr, dval, dte = mf_demux_data_prep(H.TRACE_LENGTH)
mf_env, _ = search_matched_filter_for_all_qubits_demux(dtr, best_bc=BOX)
trf = np.concatenate((matched_filter_preprocess_demux(dtr, mf_env, scramble=False), rmf.predict(data_type=0)), axis=2)
vaf = np.concatenate((matched_filter_preprocess_demux(dval, mf_env, scramble=False), rmf.predict(data_type=1)), axis=2)
tef = np.concatenate((matched_filter_preprocess_demux(dte, mf_env, scramble=False), rmf.predict(data_type=2)), axis=2)
trl = T.utils.data.DataLoader(MFOutputDataset(trf), batch_size=512, shuffle=True)
vall = T.utils.data.DataLoader(MFOutputDataset(vaf), batch_size=512, shuffle=False)
tel = T.utils.data.DataLoader(MFOutputDataset(tef), batch_size=4096, shuffle=False)
print("[heads] features built", flush=True)


def bits_from_packed(t):  # (N,) long -> (N,5)
    return T.stack([(t >> q) & 1 for q in range(5)], 1)


def train_eval(kind, seed):
    T.manual_seed(seed); np.random.seed(seed)
    if kind == "softmax":
        net = Net_rmf().to(DEV).train(); crit = nn.CrossEntropyLoss()
    else:
        net = NetBin().to(DEV).train(); crit = nn.BCEWithLogitsLoss(reduction="sum")
    opt = T.optim.Adam(net.parameters(), lr=0.01)
    best, bs = -1, None
    for ep in range(100):
        adjust_learning_rate(0.01, opt, ep)
        net.train()
        for b in trl:
            x = b['predictors'].to(DEV)
            if kind == "softmax":
                loss = crit(net(x), b['target'].to(DEV))
            else:
                yb = bits_from_packed(b['target']).float().to(DEV)
                loss = crit(net(x), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        # val: mean per-qubit bit accuracy
        net.eval(); cor = tot = 0
        with T.no_grad():
            for b in vall:
                x = b['predictors'].to(DEV); tb = bits_from_packed(b['target']).numpy()
                if kind == "softmax":
                    p = net(x).argmax(1).cpu().numpy(); pb = np.stack([(p>>q)&1 for q in range(5)], 1)
                else:
                    pb = (T.sigmoid(net(x)).cpu().numpy() > 0.5).astype(int)
                cor += (pb == tb).sum(); tot += tb.size
        va = cor/tot
        if va >= best: best = va; bs = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(bs); net.eval()
    P, Tr = [], []
    with T.no_grad():
        for b in tel:
            x = b['predictors'].to(DEV); Tr.append(bits_from_packed(b['target']).numpy())
            if kind == "softmax":
                p = net(x).argmax(1).cpu().numpy(); P.append(np.stack([(p>>q)&1 for q in range(5)], 1))
            else:
                P.append((T.sigmoid(net(x)).cpu().numpy() > 0.5).astype(int))
    pb = np.concatenate(P); tb = np.concatenate(Tr)
    fq = per_qubit_fid(tb, pb); g = float(np.prod(fq) ** 0.2)
    return fq, g


rows = {"softmax": [], "binary": []}
for s in range(1, NSEED + 1):
    for kind, key in [("softmax", "softmax"), ("binary", "binary")]:
        fq, g = train_eval(kind, s)
        rows[key].append(fq + [g])
        flag = "  <-- FAIL" if g < FAIL else ""
        print(f"[heads] seed {s} {key:7s} F5Q {g:.4f}  Q {[round(a,3) for a in fq]}{flag}", flush=True)

lab = ["Q1", "Q2", "Q3", "Q4", "Q5", "F5Q"]
out = []
for key in ("softmax", "binary"):
    a = np.array(rows[key]); ok = a[a[:, 5] >= FAIL]; nf = len(a) - len(ok)
    m, sd = ok.mean(0), ok.std(0)
    print(f"\n[heads] {key}: {nf}/{len(a)} fail | " + "  ".join(f"{lab[i]}={m[i]:.4f}±{sd[i]:.4f}" for i in range(6)), flush=True)
    out.append([key, nf, len(a)] + [f"{m[i]:.4f}" for i in range(6)] + [f"{sd[i]:.4f}" for i in range(6)])
with open(os.path.join(REPORTS, "herqules_head_ablation.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["head", "n_fail", "n"] + [f"{x}_mean" for x in lab] + [f"{x}_std" for x in lab])
    w.writerows(out)
print("[heads] done", flush=True)
