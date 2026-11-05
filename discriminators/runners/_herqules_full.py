"""Fixed HERQULES (no shot scrambling), CLEAN 32-class Net_rmf, on the FULL
dataset (all ~50000 shots/state from the combined train+test pool). Build the
MF/RMF features ONCE, then retrain the net per seed to measure fidelity AND
run-to-run variance with full data.

Full split: SUBSET_NUM_TRAIN_VAL + SUBSET_NUM_TEST = min_available (=50000/state),
NUM_VAL_RATIO=0.35. 1000 ns (500 samples).
"""
import os, csv, numpy as np, torch as T
import os as _os, sys as _sys  # [unify] discriminators/ on sys.path (repo layout)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from disc_paths import REPORTS_OUT, ensure_dir  # [unify] replaces the old docker mount paths
import trainers.HERQULES_original as H
from trainers.HERQULES_original import (
    preclassifier, relaxation_mf_classifier, mf_demux_data_prep,
    Net_rmf, MFOutputDataset, adjust_learning_rate)
from matched_filter import search_matched_filter_for_all_qubits_demux, matched_filter_preprocess_demux

# FULL dataset: 20000 train+val + 30000 test per state = 50000 (= all combined)
H.SUBSET_NUM_TRAIN_VAL = 20000
H.SUBSET_NUM_TEST = 30000
_d = [0.35 if (isinstance(x, float) and abs(x-0.53) < 1e-9) else x
      for x in H.get_train_val_and_test_set.__defaults__]
H.get_train_val_and_test_set.__defaults__ = tuple(_d)
REPORTS = ensure_dir(REPORTS_OUT); BOX = [1, 1, 9, 2, 9]  # [unify] was REPORTS = \"/app/optimization_reports\"
DEV = T.device("cuda" if T.cuda.is_available() else "cpu")
NSEED = int(os.environ.get("NSEED", "5")); FAIL = 0.85
print(f"[full] device={DEV} seeds=1..{NSEED} TRAIN_VAL={H.SUBSET_NUM_TRAIN_VAL} TEST={H.SUBSET_NUM_TEST}", flush=True)


def bits(t): return np.stack([(t >> q) & 1 for q in range(5)], 1)


def per_qubit_fid(tb, pb):
    return [float(1 - ((pb[tb[:, q]==1, q]==0).mean() + (pb[tb[:, q]==0, q]==1).mean())/2) for q in range(5)]


# ---- build MF/RMF features ONCE on full data (no scramble) ----
pre = preclassifier(); pre.fit()
rmf = relaxation_mf_classifier(); rmf.fit(pre.get_traces(), boxcars=BOX)
dtr, dval, dte = mf_demux_data_prep(H.TRACE_LENGTH)
mf_env, _ = search_matched_filter_for_all_qubits_demux(dtr, best_bc=BOX)
trf = np.concatenate((matched_filter_preprocess_demux(dtr, mf_env, scramble=False), rmf.predict(data_type=0)), axis=2)
vaf = np.concatenate((matched_filter_preprocess_demux(dval, mf_env, scramble=False), rmf.predict(data_type=1)), axis=2)
tef = np.concatenate((matched_filter_preprocess_demux(dte, mf_env, scramble=False), rmf.predict(data_type=2)), axis=2)
trl = T.utils.data.DataLoader(MFOutputDataset(trf), batch_size=512, shuffle=True)
vall = T.utils.data.DataLoader(MFOutputDataset(vaf), batch_size=4096, shuffle=False)
tel = T.utils.data.DataLoader(MFOutputDataset(tef), batch_size=4096, shuffle=False)
print(f"[full] features built (train {len(MFOutputDataset(trf))}, test {len(MFOutputDataset(tef))})", flush=True)


def run(seed):
    T.manual_seed(seed); np.random.seed(seed)
    net = Net_rmf().to(DEV).train(); opt = T.optim.Adam(net.parameters(), lr=0.01); crit = T.nn.CrossEntropyLoss()
    best, bs = -1, None
    for ep in range(100):
        adjust_learning_rate(0.01, opt, ep); net.train()
        for b in trl:
            loss = crit(net(b['predictors'].to(DEV)), b['target'].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); cor = tot = 0
        with T.no_grad():
            for b in vall:
                p = net(b['predictors'].to(DEV)).argmax(1).cpu().numpy()
                pb = np.stack([(p >> q) & 1 for q in range(5)], 1); tb = bits(b['target'].numpy())
                cor += (pb == tb).sum(); tot += tb.size
        if cor/tot >= best: best = cor/tot; bs = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(bs); net.eval()
    P, Tr = [], []
    with T.no_grad():
        for b in tel:
            p = net(b['predictors'].to(DEV)).argmax(1).cpu().numpy()
            P.append(np.stack([(p >> q) & 1 for q in range(5)], 1)); Tr.append(bits(b['target'].numpy()))
    pb = np.concatenate(P); tb = np.concatenate(Tr)
    fq = per_qubit_fid(tb, pb); return fq, float(np.prod(fq) ** 0.2)


rows = []
for s in range(1, NSEED + 1):
    fq, g = run(s); rows.append(fq + [g])
    flag = "  <-- FAIL" if g < FAIL else ""
    print(f"[full] seed {s} F5Q {g:.4f}  Q {[round(a,3) for a in fq]}{flag}", flush=True)
a = np.array(rows); ok = a[a[:, 5] >= FAIL]; nf = len(a) - len(ok)
lab = ["Q1", "Q2", "Q3", "Q4", "Q5", "F5Q"]; m, sd = ok.mean(0), ok.std(0)
print(f"\n[full] {nf}/{len(a)} fail | " + "  ".join(f"{lab[i]}={m[i]:.4f}±{sd[i]:.4f}" for i in range(6)), flush=True)
with open(os.path.join(REPORTS, "herqules_full_dataset.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["seed"] + lab)
    for i, r in enumerate(rows): w.writerow([i+1] + [f"{x:.4f}" for x in r])
    w.writerow(["mean"] + [f"{m[i]:.4f}" for i in range(6)]); w.writerow(["std"] + [f"{sd[i]:.4f}" for i in range(6)])
print("[full] done", flush=True)
