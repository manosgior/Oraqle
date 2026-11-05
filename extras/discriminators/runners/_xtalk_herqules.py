"""Cross-fidelity + accuracy-vs-N for full HERQULES (MF-RMF-NN), no-scramble.
One inference pass over the full test set -> per-shot predicted 5-bit -> both metrics.
"""
import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import os, csv, numpy as np, torch as T
from itertools import combinations
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
T.manual_seed(1); np.random.seed(1)
SC = os.environ.get("XTALK_SCRAMBLE", "0") == "1"   # True = competitor's shot-mixing
MODE = "scrambled" if SC else "noscramble"
print(f"[xtalk:HERQULES] mode={MODE} (scramble={SC})", flush=True)


def xfid(tb, pb):
    F = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if i == j: continue
            m0 = tb[:, j] == 0; m1 = tb[:, j] == 1
            F[i, j] = 1 - ((pb[m0, i] == 1).mean() + (pb[m1, i] == 0).mean())
    return {d: float(np.mean([abs(F[i, j]) for i in range(5) for j in range(5)
                              if abs(i-j) == d])) for d in (1, 2, 3, 4)}


def fvsN(tb, pb):
    res = {}
    for N in range(1, 6):
        a = []
        for S in combinations(range(5), N):
            spec = [q for q in range(5) if q not in S]
            mask = np.all(tb[:, spec] == 0, 1) if spec else np.ones(len(tb), bool)
            t, p = tb[mask], pb[mask]
            for i in S:
                m1 = t[:, i] == 1; m0 = t[:, i] == 0
                a.append(1 - ((p[m1, i] == 0).mean() + (p[m0, i] == 1).mean())/2)
        res[N] = float(np.mean(a))
    return res


pre = preclassifier(); pre.fit()
rmf = relaxation_mf_classifier(); rmf.fit(pre.get_traces(), boxcars=BOX)
dtr, dval, dte = mf_demux_data_prep(H.TRACE_LENGTH)
mf_env, _ = search_matched_filter_for_all_qubits_demux(dtr, best_bc=BOX)
trf = np.concatenate((matched_filter_preprocess_demux(dtr, mf_env, scramble=SC), rmf.predict(data_type=0)), axis=2)
vaf = np.concatenate((matched_filter_preprocess_demux(dval, mf_env, scramble=SC), rmf.predict(data_type=1)), axis=2)
trl = T.utils.data.DataLoader(MFOutputDataset(trf), batch_size=512, shuffle=True)
vall = T.utils.data.DataLoader(MFOutputDataset(vaf), batch_size=512, shuffle=False)
net = Net_rmf().train(); opt = T.optim.Adam(net.parameters(), lr=0.01); crit = T.nn.CrossEntropyLoss()
best, bs = -1, None
for ep in range(100):
    adjust_learning_rate(0.01, opt, ep)
    for b in trl:
        l = crit(net(b['predictors']), b['target']); opt.zero_grad(); l.backward(); opt.step()
    net.eval(); cor = tot = 0
    with T.no_grad():
        for b in vall:
            cor += (net(b['predictors']).argmax(1) == b['target']).sum().item(); tot += len(b['target'])
    if cor/tot >= best: best = cor/tot; bs = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.train()
net.load_state_dict(bs); net.eval()
print(f"[xtalk:HERQULES] trained (val acc {best:.4f})", flush=True)

tef = np.concatenate((matched_filter_preprocess_demux(dte, mf_env, scramble=SC), rmf.predict(data_type=2)), axis=2)
ld = T.utils.data.DataLoader(MFOutputDataset(tef), batch_size=4096, shuffle=False)
P, Tr = [], []
with T.no_grad():
    for b in ld:
        P.append(net(b['predictors']).argmax(1).numpy()); Tr.append(b['target'].numpy())
P = np.concatenate(P); Tr = np.concatenate(Tr)
pb = np.stack([(P >> q) & 1 for q in range(5)], 1)
tb = np.stack([(Tr >> q) & 1 for q in range(5)], 1)
cf = xfid(tb, pb); fn = fvsN(tb, pb)
print("[xtalk:HERQULES] cross-fidelity |F^CF| by dist:", {k: round(v, 4) for k, v in cf.items()}, flush=True)
print("[xtalk:HERQULES] <F> vs N:", {k: round(v, 4) for k, v in fn.items()}, flush=True)
with open(os.path.join(REPORTS, f"xtalk_HERQULES_{MODE}.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["metric", "k", "value"])
    for k, v in cf.items(): w.writerow(["crossfid_dist", k, f"{v:.5f}"])
    for k, v in fn.items(): w.writerow(["F_vs_N", k, f"{v:.5f}"])
print("[xtalk:HERQULES] done", flush=True)
