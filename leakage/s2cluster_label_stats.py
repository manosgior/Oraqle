"""s2cluster_label_stats.py -- audit the prior three-level (|2>) labels.

Reads (read-only) the ternary labels produced in May 2026 by
Oraqle/Discriminators/runners/build_three_level_labels.py:

    three_level_labels.hdf5 : y3_train (480000,5) int8, y3_test (1120000,5) int8,
                              y_train / y_test (prepared 5-bit state, 0..31)
    inflated_p10/labels.hdf5: same layout after inflate_leakage.py --target-leak 0.10

and reports, per qubit (code q0..q4 = paper Q1..Q5, bit q of y):
  * fraction labelled 2, overall / by split / by prepared bit, with Wilson 95% CI
  * per prepared 5-bit state (32 x 5 table of counts)
  * by every other qubit's prepared bit (crosstalk hint)
  * multiplicity: how many shots carry >=1, >=2, ... qubits labelled 2
  * pairwise co-occurrence of label-2 between qubits (lift vs independence)

Also checks that y_train / y_test in the labels file equal the raw-file labels.
No labels are produced here; these are statistics of a heuristic labeler
(perp > k*sigma_perp in the time-averaged IQ plane), NOT a leakage measurement.

Run inside oraqle-torch:latest on graham:
  docker run --rm --user $(id -u):$(id -g) \
    -v ~/qubit_readout_klinq/data/five_qubit_data:/data:ro \
    -v ~/oraqle-ics-sims/leakage/s2cluster:/out \
    -v ~/oraqle-ics-sims/leakage/s2cluster:/code:ro \
    oraqle-torch:latest python /code/s2cluster_label_stats.py --data /data --out /out
"""
from __future__ import annotations

import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import argparse
import json
import math
import os

import h5py
import numpy as np

NQ = 5
PAPER = {q: f"Q{q+1}" for q in range(NQ)}
GAUSS_TAIL = {3.0: 2 * 0.0013498980316301, 3.5: 2 * 0.00023262907903552,
              4.0: 2 * 3.1671241833e-05}


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return [float("nan")] * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [p, max(0.0, c - h), min(1.0, c + h)]


def rate(mask_lab2: np.ndarray, sel: np.ndarray) -> dict:
    n = int(sel.sum())
    k = int((mask_lab2 & sel).sum())
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "ci95": [lo, hi]}


def stats_for(y3: np.ndarray, y: np.ndarray, split: np.ndarray, tag: str) -> dict:
    y = y.astype(np.int64)
    bits = np.stack([(y >> q) & 1 for q in range(NQ)], axis=1)
    lab2 = (y3 == 2)
    out = {"tag": tag, "n_shots": int(y3.shape[0]),
           "label_values": {str(v): int(c) for v, c in zip(*np.unique(y3, return_counts=True))},
           "per_qubit": {}, "multiplicity": {}, "cooccurrence": {}}
    allsel = np.ones(len(y), bool)
    for q in range(NQ):
        d = {"paper_name": PAPER[q], "code_name": f"q{q}"}
        d["overall"] = rate(lab2[:, q], allsel)
        for sname in np.unique(split):
            ss = split == sname
            d[f"split_{sname}"] = rate(lab2[:, q], ss)
            for b in (0, 1):
                d[f"split_{sname}_prep{b}"] = rate(lab2[:, q], ss & (bits[:, q] == b))
        for b in (0, 1):
            d[f"prep{b}"] = rate(lab2[:, q], bits[:, q] == b)
        # labels that disagree with the prepared bit OTHER than 2 (should be 0:
        # the labeler copies the prepared bit and only overrides to 2)
        nonleak = ~lab2[:, q]
        d["n_label_ne_prepared_excluding_2"] = int((y3[nonleak, q] != bits[nonleak, q]).sum())
        # by other qubits' prepared bits (crosstalk hint)
        xt = {}
        for j in range(NQ):
            if j == q:
                continue
            for b in (0, 1):
                for bq in (0, 1):
                    sel = (bits[:, j] == b) & (bits[:, q] == bq)
                    xt[f"{PAPER[j]}(q{j})={b}|own={bq}"] = rate(lab2[:, q], sel)
        d["by_other_prepared_bit"] = xt
        # per prepared 5-bit state
        st = {}
        for s in range(32):
            sel = y == s
            r = rate(lab2[:, q], sel)
            st[str(s)] = {"bits_q0..q4": [(s >> i) & 1 for i in range(NQ)], **r}
        d["by_prepared_state"] = st
        out["per_qubit"][f"q{q}"] = d
    m = lab2.sum(axis=1)
    for t in range(0, NQ + 1):
        out["multiplicity"][f"exactly_{t}"] = int((m == t).sum())
    for t in range(1, NQ + 1):
        out["multiplicity"][f"at_least_{t}"] = int((m >= t).sum())
    out["multiplicity"]["at_least_2_by_split"] = {
        str(s): int(((m >= 2) & (split == s)).sum()) for s in np.unique(split)}
    # which qubit combinations occur when >=2 qubits are labelled 2
    combos = {}
    for row in lab2[m >= 2]:
        key = "+".join(PAPER[q] for q in range(NQ) if row[q])
        combos[key] = combos.get(key, 0) + 1
    out["multiplicity"]["combos_at_least_2"] = dict(sorted(combos.items(), key=lambda kv: -kv[1]))
    N = len(y)
    for a in range(NQ):
        for b in range(a + 1, NQ):
            both = int((lab2[:, a] & lab2[:, b]).sum())
            pa, pb = lab2[:, a].mean(), lab2[:, b].mean()
            exp = pa * pb * N
            out["cooccurrence"][f"{PAPER[a]}&{PAPER[b]}"] = {
                "observed": both, "expected_if_independent": float(exp),
                "lift": float(both / exp) if exp > 0 else float("nan")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(LP.DATA))
    ap.add_argument("--out", default=str(LP.stage("s2cluster")))
    a = ap.parse_args()
    res = {"source": "three_level_labels.hdf5 (build_three_level_labels.py, 2026-05-27)",
           "note": ("Rates are fractions of shots whose label was overridden to 2 by the "
                    "perp>k*sigma_perp heuristic on the first 500 samples (1 us). "
                    "Denominators = shots with the stated prepared condition. "
                    "CI = Wilson 95% on the binomial count (statistical only; the "
                    "labeler's systematic error is not included).")}
    p = os.path.join(a.data, "three_level_labels.hdf5")
    with h5py.File(p, "r") as f:
        res["attrs"] = {k: (v.tolist() if hasattr(v, "tolist") else str(v)) for k, v in f.attrs.items()}
        res["datasets"] = {k: [list(f[k].shape), str(f[k].dtype)] for k in f.keys()}
        y3tr, y3te = f["y3_train"][:], f["y3_test"][:]
        ytr, yte = f["y_train"][:], f["y_test"][:]
    # consistency with the raw files
    with h5py.File(os.path.join(a.data, "DRaw_C_Tr_v0-001"), "r") as f:
        raw_ytr = f["y_train"][:]
    with h5py.File(os.path.join(a.data, "DRaw_C_Te_v0-002"), "r") as f:
        raw_yte = f["y_test"][:]
    res["labels_y_equal_raw_y"] = {"train": bool(np.array_equal(ytr, raw_ytr)),
                                   "test": bool(np.array_equal(yte, raw_yte))}
    y3 = np.concatenate([y3tr, y3te])
    y = np.concatenate([ytr, yte])
    split = np.array(["train"] * len(ytr) + ["test"] * len(yte))
    res["original"] = stats_for(y3, y, split, "pool(train+test)")
    res["gaussian_floor_two_sided"] = {str(k): v for k, v in GAUSS_TAIL.items()}

    # inflated_p10
    pi = os.path.join(a.data, "inflated_p10", "labels.hdf5")
    if os.path.exists(pi):
        with h5py.File(pi, "r") as f:
            ds = {k: [list(f[k].shape), str(f[k].dtype)] for k in f.keys()}
            i3tr, i3te = f["y3_train"][:], f["y3_test"][:]
            itr, ite = f["y_train"][:], f["y_test"][:]
        isplit = np.array(["train"] * len(itr) + ["test"] * len(ite))
        st = stats_for(np.concatenate([i3tr, i3te]), np.concatenate([itr, ite]), isplit,
                       "inflated_p10 pool")
        # keep the inflated summary short
        res["inflated_p10"] = {
            "datasets": ds,
            "per_qubit_rate": {f"q{q}": {"paper": PAPER[q],
                                         "train": rate(i3tr[:, q] == 2, np.ones(len(i3tr), bool)),
                                         "test": rate(i3te[:, q] == 2, np.ones(len(i3te), bool)),
                                         "prep0": st["per_qubit"][f"q{q}"]["prep0"],
                                         "prep1": st["per_qubit"][f"q{q}"]["prep1"]}
                               for q in range(NQ)},
            "rows_with_any_2": {"train": int((i3tr == 2).any(1).sum()),
                                "test": int((i3te == 2).any(1).sum())},
            "multiplicity": st["multiplicity"],
            "prepared_state_counts_train": np.bincount(itr.astype(np.int64), minlength=32).tolist(),
            "prepared_state_counts_test": np.bincount(ite.astype(np.int64), minlength=32).tolist(),
        }
    os.makedirs(a.out, exist_ok=True)
    op = os.path.join(a.out, "s2cluster_label_stats.json")
    with open(op, "w") as f:
        json.dump(res, f, indent=1)
    # short console summary
    for q in range(NQ):
        d = res["original"]["per_qubit"][f"q{q}"]
        print(f"{PAPER[q]}(q{q}) all={d['overall']['k']}/{d['overall']['n']}={100*d['overall']['rate']:.3f}% "
              f"prep0={100*d['prep0']['rate']:.3f}% prep1={100*d['prep1']['rate']:.3f}% "
              f"train={100*d['split_train']['rate']:.3f}% test={100*d['split_test']['rate']:.3f}% "
              f"label!=prep(excl2)={d['n_label_ne_prepared_excluding_2']}")
    print("multiplicity", res["original"]["multiplicity"])
    print("raw-y equal", res["labels_y_equal_raw_y"])
    print("wrote", op)


if __name__ == "__main__":
    main()
