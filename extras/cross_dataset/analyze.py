"""Collect the runs into one summary: fidelity tables, paired comparisons, and each takeaway's check.

    python -I analyze.py          # writes $ORAQLE_WORK/cross_dataset/summary.json

Comparisons between models use the per-shot scores on the shared test shots, with a stratified bootstrap
of the fidelity difference, so that test-set noise common to both models cancels.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from common import CACHE, RUNS, XD  # noqa: E402

LARGE = ["fnn", "transformer"]  # the paper's large models
SMALL = ["herqules", "cnn", "klinq", "qubic"]  # deployable on the FPGA in the paper
ORDER = ["linear", "qubic", "herqules", "klinq", "fnn", "transformer", "cnn"]


def load_runs():
    runs = {}
    for p in glob.glob(str(RUNS / "*" / "*.json")):
        r = json.load(open(p))
        r["path"] = p
        if r.get("epochs_override") or r.get("tag") == "smoke":
            continue
        key = (r["dataset"], r["model"] + (("_" + r["tag"]) if r.get("tag") else ""), int(r["T_ns"]))
        runs.setdefault(key, []).append(r)
    # The published Linear Threshold row matches no filter, so the unfiltered runs are "linear" where they
    # exist. The first runs, with the causal filter, stay available as "linear_lp".
    for (d, m, T) in [k for k in runs if k[1] == "linear_nolp"]:
        if (d, "linear", T) in runs:
            runs[(d, "linear_lp", T)] = runs.pop((d, "linear", T))
        runs[(d, "linear", T)] = runs.pop((d, "linear_nolp", T))
    return runs


def scores(r):
    return np.load(r["path"].replace(".json", "_scores.npy"), allow_pickle=False)


def boot_diff(y, sa, sb, B=1000, seed=0):
    """F(a) - F(b) with a stratified bootstrap over test shots. Returns (diff, lo, hi)."""
    pa, pb = sa > 0, sb > 0
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    # A shot only matters through (a right, b right); count those per class for speed.
    ok_a0, ok_b0 = ~pa[i0], ~pb[i0]
    ok_a1, ok_b1 = pa[i1], pb[i1]
    d0 = ok_a0.astype(int) - ok_b0.astype(int)
    d1 = ok_a1.astype(int) - ok_b1.astype(int)
    diff = 0.5 * (d0.mean() + d1.mean())
    rng = np.random.default_rng(seed)
    # multinomial on the three values -1, 0, 1 is equivalent to resampling the shots
    v = np.array([-1, 0, 1])
    c0 = np.array([(d0 == x).sum() for x in v])
    c1 = np.array([(d1 == x).sum() for x in v])
    m0 = rng.multinomial(len(d0), c0 / len(d0), B) @ v / len(d0)
    m1 = rng.multinomial(len(d1), c1 / len(d1), B) @ v / len(d1)
    lo, hi = np.percentile(0.5 * (m0 + m1), [2.5, 97.5])
    return float(diff), float(lo), float(hi)


def f_table(runs, ds):
    tab = {}
    for (d, m, T), rs in runs.items():
        if d != ds:
            continue
        Fs = [r["F"] for r in rs]
        tab.setdefault(m, {})[T] = dict(F=float(np.mean(Fs)), sd=float(np.std(Fs)) if len(Fs) > 1 else None, n_seeds=len(Fs),
                                        params=rs[0].get("params"), p01=float(np.mean([r["p01"] for r in rs])),
                                        p10=float(np.mean([r["p10"] for r in rs])), T_actual=rs[0].get("T_actual_ns", T))
    return tab


def compare(runs, ds, T, pairs):
    y = np.load(CACHE / ds / "yte.npy", allow_pickle=False).astype(int)
    out = []
    for a, b in pairs:
        ra, rb = runs.get((ds, a, T)), runs.get((ds, b, T))
        if not ra or not rb:
            continue
        d, lo, hi = boot_diff(y, scores(ra[0]), scores(rb[0]))
        out.append(dict(a=a, b=b, T=T, diff_pp=100 * d, lo_pp=100 * lo, hi_pp=100 * hi))
    return out


def takeaways(ds_name, tab):
    """The numbers behind takeaways 3 and 4 for one dataset."""
    res = {}
    models = [m for m in ORDER if m in tab]
    # Takeaway 3: what does truncating to 600 ns cost, relative to the best duration of that model?
    t3 = {}
    for m in models:
        Ts = sorted(tab[m])
        best_T = max(Ts, key=lambda t: tab[m][t]["F"])
        e_best = 1 - tab[m][best_T]["F"]
        row = dict(best_T=best_T, F_best=tab[m][best_T]["F"])
        for t in (600, 800, 1000):
            if t in tab[m]:
                row[f"err_ratio_{t}"] = (1 - tab[m][t]["F"]) / e_best
        t3[m] = row
    res["truncation_cost"] = t3
    # Takeaway 4: per duration, who is best, and how far are the small models from the large ones?
    t4 = {}
    for t in sorted({t for m in models for t in tab[m]}):
        vals = {m: tab[m][t]["F"] for m in models if t in tab[m]}
        ml = {m: v for m, v in vals.items() if m != "linear"}
        if ml:
            t4[t] = dict(best=max(ml, key=ml.get), F=vals, gain_over_linear_pp=(100 * (max(ml.values()) - vals["linear"])) if "linear" in vals else None)
    res["by_duration"] = t4
    return res


def main():
    runs = load_runs()
    out = dict(datasets={})
    for ds in sorted({k[0] for k in runs}):
        tab = f_table(runs, ds)
        Ts = sorted({t for m in tab for t in tab[m]})
        pairs = [("cnn", "fnn"), ("cnn", "transformer"), ("herqules", "fnn"), ("herqules", "transformer"), ("cnn", "herqules"),
                 ("cnn", "klinq"), ("klinq", "fnn"), ("cnn", "qubic"), ("cnn", "linear"), ("fnn", "linear"), ("fnn", "transformer")]
        cmp_ = [c for t in Ts for c in compare(runs, ds, t, pairs)]
        out["datasets"][ds] = dict(table=tab, comparisons=cmp_, takeaways=takeaways(ds, tab),
                                   meta=json.loads((CACHE / ds / "meta.json").read_text()))
    p = XD / "error_modes.json"
    if p.exists():
        out["error_modes"] = json.loads(p.read_text())
    for name in ("qick", "cao", "hassani", "gunyho", "zheng"):
        q = XD / f"linear_{name}.json"
        if q.exists():
            out[f"linear_{name}"] = json.loads(q.read_text())
    (XD / "summary.json").write_text(json.dumps(out))
    for ds, v in out["datasets"].items():
        print("==", ds)
        Ts = sorted({t for m in v["table"] for t in v["table"][m]})
        print("model      " + " ".join(f"{t:>7}" for t in Ts))
        for m in sorted(v["table"], key=lambda x: ORDER.index(x.split("_")[0]) if x.split("_")[0] in ORDER else 99):
            print(f"{m:<11}" + " ".join((f"{v['table'][m][t]['F']:7.4f}" if t in v["table"][m] else "      .") for t in Ts))


if __name__ == "__main__":
    main()
