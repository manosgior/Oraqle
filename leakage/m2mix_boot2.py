"""m2mix_boot2.py -- bootstrap CIs and parametric LR nulls for the m2mix mixture fits (v2).

(a) nonparametric bootstrap ('np'): resample the binned counts multinomially per prepared bit
    (N_b fixed), refit M1, M1f and M2f warm-started from the full-data optimum, record all
    weights and the extra-component locations (relative to the fixed data anchors).
(b) parametric null for adding G2 to flexible cores ('nullA'): simulate from the fitted M0f,
    fit M0f and M1f (G2 inits: residual-excess peaks of the simulated M0f fit + the observed
    M1f G2 location), LR = 2(LL_M1f - LL_M0f).
(c) parametric null for the far-survival test ('nullB'): simulate from the fitted M1f_nofar,
    fit M1f_nofar and M2f (G3 inits: observed far location + residual peaks),
    LR = 2(LL_M2f - LL_M1f_nofar).
Usage: python m2mix_boot2.py /in /fit2 /out  kind:split:q:win:seed0:nrep ...
"""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2mix_core2 as C

IN, FIT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]


def load(split, q, w):
    Z = np.load(f"{IN}/m2mix_winmeans_{split}.npz")
    y = Z["y"]
    x = Z[f"x_{w}"][:, q, :].astype(np.float64) - Z[f"shift_{w}"][y, q, :]
    b = (y >> q) & 1
    sig0 = float(Z[f"sig0_{w}"][q])
    core = Z[f"core_{w}"][:, q, :]
    c0 = core[[s for s in range(32) if not (s >> q) & 1]].mean(0)
    c1 = core[[s for s in range(32) if (s >> q) & 1]].mean(0)
    return x, b, sig0, c0, c1


SPECS = {"M0f": C.Spec(flex=True), "M1": C.Spec(g2="shared"), "M1f": C.Spec(flex=True, g2="shared"),
         "M2f": C.Spec(flex=True, g2="shared2"), "M1f_nofar": C.Spec(flex=True, g2="shared")}


def summ(nm, P, LL, data):
    d = C.describe(P, SPECS[nm], data, LL)
    out = {"LL": LL if np.isfinite(LL) else None, "weights": d["weights"], "sep": d["sep"], "lam": d["lam"]}
    for g in ("G2", "G3"):
        if g in d:
            out[g] = {k: d[g][k] for k in ("mu", "along_anchor_sep", "perp_anchor_sep", "collapsed",
                                           "BC_with_G0", "BC_with_G1", "sigma_ratio_vs_sig0")}
    if "far" in d:
        out["far"] = d["far"]
    return out


def simulate(P, spec, data, rng):
    lp, _, _ = C.full_logp(P, spec, data)
    full = np.zeros((2, data.full_X.shape[0]), np.int64)
    for b in (0, 1):
        p = np.exp(lp[b] - lp[b].max())
        full[b] = rng.multinomial(int(data.N0[b]), p / p.sum())
    return full


def task(arg):
    kind, split, q, w, seed0, nrep = arg
    torch.set_num_threads(1)
    J = json.load(open(f"{FIT}/m2mix_fit2_{split}_q{q}_{w}.json"))
    Pbest = {k: C.from_json(v) for k, v in J["params"].items()}
    x, b, sig0, c0, c1 = load(split, q, w)
    data = C.Binned(x, b, sig0, (c0, c1))
    data.N0 = data.N.copy()
    full0 = data.cur_full.copy()
    out = []
    for r in range(seed0, seed0 + nrep):
        rng = np.random.default_rng(1000 * q + 100 * ["early", "1us", "2us"].index(w) + r + (0 if kind == "np" else 77777 * (1 + ["nullA", "nullB"].index(kind))))
        t0 = time.time()
        rec = {"rep": r}
        if kind == "np":
            full = np.zeros_like(full0)
            for bb in (0, 1):
                p = full0[bb] / full0[bb].sum()
                full[bb] = rng.multinomial(int(full0[bb].sum()), p)
            data.set_counts(full)
            for nm in ("M1", "M1f", "M2f"):
                P, LL = C.fit(C.tame(Pbest[nm]), SPECS[nm], data)
                rec[nm] = summ(nm, P, LL, data)
        elif kind == "nullA":
            data.set_counts(full0)
            full = simulate(Pbest["M0f"], SPECS["M0f"], data, rng)
            data.set_counts(full)
            P0, LL0 = C.fit(C.tame(Pbest["M0f"]), SPECS["M0f"], data)
            starts = [np.asarray(J["results"]["M1f"]["G2"]["mu"])] + [xy for xy, how, ex in C.g2_inits(P0, SPECS["M0f"], data)][:4]
            best = (None, -np.inf)
            for xy in starts:
                P, LL = C.fit(C.add_g2(P0, xy, data), SPECS["M1f"], data)
                if LL > best[1]:
                    best = (P, LL)
            rec["LL0"], rec["LL1"] = LL0, best[1]
            rec["LR"] = 2 * (best[1] - LL0)
            rec["M1f"] = summ("M1f", best[0], best[1], data) if best[0] is not None else None
        elif kind == "nullB":
            data.set_counts(full0)
            full = simulate(Pbest["M1f_nofar"], SPECS["M1f_nofar"], data, rng)
            data.set_counts(full)
            Pn, LLn = C.fit(C.tame(Pbest["M1f_nofar"]), SPECS["M1f_nofar"], data)
            far = J["results"]["M2f"]["far"]
            starts = [np.asarray(J["results"]["M2f"][far]["mu"])] + [xy for xy, how, ex in C.g2_inits(Pn, SPECS["M1f_nofar"], data, seed=r)][:3]
            best = (None, -np.inf)
            for xy in starts:
                P, LL = C.fit(C.add_g3(Pn, xy, data), SPECS["M2f"], data)
                if LL > best[1]:
                    best = (P, LL)
            rec["LLnofar"], rec["LL2"] = LLn, best[1]
            rec["LR"] = 2 * (best[1] - LLn)
            rec["M2f"] = summ("M2f", best[0], best[1], data) if best[0] is not None else None
        rec["sec"] = time.time() - t0
        out.append(rec)
    fn = f"{OUT}/m2mix_boot2_{kind}_{split}_q{q}_{w}_s{seed0}.json"
    json.dump({"kind": kind, "split": split, "q": q, "window": w, "reps": out}, open(fn, "w"), indent=1)
    return arg


if __name__ == "__main__":
    tasks = []
    for t in sys.argv[4:]:
        k, s, q, w, s0, n = t.split(":")
        tasks.append((k, s, int(q), w, int(s0), int(n)))
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(min(len(tasks), int(os.environ.get("NPROC", 30)))) as pool:
        for r in pool.imap_unordered(task, tasks):
            print("finished", r, flush=True)
