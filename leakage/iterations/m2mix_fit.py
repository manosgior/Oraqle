"""m2mix_fit.py -- main mixture fits per (qubit, window, split).  Agent tag m2mix.

Models (see m2mix_core): M0 = G0+G1+R ; M1 = M0+G2 ; M0f/M1f = flexible (2-sub, unimodal)
cores ; M1split / M1fsplit = G2 location free per prepared bit ; M1r12 / M1fr12 = M1(f)
plus a G1<->G2 bridge.  Several G2 initialisations (residual peaks of the no-G2 fit) are
tried and all candidates are recorded.
Usage: python m2mix_fit.py /in(out of prep) /out  [tasks like test:2:1us ...]  (default: all)
"""
import sys, os, json, time, itertools, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2mix_core as C

IN, OUT = sys.argv[1], sys.argv[2]
WINS = ["early", "1us", "2us"]


def load(split, q, w, correct=True):
    Z = np.load(f"{IN}/m2mix_winmeans_{split}.npz")
    y = Z["y"]
    x = Z[f"x_{w}"][:, q, :].astype(np.float64)
    if correct:
        x = x - Z[f"shift_{w}"][y, q, :]
    b = (y >> q) & 1
    sig0 = float(Z[f"sig0_{w}"][q])
    core = Z[f"core_{w}"][:, q, :]
    c0 = core[[s for s in range(32) if not (s >> q) & 1]].mean(0)
    c1 = core[[s for s in range(32) if (s >> q) & 1]].mean(0)
    return x, b, y, sig0, c0, c1


def best_of(cands):
    return max(cands, key=lambda t: t[1] if np.isfinite(t[1]) else -np.inf)


def run_models(data, log, tag=""):
    res, params, cands_rec = {}, {}, {}
    t0 = time.time()
    s0, s1 = C.Spec(), C.Spec(g2="shared")
    sf0, sf1 = C.Spec(flex=True), C.Spec(flex=True, g2="shared")
    P0, LL0 = C.fit(C.init_M0(data), s0, data)
    res["M0"], params["M0"] = C.describe(P0, s0, data, LL0), P0
    log(f"{tag} M0 LL={LL0:.1f} ({time.time()-t0:.0f}s)")
    inits = C.g2_inits(P0, s0, data)
    cands = []
    for xy, how, ex in inits:
        P, LL = C.fit(C.add_g2(P0, xy, data), s1, data)
        cands.append((P, LL, f"M0+G2@{how}{np.round(xy, 2).tolist()}"))
    # flexible cores
    rng = np.random.default_rng(1)
    c = [C.fit(C.add_flex(P0, data), sf0, data), C.fit(C.add_flex(P0, data, rng), sf0, data)]
    P0f, LL0f = best_of(c)
    res["M0f"], params["M0f"] = C.describe(P0f, sf0, data, LL0f), P0f
    log(f"{tag} M0f LL={LL0f:.1f} ({time.time()-t0:.0f}s)")
    P1, LL1 = best_of(cands)[:2]
    fc = [(*C.fit(C.add_flex(P1, data), sf1, data), "M1+flex")]
    for xy, how, ex in C.g2_inits(P0f, sf0, data, seed=1):
        fc.append((*C.fit(C.add_g2(P0f, xy, data), sf1, data), f"M0f+G2@{how}{np.round(xy, 2).tolist()}"))
    P1f, LL1f = best_of(fc)[:2]
    # cross-seed: M1 from the best M1f G2 location
    cands.append((*C.fit(C.add_g2(P0, P1f["mu2"], data), s1, data), "M0+G2@M1f"))
    P1, LL1 = best_of(cands)[:2]
    fc.append((*C.fit(C.add_g2(P0f, P1["mu2"], data), sf1, data), "M0f+G2@M1"))
    P1f, LL1f = best_of(fc)[:2]
    cands_rec["M1"] = [{"init": c[2], "LL": c[1], "mu2": c[0]["mu2"].tolist(),
                        "w2": np.exp(c[0]["logits"][:, 3] - np.log(np.exp(c[0]["logits"]).sum(1))).tolist()} for c in cands]
    cands_rec["M1f"] = [{"init": c[2], "LL": c[1], "mu2": c[0]["mu2"].tolist(),
                         "w2": np.exp(c[0]["logits"][:, 3] - np.log(np.exp(c[0]["logits"]).sum(1))).tolist()} for c in fc]
    res["M1"], params["M1"] = C.describe(P1, s1, data, LL1), P1
    res["M1f"], params["M1f"] = C.describe(P1f, sf1, data, LL1f), P1f
    log(f"{tag} M1 LL={LL1:.1f} mu2={P1['mu2']}  M1f LL={LL1f:.1f} mu2={P1f['mu2']} ({time.time()-t0:.0f}s)")
    # split G2 location per prepared bit
    for nm, base, flex in (("M1split", P1, False), ("M1fsplit", P1f, True)):
        sp = C.Spec(flex=flex, g2="split")
        P, LL = C.fit(C.to_split(base), sp, data)
        res[nm], params[nm] = C.describe(P, sp, data, LL), P
        log(f"{tag} {nm} LL={LL:.1f} ({time.time()-t0:.0f}s)")
    # G1<->G2 bridge
    for nm, base, flex in (("M1r12", P1, False), ("M1fr12", P1f, True)):
        sp = C.Spec(flex=flex, g2="shared", r12=True)
        P, LL = C.fit(C.add_r12(base), sp, data)
        res[nm], params[nm] = C.describe(P, sp, data, LL), P
        log(f"{tag} {nm} LL={LL:.1f} mu2={P['mu2']} ({time.time()-t0:.0f}s)")
    cmp = {}
    for a, b in (("M0", "M1"), ("M0f", "M1f"), ("M0", "M0f"), ("M1", "M1f"), ("M1", "M1r12"), ("M1f", "M1fr12"),
                 ("M1", "M1split"), ("M1f", "M1fsplit"), ("M0f", "M1fr12")):
        cmp[f"{a}->{b}"] = {"LR": 2 * (res[b]["LL"] - res[a]["LL"]), "dk": res[b]["k"] - res[a]["k"],
                            "dBIC": res[b]["BIC"] - res[a]["BIC"]}
    return res, params, cands_rec, cmp


def plotdata(data, specs_params):
    out = {"counts": data.cur_full.reshape(2, data.nx, data.ny).astype(np.int32), "gx": data.gx, "gy": data.gy,
           "h": data.h, "N": data.N}
    area = data.h ** 2
    for nm, (sp, P) in specs_params.items():
        lp, GL, lpi = C.full_logp(P, sp, data)
        out[f"{nm}_tot"] = (np.exp(lp) * area * data.N[:, None]).reshape(2, data.nx, data.ny).astype(np.float32)
        grp = np.exp(GL[None] + lpi[:, None, :]) * area * data.N[:, None, None]  # (2,B,K)
        out[f"{nm}_groups"] = np.moveaxis(grp, 2, 1).reshape(2, len(sp.groups), data.nx, data.ny).astype(np.float32)
        out[f"{nm}_groupnames"] = np.array(sp.groups)
    return out


def task(arg):
    split, q, w = arg
    torch.set_num_threads(2)
    lf = open(f"{OUT}/logs/m2mix_fit_{split}_q{q}_{w}.log", "w")

    def log(s):
        lf.write(s + "\n"); lf.flush()

    x, b, y, sig0, c0, c1 = load(split, q, w)
    data = C.Binned(x, b, sig0, (c0, c1))
    log(f"bins {data.nx}x{data.ny} h={data.h:.4f} nz={len(data.nz)} N={data.N}")
    res, params, cands, cmp = run_models(data, log, f"{split} q{q} {w}")
    out = {"split": split, "qubit_code": q, "qubit_paper": f"Q{q+1}", "window": w, "sig0": sig0,
           "grid": {"h": data.h, "nx": data.nx, "ny": data.ny, "lo": data.lo.tolist()},
           "anchors": [c0.tolist(), c1.tolist()],
           "results": res, "compare": cmp, "candidates": cands,
           "params": {k: C.jsonable(v) for k, v in params.items()}}
    json.dump(out, open(f"{OUT}/m2mix_fit_{split}_q{q}_{w}.json", "w"), indent=1)
    if split == "test":
        sp = {"M0": (C.Spec(), params["M0"]), "M1": (C.Spec(g2="shared"), params["M1"]),
              "M1f": (C.Spec(flex=True, g2="shared"), params["M1f"]),
              "M1fr12": (C.Spec(flex=True, g2="shared", r12=True), params["M1fr12"])}
        np.savez_compressed(f"{OUT}/m2mix_plot_{split}_q{q}_{w}.npz", **plotdata(data, sp))
    log("done")
    return arg


if __name__ == "__main__":
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    if len(sys.argv) > 3:
        tasks = [(t.split(":")[0], int(t.split(":")[1]), t.split(":")[2]) for t in sys.argv[3:]]
    else:
        tasks = list(itertools.product(["test", "train"], range(5), WINS))
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(min(len(tasks), 30)) as pool:
        for r in pool.imap_unordered(task, tasks):
            print("finished", r, flush=True)
