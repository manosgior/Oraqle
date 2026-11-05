"""m2mix_fit2.py -- main mixture fits per (split, qubit, window).  Agent tag m2mix, v2.

Models (m2mix_core2):
  M0   = G0 + G1 + R                     (plain Gaussian cores, T1/excitation bridge)
  M1   = M0 + G2                         (free 2-D Gaussian = candidate |2>)
  M0f  = flexible cores (each core a 2-Gaussian mixture, 2nd sub within 2 Mahalanobis units)
  M1f  = M0f + G2
  M2f  = M0f + G2 + G3                   (two free extra Gaussians)
  M1split / M1fsplit : G2 location free per prepared bit
  M1r12 / M1fr12     : + bridge G1<->G2 (mid-window 1->2 transitions)
  M1f_nofar          : M2f with its 'far' extra component removed, refit (the far-survival test)
Several G2/G3 initialisations (residual-excess peaks, k-means of outlying bins, cross seeds)
are tried; all candidates are recorded.
v2 vs pilot m2mix_fit.py: uses m2mix_core2 (NaN/overflow fix), adds M2f + M1f_nofar, saves
per-bin responsibilities, threads=1.
Usage: python m2mix_fit2.py /in /out [split:q:win ...]   (default: all 30)
"""
import sys, os, json, time, itertools, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2mix_core2 as C

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


def w_extra(P, j):
    lg = P["logits"]
    return np.exp(lg[:, j] - np.logaddexp.reduce(lg, axis=1)).tolist()


def rec(c, j=3):
    out = {"init": c[2], "LL": c[1] if np.isfinite(c[1]) else None, "mu2": np.asarray(c[0]["mu2"]).tolist(),
           "w2": w_extra(c[0], j)}
    if "mu3" in c[0]:
        out["mu3"] = np.asarray(c[0]["mu3"]).tolist()
        out["w3"] = w_extra(c[0], 4)
    return out


def dedupe(points, sig0, exclude=(), r=1.5):
    out = []
    for p in points:
        p = np.asarray(p, float)
        if not np.all(np.isfinite(p)):
            continue
        if all(np.linalg.norm(p - o) > r * sig0 for o in list(out) + [np.asarray(e) for e in exclude]):
            out.append(p)
    return out


def run_models(data, log, tag=""):
    res, params, cands_rec = {}, {}, {}
    t0 = time.time()
    specs = {"M0": C.Spec(), "M1": C.Spec(g2="shared"), "M0f": C.Spec(flex=True), "M1f": C.Spec(flex=True, g2="shared"),
             "M2f": C.Spec(flex=True, g2="shared2"), "M1f_nofar": C.Spec(flex=True, g2="shared"),
             "M1split": C.Spec(g2="split"), "M1fsplit": C.Spec(flex=True, g2="split"),
             "M1r12": C.Spec(g2="shared", r12=True), "M1fr12": C.Spec(flex=True, g2="shared", r12=True)}

    def put(nm, P, LL):
        res[nm], params[nm] = C.describe(P, specs[nm], data, LL), P

    P0, LL0 = C.fit(C.init_M0(data), specs["M0"], data)
    put("M0", P0, LL0)
    log(f"{tag} M0 LL={LL0:.1f} ({time.time()-t0:.0f}s)")
    cands = []
    for xy, how, ex in C.g2_inits(P0, specs["M0"], data):
        P, LL = C.fit(C.add_g2(P0, xy, data), specs["M1"], data)
        cands.append((P, LL, f"M0+G2@{how}{np.round(xy, 2).tolist()}"))
    rng = np.random.default_rng(1)
    c = [C.fit(C.add_flex(P0, data), specs["M0f"], data), C.fit(C.add_flex(P0, data, rng), specs["M0f"], data)]
    P0f, LL0f = best_of(c)
    put("M0f", P0f, LL0f)
    log(f"{tag} M0f LL={LL0f:.1f} ({time.time()-t0:.0f}s)")
    P1, LL1 = best_of(cands)[:2]
    fc = [(*C.fit(C.add_flex(P1, data), specs["M1f"], data), "M1+flex")]
    for xy, how, ex in C.g2_inits(P0f, specs["M0f"], data, seed=1):
        fc.append((*C.fit(C.add_g2(P0f, xy, data), specs["M1f"], data), f"M0f+G2@{how}{np.round(xy, 2).tolist()}"))
    P1f, LL1f = best_of(fc)[:2]
    if np.all(np.isfinite(P1f["mu2"])):
        cands.append((*C.fit(C.add_g2(P0, P1f["mu2"], data), specs["M1"], data), "M0+G2@M1f"))
    P1, LL1 = best_of(cands)[:2]
    if np.all(np.isfinite(P1["mu2"])):
        fc.append((*C.fit(C.add_g2(P0f, P1["mu2"], data), specs["M1f"], data), "M0f+G2@M1"))
    P1f, LL1f = best_of(fc)[:2]
    cands_rec["M1"] = [rec(c) for c in cands]
    cands_rec["M1f"] = [rec(c) for c in fc]
    put("M1", P1, LL1)
    put("M1f", P1f, LL1f)
    log(f"{tag} M1 LL={LL1:.1f} mu2={P1['mu2']}  M1f LL={LL1f:.1f} mu2={P1f['mu2']} ({time.time()-t0:.0f}s)")
    # ---- M2f: best M1f + a second free Gaussian at every other distinct candidate location
    pts = [cc[0]["mu2"] for cc in sorted(cands + fc, key=lambda t: -(t[1] if np.isfinite(t[1]) else -1e300))]
    pts += [xy for xy, how, ex in C.g2_inits(P1f, specs["M1f"], data, seed=2)]
    starts = dedupe(pts, data.sig0, exclude=[P1f["mu2"]])[:6]
    c2 = []
    for xy in starts:
        P, LL = C.fit(C.add_g3(P1f, xy, data), specs["M2f"], data)
        c2.append((P, LL, f"M1f+G3@{np.round(xy, 2).tolist()}"))
    if not c2:
        c2.append((*C.fit(C.add_g3(P1f, P1f["mu2"] + 3 * data.sig0, data), specs["M2f"], data), "M1f+G3@offset"))
    P2f, LL2f = best_of(c2)[:2]
    cands_rec["M2f"] = [rec(c) for c in c2]
    put("M2f", P2f, LL2f)
    far = res["M2f"]["far"]
    keep = "3" if far == "G2" else "2"
    Pn, LLn = C.fit(C.drop_extra(P2f, keep), specs["M1f_nofar"], data)
    put("M1f_nofar", Pn, LLn)
    res["M2f"]["far_component"] = far
    log(f"{tag} M2f LL={LL2f:.1f} far={far} {res['M2f'][far]['mu']} w={res['M2f']['weights'][far]}  "
        f"M1f_nofar LL={LLn:.1f} ({time.time()-t0:.0f}s)")
    # ---- split G2 location per prepared bit
    for nm, base in (("M1split", P1), ("M1fsplit", P1f)):
        P, LL = C.fit(C.to_split(base), specs[nm], data)
        put(nm, P, LL)
        log(f"{tag} {nm} LL={LL:.1f} ({time.time()-t0:.0f}s)")
    # ---- G1<->G2 bridge
    for nm, base in (("M1r12", P1), ("M1fr12", P1f)):
        P, LL = C.fit(C.add_r12(base), specs[nm], data)
        put(nm, P, LL)
        log(f"{tag} {nm} LL={LL:.1f} mu2={P['mu2']} ({time.time()-t0:.0f}s)")
    cmp = {}
    for a, b in (("M0", "M1"), ("M0f", "M1f"), ("M0", "M0f"), ("M1", "M1f"), ("M1f", "M2f"), ("M1f_nofar", "M2f"),
                 ("M0f", "M2f"), ("M1", "M1r12"), ("M1f", "M1fr12"), ("M1", "M1split"), ("M1f", "M1fsplit")):
        cmp[f"{a}->{b}"] = {"LR": 2 * (res[b]["LL"] - res[a]["LL"]), "dk": res[b]["k"] - res[a]["k"],
                            "dBIC": res[b]["BIC"] - res[a]["BIC"]}
    return res, params, cands_rec, cmp, specs


def plotdata(data, specs, params, names):
    out = {"counts": data.cur_full.reshape(2, data.nx, data.ny).astype(np.int32), "gx": data.gx, "gy": data.gy,
           "h": data.h, "N": data.N}
    area = data.h ** 2
    for nm in names:
        sp, P = specs[nm], params[nm]
        lp, GL, lpi = C.full_logp(P, sp, data)
        out[f"{nm}_tot"] = (np.exp(lp) * area * data.N[:, None]).reshape(2, data.nx, data.ny).astype(np.float32)
        grp = np.exp(GL[None] + lpi[:, None, :]) * area * data.N[:, None, None]  # (2,B,K)
        out[f"{nm}_groups"] = np.moveaxis(grp, 2, 1).reshape(2, len(sp.groups), data.nx, data.ny).astype(np.float32)
        out[f"{nm}_groupnames"] = np.array(sp.groups)
        resp = np.exp(GL[None] + lpi[:, None, :] - lp[:, :, None])  # (2,B,K) posterior per bin
        out[f"{nm}_resp"] = np.moveaxis(resp, 2, 1).reshape(2, len(sp.groups), data.nx, data.ny).astype(np.float32)
    return out


def task(arg):
    split, q, w = arg
    torch.set_num_threads(1)
    lf = open(f"{OUT}/logs/m2mix_fit2_{split}_q{q}_{w}.log", "w")

    def log(s):
        lf.write(s + "\n"); lf.flush()

    x, b, y, sig0, c0, c1 = load(split, q, w)
    data = C.Binned(x, b, sig0, (c0, c1))
    log(f"bins {data.nx}x{data.ny} h={data.h:.4f} nz={len(data.nz)} N={data.N}")
    res, params, cands, cmp, specs = run_models(data, log, f"{split} q{q} {w}")
    out = {"split": split, "qubit_code": q, "qubit_paper": f"Q{q+1}", "window": w, "sig0": sig0,
           "grid": {"h": data.h, "nx": data.nx, "ny": data.ny, "lo": data.lo.tolist()},
           "anchors": [c0.tolist(), c1.tolist()],
           "results": res, "compare": cmp, "candidates": cands,
           "params": {k: C.jsonable(v) for k, v in params.items()}}
    json.dump(out, open(f"{OUT}/m2mix_fit2_{split}_q{q}_{w}.json", "w"), indent=1)
    np.savez_compressed(f"{OUT}/m2mix_plot2_{split}_q{q}_{w}.npz",
                        **plotdata(data, specs, params, ["M0", "M1", "M0f", "M1f", "M2f", "M1fr12"]))
    log("done")
    return arg


if __name__ == "__main__":
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    if len(sys.argv) > 3:
        tasks = [(t.split(":")[0], int(t.split(":")[1]), t.split(":")[2]) for t in sys.argv[3:]]
    else:
        tasks = list(itertools.product(["test", "train"], range(5), WINS))
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(min(len(tasks), int(os.environ.get("NPROC", 24)))) as pool:
        for r in pool.imap_unordered(task, tasks):
            print("finished", r, flush=True)
