"""m2mix_coinc2.py -- per-shot coincidence of the M2f extra components across qubits (TEST).

For every qubit and window (1us, 2us) each shot gets the M2f per-bin posterior (same grid and
neighbour-shift correction as the fit). 'X' = hard assignment (resp > 0.5) to any NON-collapsed
extra component of M2f; 'far' = the far component only.
For each target t and source s, restricted to prepared bit_t = 1 and bit_s = 1:
    P(X_t | X_s) vs P(X_t | not X_s) -> lift and counts
and P(X_t | bit_s = 1) vs P(X_t | bit_s = 0) (prepared-state dependence).
A drive-crosstalk mechanism (source's pi pulse exciting the target's e-f transition) predicts
a large bit_s effect with lift ~1 given bit_s = 1; signal leakage from the source's far-state
readout into the target channel predicts lift >> 1.
Usage: python m2mix_coinc2.py /in /fit2 /out
"""
import sys, json, numpy as np

IN, FIT, OUT = sys.argv[1:4]
Z = np.load(f"{IN}/m2mix_winmeans_test.npz")
y = Z["y"]
bits = (y[:, None] >> np.arange(5)) & 1
res = {}
for w in ("1us", "2us"):
    X, FAR, info = {}, {}, {}
    for q in range(5):
        J = json.load(open(f"{FIT}/m2mix_fit2_test_q{q}_{w}.json"))
        P = np.load(f"{FIT}/m2mix_plot2_test_q{q}_{w}.npz")
        g = J["grid"]
        x = Z[f"x_{w}"][:, q, :].astype(np.float64) - Z[f"shift_{w}"][y, q, :]
        ix = np.clip(((x - np.asarray(g["lo"])) / g["h"]).astype(np.int64), 0, [g["nx"] - 1, g["ny"] - 1])
        names = [str(s) for s in P["M2f_groupnames"]]
        R = P["M2f_resp"][bits[:, q], :, ix[:, 0], ix[:, 1]]
        M2 = J["results"]["M2f"]
        ext = [n for n in ("G2", "G3") if not M2[n]["collapsed"]]
        far = M2["far"]
        X[q] = (R[:, [names.index(n) for n in ext]].sum(1) > 0.5) if ext else np.zeros(len(y), bool)
        FAR[q] = R[:, names.index(far)] > 0.5
        info[q] = {"extras_noncollapsed": ext, "far": far}
    out = {"info": info}
    for lab, A in (("X", X), ("far", FAR)):
        tab = {}
        for t in range(5):
            for s in range(5):
                if s == t:
                    continue
                m11 = (bits[:, t] == 1) & (bits[:, s] == 1)
                m10 = (bits[:, t] == 1) & (bits[:, s] == 0)
                a, bsel = A[t], A[s]
                n_s = int((m11 & bsel).sum()); n_ns = int((m11 & ~bsel).sum())
                k_s = int((m11 & bsel & a).sum()); k_ns = int((m11 & ~bsel & a).sum())
                p_s = k_s / max(n_s, 1); p_ns = k_ns / max(n_ns, 1)
                tab[f"t{t}_s{s}"] = {
                    "P_t_given_sbit1": float(a[m11].mean()), "P_t_given_sbit0": float(a[m10].mean()),
                    "bit_ratio": float(a[m11].mean() / max(a[m10].mean(), 1e-12)),
                    "n_s_flag": n_s, "k_t_and_s_flag": k_s, "P_t_given_s_flag": p_s,
                    "n_s_noflag": n_ns, "k_t_and_s_noflag": k_ns, "P_t_given_s_noflag": p_ns,
                    "lift": float(p_s / p_ns) if p_ns > 0 else None}
        out[lab] = tab
    res[w] = out
    for t in range(5):
        for s in range(5):
            if s != t:
                d = out["X"][f"t{t}_s{s}"]
                if d["bit_ratio"] > 1.5 or (d["lift"] or 0) > 3:
                    print(w, f"target Q{t+1} source Q{s+1}", json.dumps(d), flush=True)
json.dump(res, open(f"{OUT}/m2mix_coinc2.json", "w"), indent=1)
print("done")
