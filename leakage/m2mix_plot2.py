"""m2mix_plot2.py -- local plots of the m2mix v2 fits (TEST split). One figure per qubit:
rows = windows (early 0-200 ns, paper '1 us' = samples 0-499, full 2 us); columns = data prep0,
data prep1 (log counts, with M2f component ellipses and the M1 G2 location), Pearson residual
of M2f (prep1), projection on the |0>-|1> anchor axis and on the perpendicular (both in units
of the anchor separation).  Bottom row: time traces (1 us fit, M2f responsibilities > 0.8) and
the acquisition-order share of the far component.
Usage: python m2mix_plot2.py <dir with m2mix_fit2_*.json, m2mix_plot2_*.npz, m2mix_traj2_*> <outdir>
"""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm

D, OUT = sys.argv[1], sys.argv[2]
COL = {"G0": "#2a78d6", "G1": "#eb6834", "R": "#1baf7a", "G2": "#6250d6", "G3": "#eda100", "R12": "#e87ba4",
       "tot": "#0b0b0b"}
WINS = [("early", "early 0-200 ns"), ("1us", "paper '1 us' (samples 0-499)"), ("2us", "full 2 us")]


def ell(ax, mu, S, color, n=2.0, ls="-"):
    S = np.asarray(S); vals, vecs = np.linalg.eigh(S)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(mu, 2 * n * np.sqrt(vals[1]), 2 * n * np.sqrt(vals[0]), angle=ang, fill=False,
                         ec=color, lw=1.6, ls=ls))


def proj(P, J, key):
    A0, A1 = np.array(J["anchors"][0]), np.array(J["anchors"][1])
    sep = np.linalg.norm(A1 - A0); e = (A1 - A0) / sep; ep = np.array([-e[1], e[0]])
    GX, GY = np.meshgrid(P["gx"], P["gy"], indexing="ij")
    X = np.stack([GX, GY], -1) - A0
    u, v = X @ e / sep, X @ ep / sep
    return u, v, sep


for q in range(5):
    fig, axs = plt.subplots(4, 5, figsize=(26, 19))
    handles = {}
    for r, (w, wl) in enumerate(WINS):
        try:
            J = json.load(open(f"{D}/m2mix_fit2_test_q{q}_{w}.json"))
            P = np.load(f"{D}/m2mix_plot2_test_q{q}_{w}.npz")
        except FileNotFoundError:
            continue
        R2 = J["results"]["M2f"]; R1 = J["results"]["M1"]
        far = R2["far"]
        ext = [P["gx"][0], P["gx"][-1], P["gy"][0], P["gy"][-1]]
        cnt = P["counts"].astype(float)
        for bb in (0, 1):
            ax = axs[r, bb]
            im = ax.imshow(np.where(cnt[bb] > 0, cnt[bb], np.nan).T, origin="lower", extent=ext, aspect="auto",
                           cmap="Blues", norm=LogNorm(vmin=1, vmax=cnt.max()))
            for k, g in ((0, "G0"), (1, "G1")):
                ell(ax, R2[f"mu{k}"], R2[f"S{k}"], COL[g])
            for g in ("G2", "G3"):
                ell(ax, R2[g]["mu"], R2[g]["cov"], COL[g], ls="-" if g == far else "--")
            ax.plot(*np.array([R2["mu0"], R2["mu1"]]).T, color=COL["R"], lw=1.2)
            ax.plot(*R1["G2"]["mu"], "x", color="#e34948", ms=10, mew=2.5)
            ax.plot(*np.array(J["anchors"]).T, "+", color="#0b0b0b", ms=12, mew=2)
            wt = R2["weights"]
            ax.set_title(f"Q{q+1}/q{q} {wl}: data prepared {bb}  (N={int(P['N'][bb])})\n"
                         f"M2f far={far} w={100*wt[far][bb]:.3f}%  other={100*wt['G3' if far=='G2' else 'G2'][bb]:.3f}%",
                         fontsize=9.5)
            ax.set_xlabel("along |0>->|1> axis (ADC units)"); ax.set_ylabel("perpendicular (ADC units)")
        ax = axs[r, 2]
        mod = P["M2f_tot"][1]
        res = np.where(mod > 0.5, (cnt[1] - mod) / np.sqrt(np.maximum(mod, 1e-9)), np.nan)
        res[(cnt[1] == 0) & (mod < 0.5)] = np.nan
        im = ax.imshow(res.T, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", vmin=-8, vmax=8)
        fig.colorbar(im, ax=ax, fraction=0.04, label="(data - model)/sqrt(model)")
        ax.set_title(f"M2f Pearson residual, prepared 1\nLL gains: M0->M1 {J['compare']['M0->M1']['LR']:.0f}, "
                     f"M0f->M1f {J['compare']['M0f->M1f']['LR']:.0f}, far test {J['compare']['M1f_nofar->M2f']['LR']:.0f}",
                     fontsize=9.5)
        ax.set_xlabel("along (ADC units)")
        u, v, sep = proj(P, J, "M2f")
        names = [str(s) for s in P["M2f_groupnames"]]
        for c, coord, lab in ((3, u, "along, units of |0>-|1> separation (0 = |0>, 1 = |1>)"),
                              (4, v, "perpendicular, units of separation")):
            ax = axs[r, c]
            hb = P["h"] / sep
            edges = np.arange(coord.min() - hb, coord.max() + 2 * hb, hb)
            ctr = 0.5 * (edges[1:] + edges[:-1])
            for bb, mk in ((0, "o"), (1, "s")):
                hd = np.histogram(coord.ravel(), edges, weights=cnt[bb].ravel())[0]
                hm = np.histogram(coord.ravel(), edges, weights=P["M2f_tot"][bb].ravel())[0]
                a = 0.35 if bb == 0 else 1.0
                h1, = ax.plot(ctr, hd, mk, ms=2.5, color="#52514e", alpha=a, label=f"data prep {bb}")
                h2, = ax.plot(ctr, hm, "-", color=COL["tot"], lw=1.0, alpha=a, label="M2f total")
                handles[f"data prep {bb}"] = h1; handles["M2f total"] = h2
                if bb == 1 or True:
                    for i, g in enumerate(names):
                        hg = np.histogram(coord.ravel(), edges, weights=P["M2f_groups"][bb, i].ravel())[0]
                        hh, = ax.plot(ctr, hg, "-" if bb == 1 else ":", color=COL[g], lw=1.6,
                                      label=f"{g}{' (far)' if g == far else ''}")
                        if bb == 1:
                            handles[g] = hh
            ax.set_yscale("log"); ax.set_ylim(0.5, None)
            ax.set_xlabel(lab); ax.set_ylabel("shots per bin")
            ax.set_title(f"projection (solid: prep 1, dotted/faint: prep 0)", fontsize=9.5)
    # bottom row: time traces + acquisition order (1 us fit)
    try:
        T = np.load(f"{D}/m2mix_traj2_q{q}_1us.npz")
        S = json.load(open(f"{D}/m2mix_traj2_summary.json"))[f"q{q}_1us"]["M2f"]
        J = json.load(open(f"{D}/m2mix_fit2_test_q{q}_1us.json"))
        names = [str(s) for s in T["M2f_names"]]
        far = S["far"]
        t = np.arange(20) * 100 + 50
        A0, A1 = np.array(J["anchors"][0]), np.array(J["anchors"][1])
        for c, (bb, comp) in enumerate(((1, 0), (1, 1), (0, 0), (0, 1))):
            ax = axs[3, c]
            for i, g in enumerate(names):
                n = T["M2f_nhard"][bb, i]
                if n < 30:
                    continue
                m, sd = T["M2f_hard"][bb, i, :, comp], T["M2f_hardsd"][bb, i, :, comp]
                ax.plot(t, m, "-o", ms=3, color=COL[g], label=f"{g} (n={n})")
                ax.fill_between(t, m - sd, m + sd, color=COL[g], alpha=0.12)
            ax.axvline(1000, color="#52514e", lw=0.8, ls="--")
            ax.set_xlabel("time bin centre (ns); dashed = end of paper '1 us' window")
            ax.set_ylabel(["along", "perpendicular"][comp] + " (ADC units)")
            ax.set_title(f"mean trace of shots with M2f responsibility > 0.8, prepared {bb} (+-1 sd)", fontsize=9.5)
            ax.legend(fontsize=8, loc="best")
        ax = axs[3, 4]
        for bb, mk in ((0, "o"), (1, "s")):
            for g in ("G2", "G3"):
                if g not in S:
                    continue
                d = S[g][f"prep{bb}"]
                ax.plot(np.arange(14), 100 * np.array(d["chunk_share_mean_over_states"]), "-" + mk, color=COL[g],
                        alpha=1 if bb else 0.5, label=f"{g}{' (far)' if g == far else ''} prep {bb}: chi2/dof={d['chunk_overdispersion'] or float('nan'):.1f}")
        ax.set_xlabel("acquisition-order chunk within each prepared state (2500 shots each)")
        ax.set_ylabel("share of shots assigned (resp > 0.5), %")
        ax.set_yscale("log")
        ax.set_title("does the extra component's share drift with shot order?", fontsize=9.5)
        ax.legend(fontsize=8)
    except FileNotFoundError:
        pass
    order = [k for k in ("data prep 0", "data prep 1", "M2f total", "G0", "G1", "R", "G2", "G3") if k in handles]
    fig.legend([handles[k] for k in order], order, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(order),
               fontsize=11, frameon=False)
    fig.suptitle(f"Q{q+1} (code q{q}) semi-supervised mixture fits, TEST split (560000 shots per prepared bit). "
                 f"Ellipses = 2 sigma; solid violet/yellow = far / other extra Gaussian of M2f; red x = M1 G2; + = data anchors",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    fig.savefig(f"{OUT}/m2mix_q{q}.png", dpi=80)
    plt.close(fig)
    print("wrote", f"{OUT}/m2mix_q{q}.png")
