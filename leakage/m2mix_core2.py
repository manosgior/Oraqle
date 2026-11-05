"""m2mix_core2.py -- semi-supervised 2-D mixture model for leakage (agent tag m2mix), v2.

v2 changes vs m2mix_core.py (pilot): (1) add_g2/add_r12 use logsumexp (the pilot overflowed
and produced NaN for every G2 model of Q2/q1); (2) fit() keeps the last finite parameter set
and stops when the LL gain per round is < tol (default 0.01 nats); (3) new g2="shared2" model
(two free extra Gaussians G2, G3) to test whether a far component survives when another
extra Gaussian is available to absorb core non-Gaussianity; (4) every extra component is
also located relative to the FIXED data anchors (per-state mean-shift modes averaged over
the 16 states with that bit), not only the fitted (drifting) core means; collapse flags.

Data: per qubit, the 2-D window mean x = (along, perp) of every shot and the prepared
bit b of that qubit. Shots are binned on a fine grid (h = sigma0/4, sigma0 = robust
|0> noise sigma), so the likelihood is a count-weighted likelihood on bin centres;
the bin quantisation variance h^2/12 is added to every component covariance.

Top-level components (shapes SHARED by prepared 0 and prepared 1, weights per prepared bit):
  G0   |0> core. Main Gaussian whose mean is ANCHORED within 1.5 sigma0 of the mean-shift
       mode of the prepared-0 data (prevents label switching: without the anchor a single
       broad "G1" moves onto Q3's far cloud and G2 takes the |1> core).
       "flex" variant: 2-Gaussian mixture whose 2nd sub-component mean is constrained to
       within Dmax=2 Mahalanobis units (of the main sub) of the main mean -> can absorb
       skew / heavy tails, cannot form a separate cluster.
  G1   |1> core, same (anchored to the prepared-1 mode).
  R    T1 / excitation bridge: Gaussian-convolved line segment between the main G0 and
       G1 means, mu(s) = (1-s) mu0 + s mu1, s in (0,1) on an M-point grid, density of s
       proportional to exp(lam s), covariance interpolated (1-s) S0 + s S1.
  G2   optional free 2-D Gaussian (candidate |2>). 'split' variant: separate G2a (only in
       prepared 0) and G2b (only in prepared 1) to test location consistency.
  R12  optional bridge between G1 and G2 (mid-window 1<->2 transitions).
All free covariances have Cholesky diagonals >= 0.5 sigma0 (every stationary state's
window mean carries at least the amplifier noise).
"""
import math, numpy as np, torch
from scipy.special import logsumexp

torch.set_default_dtype(torch.float64)
LOG2PI = math.log(2 * math.pi)
ANCHOR_R = 1.5     # sigma0 units


# ----------------------------------------------------------------------------- binning
class Binned:
    def __init__(self, x, b, sig0, anchors, hfac=0.25, max_bins=600):
        x = np.asarray(x, np.float64)
        b = np.asarray(b).astype(np.int64)
        h = sig0 * hfac
        lo = x.min(0) - 3 * h
        hi = x.max(0) + 3 * h
        while np.prod(np.ceil((hi - lo) / h)) > max_bins ** 2:
            h *= 1.25
        nx, ny = [int(v) for v in np.ceil((hi - lo) / h)]
        ix = np.clip(((x - lo) / h).astype(np.int64), 0, [nx - 1, ny - 1])
        flat = ix[:, 0] * ny + ix[:, 1]
        full = np.zeros((2, nx * ny), np.int64)
        for bb in (0, 1):
            full[bb] = np.bincount(flat[b == bb], minlength=nx * ny)
        self.h, self.lo, self.nx, self.ny, self.sig0 = h, lo, nx, ny, sig0
        self.anchors = [torch.tensor(np.asarray(a, float)) for a in anchors]
        self.N = full.sum(1)                          # shots per prepared bit
        gx = lo[0] + (np.arange(nx) + 0.5) * h
        gy = lo[1] + (np.arange(ny) + 0.5) * h
        GX, GY = np.meshgrid(gx, gy, indexing="ij")
        self.full_X = np.stack([GX.ravel(), GY.ravel()], 1)
        self.gx, self.gy = gx, gy
        self.set_counts(full)

    def set_counts(self, full):
        """(re)select the union of non-empty bins for the counts `full` (2, nx*ny)."""
        nz = np.nonzero(full.sum(0))[0]
        self.nz = nz
        self.X = torch.tensor(self.full_X[nz])
        self.C = torch.tensor(full[:, nz].astype(np.float64))
        self.Ntot = float(full.sum())
        self.N = full.sum(1)
        self.logarea = 2 * math.log(self.h)
        self.cur_full = full


# ----------------------------------------------------------------------------- spec
class Spec:
    def __init__(self, flex=False, g2=None, r12=False, M=40, M12=40, Dmax=2.0):
        self.flex, self.g2, self.r12, self.M, self.M12, self.Dmax = flex, g2, r12, M, M12, Dmax
        groups = ["G0", "G1", "R"]
        if g2 == "shared":
            groups.append("G2")
        elif g2 == "shared2":
            groups += ["G2", "G3"]
        elif g2 == "split":
            groups += ["G2a", "G2b"]
        if r12:
            assert g2 == "shared"
            groups.append("R12")
        self.groups = groups
        mask = np.zeros((2, len(groups)))
        if g2 == "split":
            mask[1, groups.index("G2a")] = -np.inf
            mask[0, groups.index("G2b")] = -np.inf
        self.mask = torch.tensor(mask)

    @property
    def name(self):
        n = "M1" if self.g2 else "M0"
        if self.g2 == "shared2":
            n = "M2"
        if self.g2 == "split":
            n += "split"
        if self.flex:
            n += "f"
        if self.r12:
            n += "r12"
        return n

    def n_params(self):
        k = 5 + 5 + 1                      # G0, G1 (mean 2 + cov 3), lam
        if self.flex:
            k += 2 * 6                     # offset 2 + cov 3 + sub weight 1, per core
        if self.g2 == "shared":
            k += 5
        elif self.g2 in ("split", "shared2"):
            k += 10
        if self.r12:
            k += 1
        allowed = np.isfinite(self.mask.numpy()).sum(1)
        k += int((allowed - 1).sum())
        return k


# ----------------------------------------------------------------------------- params
def cov_from(ch, floor):
    l11 = floor + torch.exp(ch[0])
    l22 = floor + torch.exp(ch[1])
    l21 = ch[2]
    return l11 * l11, l11 * l21, l21 * l21 + l22 * l22, (l11, l21, l22)


def ch_from_cov(S, floor):
    S = np.asarray(S, float)
    L = np.linalg.cholesky(S + 1e-12 * np.eye(2))
    l11 = max(L[0, 0] - floor, 1e-3 * floor)
    l22 = max(L[1, 1] - floor, 1e-3 * floor)
    return np.array([math.log(l11), math.log(l22), L[1, 0]])


def bounded(v, R):
    r = torch.sqrt((v * v).sum() + 1e-12)
    return v * (R * torch.tanh(r) / r)


def unbounded(off, R):
    """inverse of bounded() for an offset vector (numpy)."""
    off = np.asarray(off, float)
    n = np.linalg.norm(off)
    if n < 1e-9:
        return np.zeros(2)
    t = min(n / R, 0.999)
    r = np.arctanh(t)
    return off / n * r


def core_mean(P, k, data):
    return data.anchors[k] + bounded(P[f"m{k}"], ANCHOR_R * data.sig0)


def gauss_ld(X, mu, a, b, d):
    dx = X[:, None, 0] - mu[None, :, 0]
    dy = X[:, None, 1] - mu[None, :, 1]
    det = a * d - b * b
    q = (d * dx * dx - 2 * b * dx * dy + a * dy * dy) / det
    return -LOG2PI - 0.5 * torch.log(det) - 0.5 * q


def components(P, spec, data):
    """dict group -> (mu (m,2), a,b,d (m,), logw (m,)), plus the main core params."""
    sig0, h = data.sig0, data.h
    fl = 0.5 * sig0
    qv = h * h / 12.0
    out = {}
    cores = {}
    for k in (0, 1):
        mu = core_mean(P, k, data)
        a, b, d, L = cov_from(P[f"ch{k}"], fl)
        cores[k] = (mu, a, b, d)
        mus, A, B, D = [mu[None]], [a[None]], [b[None]], [d[None]]
        if spec.flex:
            u = bounded(P[f"v{k}"], spec.Dmax)
            off = torch.stack([L[0] * u[0], L[1] * u[0] + L[2] * u[1]])
            a2, b2, d2, _ = cov_from(P[f"ch{k}b"], fl)
            mus.append((mu + off)[None]); A.append(a2[None]); B.append(b2[None]); D.append(d2[None])
            w = torch.sigmoid(P[f"a{k}"])
            lw = torch.log(torch.cat([(1 - w)[None], w[None]]) + 1e-300)
        else:
            lw = torch.zeros(1)
        out[f"G{k}"] = (torch.cat(mus), torch.cat(A) + qv, torch.cat(B), torch.cat(D) + qv, lw)
    s = (torch.arange(spec.M, dtype=torch.float64) + 0.5) / spec.M
    (m0, a0, b0, d0), (m1, a1, b1, d1) = cores[0], cores[1]
    muR = (1 - s)[:, None] * m0[None] + s[:, None] * m1[None]
    out["R"] = (muR, (1 - s) * a0 + s * a1 + qv, (1 - s) * b0 + s * b1, (1 - s) * d0 + s * d1 + qv,
                torch.log_softmax(P["lam"] * s, 0))
    if spec.g2 == "shared":
        a2, b2, d2, _ = cov_from(P["ch2"], fl)
        out["G2"] = (P["mu2"][None], (a2 + qv)[None], b2[None], (d2 + qv)[None], torch.zeros(1))
        if spec.r12:
            s2 = (torch.arange(spec.M12, dtype=torch.float64) + 0.5) / spec.M12
            mu = (1 - s2)[:, None] * m1[None] + s2[:, None] * P["mu2"][None]
            out["R12"] = (mu, (1 - s2) * a1 + s2 * a2 + qv, (1 - s2) * b1 + s2 * b2,
                          (1 - s2) * d1 + s2 * d2 + qv, torch.log_softmax(P["lam12"] * s2, 0))
    elif spec.g2 == "split":
        for nm in ("a", "b"):
            a2, b2, d2, _ = cov_from(P[f"ch2{nm}"], fl)
            out[f"G2{nm}"] = (P[f"mu2{nm}"][None], (a2 + qv)[None], b2[None], (d2 + qv)[None], torch.zeros(1))
    elif spec.g2 == "shared2":
        for nm in ("2", "3"):
            a2, b2, d2, _ = cov_from(P[f"ch{nm}"], fl)
            out[f"G{nm}"] = (P[f"mu{nm}"][None], (a2 + qv)[None], b2[None], (d2 + qv)[None], torch.zeros(1))
    return out, cores


def group_logdens(P, spec, X, data):
    comps, _ = components(P, spec, data)
    GL = []
    for g in spec.groups:
        mu, a, b, d, lw = comps[g]
        GL.append(torch.logsumexp(gauss_ld(X, mu, a, b, d) + lw[None], 1))
    return torch.stack(GL, 1)  # (B, K)


def log_pi(P, spec):
    return torch.log_softmax(P["logits"] + spec.mask, 1)  # (2, K)


def logp_bits(P, spec, X, data):
    GL = group_logdens(P, spec, X, data)
    lpi = log_pi(P, spec)
    return torch.logsumexp(GL[None] + lpi[:, None, :], 2), GL, lpi  # (2,B)


def nll(P, spec, data):
    lp, _, _ = logp_bits(P, spec, data.X, data)
    return -(data.C * lp).sum()


# ----------------------------------------------------------------------------- fitting
def to_torch(Pnp, grad=True):
    return {k: torch.tensor(np.asarray(v, float), requires_grad=grad) for k, v in Pnp.items()}


def to_np(P):
    return {k: v.detach().numpy().copy() for k, v in P.items()}


def fit(Pnp, spec, data, max_rounds=40, iters=200, tol=1e-2):
    P = to_torch(Pnp)
    params = list(P.values())
    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, tolerance_grad=1e-10,
                            tolerance_change=1e-13, history_size=50, line_search_fn="strong_wolfe")
    scale = 1.0 / data.Ntot

    def closure():
        opt.zero_grad()
        loss = nll(P, spec, data) * scale
        loss.backward()
        return loss

    with torch.no_grad():
        prev = float(nll(P, spec, data))
    good = to_np(P) if np.isfinite(prev) else None
    for r in range(max_rounds):
        try:
            opt.step(closure)
        except Exception:
            break
        with torch.no_grad():
            cur = float(nll(P, spec, data))
        finite = np.isfinite(cur) and all(np.isfinite(v.detach().numpy()).all() for v in P.values())
        if not finite:
            break
        good = to_np(P)
        if np.isfinite(prev) and abs(prev - cur) < tol:
            break
        prev = cur
    if good is None:
        return to_np(P), -np.inf
    P = to_torch(good, grad=False)
    with torch.no_grad():
        cur = float(nll(P, spec, data))
    LL = -cur + data.logarea * data.Ntot
    if not np.isfinite(LL):
        LL = -np.inf
    return good, LL


# ----------------------------------------------------------------------------- inits
def tame(P, floor=-14.0):
    """renormalise logits to log-weights and floor them at log(1e-6), so that a weight driven
    to exactly 0 (logits ~1e5 after L-BFGS) can regrow when the params seed another fit."""
    P = {k: np.array(v, copy=True) for k, v in P.items()}
    lg = P["logits"] - logsumexp(P["logits"], 1, keepdims=True)
    P["logits"] = np.maximum(lg, floor)
    return P


def init_M0(data):
    sig0 = data.sig0
    S = np.eye(2) * sig0 ** 2
    fl = 0.5 * sig0
    lg = np.log(np.array([[0.95, 0.03, 0.02], [0.05, 0.85, 0.10]]))
    return {"m0": np.zeros(2), "ch0": ch_from_cov(S, fl), "m1": np.zeros(2),
            "ch1": ch_from_cov(S, fl), "lam": np.array(0.0), "logits": lg}


def add_flex(P, data, rng=None):
    P = tame(P)
    fl = 0.5 * data.sig0
    for k in (0, 1):
        a, b, d, _ = cov_from(torch.tensor(P[f"ch{k}"]), fl)
        S = np.array([[float(a), float(b)], [float(b), float(d)]])
        P[f"v{k}"] = np.array([0.05, 0.05]) if rng is None else rng.normal(0, 0.3, 2)
        P[f"ch{k}b"] = ch_from_cov(2.0 * S, fl)
        P[f"a{k}"] = np.array(-2.0)
    return P


def add_g2(P, mu2, data, w=0.01, cov_scale=1.5):
    P = tame(P)
    P["mu2"] = np.array(mu2, float)
    P["ch2"] = ch_from_cov(np.eye(2) * (cov_scale * data.sig0) ** 2, 0.5 * data.sig0)
    lg = P["logits"]
    P["logits"] = np.concatenate([lg[:, :3], np.full((2, 1), np.log(w)) + logsumexp(lg[:, :3], 1, keepdims=True), lg[:, 3:]], 1)
    return P


def add_g3(P, mu3, data, w=0.01, cov_scale=1.5):
    """M1(f) shared params -> M2(f) shared2 init: append G3 (logit slot 4)."""
    P = {k: np.array(v, copy=True) for k, v in tame(P).items() if k not in ("lam12",)}
    P["mu3"] = np.array(mu3, float)
    P["ch3"] = ch_from_cov(np.eye(2) * (cov_scale * data.sig0) ** 2, 0.5 * data.sig0)
    lg = P["logits"][:, :4]
    P["logits"] = np.concatenate([lg, np.full((2, 1), np.log(w)) + logsumexp(lg, 1, keepdims=True)], 1)
    return P


def drop_extra(P, keep):
    """M2(f) params -> M1(f) params keeping extra component `keep` ('2' or '3') as G2."""
    Q = {k: np.array(v, copy=True) for k, v in P.items() if k not in ("mu2", "ch2", "mu3", "ch3")}
    Q["mu2"], Q["ch2"] = P[f"mu{keep}"].copy(), P[f"ch{keep}"].copy()
    j = 3 if keep == "2" else 4
    Q["logits"] = P["logits"][:, [0, 1, 2, j]].copy()
    return tame(Q)


def add_r12(P, w=0.005):
    P = tame(P)
    P["lam12"] = np.array(0.0)
    P["logits"] = np.concatenate([P["logits"], np.full((2, 1), np.log(w)) + logsumexp(P["logits"], 1, keepdims=True)], 1)
    return P


def to_split(P):
    """M1 (shared G2) params -> M1split init (G2a, G2b both at the shared G2)."""
    Q = {k: np.array(v, copy=True) for k, v in P.items() if k not in ("mu2", "ch2", "lam12")}
    for nm in ("a", "b"):
        Q[f"mu2{nm}"] = P["mu2"].copy()
        Q[f"ch2{nm}"] = P["ch2"].copy()
    lg = P["logits"][:, :4]
    Q["logits"] = np.concatenate([lg, lg[:, 3:4]], 1)
    return tame(Q)


# ----------------------------------------------------------------------------- evaluation
def full_logp(Pnp, spec, data, X=None):
    P = to_torch(Pnp, grad=False)
    X = torch.tensor(data.full_X) if X is None else torch.tensor(np.asarray(X, float))
    with torch.no_grad():
        out_lp, out_GL = [], []
        for i in range(0, X.shape[0], 100000):
            lp, GL, lpi = logp_bits(P, spec, X[i:i + 100000], data)
            out_lp.append(lp); out_GL.append(GL)
        return torch.cat(out_lp, 1).numpy(), torch.cat(out_GL, 0).numpy(), lpi.numpy()


def main_means(Pnp, data):
    P = to_torch(Pnp, grad=False)
    with torch.no_grad():
        return core_mean(P, 0, data).numpy(), core_mean(P, 1, data).numpy()


def g2_inits(Pnp, spec, data, npk=4, min_sep=2.0, smooth=1.0, kmeans_k=3, out_r=4.0, seed=0):
    """candidate G2 start positions: (a) peaks of the smoothed positive (data - model)
    excess, (b) count-weighted k-means centres of the bins > out_r sigma0 from both core
    means (outlier populations). Deduplicated within 1.5 sigma0."""
    from scipy.ndimage import gaussian_filter
    lp, _, _ = full_logp(Pnp, spec, data)
    area = data.h ** 2
    ex = np.zeros(data.full_X.shape[0])
    for b in (0, 1):
        e = np.exp(lp[b]) * area * data.N[b]
        ex += (data.cur_full[b] - e) / data.N[b]
    img = gaussian_filter(ex.reshape(data.nx, data.ny), smooth * data.sig0 / data.h)
    GX, GY = np.meshgrid(data.gx, data.gy, indexing="ij")
    cands = []
    work = img.copy()
    for _ in range(npk):
        i = np.unravel_index(np.argmax(work), work.shape)
        if not np.isfinite(work[i]) or work[i] <= 0:
            break
        xy = np.array([data.gx[i[0]], data.gy[i[1]]])
        cands.append((xy, "excess", float(work[i])))
        work[(GX - xy[0]) ** 2 + (GY - xy[1]) ** 2 < (min_sep * data.sig0) ** 2] = -np.inf
    m0, m1 = main_means(Pnp, data)
    X = data.full_X
    w = data.cur_full.sum(0).astype(float)
    far = (np.linalg.norm(X - m0, axis=1) > out_r * data.sig0) & (np.linalg.norm(X - m1, axis=1) > out_r * data.sig0) & (w > 0)
    if far.sum() > 10 and w[far].sum() > 50:
        Xf, wf = X[far], w[far]
        rng = np.random.default_rng(seed)
        for k in range(1, kmeans_k + 1):
            cen = Xf[rng.choice(len(Xf), k, replace=False, p=wf / wf.sum())]
            for _ in range(50):
                lab = np.argmin(((Xf[:, None] - cen[None]) ** 2).sum(2), 1)
                cen = np.array([np.average(Xf[lab == j], axis=0, weights=wf[lab == j]) if (lab == j).any() else cen[j]
                                for j in range(k)])
            for j in range(k):
                cands.append((cen[j], f"kmeans{k}", float(wf[lab == j].sum())))
    out = []
    for c in cands:
        if all(np.linalg.norm(c[0] - o[0]) > 1.5 * data.sig0 for o in out):
            out.append(c)
    return out


def cov_np(ch, sig0):
    a, b, d, _ = cov_from(torch.tensor(np.asarray(ch, float)), 0.5 * sig0)
    return np.array([[float(a), float(b)], [float(b), float(d)]])


def bhatt(m1, S1, m2, S2):
    m1, m2 = np.asarray(m1), np.asarray(m2)
    S = 0.5 * (np.asarray(S1) + np.asarray(S2))
    dm = m1 - m2
    DB = 0.125 * dm @ np.linalg.solve(S, dm) + 0.5 * math.log(
        np.linalg.det(S) / math.sqrt(np.linalg.det(S1) * np.linalg.det(S2)))
    return float(math.exp(-DB))


def describe(Pnp, spec, data, LL):
    sig0 = data.sig0
    mu0, mu1 = main_means(Pnp, data)
    S0, S1 = cov_np(Pnp["ch0"], sig0), cov_np(Pnp["ch1"], sig0)
    dv = mu1 - mu0
    sep = float(np.linalg.norm(dv))
    e = dv / sep
    ep = np.array([-e[1], e[0]])
    pi = np.exp(log_pi(to_torch(Pnp, grad=False), spec).numpy())
    k = spec.n_params()
    Nt = float(data.N.sum())
    d = {"model": spec.name, "LL": LL, "k": k, "BIC": -2 * LL + k * math.log(Nt), "N_per_bit": data.N.tolist(),
         "mu0": mu0.tolist(), "mu1": mu1.tolist(), "S0": S0.tolist(), "S1": S1.tolist(), "sep": sep,
         "anchor_offset_sig0": [float(np.linalg.norm(mu0 - data.anchors[0].numpy()) / sig0),
                                float(np.linalg.norm(mu1 - data.anchors[1].numpy()) / sig0)],
         "axis_angle_deg": float(np.degrees(np.arctan2(dv[1], dv[0]))), "lam": float(Pnp["lam"]),
         "weights": {g: [float(pi[0, i]), float(pi[1, i])] for i, g in enumerate(spec.groups)},
         "sig0": sig0, "h": data.h, "SNR_sep_over_sig0": sep / sig0,
         "sigma_core_over_sig0": [float(np.linalg.det(S0) ** 0.25 / sig0), float(np.linalg.det(S1) ** 0.25 / sig0)]}
    if spec.flex:
        comps, _ = components(to_torch(Pnp, grad=False), spec, data)
        for kk in (0, 1):
            mu, a, b, dd, lw = [t.detach() for t in comps[f"G{kk}"]]
            d[f"G{kk}_sub"] = {"mus": mu.numpy().tolist(), "w": np.exp(lw.numpy()).tolist(),
                               "offset_over_sig0": float(np.linalg.norm(mu[1].numpy() - mu[0].numpy()) / sig0),
                               "sub_sigma_over_sig0": float((a[1] * dd[1] - b[1] ** 2).sqrt().sqrt() / sig0)}

    A0, A1 = data.anchors[0].numpy(), data.anchors[1].numpy()
    dA = A1 - A0
    sepA = float(np.linalg.norm(dA))
    eA = dA / sepA
    epA = np.array([-eA[1], eA[0]])
    d["anchor_sep"] = sepA

    def loc(mu, S):
        mu = np.asarray(mu)
        rel = mu - mu0
        r1 = mu - mu1
        rA = mu - A0
        Si0, Si1 = np.linalg.inv(S0), np.linalg.inv(S1)
        bc0, bc1 = bhatt(mu0, S0, mu, S), bhatt(mu1, S1, mu, S)
        dm0, dm1 = float(np.sqrt(rel @ Si0 @ rel)), float(np.sqrt(r1 @ Si1 @ r1))
        return {"mu": mu.tolist(),
                "along_anchor_sep": float(rA @ eA / sepA), "perp_anchor_sep": float(rA @ epA / sepA),
                "along_anchor_sig0": float(rA @ eA / sig0), "perp_anchor_sig0": float(rA @ epA / sig0),
                "collapsed": bool(max(bc0, bc1) > 0.5 or min(dm0, dm1) < 2.0),
                "along_sep": float(rel @ e / sep), "perp_sep": float(rel @ ep / sep),
                "along_sig0": float(rel @ e / sig0), "perp_sig0": float(rel @ ep / sig0),
                "dist_from_G1_sep": float(np.linalg.norm(r1) / sep),
                "D_M_from_G0": float(np.sqrt(rel @ Si0 @ rel)), "D_M_from_G1": float(np.sqrt(r1 @ Si1 @ r1)),
                "cov": np.asarray(S).tolist(), "sigma_ratio_vs_sig0": float(np.linalg.det(S) ** 0.25 / sig0),
                "BC_with_G0": bhatt(mu0, S0, mu, S), "BC_with_G1": bhatt(mu1, S1, mu, S)}

    if spec.g2 == "shared":
        d["G2"] = loc(Pnp["mu2"], cov_np(Pnp["ch2"], sig0))
        if spec.r12:
            d["lam12"] = float(Pnp["lam12"])
    elif spec.g2 == "shared2":
        d["G2"] = loc(Pnp["mu2"], cov_np(Pnp["ch2"], sig0))
        d["G3"] = loc(Pnp["mu3"], cov_np(Pnp["ch3"], sig0))
        sc = {g: max(d[g]["BC_with_G0"], d[g]["BC_with_G1"]) for g in ("G2", "G3")}
        d["far"] = min(sc, key=sc.get)      # the extra component least overlapping either core
    elif spec.g2 == "split":
        d["G2a_prep0"] = loc(Pnp["mu2a"], cov_np(Pnp["ch2a"], sig0))
        d["G2b_prep1"] = loc(Pnp["mu2b"], cov_np(Pnp["ch2b"], sig0))
    return d


def jsonable(Pnp):
    return {k: np.asarray(v).tolist() for k, v in Pnp.items()}


def from_json(d):
    return {k: np.asarray(v, float) for k, v in d.items()}
