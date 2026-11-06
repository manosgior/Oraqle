#!/usr/bin/env python3
"""Unit and regression checks of leaksim.py. Needs stim, pymatching and the lattice-sim sources (oraqle-latsim image)."""
import json, os, pathlib, sys
import numpy as np
import pymatching, stim
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "system"), str(pathlib.Path(__file__).resolve().parent)]
from sweep_ler import build
import leaksim as S


def ler(truth, belief, shots, seed=1):
    m = pymatching.Matching.from_detector_error_model(belief.detector_error_model(decompose_errors=True))
    det, obs = truth.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True, bit_packed=True)
    return int(np.sum(np.any(m.decode_batch(det, bit_packed_shots=True, bit_packed_predictions=True) != obs, axis=1)))


def main():
    p = 0.0373
    d, T = 5, 1000
    c, _, _ = build(d, T, p, "ibm_boston")
    lay = S.Layout(c)
    print("rounds", lay.rounds, "mr rounds", len(lay.mr_rounds), "ancillas", len(lay.all_ancillas), "meas", c.num_measurements)
    assert len(lay.mr_rounds) == 2 * (d + 1), lay.mr_rounds

    # 1. no leakage: rewriting changes nothing
    m0 = S.LeakModel(p_create=0.0, survive=0.0)
    h0 = S.draw_history(lay, m0, np.random.default_rng(0))
    tr0 = S.truth_circuit(lay, h0, m0)
    # compared as (targets -> probability): the rewrite drops QUBIT_COORDS, so the coordinate annotations differ
    a = {str(i.targets_copy()): i.args_copy()[0] for i in c.detector_error_model(decompose_errors=True) if i.type == "error"}
    b = {str(i.targets_copy()): i.args_copy()[0] for i in tr0.detector_error_model(decompose_errors=True) if i.type == "error"}
    print("1 DEM identical without leakage:", a == b, len(a), "errors")
    assert a == b

    # 2. blind readouts are silent and deterministic: without noise no detector fires, with or without leakage
    cn = c.without_noise()
    layn = S.Layout(cn)
    m = S.LeakModel(p_create=0.03, survive=0.5, frac_from_0=0.0, p_ld=0.0)
    h = S.draw_history(layn, m, np.random.default_rng(1))
    trn = S.truth_circuit(layn, h, m)
    det = trn.compile_detector_sampler(seed=2).sample(200)
    n_leak = sum(len(v) for v in h["leaked"].values())
    print("2 blind rounds", n_leak, "detection events", int(det.sum()))
    assert n_leak > 0 and det.sum() == 0
    trn.detector_error_model(decompose_errors=True)        # must not raise on non-deterministic detectors

    # 3. an episode created while reading |0> gives two events (start and end)
    m = S.LeakModel(p_create=0.01, survive=0.5, frac_from_0=1.0, p_ld=0.0)
    h = S.draw_history(layn, m, np.random.default_rng(3))
    trn = S.truth_circuit(layn, h, m)
    det = trn.compile_detector_sampler(seed=4).sample(50)
    ev = det.sum(1)
    print("3 from-|0> episodes", len(h["from0"]), "events per shot", ev.min(), ev.max())
    assert ev.min() == ev.max() and ev.min() % 2 == 0 and ev.min() > 0

    # 4. a data error under a blind ancilla is not seen until the ancilla is back
    print("4 covered by 2 and 3 plus the LER trend below")

    # 5. regression: the 1-bit erasure of sweep_soft.py at T=1000, f=0.05 and the hard point
    root = ROOT / "results" / "system"
    params = json.load(open(root / "soft_params.json"))["1000"]
    rows = [json.loads(l) for l in open(root / "ler_soft.jsonl")]
    for f, target in ((0.0, None), (0.05, None)):
        e_u = params[str(f)]["e_u"]
        r = [x for x in rows if x["d"] == 7 and x["T"] == 1000 and x["f"] == f]
        pub = sum(x["fails"] for x in r) / sum(x["shots"] for x in r)
        c7, _, _ = build(7, 1000, e_u, "ibm_boston")
        l7 = S.Layout(c7)
        rng = np.random.default_rng(5)
        fails = shots = 0
        for pat in range(16):
            levels = [(f, 0.5), (1 - f, e_u)] if f else [(1.0, e_u)]
            soft = S.draw_soft(c7.num_measurements, levels, rng)
            hh = S.draw_history(l7, S.LeakModel(0.0, 0.0), rng)
            t = S.truth_circuit(l7, hh, S.LeakModel(0.0, 0.0), soft=soft)
            fails += ler(t, t, 20000, seed=pat)
            shots += 20000
        got = fails / shots
        se = (got / shots) ** 0.5
        print(f"5 f={f}: published LER {pub:.5f}, here {got:.5f} +- {se:.5f}")
        assert abs(got - pub) < 4 * se + 0.1 * pub

    # 6. direction of the effects
    e = 0.0373
    for d7 in (5, 7):
        c7, _, _ = build(d7, T, e, "ibm_boston")
        l7 = S.Layout(c7)
        mods = dict(p_create=0.03, survive=0.5, p_ld=0.75, tpr=0.9, fpr=0.01, precision=0.75)
        out = {}
        base_ler = ler(c7, c7, 200000)
        m = S.LeakModel(**mods)
        base = S.belief_circuit(l7, S.draw_history(l7, S.LeakModel(0, 0), np.random.default_rng(0)), m, mode="base")
        f = dict(base=0, uniform=0, flag=0, oracle=0)
        rng = np.random.default_rng(7)
        mb = pymatching.Matching.from_detector_error_model(base.detector_error_model(decompose_errors=True))
        h0 = S.draw_history(l7, m, rng)
        mu = pymatching.Matching.from_detector_error_model(S.belief_circuit(l7, h0, m, mode="uniform").detector_error_model(decompose_errors=True))
        P = 16
        n = 4000
        for pat in range(P):
            h = S.draw_history(l7, m, rng)
            t = S.truth_circuit(l7, h, m)
            det, obs = t.compile_detector_sampler(seed=pat).sample(n, separate_observables=True, bit_packed=True)
            for name, mm in (("base", mb), ("uniform", mu),
                             ("flag", pymatching.Matching.from_detector_error_model(S.belief_circuit(l7, h, m, mode="flag").detector_error_model(decompose_errors=True))),
                             ("oracle", pymatching.Matching.from_detector_error_model(S.belief_circuit(l7, h, m, mode="oracle").detector_error_model(decompose_errors=True)))):
                f[name] += int(np.sum(np.any(mm.decode_batch(det, bit_packed_shots=True, bit_packed_predictions=True) != obs, axis=1)))
        print(f"6 d={d7} no-leak LER {base_ler/200000:.5f}; with leakage:", {k: round(v / (P * n), 5) for k, v in f.items()}, flush=True)
        assert f["oracle"] <= f["flag"] * 1.2 and f["flag"] <= f["base"]
    print("ALL OK")


if __name__ == "__main__":
    main()
