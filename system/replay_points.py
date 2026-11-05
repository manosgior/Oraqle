"""Tier-2 smoke test for §7: re-run a few cheap published sweep jobs exactly and compare with results/system.

Every sweep job is seeded from its point alone (zlib.crc32 of (d, T, p/f, preset, chunk/pattern)), so with the same
stim / PyMatching / ldpc versions (the oraqle-latsim:dec image) a re-run reproduces the published shots and failures
bit for bit. This script rebuilds the job tuple of published lines exactly as the sweep scripts built it (job sizes from
the paper runs' settings, documented in docs/reproduce/qec_system.md) and re-runs them:

  grid      sweep_ler.run      ler_grid.jsonl     (--set grid: target 200, cap scale 1)
  disc_hi   sweep_ler.run      ler_disc_hi.jsonl  (--set disc --target-fails 1000 --cap-scale 5)
  soft      sweep_soft.run     ler_soft.jsonl     (flagged measurements as erasures; e_u from soft_params.json)
  flags     sweep_soft.flag_params on soft/cnn_soft_*ns.npz == soft_params.json (numpy only)
  decoders  sweep_decoders.run ler_decoders.jsonl (all 7 decoders incl. union-find, paired shots)
  timing    detection events of decode_timing.py (seed d*1000+T, 20,000 shots) == events_per_round (timings differ)

Only d = 5 points with few shots are picked (seconds each). Usage (repo root, oraqle-latsim:dec image):
  docker run --rm -v $PWD:/repo:ro -w /repo oraqle-latsim:dec python system/replay_points.py [--n 2] [--sets ...]
Exit status 1 if any replayed job differs.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
from oraqle_env import RESULTS  # noqa: E402


def lines(path):
    return [json.loads(l) for l in open(path)]


def pick(rows, n, key=lambda r: r["shots"]):
    rows = sorted(rows, key=key)
    step = max(1, len(rows) // n)
    return rows[::step][:n]


def replay_ler(data, name, target, cap_scale, n):
    import sweep_ler as L
    rows = [r for r in lines(os.path.join(data, f"ler_{name}.jsonl")) if r["d"] == 5 and r["shots"] <= 40000]
    bad = 0
    for r in pick(rows, n):
        k = L.CHUNKS[r["d"]]
        cap = L.SHOT_CAP[r["d"]] * cap_scale
        job = (r["d"], r["T"], r["p"], r["preset"], r["label"], r["chunk"], int(cap / k), -(-target // k))
        t0 = time.time()
        got = L.run(job)
        ok = (got["shots"], got["fails"], got["detectors"], got["circuit_ns"]) == \
             (r["shots"], r["fails"], r["detectors"], r["circuit_ns"])
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {name:<8} d={r['d']} T={r['T']} p={r['p']:.4f} {r['label']:<12} chunk={r['chunk']}: "
              f"published {r['fails']}/{r['shots']}  replay {got['fails']}/{got['shots']}  ({time.time() - t0:.1f}s)",
              flush=True)
    return bad


def replay_soft(data, n):
    import sweep_soft as S
    from sweep_ler import CHUNKS, SHOT_CAP
    rows = [r for r in lines(os.path.join(data, "ler_soft.jsonl")) if r["d"] == 5 and r["shots"] <= 40000 and r["f"] > 0]
    bad = 0
    for r in pick(rows, n):
        nn = S.PATTERNS if r["f"] > 0 else CHUNKS[r["d"]]
        job = (r["d"], r["T"], r["f"], r["p"], r["pattern"], int(SHOT_CAP[r["d"]] / nn), -(-S.TARGET_FAILS // nn))
        t0 = time.time()
        got = S.run(job)
        ok = (got["shots"], got["fails"]) == (r["shots"], r["fails"])
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} soft     d={r['d']} T={r['T']} f={r['f']} pattern={r['pattern']}: "
              f"published {r['fails']}/{r['shots']}  replay {got['fails']}/{got['shots']}  ({time.time() - t0:.1f}s)",
              flush=True)
    return bad


def replay_flags(data):
    import sweep_soft as S
    pub = json.load(open(os.path.join(data, "soft_params.json")))
    bad = 0
    for fn in sorted(os.listdir(os.path.join(data, "soft"))):
        if fn.endswith(".npz"):
            T, p = S.flag_params(os.path.join(data, "soft", fn))
            ok = json.loads(json.dumps({str(f): v for f, v in p.items()})) == pub[str(T)]
            bad += not ok
            print(f"{'ok ' if ok else 'BAD'} flags    {fn}: e_u/e_flagged per flag fraction == soft_params.json[{T}]", flush=True)
    return bad


def replay_decoders(data, n):
    import sweep_decoders as D
    rows = [r for r in lines(os.path.join(data, "ler_decoders.jsonl")) if r["d"] == 5 and r["shots"] <= 20000]
    bad = 0
    for r in pick(rows, n):
        nn = D.PATTERNS if r["f"] > 0 else D.CHUNKS[r["d"]]
        job = (r["d"], r["T"], r["f"], r["p"], r["pattern"], int(D.SHOT_CAP[r["d"]] / nn), -(-D.TARGET_FAILS // nn),
               D.UF_CAP_FRAC)
        t0 = time.time()
        got = D.run(job)
        ok = (got["shots"], got["uf_shots"], got["fails"]) == (r["shots"], r["uf_shots"], r["fails"])
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} decoders d={r['d']} T={r['T']} f={r['f']} pattern={r['pattern']}: "
              f"published {r['fails']} / {r['shots']} ({r['uf_shots']} UF)\n"
              f"{'':13}replay    {got['fails']} / {got['shots']} ({got['uf_shots']} UF)  ({time.time() - t0:.1f}s)",
              flush=True)
    return bad


def replay_timing(data, Ts):
    from discriminators import measure_error
    from sweep_ler import build
    pub = {r["T"]: r for r in lines(os.path.join(data, "decode_timing_d5.jsonl"))}
    bad = 0
    for T in Ts:
        r = pub[T]
        c, _, t_circ = build(5, T, measure_error(r["model"], T), r["preset"])
        det, _ = c.compile_detector_sampler(seed=5 * 1000 + T).sample(r["shots"], separate_observables=True)
        epr = float(det.sum(axis=1).mean() / r["rounds"])
        ok = epr == r["events_per_round"] and c.num_detectors == r["detectors"] and t_circ == r["circuit_ns"]
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} timing   d=5 T={T}: events/round published {r['events_per_round']} replay {epr}"
              f" (decode times are machine-dependent and not compared)", flush=True)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(RESULTS / "system"))
    ap.add_argument("--n", type=int, default=2, help="jobs per set")
    ap.add_argument("--sets", nargs="*", default=["grid", "disc_hi", "flags", "soft", "decoders", "timing"])
    a = ap.parse_args()
    bad = 0
    if "grid" in a.sets:
        bad += replay_ler(a.data, "grid", 200, 1.0, a.n)
    if "disc_hi" in a.sets:
        bad += replay_ler(a.data, "disc_hi", 1000, 5.0, a.n)
    if "flags" in a.sets:
        bad += replay_flags(a.data)
    if "soft" in a.sets:
        bad += replay_soft(a.data, a.n)
    if "decoders" in a.sets:
        bad += replay_decoders(a.data, a.n)
    if "timing" in a.sets:
        bad += replay_timing(a.data, [400, 1000])
    print(f"\n{bad} replayed job(s) differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
