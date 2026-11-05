"""Producer of UNIFORM_FRACTION in system_cost.py: detection-event density of uniform circuit noise.

system_cost.py places every readout length on the Helios latency curve of qShell (measured versus a physical
error rate p) by matching the detection-event density of our d = 9 lattice-sim timing runs to that of a
standard Stim rotated memory-Z circuit (d = 9, 9 rounds) with uniform circuit noise p:

    UNIFORM_P        = [1e-3, 3e-3, 5e-3, 1e-2, 2e-2]
    UNIFORM_FRACTION = [0.0166, 0.0481, 0.0774, 0.1428, 0.2448]   # mean detection fraction per detector

Provenance. The constants had no producer script in any repository. They were produced by an inline run in
Claude session 94019386 (transcript ~/.claude/projects/-home-manosgior-Documents-GitHub-oraqle-ics/
94019386-7e9a-4302-8276-39a280e43d5e.jsonl, line 1039, 2026-10-26T11:57Z) with stim 1.16.0 from the MCMit
.venv, from the oraqle-ics checkout. This script is that run, unchanged in substance:

    for p in [1e-3, 3e-3, 5e-3, 1e-2, 2e-2]:
        c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=9, rounds=9,
                after_clifford_depolarization=p, before_round_data_depolarization=p,
                before_measure_flip_probability=p, after_reset_flip_probability=p)
        det = c.compile_detector_sampler(seed=1).sample(20000)
        print(det.mean())
    # and, from data/sims/decode_timing_d9.jsonl: events_per_round * rounds / detectors per readout length

which printed 0.0166, 0.0481, 0.0774, 0.1428, 0.2448 and, for the d = 9 timing runs at T = 200..1000 ns,
0.3298, 0.1590, 0.1126, 0.1044, 0.1019.

Reproducibility. Stim's sampler is deterministic for a fixed seed only with the same stim version (and the same
SIMD code path); the 4-decimal values are reproduced exactly with stim 1.16.0 on x86-64 (AVX2). With another
stim build the values move by a few 1e-5 (20,000 shots x 720 detectors); use --tol to accept that.

Caveat (documented, not changed): qShell measured Helios under SI1000 noise with parameter p, while this
mapping uses uniform depolarizing noise p. The "equivalent p" is therefore a uniform-noise p used to index an
SI1000(p) latency curve; see results/external/COMPARISON_qshell_vs_sec7.md.

Usage:
  python system/derive_uniform_fraction.py [--timing results/system/decode_timing_d9.jsonl]
         [--out results/external/uniform_fraction_d9.json | --no-write] [--tol 0]
Docker (stim 1.16.0):
  docker run --rm -v $REPO:/repo:ro -e ORAQLE_RESULTS=/repo/results -w /repo oraqle-latsim \
      python system/derive_uniform_fraction.py --no-write
Exit status 1 if a rounded value differs from the hard-coded constants (beyond --tol).
"""
import argparse
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oraqle_env import RESULTS  # noqa: E402

UNIFORM_P = [1e-3, 3e-3, 5e-3, 1e-2, 2e-2]
# Copied from oraqle-ics/sims/system_cost.py @ d7cd4e9 (lines 69-70); checked against system/system_cost.py below.
EXPECTED_UNIFORM_FRACTION = [0.0166, 0.0481, 0.0774, 0.1428, 0.2448]
# Printed by the 2026-10-26 run for data/sims/decode_timing_d9.jsonl (d = 9, MCMit-CNN, T = 200..1000 ns).
EXPECTED_TIMING_FRACTION = {200: 0.3298, 400: 0.1590, 600: 0.1126, 800: 0.1044, 1000: 0.1019}
DISTANCE, ROUNDS, SHOTS, SEED = 9, 9, 20000, 1


def uniform_fraction(p):
    import stim
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=DISTANCE, rounds=ROUNDS,
                               after_clifford_depolarization=p, before_round_data_depolarization=p,
                               before_measure_flip_probability=p, after_reset_flip_probability=p)
    det = c.compile_detector_sampler(seed=SEED).sample(SHOTS)
    return float(det.mean()), c.num_detectors


def interp_log(x, xs, ys):
    """Same as system_cost.interp_log: piecewise-linear in log x, extrapolated from the last segment."""
    lx, lxs = math.log(x), [math.log(v) for v in xs]
    i = max(1, min(len(xs) - 1, next((j for j, v in enumerate(lxs) if v >= lx), len(xs) - 1)))
    t = (lx - lxs[i - 1]) / (lxs[i] - lxs[i - 1])
    return ys[i - 1] + t * (ys[i] - ys[i - 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--timing", default=str(RESULTS / "system" / "decode_timing_d9.jsonl"),
                    help="d = 9 decode-timing file (decode_timing.py output)")
    ap.add_argument("--out", default=str(RESULTS / "external" / "uniform_fraction_d9.json"))
    ap.add_argument("--no-write", action="store_true", help="print only (e.g. read-only repo mount)")
    ap.add_argument("--tol", type=float, default=0.0, help="accepted |difference| after rounding to 4 decimals")
    a = ap.parse_args()

    import stim
    ok = True
    rows = []
    print(f"stim {stim.__version__}: surface_code:rotated_memory_z d={DISTANCE} rounds={ROUNDS}, "
          f"{SHOTS} shots, seed {SEED}")
    for p, exp in zip(UNIFORM_P, EXPECTED_UNIFORM_FRACTION):
        frac, ndet = uniform_fraction(p)
        r = round(frac, 4)
        match = abs(r - exp) <= a.tol + 1e-12
        ok &= match
        rows.append(dict(p=p, detection_fraction=frac, rounded=r, expected=exp, detectors=ndet, match=match))
        print(f"  uniform p={p:<7g} detection fraction {frac:.6f} -> {r:.4f}  (system_cost.py {exp:.4f})"
              f"  {'OK' if match else 'MISMATCH'}")

    # the constants in system/system_cost.py, if it is importable, must equal the ones checked here
    try:
        sys.path.insert(0, str(ROOT / "system"))
        import system_cost
        same = (list(system_cost.UNIFORM_P) == UNIFORM_P
                and list(system_cost.UNIFORM_FRACTION) == EXPECTED_UNIFORM_FRACTION)
        print(f"system/system_cost.py constants {'identical' if same else 'DIFFER'} to the ones checked")
        ok &= same
    except Exception as e:  # system_cost needs numpy and its sibling modules; not fatal here
        print(f"(system/system_cost.py not imported: {type(e).__name__}: {e})")

    timing = []
    tpath = pathlib.Path(a.timing)
    if tpath.exists():
        print(f"d=9 timing runs ({tpath}): detection fraction = events_per_round * rounds / detectors")
        fr = EXPECTED_UNIFORM_FRACTION                  # the rounded constants system_cost.py interpolates on
        for line in open(tpath):
            r = json.loads(line)
            frac = r["events_per_round"] * r["rounds"] / r["detectors"]
            p_eq = math.exp(interp_log(frac, fr, [math.log(p) for p in UNIFORM_P]))
            exp = EXPECTED_TIMING_FRACTION.get(r["T"])
            match = exp is None or abs(round(frac, 4) - exp) <= a.tol + 1e-12
            ok &= match
            timing.append(dict(T=r["T"], d=r["d"], detection_fraction=frac, expected=exp,
                               p_equiv_uniform=p_eq, match=match))
            print(f"  T={r['T']:5}: {frac:.4f} (2026-10-26 run: {exp})  equivalent uniform p {p_eq:.4f}"
                  f"  {'OK' if match else 'MISMATCH'}")
    else:
        print(f"(timing file {tpath} not found; skipped the d=9 timing part)")

    out = dict(producer="system/derive_uniform_fraction.py", stim_version=stim.__version__,
               circuit=f"surface_code:rotated_memory_z d={DISTANCE} rounds={ROUNDS}", shots=SHOTS, seed=SEED,
               noise="after_clifford_depolarization = before_round_data_depolarization = "
                     "before_measure_flip_probability = after_reset_flip_probability = p",
               uniform=rows, timing_d9=timing, all_match=bool(ok),
               provenance="inline run, transcript 94019386-7e9a-4302-8276-39a280e43d5e.jsonl line 1039, "
                          "2026-10-26T11:57Z, stim 1.16.0")
    if not a.no_write:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    print("ALL MATCH" if ok else "MISMATCH against the hard-coded constants")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
