"""Sensitivity check (NOT used by the paper): how much does the uniform-noise mapping of system_cost.py move
Helios's latency, given that qShell measured Helios under SI1000 noise?

system_cost.helios_latency() maps the detection fraction of our d=9 timing runs to an "equivalent p" of UNIFORM
circuit noise (UNIFORM_FRACTION, system/derive_uniform_fraction.py) and then reads qShell's p99 at that p, which
qShell measured under SI1000(p) (qLDPC SI1000NoiseModel). This script builds an approximate SI1000 version of the
same Stim circuit (CX -> DEPOLARIZE2(p); 1q gates -> DEPOLARIZE1(p/10); reset -> flip 2p; measurement -> flip 5p;
idle per TICK -> DEPOLARIZE1(p/10), 2p during measure/reset layers), maps the same detection fractions through it
and recomputes Helios's p99 and the reaction wait. Approximation only (Stim's circuit uses CX, not CZ + H).

Usage (stim, e.g. oraqle-latsim image): python results/external/si1000_sensitivity.py
"""
import math
import stim

UNIFORM_P = [1e-3, 3e-3, 5e-3, 1e-2, 2e-2]
UNIFORM_FRACTION = [0.0166, 0.0481, 0.0774, 0.1428, 0.2448]          # system_cost.py
QSHELL_P = [1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2]
QSHELL_P99_NS = [1680, 1720, 1804, 1980, 2132, 2360, 2592]
TIMING_FRACTION = {200: 0.3298, 400: 0.1590, 600: 0.1126, 800: 0.1044, 1000: 0.1019}   # decode_timing_d9.jsonl
CNN_NS, FRAME_NS, GATES_NS, HELIOS_P99_NS, WEIGHTED = 74, 100, 340, 308, 38 / 15


def interp_log(x, xs, ys):
    lx, lxs = math.log(x), [math.log(v) for v in xs]
    i = max(1, min(len(xs) - 1, next((j for j, v in enumerate(lxs) if v >= lx), len(xs) - 1)))
    t = (lx - lxs[i - 1]) / (lxs[i] - lxs[i - 1])
    return ys[i - 1] + t * (ys[i] - ys[i - 1])


def si1000(c0, p):
    allq = set(range(c0.num_qubits))
    out = stim.Circuit()
    for layer in str(c0.flattened()).split("TICK"):
        lc = stim.Circuit(layer)
        touched, meas_reset = set(), False
        for inst in lc:
            if inst.name in ("QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"):
                out.append(inst)
                continue
            t = [x.value for x in inst.targets_copy()]
            touched |= set(t)
            if inst.name in ("M", "MR", "MX"):
                meas_reset = True
                out.append("X_ERROR", t, 5 * p)
                out.append(inst)
                if inst.name == "MR":
                    out.append("X_ERROR", t, 2 * p)
            elif inst.name in ("R", "RX"):
                meas_reset = True
                out.append(inst)
                out.append("X_ERROR" if inst.name == "R" else "Z_ERROR", t, 2 * p)
            elif inst.name in ("CX", "CZ"):
                out.append(inst)
                out.append("DEPOLARIZE2", t, p)
            elif inst.name in ("H", "S", "SQRT_X", "X", "Y", "Z"):
                out.append(inst)
                out.append("DEPOLARIZE1", t, p / 10)
            else:
                out.append(inst)
        idle = sorted(allq - touched)
        if idle and len(lc):
            out.append("DEPOLARIZE1", idle, 2 * p if meas_reset else p / 10)
        out.append("TICK")
    return out


def main():
    c0 = stim.Circuit.generated("surface_code:rotated_memory_z", distance=9, rounds=9)
    si = []
    for p in UNIFORM_P:
        si.append(float(si1000(c0, p).compile_detector_sampler(seed=1).sample(20000).mean()))
        print(f"p={p:<6g} detection fraction uniform {UNIFORM_FRACTION[UNIFORM_P.index(p)]:.4f}  SI1000~ {si[-1]:.4f}")
    print(f"{'T':>5} {'map':>8} {'p_equiv':>8} {'p99 ns':>7} {'R unw':>7} {'w':>2} {'R w16':>7} {'w':>2}")
    for T, f in TIMING_FRACTION.items():
        for name, tab in (("uniform", UNIFORM_FRACTION), ("SI1000~", si)):
            p = math.exp(interp_log(f, tab, [math.log(x) for x in UNIFORM_P]))
            p99 = HELIOS_P99_NS + interp_log(p, QSHELL_P, QSHELL_P99_NS) - QSHELL_P99_NS[2]
            r_u, r_w = CNN_NS + p99 + FRAME_NS, CNN_NS + p99 * WEIGHTED + FRAME_NS
            t_r = T + GATES_NS
            print(f"{T:>5} {name:>8} {p:>8.4f} {p99:>7.0f} {r_u / 1e3:>6.2f}u {max(1, math.ceil(r_u / t_r)):>2} "
                  f"{r_w / 1e3:>6.2f}u {max(1, math.ceil(r_w / t_r)):>2}")


if __name__ == "__main__":
    main()
