"""s3lit: physics prior for |2> in the Lienhard et al. 5-qubit dataset (INFERRED, not measured).

Inputs are Lienhard et al., PR Applied 17, 014024 (arXiv 2102.12481v2), Tab. III (f01, alpha, T1) and
Tab. V (P(1|empty) after 1 us MF). Assumes a Boltzmann population at one effective temperature per qubit
(P(1|empty) also contains prep/readout error, so T_eff and everything below are UPPER bounds),
|2>->|1> decay at 2/T1, and detailed balance for the 1->2 upward rate. No readout-induced (MIST) or
pi-pulse leakage is included.
"""
import math

h, k = 6.62607015e-34, 1.380649e-23
QUBITS = [  # paper name / code name, f01 (GHz), alpha (MHz), P(1|empty), T1 (us)
    ("Q1/q0", 5.092, -212, 0.005, 40.8),
    ("Q2/q1", 4.404, -216, 0.003, 6.4),
    ("Q3/q2", 5.000, -204, 0.006, 21.4),
    ("Q4/q3", 4.309, -214, 0.009, 11.8),
    ("Q5/q4", 5.165, -200, 0.003, 23.4),
]
for name, f01, a, p1, t1 in QUBITS:
    f01 *= 1e9
    f12 = f01 + a * 1e6
    T = h * f01 / (k * math.log((1 - p1) / p1))
    r12 = math.exp(-h * f12 / (k * T))
    p2 = p1 * r12                              # thermal |2> in |0>-prepared shots
    up = (2 / (t1 * 1e-6)) * r12 * 1e-6        # P(1->2) within 1 us for |1>-prepared shots
    dec = 1 - math.exp(-2.0 / t1)              # P(|2> decays within 1 us)
    print(f"{name}: T_eff={T*1e3:.1f} mK  P2_thermal={p2:.1e}  P(1->2 in 1us)={up:.1e}  P(2 decays in 1us)={dec:.2f}")
