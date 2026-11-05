"""Workload models for the time-to-solution study (part B).

Taken from CXL-Quantum/plots/workloads.py and extended with two larger instances. Every workload is
reduced to three numbers: logical qubits N_L, feed-forward steps on the critical path (each a logical
operation of d+1 rounds followed by the wait for its decoded outcome) and Clifford-only cycles on the
critical path (d+1 rounds, no wait). Magic-state factories and routing space are not counted in N_L.

  Grover-n     Toffoli-ladder oracle and diffusion, 4(n-2) sequential Toffolis and ~5 Clifford-only
               cycles per iteration, floor(pi/4 sqrt(2^n)) iterations; N_L = 2n - 2
  Adder-n      Cuccaro ripple-carry adder: 2n sequential Toffolis; N_L = 2n + 2
  Ising-N      1D transverse-field Ising chain, first-order Trotter, r steps of three rotation layers;
               each rotation is ~3 log2(1/eps) sequential T gates (Ross-Selinger); N_L = N
  QFT-n        approximate QFT, 2n - 3 layers of two sequential synthesized rotations, n Hadamard
               cycles; N_L = n
"""
import math

EPS = 1e-6                                     # precision per synthesized rotation
T_PER_ROT = math.ceil(3 * math.log2(1 / EPS))  # 60 sequential T gates per rotation


def grover(n):
    it = math.floor(math.pi / 4 * math.sqrt(2 ** n))
    return dict(lq=2 * n - 2, steps=it * 4 * (n - 2), cliff=it * 5)


def adder(n):
    return dict(lq=2 * n + 2, steps=2 * n, cliff=0)


def ising(n, r=100):
    return dict(lq=n, steps=r * 3 * T_PER_ROT, cliff=0)


def qft(n):
    return dict(lq=n, steps=(2 * n - 3) * 2 * T_PER_ROT, cliff=n)


WORKLOADS = {"Adder-14": adder(14), "QFT-30": qft(30), "Grover-16": grover(16), "Ising-30": ising(30),
             "Adder-1024": adder(1024), "Ising-100": ising(100, r=1000)}
