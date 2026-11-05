"""Workload models for the evaluation's program-cost panels (fig:eval-cost b, c).

Every workload is reduced to three numbers: logical qubits (N_L), feed-forward steps on the critical
path (D; each a logical cycle of d rounds plus the wait for its decision) and Clifford-only cycles on the
critical path (C; d rounds, no wait). Magic-state factories are not counted in N_L.

  Grover-n     search over n qubits with a Toffoli ladder for each (n-1)-controlled Z (n-2 clean
               ancillas): 4(n-2) sequential Toffolis and ~5 Clifford-only cycles per iteration,
               floor(pi/4 sqrt(2^n)) iterations; N_L = 2n - 2 (feedforward_density.py).
  Adder-n      Cuccaro ripple-carry adder: 2n sequential Toffolis; N_L = 2n + 2.
  Ising-N      1D transverse-field Ising chain, first-order Trotter, r steps; each step is three layers of
               rotations (even ZZ bonds, odd ZZ bonds, transverse X), parallel within a layer; each
               rotation is synthesized to precision eps with ~3 log2(1/eps) sequential T gates
               (Ross-Selinger), and every T is a feed-forward point; N_L = N.
  QFT-n        approximate QFT (Coppersmith): controlled rotations below pi / 2^b dropped,
               b = ceil(log2 n) + 3; the rest scheduled in 2n - 3 layers, each controlled rotation two
               sequential synthesized rotations on its target; n Hadamard cycles; N_L = n. QPE ends in
               this inverse QFT after the problem-specific controlled powers of U.
"""
import math

EPS = 1e-6                                   # precision per synthesized rotation
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


WORKLOADS = {"Grover-16": grover(16), "Adder-14": adder(14), "Ising-30": ising(30), "QFT-30": qft(30)}

if __name__ == "__main__":
    for name, w in WORKLOADS.items():
        f = w["steps"] / (w["steps"] + w["cliff"])
        print(f"{name:10s} N_L {w['lq']:3d}  feed-forward steps {w['steps']:7d}  Clifford cycles {w['cliff']:5d}  waiting share {f:.3f}")
