"""Chemistry and materials instances from published resource estimates, in the workload form of system/workloads.py.

A workload is three numbers: logical qubits N_L, feed-forward steps on the critical path and Clifford-only
cycles. Published estimates give gate counts, not a critical path, so every non-Clifford gate is taken as one
sequential feed-forward step (a Toffoli is one step, as in system/workloads.py's adders; a rotation is
T_PER_ROT sequential T gates, as in its Ising model). That is an upper bound on the runtime: gates that
could run in parallel are serialized. N_L is the algorithm's logical qubits; routing space and magic-state
factories are not counted, as in system/workloads.py.

Each entry keeps its source and how the numbers were checked (2026-11-12).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "system"))
from workloads import T_PER_ROT  # noqa: E402


def from_counts(lq, toffoli=0, t=0, rotations=0, **meta):
    return dict(lq=lq, steps=int(toffoli + t + rotations * T_PER_ROT), cliff=0,
                counts=dict(toffoli=toffoli, t=t, rotations=rotations), **meta)


APPS = {
    # Materials: 2D transverse-field Ising dynamics on a 10 x 10 lattice.
    "Dynamics-10x10": from_counts(
        lq=100, rotations=30_100,
        field="materials (quantum dynamics)",
        source="Beverland et al., arXiv:2211.07629, Appendix F",
        checked="via the Qualtran reproduction of Appendix F (qualtran.readthedocs.io, beverland_et_al_model): "
                "100 algorithm qubits, 30,100 rotations, no T or Toffoli gates; 230 logical qubits with layout"),
    # Chemistry: the quantum-chemistry application of the same paper.
    "Chemistry-Beverland": from_counts(
        lq=1318, toffoli=1.35e11, t=5.53e7, rotations=2.06e8,
        field="chemistry",
        source="Beverland et al., arXiv:2211.07629, Appendix F",
        checked="via the Qualtran reproduction: 1,318 algorithm qubits, 1.35e11 Toffoli, 5.53e7 T, 2.06e8 "
                "rotations; 2,740 logical qubits with layout. Not read from the paper's own table."),
    # Chemistry: FeMoco (nitrogenase cofactor), tensor hypercontraction.
    "FeMoco-Reiher": from_counts(
        lq=2142, toffoli=5.3e9,
        field="chemistry",
        source="Lee et al., PRX Quantum 2, 030305 (2021), Table 3, THC row",
        checked="read from the arXiv HTML of 2011.03494: 2,142 logical qubits, 5.3e9 Toffoli"),
    "FeMoco-Li": from_counts(
        lq=2196, toffoli=3.2e10,
        field="chemistry",
        source="Lee et al., PRX Quantum 2, 030305 (2021), Table 3, THC row",
        checked="read from the arXiv HTML of 2011.03494: 2,196 logical qubits, 3.2e10 Toffoli"),
}
