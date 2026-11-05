# probe_time.py: cluster-only helper from steve/graham ~/oraqle-ics-sims/probe_time.py (no git); path-plumbed copy.
# Original (verbatim) in system/helpers/original/probe_time.py. Run from the repo root in oraqle-latsim:
#   docker run --rm -v $PWD:/repo -w /repo oraqle-latsim python system/helpers/probe_time.py
import os, sys; _R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LATSIM = os.environ.get("LATSIM", os.path.join(_R, "qec", "lattice-sim", "sim"))  # was /work/lattice-sim/sim
import sys; sys.path.insert(0,LATSIM)
from circuit_4 import circuit
def mk(T,cx,rounds,merge,d=5):
    return circuit(distance=d,num_patches_x=20,num_patches_y=20,spacing=1,disable_noise=False,
        fixed_t1=284.95,fixed_t2=322.68,fixed_cnot_latency=cx,fixed_measure_latency=T,
        fixed_cnot_noise=1.191e-3,fixed_measure_noise=0.04,idle_multiplier=3,rounds_per_op=rounds,
        basis="Z",ls_basis="X",merge=merge,sync=None).from_string("qreg q[2];")
for merge in (False,True):
  for cx in (70,140):
    for T in (400,1000):
      a=mk(T,cx,5,merge).time; b=mk(T,cx,10,merge).time
      print(f"merge={merge} cx={cx} T={T}: per-round-pair {(b-a)/5:.0f} ns, total(5)={a}")
s=mk(1000,70,6,True)
c=str(s.ckt); import collections
print(collections.Counter(l.split()[0] for l in c.splitlines() if l.strip() and not l.startswith(("DETECTOR","OBSERVABLE","QUBIT","SHIFT"))).most_common(15))
ler,_=s.get_error_rate(ckt=s.ckt,num_shots=200000); print("LER d5 T1000 (rounds 6, meas .0428):",ler)
