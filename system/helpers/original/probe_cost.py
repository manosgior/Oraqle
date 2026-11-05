import sys,time; sys.path.insert(0,"/work/lattice-sim/sim")
import numpy as np, stim, pymatching
from circuit_4 import circuit
for d in (9,13,17):
    t0=time.time()
    s=circuit(distance=d,num_patches_x=20,num_patches_y=20,spacing=1,disable_noise=False,
        fixed_t1=284.95,fixed_t2=322.68,fixed_cnot_latency=70,fixed_measure_latency=600,
        fixed_cnot_noise=1.191e-3,fixed_measure_noise=0.05,idle_multiplier=3,rounds_per_op=d+1,
        basis="Z",ls_basis="X",merge=True,sync=None).from_string("qreg q[2];")
    t1=time.time(); c=stim.Circuit(str(s.ckt)) if not isinstance(s.ckt,stim.Circuit) else s.ckt
    dem=c.detector_error_model(decompose_errors=True); m=pymatching.Matching.from_detector_error_model(dem)
    t2=time.time(); smp=c.compile_detector_sampler(seed=1)
    det,obs=smp.sample(20000,separate_observables=True,bit_packed=True)
    t3=time.time(); pred=m.decode_batch(det,bit_packed_shots=True,bit_packed_predictions=True); t4=time.time()
    print(f"d={d}: build {t1-t0:.1f}s dem {t2-t1:.1f}s sample {(t3-t2)/2e4*1e6:.1f}us/shot decode {(t4-t3)/2e4*1e6:.1f}us/shot dets={c.num_detectors} obs={c.num_observables} qubits={c.num_qubits} time={s.time}ns fails={int(np.any(pred!=obs,axis=1).sum())}",flush=True)
