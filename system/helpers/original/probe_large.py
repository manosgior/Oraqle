import sys,time; sys.path.insert(0,"/work/lattice-sim/sim"); sys.path.insert(0,"/work/sims")
import circuit_4
circuit_4.circuit.assign_decoder_latency = lambda self: setattr(self, "decoder_latency", 0)
import numpy as np, stim, pymatching
from circuit_4 import circuit
for d in (25,33):
    t0=time.time()
    s=circuit(distance=d,num_patches_x=20,num_patches_y=20,spacing=1,disable_noise=False,fixed_t1=284.95,fixed_t2=322.68,fixed_cnot_latency=70,fixed_measure_latency=600,fixed_cnot_noise=1.191e-3,fixed_measure_noise=0.0506,idle_multiplier=3,rounds_per_op=d+1,basis="Z",ls_basis="X",merge=True,sync=None).from_string("qreg q[2];")
    t1=time.time(); c=s.ckt if isinstance(s.ckt,stim.Circuit) else stim.Circuit(str(s.ckt))
    m=pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True)); t2=time.time()
    det,obs=c.compile_detector_sampler(seed=1).sample(200,separate_observables=True,bit_packed=True)
    t3=time.time(); m.decode_batch(det,bit_packed_shots=True); t4=time.time()
    print(d,f"build {t1-t0:.0f}s dem {t2-t1:.0f}s decode {(t4-t3)/200*1e3:.1f} ms/shot dets {c.num_detectors}",flush=True)
