# test_uf.py: cluster-only helper from steve/graham ~/oraqle-ics-sims/test_uf.py (no git); path-plumbed copy.
# Original (verbatim) in system/helpers/original/test_uf.py. Run from the repo root in oraqle-latsim:dec:
#   docker run --rm -v $PWD:/repo -w /repo oraqle-latsim:dec python system/helpers/test_uf.py
import os, sys; _R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_R, "system"))  # was /work/sims
import numpy as np, stim, pymatching
from sweep_decoders import uf_setup
for p in (1e-3, 3e-3):
  for d in (3,5,7):
    c=stim.Circuit.generated("surface_code:rotated_memory_z",distance=d,rounds=d,after_clifford_depolarization=p,before_round_data_depolarization=p,before_measure_flip_probability=p,after_reset_flip_probability=p)
    dem=c.detector_error_model(decompose_errors=True); m=pymatching.Matching.from_detector_error_model(dem)
    uf,L,llr=uf_setup(m,c.num_detectors,c.num_observables)
    det,obs=c.compile_detector_sampler(seed=1).sample(4000,separate_observables=True)
    fw=int(np.any(m.decode_batch(det)!=obs,axis=1).sum())
    H=None
    fu=fl=bad=0
    for i in range(len(det)):
        e=uf.decode(det[i].astype(np.uint8)); fu+=int(np.any((L@e)%2!=obs[i]))
        e2=uf.decode(det[i].astype(np.uint8),llrs=llr); fl+=int(np.any((L@e2)%2!=obs[i]))
    print(f"p={p} d={d}: mwpm {fw}  uf_u {fu}  uf_w {fl}  /4000",flush=True)
