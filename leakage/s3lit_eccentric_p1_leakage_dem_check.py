# Does a probability-1 Pauli (as ECCentric inserts for leakage) survive into the DEM, and how does pymatching treat it?
import stim, pymatching, numpy as np
base = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=5, before_round_data_depolarization=0.01)
flat = base.flattened()
data = sorted({t.value for inst in flat if inst.name == "DEPOLARIZE1" for t in inst.targets_copy()})
c = stim.Circuit(); n_dep = 0
for inst in flat:
    c.append(inst)
    if inst.name == "DEPOLARIZE1":
        n_dep += 1
        if n_dep == 2:
            c.append("X_ERROR", [data[2]], 1.0)
for name, circ in (("baseline", base), ("with X_ERROR(1.0) on data", c)):
    dem = circ.detector_error_model(decompose_errors=True)
    big = [str(i) for i in dem.flattened() if i.type == "error" and i.args_copy()[0] > 0.5]
    m = pymatching.Matching.from_detector_error_model(dem)
    det, obs = circ.compile_detector_sampler(seed=1).sample(200000, separate_observables=True)
    pred = m.decode_batch(det)
    print(name, "| DEM terms p>0.5:", big[:3], "| max detector rate:", det.mean(0).max().round(3), "| LER:", np.mean(np.any(pred != obs, axis=1)))
