"""Retrain HERQULES at one readout length in BOTH modes (fake/scrambled and
realistic/per-trace) and run the matching scrambled error-mode analysis for
each, all in one process so the 24 GB demodulation is paid once.

For each mode we:
  1. train(scramble=mode) -> checkpoint (HERQULES_len{L}.pth) + rmf.pkl (cwd),
     consistent with each other,
  2. immediately run analyze in the matching mode (scrambled=mode), which loads
     that just-saved checkpoint+rmf, BEFORE the next train overwrites them.

Outputs (to /app/optimization_reports, mounted -> oraqle_reports):
  - herqules_error_modes_len{L}_scrambled_*.csv   (fake)
  - herqules_error_modes_len{L}_pertrace_*.csv    (realistic)
Length via HERQULES_TRACE_LENGTH (read at import by HERQULES_original).
"""
import os as _os, sys as _sys  # [unify] discriminators/ on sys.path (repo layout)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from disc_paths import REPORTS_OUT, ensure_dir  # [unify] replaces the old docker mount paths
import shutil
import trainers.HERQULES_original as H
import runners.analyze_herqules_errors as A

L = H.TRACE_LENGTH
A.TRACE_LENGTH = L

for scramble in (True, False):
    tag = "fake/scrambled" if scramble else "realistic/per-trace"
    print(f"[taa] ===== length {L} | mode = {tag} =====", flush=True)
    print(f"[taa] training (scramble={scramble}) ...", flush=True)
    H.train(run_semi_sup=False, run_pre_filter=False, run_rmf=True,
            scramble=scramble)
    # Persist the consistent rmf (same for both modes; kept for reproducibility).
    shutil.copy("rmf.pkl", f"{ensure_dir(REPORTS_OUT)}/rmf_len{L}.pkl")  # [unify]
    print(f"[taa] analysing (scrambled={scramble}) ...", flush=True)
    A.run(scrambled=scramble)

print("[taa] done", flush=True)
