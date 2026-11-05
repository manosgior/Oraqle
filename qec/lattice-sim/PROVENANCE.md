# qec/lattice-sim — provenance

lattice-sim (Satvik Maurya, MIT License, `LICENSE` kept) is the stabilizer-circuit simulator of §7
(system_bottleneck.tex): `system/sweep_ler.py` and every script built on it import `sim/circuit_4.py` from here
(`LATSIM` environment variable, default `<repo>/qec/lattice-sim/sim`).

| | |
|---|---|
| Vendored from | `MCMit/lattice-sim` (github.com/manosgior/MCMit), commit **16158ed** (2026-07-12, "Vendor lattice-sim's plotting scripts and results, drop eccentric submodule"); the simulator `sim/` is unchanged since **32f98a3** (the commit that vendored lattice-sim with history from TUM-DSE/eccentric `mcm-latency-eccentric:synchronization-artifact/`); `git diff 32f98a3 16158ed -- lattice-sim/sim` is empty. Exported with `git archive 16158ed lattice-sim`. |
| Copy actually run | steve/graham (shared NFS) `~/oraqle-ics-sims/lattice-sim` (no git, mounted at `/work/lattice-sim` in the 2026-10-26..29 runs). |
| Identity check (2026-11-05) | md5 of every file of this directory == the cluster copy (80 files, 0 differences, `__pycache__` ignored). |
| Changes | none to any file. Three notebooks of MCMit 16158ed that the cluster copy does not have were left out: `plots.ipynb`, `plotting/mcm_plot.ipynb`, `synchronization.ipynb` (4.8 MB, not used by Oraqle). |
| Not used by the paper | lattice-sim's own `Dockerfile` / `requirements.txt` (Python 3.9, stim 1.14, PyMatching 2.2.1) and the top-level MCMit scripts and CSVs; the paper used the `oraqle-latsim` image (`docker/latsim`, stim 1.16.0, PyMatching 2.4.0). |
| Reproduction check | `system/replay_points.py` re-runs published sweep jobs with this copy in oraqle-latsim(:dec) and gets bit-identical shots/failures (steve, 2026-11-05; docs/reproduce/qec_system.md). |
