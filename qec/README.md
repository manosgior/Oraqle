# qec/ — QEC simulators used by the paper

| Directory | What | Paper part |
|---|---|---|
| `lattice-sim/` | vendored lattice-sim (MCMit @16158ed, identical to the copy run on the cluster) | §7 (system_bottleneck) via `system/`; App. "ECCentric vs. lattice-sim" (lattice-sim half, through ECCentric's own copy @7858c2c) |
| `eccentric/` | `fetch_eccentric.sh` (TUM-DSE/eccentric @da7164c + submodules, pinned, public), the path-plumbed Oraqle driver overlay, the plotting code of the §6/appendix figures (incl. the recovered `plot_paper_pairs.py`), `paper_numbers.py` | §6 (qec_impact) and appendix A.13-A.17 |

How to reproduce and verify: `docs/reproduce/qec_system.md`. Images: `docker/README.md`.
