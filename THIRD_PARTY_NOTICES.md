# Third-party notices

Oraqle is released under the MIT License (see `LICENSE`). The following parts were adapted from or
contributed by third parties. They keep their own licenses and attributions. Files adapted during
consolidation are marked with `[unify]` comments, and each folder's `PROVENANCE.md` records its
origin and changes.

| Path | Origin | License |
|---|---|---|
| `qec/lattice-sim/` | lattice-sim by Satvik Maurya, vendored via MCMit (commit 16158ed; `sim/` unchanged since 32f98a3) | MIT, © 2025 Satvik Maurya (`qec/lattice-sim/LICENSE`, also `LICENSES/MIT-lattice-sim.txt`) |
| `discriminators/trainers/HERQULES_original.py`, `discriminators/HERQULES.py`, `discriminators/networks/HERQULES.py`, `discriminators/matched_filter.py`, `discriminators/trainers/train_HERQULES.py` | adapted from HERQULES, https://github.com/satvikmaurya/HERQULES (Maurya et al., ISCA 2023); modified for this study | Apache-2.0 (`LICENSES/Apache-2.0.txt`) |
| `discriminators/networks/KLiNQ_StudentModel.py`, `discriminators/networks/KLiNQ_TeacherModel.py`, `discriminators/networks/Qubic.py`, `discriminators/trainers/KnowledgeDistillationTrainer_KLiNQ.py`, `discriminators/runners/reproduce_klinq_paper.py` | adapted from KLiNQ, https://github.com/xiguo1030/KLiNQ (Guo et al.); modified for this study | Apache-2.0 (`LICENSES/Apache-2.0.txt`) |
| `fpga/hls4ml/` | hls4ml synthesis scripts and Vitis HLS reports by Felix Gust, https://github.com/gustifix/mcmit-model (commit 56cfcc0) | included with the author's permission; license pending the author's confirmation |
| `qec/eccentric/` (driver and plot scripts) | written for this study on top of ECCentric, https://github.com/TUM-DSE/eccentric (branch `mcm-latency-eccentric`); ECCentric itself is fetched at pinned commits by `fetch_eccentric.sh`, not redistributed | upstream: see the ECCentric repository |

Not redistributed (fetched or installed at build time, under their own licenses): ECCentric and its
submodules (qiskit-qec, more-bacon-less-threshold), Stim, PyMatching, ldpc, PyTorch, hls4ml and the
other Python packages listed in `docker/`.

The raw dataset (Lienhard et al., Phys. Rev. Applied 17, 014024 (2022)) is not part of this repository
and is not covered by its license. The checkpoints in `checkpoints/` are models trained on that dataset.
