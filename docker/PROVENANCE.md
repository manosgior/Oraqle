# docker/ — provenance

None of these Dockerfiles was in a git repository before. Pins come from the images that ran the paper, inspected on
2026-11-05 with `docker run --rm --name unify-qec-freeze-* <image> sh -c "python --version; pip freeze"` (outputs in
`freeze/`). The base-image digests of the Oct 2026 images were not kept (the `python:3.11-slim` base is no longer on
the hosts), so the base is pinned to the Python patch release the images report.

| Dockerfile | Original (host:path, date) | Original image (id) | Changes |
|---|---|---|---|
| `latsim/Dockerfile` | steve/graham `~/oraqle-ics-sims/docker/Dockerfile` (2026-10-26) | oraqle-latsim, steve cc757391b5b1 (Python 3.11.17) and graham 0456515d488a (3.11.11); identical `pip freeze` | `FROM python:3.11.17-slim`; installs `requirements.lock` (= steve freeze) instead of the partly unpinned list |
| `latsim/Dockerfile.dec` | steve `~/oraqle-ics-sims/docker/Dockerfile.dec` (2026-10-29) | oraqle-latsim:dec, steve ff7469718be1 | installs `requirements-dec.lock` (= its freeze) instead of `ldpc==2.4.1 scipy` |
| `torch/Dockerfile` | graham `~/oraqle-ics-sims/docker-torch/Dockerfile` (2026-10-26) | oraqle-torch, graham f40dbb670f94 (3.11.11) | `FROM python:3.11.11-slim`; torch 2.5.1 from the CPU index, rest from `requirements.lock` |
| `torch-gpu/Dockerfile` | steve `~/oraqle-ics-sims/docker-torch-gpu/Dockerfile` (2026-10-26) | oraqle-torch-gpu, steve 8e0bef81a5fc (3.11.17) | `FROM python:3.11.17-slim`; torch 2.5.1 from the cu124 index, rest (incl. nvidia-* runtime wheels) from `requirements.lock` |
| `oraqle-cuda/Dockerfile` | repository root `Oraqle/Dockerfile` @87d3162 (== graham `~/Oraqle/Dockerfile`) | `oraqle` — **gone** from graham and steve | moved here; `COPY Discriminators/requirements.txt` -> `discriminators/requirements.txt`; header comment. Not pinnable (image lost, requirements unpinned) |
| `eccentric/` | TUM-DSE/eccentric @da7164c `Dockerfile` | `eccentric` (2026-06-09 build on graham; gone) | see `eccentric/PROVENANCE.md` (digest-pinned base, lock rebuilt from the build log) |

Rebuild check (steve, 2026-11-05): `docker build -t unify-qec-latsim docker/latsim` and
`docker build -f docker/latsim/Dockerfile.dec -t unify-qec-latsim-dec docker/latsim` (on top of the original
oraqle-latsim tag) -> `pip freeze` equal to the locks; `system/replay_points.py` in the rebuilt images reproduced the published sweep jobs bit for bit; the two test
images were removed afterwards. The torch images were not rebuilt (their locks are the freezes of the originals).
