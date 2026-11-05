# ECCentric image (`eccentric`)

Runs every ECCentric simulation of the paper: §6 (`qec_impact.tex`) and the appendix six-code / simulator-comparison
sweeps. Not needed to regenerate the figures or numbers from `results/qec` (that needs only numpy + matplotlib).

## Build

```bash
qec/eccentric/fetch_eccentric.sh            # -> work/eccentric (eccentric @da7164c + submodules, overlay, these pins)
docker build -f work/eccentric/Dockerfile.oraqle -t eccentric work/eccentric      # ~3 min, 2.9 GB
```
`Dockerfile` here is eccentric's own Dockerfile with two pins: the base image digest of the original build
(`python:3.11-bookworm@sha256:121d86b6…`, Python 3.11.15) and `requirements.lock`, the exact package set of that
build (235 packages; numpy 1.26.4, tqec 0.2.0, stim 1.14.0, PyMatching 2.3.1, ldpc 0.1.60, bposd 1.6, qiskit 1.4.2,
sinter 1.14.0, matplotlib 3.9.3). The qiskit_qec C extension is compiled in place, as upstream does. A rebuild on
graham on 2026-11-05 reproduced the lock exactly (`pip freeze` of the new image = the lock).

## Run recipes

The image's working directory is `/qec` (the checkout). Drivers write `experiment_results/<group>/*.csv` relative to
it, so mount an empty host directory there; mount the repository's `results/` read-only for `master_fidelity.csv`.
No dataset is needed (the readout enters only through the per-qubit fidelity tables).

```bash
mkdir -p work/ecc_out
# §6.2/6.3 six-code sweeps (LER_MODE = current | boston_lowt_backlog | highfid | futuristic | futuristic_fixedreadout)
docker run --rm -v "$PWD/work/ecc_out:/qec/experiment_results" -e LER_MODE=current \
  eccentric python -u run_ler_noise_models.py
# §6.1 discriminator sweep (d = 5; for d = 12: -e LER_DIST=12 -e LER_CODES=surface,bacon,hh -e LER_INCLUDE_CNN=1)
docker run --rm -v "$PWD/work/ecc_out:/qec/experiment_results" -v "$PWD/results:/results:ro" \
  -e ORAQLE_RESULTS=/results -e LER_DIST=5 eccentric python -u run_ler_discriminators.py
# one-point smoke test (seconds), compared with the paper CSV
docker run --rm -v "$PWD/results:/results:ro" -e ORAQLE_RESULTS=/results -e LER_DIST=5 \
  eccentric python smoke_point.py discriminators surface MCMit-T 600
```
Add `--user $(id -u):$(id -g) -e HOME=/tmp` to keep host file ownership. The full list of invocations per paper CSV
is in `qec/eccentric/README.md`. The appendix comparison driver at 7858c2c (`fetch_eccentric.sh --with-7858c2c`)
runs in the same image (tested on graham, d=5/600 ns/current: LER 0.3631 vs 0.36084 in the paper's `_nmr1` CSV):
```bash
docker run --rm -v "$PWD/work/eccentric-7858c2c/compare_readout_eccentric.py:/qec/compare_readout_eccentric_7858c2c.py:ro" \
  -v "$PWD/work/ecc_out:/qec/experiment_results" eccentric python -u compare_readout_eccentric_7858c2c.py
```
