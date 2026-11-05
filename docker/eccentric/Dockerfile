# [Oraqle artifact] ECCentric's own Dockerfile (TUM-DSE/eccentric @da7164c), pinned:
#   - base image pinned to the digest of the 2026-06-09 build on graham (docker_build.log)
#   - pip installs requirements.oraqle.lock (the exact package set of that build) instead of the
#     partly unpinned requirements.txt (numpy, tqec)
# Build context = an ECCentric checkout made by qec/eccentric/fetch_eccentric.sh, which copies this file to
# Dockerfile.oraqle and the lock to requirements.oraqle.lock:
#   docker build -f work/eccentric/Dockerfile.oraqle -t eccentric work/eccentric
# Run recipes: docker/eccentric/README.md.
#
# --- original header ---
# Reproducible environment for running ECCentric QEC benchmarks
# (evaluate_mcm_latency.py, benchmark_mcm.py, main.py, ...).
#
# Build (on the experiment server):
#   docker build -t eccentric .
#
# Run an experiment, persisting results to the host:
#   docker run --rm \
#       -v "$PWD/experiment_results:/qec/experiment_results" \
#       eccentric python evaluate_mcm_latency.py
#
# Interactive shell:
#   docker run --rm -it -v "$PWD/experiment_results:/qec/experiment_results" eccentric
#
# Notes:
# - Python 3.11 (matches the project's venv).
# - qiskit_qec is the local submodule under external/qiskit_qec; its C extension
#   is compiled in-place (the code imports it via external/qiskit_qec/src).
# - evaluate_mcm_latency.py uses the FakeIBMHeron backend, so no IBM token is
#   needed. For experiments hitting real IBM backends, pass a token, e.g.:
#       docker run --rm -e QISKIT_IBM_TOKEN=... eccentric python main.py

FROM python:3.11-bookworm@sha256:121d86b6d08752968a7dddbc708849e5f3a839bbff47f32212b46d2a1d842bab

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System build dependencies: CMake + a C/C++ toolchain for the qiskit_qec
# pybind11 extension, plus BLAS/LAPACK/GMP/FFI headers for compiled QEC wheels
# (ldpc, bposd, stim, pymatching, scipy, ...).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git pkg-config \
        libffi-dev libgmp-dev libblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /qec

# Install Python dependencies first so this expensive layer is cached across
# code changes. qiskit_qec itself is NOT here -- it is the local submodule,
# built below from source.
COPY requirements.oraqle.lock ./
RUN python -m pip install pip==26.1.2 \
    && python -m pip install -r requirements.oraqle.lock

# Copy the project (see .dockerignore for exclusions).
COPY . .

# Build the qiskit_qec C extension in-place. The project adds
# external/qiskit_qec/src to sys.path and imports the compiled module from
# there (see the sys.path.append calls in main.py / the benchmark scripts).
RUN cd external/qiskit_qec && python setup.py build_ext --inplace

# Sanity: the heavy imports must succeed at build time, not first run.
RUN python -c "import sys, os; sys.path.append('external/qiskit_qec/src'); \
    import stim, pymatching, qiskit, tqec; from codes import get_code; \
    from main import single_cnot_n_rounds; print('eccentric image OK')"

CMD ["bash"]
