# oraqle-latsim: lattice-sim + stim + PyMatching, the image of every §7 sweep (sweep_ler, sweep_soft, decode_timing,
# probes) and of verify/qec_system.py --stim.
# Origin: steve/graham ~/oraqle-ics-sims/docker/Dockerfile (2026-10-26, no git), which read
#   FROM python:3.11-slim
#   RUN pip install --no-cache-dir stim==1.16.0 PyMatching==2.4.0 numpy==2.3.1 networkx pandas scipy sinter tqdm qiskit==2.1.0 qiskit-aer==0.17.1
#   WORKDIR /work
# Pinned here to what is installed in the images that ran the paper (requirements.lock = `pip freeze` of
# oraqle-latsim on steve, cc757391b5b1, identical on graham, 0456515d488a) and to the Python patch release of the
# steve image (3.11.17; the graham image that ran decode_timing.py had 3.11.11; the base-image digest was not kept).
# Build: docker build -t oraqle-latsim docker/latsim
FROM python:3.11.17-slim
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock
WORKDIR /work
