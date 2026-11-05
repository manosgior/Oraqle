# oraqle-torch: CPU PyTorch image (leakage / error-mode analyses; also runs system/soft_readout_cnn.py on CPU, slowly).
# Origin: graham ~/oraqle-ics-sims/docker-torch/Dockerfile (2026-10-26, no git):
#   FROM python:3.11-slim
#   RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir numpy==2.1.3 h5py scipy scikit-learn
#   WORKDIR /work
# Pinned: requirements.lock = `pip freeze` of oraqle-torch on graham (f40dbb670f94) minus torch (installed from the
# CPU wheel index first); Python 3.11.11 as in that image.
# Build: docker build -t oraqle-torch docker/torch
FROM python:3.11.11-slim
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /tmp/requirements.lock
WORKDIR /work
