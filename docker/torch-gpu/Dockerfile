# oraqle-torch-gpu: CUDA 12.4 PyTorch image of the §7.6 soft-readout CNN retraining (system/soft_readout_cnn.py, H200
# on steve, 2026-10-26).
# Origin: steve ~/oraqle-ics-sims/docker-torch-gpu/Dockerfile (2026-10-26, no git):
#   FROM python:3.11-slim
#   RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && pip install --no-cache-dir numpy==2.1.3 h5py scipy scikit-learn
#   WORKDIR /work
# Pinned: requirements.lock = `pip freeze` of oraqle-torch-gpu on steve (8e0bef81a5fc) minus torch (installed from the
# cu124 wheel index first, which also brings the nvidia-* runtime wheels listed in the lock); Python 3.11.17.
# Build: docker build -t oraqle-torch-gpu docker/torch-gpu       Run with --device=nvidia.com/gpu=all (CDI) or --gpus all.
FROM python:3.11.17-slim
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 \
 && pip install --no-cache-dir -r /tmp/requirements.lock
WORKDIR /work
