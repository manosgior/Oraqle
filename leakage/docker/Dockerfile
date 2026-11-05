# oraqle-torch: the image every leakage / error-mode script ran in (graham, CPU only).
# Byte copy of graham ~/oraqle-ics-sims/docker-torch/Dockerfile (md5 31e91a05fc2d70a641abc72dd9125323), plus
# these comments. Image f40dbb670f94, built 2026-10-26. `pip freeze` inside that image (2026-11-05):
#   python 3.11.11, torch==2.5.1+cpu, numpy==2.1.3, scipy==1.17.1, scikit-learn==1.9.1, h5py==3.16.0,
#   networkx==3.6.1, sympy==1.13.1, joblib==1.6.0, threadpoolctl==3.7.0
# scipy, scikit-learn and h5py are unpinned below; pin them to the versions above for an exact rebuild.
# Build:  docker build -t oraqle-torch leakage/docker
FROM python:3.11-slim
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir numpy==2.1.3 h5py scipy scikit-learn
WORKDIR /work
