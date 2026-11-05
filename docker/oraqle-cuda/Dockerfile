# oraqle (CUDA): the image of the May-June 2026 discriminator runs (Discriminators/runners/hyper_optimize*.py, HERQULES
# runners, baseline_no_ML.py), i.e. the §4/§5 numbers that feed results/discriminators/master_fidelity.csv.
# Moved here from the repository root (Oraqle/Dockerfile @87d3162, identical on graham ~/Oraqle). Changes: the
# requirements path follows the new layout (Discriminators/ -> discriminators/); nothing else.
# NOT PINNED: the image no longer exists on graham or steve, so its package versions cannot be recovered;
# discriminators/requirements.txt is unpinned. The closest pinned environment is docker/torch-gpu (torch 2.5.1 cu124).
# Build context = repository root:  docker build -f docker/oraqle-cuda/Dockerfile -t oraqle .
# The run instructions below are the historical ones (graham paths); use docker/README.md for the new layout.
#
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update system and install Python, pip, and other necessary tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache for dependencies
COPY discriminators/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
# Note: Use .dockerignore to exclude large datasets, virtual environments, and git history
COPY . /app

# ---------------------------------------------------------------------------------------
# RUNNING INSTRUCTIONS (on graham.dos.cit.tum.de):
# ---------------------------------------------------------------------------------------
# 1. Build the image:
#    docker build -t oraqle .
#
# 2. Run the container with GPU access and data mounts:
#    docker run --gpus all -it --rm \
#      -v /home/manosgior/qubit_readout_klinq/data/five_qubit_data:/data/five_qubit_data:ro \
#      -v /home/sandra:/data/cnn:ro \
#      oraqle bash
#
# 3. Inside the container, run the hyper-optimization:
#    cd /app/Discriminators && python runners/hyper_optimize.py
#
# Note: `--gpus all` requires the NVIDIA Container Toolkit on the host.
#       `:ro` mounts data as read-only to prevent accidental modification.
# ---------------------------------------------------------------------------------------

# Default command (can be overridden when running)
CMD ["bash"]
