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
COPY Discriminators/requirements.txt /app/requirements.txt

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
