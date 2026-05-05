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
# RUNNING INSTRUCTIONS:
# ---------------------------------------------------------------------------------------
# 1. Build the image:
#    docker build -t oraqle-env .
#
# 2. Run the container with NVIDIA GPU access and volume mounting for the 30GB dataset:
#    docker run --gpus all -it --rm \
#      -v /path/to/your/local/30GB_dataset:/app/dataset \
#      oraqle-env bash
#
# Note: `--gpus all` requires the NVIDIA Container Toolkit to be installed on the host.
# The `-v` flag maps your local dataset directory to `/app/dataset` inside the container
# so you don't have to copy the 30GB dataset into the Docker image itself.
# ---------------------------------------------------------------------------------------

# Default command (can be overridden when running)
CMD ["bash"]
