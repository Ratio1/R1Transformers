# Dockerfile: DevContainer for PyTorch + Transformers on CUDA 11.8
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel

# Install system packages: git, wget, curl, and development tools
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential cmake python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables (ensure CUDA 11.8 is accessible)
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH}

# Copy Python requirements and install them
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Set up a workspace directory
WORKDIR /workspace
