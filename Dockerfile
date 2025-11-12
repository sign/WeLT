FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install fused kernel packages
RUN pip install packaging ninja psutil
# Limit Jobs, due to memory issues
RUN MAX_JOBS=4 pip install flash_attn --no-build-isolation

# System deps (git for installs; build-essential for compiling kernels; tidy apt cache)
# Rendering system deps (pango, cairo...)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential pkg-config \
      libgirepository-1.0-1 libcairo2 gir1.2-pango-1.0 libcairo2-dev libgirepository1.0-dev && \
    rm -rf /var/lib/apt/lists/*

# Install package dependencies
RUN mkdir -p /app/welt/vision && \
    touch /app/README.md
WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
RUN pip install ".[train]"

COPY welt /app/welt
COPY training /app/training
