FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (git for installs; build-essential for compiling kernels; tidy apt cache)
# Rendering system deps (pango, cairo...)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential pkg-config \
      libgirepository-1.0-1 libcairo2 gir1.2-pango-1.0 libcairo2-dev libgirepository1.0-dev && \
    rm -rf /var/lib/apt/lists/*

# Install package dependencies
RUN mkdir -p /app/welt/vision && \
    touch /app/README.md
WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
RUN pip install ".[train]"

COPY welt /app/welt
COPY welt_training /app/welt_training
