FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    # Make sure conda is first on PATH
    PATH="/opt/conda/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Set this to your target GPUs for smaller/faster CUDA kernels (example: A100 + H100)
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

SHELL ["bash", "-lc"]

# System deps (git for installs; build-essential for compiling kernels; tidy apt cache)
# Rendering system deps (pango, cairo...)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential pkg-config \
      libgirepository-1.0-1 libcairo2 gir1.2-pango-1.0 libcairo2-dev libgirepository1.0-dev && \
    rm -rf /var/lib/apt/lists/*

# Install python version
RUN conda install -y python=3.12 && conda clean -afy

# Install package dependencies
RUN mkdir -p /app/welt/vision && \
    touch /app/README.md
WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
RUN pip install ".[train]"

# Install fused kernel packages
RUN pip install ninja triton flash-attn --no-build-isolation
