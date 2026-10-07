FROM nvcr.io/nvidia/nemo:26.08.01

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Rendering system deps (pango, cairo...)
RUN apt-get update && \
    apt-get install -y --no-install-recommends pkg-config \
      libgirepository-1.0-1 libcairo2 gir1.2-pango-1.0 libcairo2-dev libgirepository1.0-dev && \
    rm -rf /var/lib/apt/lists/*

# Install package dependencies (Megatron-Bridge, vLLM, torch and transformers come with the base image)
RUN mkdir -p /app/welt /app/welt_training && touch /app/README.md
WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
RUN pip install ".[dev]"

COPY welt /app/welt
COPY welt_training /app/welt_training

CMD torchrun --nproc_per_node=${NPROC_PER_NODE:-1} -m welt_training.train $CONFIG
