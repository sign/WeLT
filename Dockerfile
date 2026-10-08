# Megatron-Bridge, Megatron-Core, Transformer Engine, vLLM, torch and transformers come with the NeMo container
FROM nvcr.io/nvidia/nemo:26.08.01

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Text rendering (pango, cairo)
RUN apt-get update && \
    apt-get install -y --no-install-recommends pkg-config \
      libgirepository-1.0-1 libcairo2 gir1.2-pango-1.0 libcairo2-dev libgirepository1.0-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Dependencies first, for layer caching (python -m pip: the NeMo image has a separate system pip)
COPY pyproject.toml README.md /app/
RUN mkdir -p welt/vision welt_training && python -m pip install ".[dev,train]" && python -m pip uninstall -y WeLT
# Rendering fonts are downloaded once, at build time
RUN python -c "from font_download import FontConfig; from font_download.example_fonts.noto_sans import FONTS_NOTO_SANS; \
    FontConfig(sources=FONTS_NOTO_SANS).get_font_dir()"

# Editable install: the commands (and the vLLM plugin) use /app, also when a checkout is mounted there
COPY . /app
RUN python -m pip install --no-deps -e .
