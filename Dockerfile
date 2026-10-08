# Megatron-Bridge, Megatron-Core, Transformer Engine, vLLM, torch and transformers come with the NeMo container
FROM nvcr.io/nvidia/nemo:26.08.01

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Text rendering: the latest Pango, cairo and PyGObject from conda-forge (as pixel-renderer installs them),
# in their own prefix, on the path of the container's Python
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-$(uname -m | sed s/x86_64/64/)/latest | \
      tar -xj -C /usr/local bin/micromamba && \
    micromamba create -y -p /opt/pango -c conda-forge python=3.12 pango pycairo pygobject && \
    micromamba clean -a -y && \
    echo /opt/pango/lib/python3.12/site-packages > "$(python -c 'import site; print(site.getsitepackages()[0])')/pango.pth"

WORKDIR /app
# Dependencies first, for layer caching (python -m pip: the NeMo image has a separate system pip)
COPY pyproject.toml README.md /app/
RUN mkdir -p welt welt_training && python -m pip install ".[dev]" && python -m pip uninstall -y WeLT
# Rendering fonts are downloaded once, at build time, rendering a word (which fails the build if Pango cannot load)
RUN python -c "from font_download import FontConfig; from font_download.example_fonts.noto_sans import FONTS_NOTO_SANS; \
    from pixel_renderer import PixelRendererProcessor; \
    print(PixelRendererProcessor(font=FontConfig(sources=FONTS_NOTO_SANS)).render_text('hello').shape)"

# Editable install: the commands (and the vLLM plugin) use /app, also when a checkout is mounted there
COPY . /app
RUN python -m pip install --no-deps -e .
