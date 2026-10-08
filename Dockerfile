# Megatron-Bridge, Megatron-Core, Transformer Engine, vLLM, torch and transformers come with the NeMo container
FROM nvcr.io/nvidia/nemo:26.08.01

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Text rendering: the latest Pango, cairo and PyGObject from conda-forge (as pixel-renderer installs them),
# in their own prefix (for the container's Python version), on the path of the container's Python.
# The prefix's own pip, setuptools and wheel are removed, so they do not shadow the container's.
RUN py=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])') && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-$(uname -m | sed s/x86_64/64/)/latest | \
      tar -xj -C /usr/local bin/micromamba && \
    micromamba create -y -p /opt/pango -c conda-forge python=$py pango pycairo pygobject && \
    micromamba clean -a -y && \
    rm -rf /opt/pango/lib/python$py/site-packages/pip* /opt/pango/lib/python$py/site-packages/setuptools* \
      /opt/pango/lib/python$py/site-packages/wheel* /opt/pango/lib/python$py/site-packages/_distutils_hack && \
    echo /opt/pango/lib/python$py/site-packages > "$(python -c 'import site; print(site.getsitepackages()[0])')/pango.pth"

WORKDIR /app
# Dependencies first, for layer caching (python -m pip: the container's Python, not the system's pip)
COPY pyproject.toml README.md /app/
RUN mkdir -p welt/vision welt_training && python -m pip install ".[dev,train]"
# Rendering fonts are downloaded once, at build time, rendering a word (which fails the build if Pango cannot load)
RUN python -c "from font_download import FontConfig; from font_download.example_fonts.noto_sans import FONTS_NOTO_SANS; \
    from pixel_renderer import PixelRendererProcessor; \
    print(PixelRendererProcessor(font=FontConfig(sources=FONTS_NOTO_SANS)).render_text('hello').shape)"

# Editable install: the commands (and the vLLM plugin) use /app, also when a checkout is mounted there
COPY . /app
RUN python -m pip install --no-deps -e .
