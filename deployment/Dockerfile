# dots.tts inference server.
#
# Build context is the REPO ROOT (engine, deployment package, vendored reference
# and tn stubs all get copied):
#
#   docker build -f deployment/Dockerfile -t dots-tts .
#   docker run --gpus all -p 8000:8000 -v $PWD/models:/models dots-tts
#
# NO nvidia/cuda base: the torch +cu126 wheels bundle their own CUDA runtime
# (cudnn/cublas/cusparse/cufft/nccl), flash-attn is a prebuilt wheel, and Triton
# ships its own ptxas -- so the only host dependency is the NVIDIA driver, which
# the nvidia-container-toolkit injects at `--gpus`. Using a cuda base would just
# ship a second, redundant copy of CUDA (it bloated the image by ~15GB). A100 is
# sm_80; deps are uv-locked (pyproject.toml + uv.lock).
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_NO_PROGRESS=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    TORCH_CUDA_ARCH_LIST="8.0" \
    # let nvidia-container-toolkit inject the driver on this non-cuda base
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# build-essential: g++ for torch.compile's inductor C++ backend (no CUDA toolkit
# needed -- Triton bundles ptxas). ffmpeg + libsndfile1: audio I/O. git/curl/ca.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl git build-essential ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# uv: fast installer; brings a standalone CPython 3.12 (UV_PYTHON_INSTALL_DIR).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Locked deps -> /opt/venv. --no-cache keeps uv's wheel cache OUT of the image
# (baking it added ~7GB). Cache layer keyed on pyproject.toml + uv.lock.
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --no-cache

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Vendored dots.tts reference, installed --no-deps: its pyproject pins
# WeTextProcessing (pynini/OpenFst, a heavy native build) which this project
# replaces with the lightweight stubs/tn package on PYTHONPATH; its real runtime
# deps are already locked and the runtime import path is gradio-free. Own layer
# (before the app COPY) so editing engine/server code doesn't rebuild it.
COPY reference/dots.tts /app/reference/dots.tts
RUN uv pip install --no-deps --no-cache /app/reference/dots.tts

# Project source (engine, HTTP layer, tn stubs).
COPY . /app

# stubs/ provides the `tn` stand-in; repo root is importable for nanovllm_dots
# and deployment.
ENV PYTHONPATH=/app/stubs:/app \
    HF_HUB_OFFLINE=1 \
    DOTS_MODEL=/models/dots.tts-mf \
    DOTS_HOST=0.0.0.0 \
    DOTS_PORT=8000

EXPOSE 8000

# Single uvicorn worker: the engine lives in-process and batches all requests.
CMD ["python", "-m", "uvicorn", "deployment.app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"]
