# WhisperX + PySceneDetect in a GPU-enabled container.
#
# Base: CUDA 12.4 + cuDNN 9 runtime. WhisperX depends on ctranslate2 >= 4.5,
# which requires CUDA 12 and cuDNN 9. Ubuntu 22.04 ships Python 3.10 which
# matches WhisperX's supported Python range.
# Note: PyTorch 2.8.0+cu128 is built against cuDNN 9.8.0, but the runtime image
# provides 9.1.0. They're compatible but with version warnings. Use CPU or float32 if
# cuDNN errors persist.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    XDG_CACHE_HOME=/models/cache \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    CUDA_HOME=/usr/local/cuda

# System deps. ffmpeg is used by both whisperx (audio decoding) and
# scenedetect (video decoding). Build torchcodec requires FFmpeg dev libs,
# but it also needs FFmpeg runtime. libavutil, libavcodec, libavformat are
# the runtime libraries torchcodec dynamically links to. libsndfile is torchaudio fallback.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        libavformat-dev \
        libavutil-dev \
        libavcodec-dev \
        libavformat58 \
        libavutil56 \
        libavcodec58 \
        libswresample3 \
        libswscale5 \
        git \
        ca-certificates \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# PyTorch, installed FIRST and on its own from the CUDA wheel index so the deps
# below see torch already satisfied and don't pull other wheels over the top.
#
# THE VERSION IS NOT FREE. whisperx 3.8.6 declares torch~=2.8.0 (i.e. >=2.8.0,
# <2.9.0), torchaudio~=2.8.0 and torchvision~=0.23.0. This used to pin
# torch==2.5.1/cu124, which does NOT satisfy that — so the `pip install -r
# requirements-docker.txt` step below quietly UPGRADED torch from PyPI, throwing
# away the CUDA build this stage exists to install. Exactly the failure the
# comment above warns about.
#
# The versions here are the ones a known-good build actually resolved to
# (recorded in a pip freeze taken from a working image): torch 2.8.0 with the
# cu128 runtime libraries, which is also the index whisperx pins upstream.
#
# Keep these in lockstep with the whisperx pin in requirements-docker.txt.
RUN pip install --no-cache-dir \
        torch==2.8.0 \
        torchaudio==2.8.0 \
        torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu128

# Everything else (requests, whisperx -> faster-whisper/ctranslate2/pyannote,
# speechbrain, numpy, scikit-learn, librosa, pyyaml)
# in one versioned manifest, so the container env is reproducible.
# Lock exact versions from a successful build with:
#   docker compose run --rm --entrypoint pip whisperx freeze > requirements-docker.lock.txt
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements-docker.txt

# Mount points. Bound at runtime via docker-compose.
RUN mkdir -p /audio /video /output /models


WORKDIR /workspace

#COPY scripts/entrypoint.sh /usr/local/bin/transcribe
#RUN chmod +x /usr/local/bin/transcribe 

# Entrypoint is overridden in docker-compose.yml for each service.
# Scripts are mounted at runtime, not copied during build.
