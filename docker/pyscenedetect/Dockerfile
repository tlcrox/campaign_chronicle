# PySceneDetect — CPU-only scene detection.
#
# This image began as a copy of the WhisperX one, from when both ran in a single
# container. That inheritance left it with a CUDA base image and a PyTorch
# install it has no use for. What actually runs here:
#
#   detect_scenes.sh        -> scenedetect (OpenCV)
#   detect_scenes_multi.sh  -> scenedetect + ffmpeg
#                              + python3 -m pipeline.scenes.extract_segments
#
# Nothing on that path imports torch, and IMPLEMENTATION_PLAN.md states scene
# detection is CPU-bound ("parallelizes freely"). Dropping torch and the CUDA
# runtime removes several GB and a GPU dependency this image never used.
#
# If a stage here ever does need CUDA, revert to
# nvidia/cuda:<ver>-cudnn-runtime-ubuntu22.04 and reinstate a torch install
# pinned to the matching CUDA wheel index — see docker/whisperx/Dockerfile for
# the version-pinning trap that applies.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg  - video decoding for scenedetect, and called directly by
#           detect_scenes_multi.sh to cut per-ROI segments.
# bash    - both entrypoints are `#!/usr/bin/env bash`; install it rather than
#           assume the slim base provides more than /bin/sh.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        bash \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# opencv-headless keeps libGL out of a slim image.
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements-docker.txt

WORKDIR /workspace

# Entrypoints are set per-service in compose/pyscenedetect/docker-compose.yml.
# Scripts and the pipeline package are bind-mounted at run time, not copied:
#   docker/pyscenedetect -> /usr/local/bin/scripts
#   src                  -> /usr/local/bin/pylib   (PYTHONPATH)
