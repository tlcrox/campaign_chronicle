"""Shared, dependency-light helpers used across pipeline steps.

Modules:
    timecode  - timestamp <-> seconds conversion (single source of truth)
    sessions  - session discovery + audio/video source detection
    mounts    - copy/clear/mirror helpers for the docker mount dirs
    docker    - run_docker_command wrapper around `docker compose`
"""
