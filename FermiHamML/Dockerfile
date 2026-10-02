FROM nvcr.io/nvidia/pytorch:26.01-py3 AS base_stage
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch?version=26.01-py3
ENV ROOT_DIR=/app
ARG MODE=release
# MODE: 'release' or 'it' (will be interpreted as debug). This is used in the docker_entrypoint.sh script
ENV MODE=${MODE}
ENV PYTHON_VERSION="3.12"
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PYTHONUNBUFFERED=1

# uv settings
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_SYSTEM_PYTHON=1
ENV UV_NO_MANAGED_PYTHON=1
ENV UV_PYTHON_DOWNLOADS=never
ENV UV_TOOL_BIN_DIR=/usr/local/bin
ENV UV_PYHTON=/usr/bin/python
ENV UV_NO_DEV=1

ENV DATA_DIR=/app/data

# Build stage
FROM base_stage AS build_stage

ARG DEBIAN_FRONTEND=noninteractive

# Setup a non-root user
RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

# System dependencies
RUN apt update -y && \
    apt install -y --no-install-recommends build-essential git dos2unix openssh-client && \
    apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Files
WORKDIR "$ROOT_DIR"
COPY *.toml *.lock *.py ./
COPY src src
COPY dist dist
COPY docker_files/*.sh docker_files/
RUN dos2unix docker_files/*.sh

# Download public key for github.com
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# Virtual environment
RUN --mount=type=ssh \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    git config --global --replace-all url."ssh://git@github.com/".insteadOf "https://github.com/" && \
    git config --global --add url."ssh://git@github.com/".insteadOf "https://git@github.com/" && \
    git config --global --add url."ssh://git@github.com/".insteadOf "git+https://git@github.com/" && \
    uv sync --locked --no-editable --no-cache --no-dev --extra cu130 --no-install-project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --no-cache --no-dev --extra cu130

# rm unnecessary files
RUN rm -rf setup.py src/*.egg-info

# Use the non-root user to run our application
USER nonroot

# Entrypoint
ENV PATH="$ROOT_DIR/.venv/bin:$PATH"
ENTRYPOINT ["bash", "docker_files/docker_entrypoint.sh"]
