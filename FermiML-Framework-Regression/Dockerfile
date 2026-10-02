FROM python:3.12-slim-trixie AS base_stage
ENV PYTHON_VERSION="3.12"
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_NO_MANAGED_PYTHON=1
ENV UV_PYTHON_DOWNLOADS=never
ENV ROOT_DIR=/app

FROM base_stage AS build_stage
ARG DEBIAN_FRONTEND=noninteractive

RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

# Extra CA certificates for TLS-intercepting build proxies (e.g. sandboxed runners): the directory is
# empty by default, so this layer is a no-op on a normal build. workflow/build_image.py drops the
# runner's CA bundle here (gitignored) when it detects a proxy. Installed before the first network step
# so that apt/pip/git/cargo all trust the interception.
COPY workflow/certs/ /usr/local/share/ca-certificates/build-proxy/
RUN find /usr/local/share/ca-certificates/build-proxy -name '*.crt' | grep -q . && update-ca-certificates || true
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# git: matchcake, torchpfaffian and pythonbasictools are pinned as git dependencies.
# cargo + rustc: torchpfaffian@dev is a maturin (Rust) build with no wheel for its pinned commit.
# apt goes over https: deb.debian.org serves it, and CONNECT-only egress proxies reject plain HTTP.
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends build-essential git cargo rustc && \
    rm -rf /var/lib/apt/lists/*

# uv from PyPI rather than the ghcr.io image: one registry less in the build's network footprint
# (ghcr blob storage is often unreachable on restricted runners; PyPI always is).
RUN pip install --no-cache-dir "uv>=0.9.2"

WORKDIR "$ROOT_DIR"
# README.md and LICENSE are not documentation here: pyproject declares `dynamic = ["readme"]` and
# `license = { file = "LICENSE" }`, so the build backend fails without them.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src src
# Dependencies first, without the project, so an edit to src/ does not re-resolve or rebuild the whole
# environment (torch and the Rust extension dominate the build).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --no-dev --no-group docs --extra cpu --no-install-project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --no-dev --no-group docs --extra cpu

USER nonroot
# Puts the console scripts (compute_ith, compute_n, generate_hashes) on PATH, which is how
# workflow/Snakefile and workflow/lib.sh invoke them inside the container.
ENV PATH="$ROOT_DIR/.venv/bin:$PATH"
# sklearn caches fetch_california_housing here; workflow/Snakefile binds a host directory onto it so
# the cache survives between jobs and no compute node has to reach the network.
ENV SCIKIT_LEARN_DATA="$ROOT_DIR/datasets"
ENTRYPOINT ["python"]
CMD ["-c", "import fermiml_framework_regression as m; print(f'FermiML-Framework-Regression {m.__version__} ready')"]
