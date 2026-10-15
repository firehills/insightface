FROM node:22-bookworm-slim AS node

FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

USER root

RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

ENV VIRTUAL_ENV=/opt/insightface-venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv "${VIRTUAL_ENV}" \
    && "${VIRTUAL_ENV}/bin/python" -m pip install --no-cache-dir --upgrade \
        pip setuptools wheel pytest pytest-cov ruff mypy build \
    && chown -R vscode:vscode "${VIRTUAL_ENV}" \
    && install -d -o vscode -g vscode /home/vscode/.insightface

USER vscode