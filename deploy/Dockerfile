# Multi-stage build; arch-specific base is selected at build time:
#   docker build --build-arg BASE=python:3.11-slim-bookworm .
# Accelerator runtimes (HailoRT / JetPack) come from the device image, not
# this container — they are mounted /dev + host libs per compose profile.
ARG BASE=python:3.11-slim-bookworm

FROM ${BASE} AS build
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir uv && uv pip install --system ".[rtsp,onnx]"

FROM ${BASE}
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin/sentinel-edge /usr/local/bin/sentinel-edge
COPY configs ./configs
# writable overlay: spool, queue, config cache, secrets
VOLUME ["/app/data"]
ENV SENTINEL_DATA_DIR=/app/data \
    SENTINEL_PROFILES_DIR=/app/configs/profiles
ENTRYPOINT ["sentinel-edge", "run"]
