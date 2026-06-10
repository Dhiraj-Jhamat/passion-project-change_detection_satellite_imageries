# UCDNet — GPU-capable training & inference container
# Requires: NVIDIA Container Toolkit for GPU (`docker compose --profile gpu up`)
FROM tensorflow/tensorflow:2.15.0-gpu

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UCDNET_PROJECT_ROOT=/app \
    UCDNET_DATA_ROOT=/data/oscd \
    UCDNET_OUTPUT_DIR=/app/src/data/processed/artifacts

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.toml README.md ./
COPY change_detection_cli ./change_detection_cli
COPY src ./src

RUN pip install poetry==1.8.3 \
    && poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

RUN mkdir -p /app/src/data/processed/artifacts /data/oscd

ENTRYPOINT ["ucdnet"]
CMD ["train", "--help"]
