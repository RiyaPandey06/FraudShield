FROM python:3.11-slim

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources && apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install .
COPY app ./app
COPY experiments ./experiments
COPY static ./static
COPY docs ./docs
COPY README.md MODEL_CARD.md ./
RUN useradd --create-home --uid 10001 fraudshield && mkdir -p /app/data /app/results && chown -R fraudshield:fraudshield /app
USER fraudshield
