# ============================
# STAGE 1 — BUILDER
# ============================

FROM python:3.10-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Rimuove metadata inutili
RUN find /install -type d -name "tests" -exec rm -rf {} + \
    && find /install -type d -name "__pycache__" -exec rm -rf {} + \
    && find /install -type f -name "*.pyc" -delete \
    && find /install -type f -name "*.pyo" -delete


# ============================
# STAGE 2 — RUNTIME
# ============================
FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# Installazione minima runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia solo i pacchetti già installati
COPY --from=builder /install /usr/local

# Strip dei binari Python e librerie per ridurre peso
RUN find /usr/local -type f -executable -exec strip --strip-unneeded {} + || true \
    && find /usr/local/lib -type f -name "*.so" -exec strip --strip-unneeded {} + || true

# HuggingFace cache esterna
ENV HF_HOME=/hf_cache

COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]
