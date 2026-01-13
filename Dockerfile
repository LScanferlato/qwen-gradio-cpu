FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Dipendenze di sistema minime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia requirements
COPY requirements.txt .

# Installa dipendenze Python (numpy<2 + tutto il resto)
RUN pip install --no-cache-dir -r requirements.txt

# Cache HuggingFace esterna
ENV HF_HOME=/hf_cache

# Copia l'app Gradio
COPY app.py .

EXPOSE 7860

CMD ["python3", "app.py"]
