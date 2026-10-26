# Image service (SDXL / Flux via diffusers, or a ComfyUI client). Needs an NVIDIA GPU for usable speed.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3-pip curl \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements/base.txt requirements/images.txt requirements/
RUN python -m pip install torch --index-url https://download.pytorch.org/whl/cu124 \
    && python -m pip install -r requirements/images.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY configs ./configs
RUN python -m pip install --no-deps . && useradd -m appuser && mkdir -p /cache/huggingface && chown -R appuser /cache /app
USER appuser
EXPOSE 8002
CMD ["uvicorn", "app.images_main:app", "--host", "0.0.0.0", "--port", "8002"]
