# Chat gateway (router + RAG + agent). CPU image: the LLM runs in the separate vLLM container.
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements/base.txt requirements/serve.txt requirements/
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements/serve.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY configs ./configs
RUN pip install --no-deps . && useradd -m appuser && mkdir -p /cache/huggingface && chown -R appuser /cache /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
