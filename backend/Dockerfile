FROM python:3.11-slim

WORKDIR /app

# System deps (including Rust for tiktoken compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Slim worker deps — only what's needed for `from worker.tasks import ...`
# to succeed at import time (no ML packages).
COPY backend/requirements.worker-slim.txt .
RUN pip install --no-cache-dir -r requirements.worker-slim.txt

COPY backend/. .
COPY worker/ ./worker/
COPY shared/ ./shared/
COPY training/ ./training/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
