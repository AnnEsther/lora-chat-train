FROM python:3.11-slim

WORKDIR /app

# System deps (including Rust for tiktoken compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY worker/requirements.txt ./worker-requirements.txt
# RUN pip install --no-cache-dir -r worker-requirements.txt

COPY backend/. .
COPY worker/ ./worker/
COPY shared/ ./shared/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
