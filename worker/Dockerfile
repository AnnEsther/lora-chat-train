FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY ../shared /app/shared
COPY ../training /app/training
COPY ../backend /app/backend

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["celery", "-A", "tasks", "worker", "--loglevel=info", "--concurrency=2"]
