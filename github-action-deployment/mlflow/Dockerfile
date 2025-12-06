FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mlflow

RUN pip install --no-cache-dir \
    mlflow[auth]==2.17.2 \
    psycopg2-binary \
    boto3 \
    pymysql \
    cryptography

RUN useradd -m -u 1000 mlflow && \
    chown -R mlflow:mlflow /mlflow

USER mlflow

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
