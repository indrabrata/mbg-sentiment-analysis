FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for better layer caching)
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy all application modules
COPY config.py ./
COPY logging_config.py ./
COPY models.py ./
COPY model_loader.py ./
COPY inference.py ./
COPY middleware.py ./
COPY endpoints.py ./
COPY app.py ./

# Copy .env file if exists (optional)
COPY .env* ./ 2>/dev/null || true

# Create data directory for predictions
RUN mkdir -p data/predictions

# Expose port
EXPOSE 8000

# Set environment variables (can be overridden)
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV API_RELOAD=false
ENV LOG_FORMAT=json
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
