FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    postgresql-client \
    libpq-dev \
    curl \
    ffmpeg \
    tesseract-ocr \
    libtesseract-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.7.1

# Copy dependency files
COPY pyproject.toml ./

# Configure poetry to not create virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies (without lock file for flexibility)
RUN poetry install --no-interaction --no-ansi --no-root || \
    (echo "Poetry install failed, trying with pip..." && \
     poetry export -f requirements.txt --output requirements.txt --without-hashes && \
     pip install --no-cache-dir -r requirements.txt)

# Copy application code
COPY . .

# Install the application
RUN poetry install --no-interaction --no-ansi || true

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

