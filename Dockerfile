# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock ./

# Configure Poetry to not create virtual environment (since we're in a container)
RUN poetry config virtualenvs.create false

# Install dependencies only (not the current project)
RUN poetry install --only main --no-root

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs uploads models

# Set permissions
RUN chmod +x start.py

# Expose port (if needed for the main app)
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["python", "start.py"]
