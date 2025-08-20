# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies including OpenCV requirements
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    python3-dev \
    curl \
    # OpenCV dependencies
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    libgtk-3-0 \
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

# Add the current directory to Python path
ENV PYTHONPATH=/app

# Expose port for the backend server
EXPOSE 8000

# Health check for the backend server
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command to run the backend server
CMD ["python", "start.py"]
