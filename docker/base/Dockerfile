# =============================================================================
# QUANTUM TRADER AI - BASE DOCKER IMAGE
# =============================================================================
# Multi-stage build for optimized production image
# Python 3.12 with performance optimizations
# =============================================================================

FROM python:3.12-slim-bookworm AS base

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.7.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_HOME=/opt/poetry

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    # Library dependencies
    libpq-dev \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    liblzma-dev \
    libsqlite3-dev \
    # Math and scientific libraries
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    # Compression
    zlib1g-dev \
    # Tools
    curl \
    wget \
    git \
    ca-certificates \
    tzdata \
    # Cleaning
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# Set timezone to UTC
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# =============================================================================
# BUILDER STAGE - Install Python dependencies
# =============================================================================

FROM base AS builder

# Copy only requirements first for better caching
COPY requirements.txt /tmp/requirements.txt
COPY docker/base/requirements-base.txt /tmp/requirements-base.txt

# Create virtual environment
RUN python -m venv /opt/venv

# Activate virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install base requirements
RUN pip install --no-cache-dir -r /tmp/requirements-base.txt

# Install main requirements
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional production dependencies
RUN pip install --no-cache-dir \
    gunicorn==21.2.0 \
    uvicorn[standard]==0.25.0 \
    uvloop==0.19.0 \
    httptools==0.6.1

# =============================================================================
# RUNTIME STAGE - Create minimal runtime image
# =============================================================================

FROM base AS runtime

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set PATH to use virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r quantum && \
    useradd -r -g quantum -u 1000 -m -s /bin/bash quantum && \
    mkdir -p /app/logs /app/data /app/models /app/checkpoints /app/config && \
    chown -R quantum:quantum /app

# Health check dependencies
RUN pip install --no-cache-dir httpx==0.25.2

# Copy application code
COPY --chown=quantum:quantum src/ /app/src/
COPY --chown=quantum:quantum config/ /app/config/
COPY --chown=quantum:quantum pyproject.toml /app/

# Install package in editable mode
RUN pip install -e /app

# Switch to non-root user
USER quantum

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command (override in specific containers)
CMD ["python", "-m", "quantum_trader"]

# =============================================================================
# LABELS
# =============================================================================

LABEL maintainer="Quantum Trader AI Team" \
      description="Base image for Quantum Trader AI platform" \
      version="1.0.0" \
      python.version="3.12"
