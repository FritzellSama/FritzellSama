# =============================================================================
# QUANTUM TRADER AI - MACHINE LEARNING DOCKER IMAGE
# =============================================================================
# GPU-enabled container for ML model training and inference
# Includes PyTorch, CUDA support, and scientific computing libraries
# =============================================================================

ARG BASE_IMAGE=quantum-trader-base:latest
ARG CUDA_VERSION=12.1.0
ARG CUDNN_VERSION=8

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-ubuntu22.04 AS cuda-base

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12

# Install Python 3.12 and base dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
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
    # Math and scientific libraries
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    # CUDA development (for PyTorch compilation)
    cuda-toolkit-12-1 \
    # Tools
    curl \
    wget \
    git \
    ca-certificates \
    tzdata \
    # Monitoring
    htop \
    nvtop \
    # Cleaning
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set timezone to UTC
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# =============================================================================
# ML DEPENDENCIES STAGE
# =============================================================================

FROM cuda-base AS ml-builder

WORKDIR /app

# Copy requirements
COPY requirements.txt /tmp/requirements.txt
COPY docker/ml/requirements-ml.txt /tmp/requirements-ml.txt

# Create virtual environment
RUN python3 -m venv /opt/venv

# Activate virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install ML and scientific computing packages
RUN pip install --no-cache-dir \
    # Deep Learning
    pytorch-lightning==2.1.3 \
    torchmetrics==1.3.0 \
    # Classical ML
    scikit-learn==1.4.0 \
    xgboost==2.0.3 \
    lightgbm==4.2.0 \
    catboost==1.2.2 \
    # Scientific computing
    numpy==1.26.3 \
    scipy==1.11.4 \
    # Model tracking
    mlflow==2.9.2 \
    wandb==0.16.2 \
    tensorboard==2.15.1 \
    # Optimization
    optuna==3.5.0 \
    hyperopt==0.2.7 \
    # Feature engineering
    featuretools==1.29.0 \
    tsfresh==0.20.1 \
    # Model interpretation
    shap==0.44.0 \
    lime==0.2.0.1

# Install base requirements
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install ML-specific requirements if exists
RUN if [ -f /tmp/requirements-ml.txt ]; then \
        pip install --no-cache-dir -r /tmp/requirements-ml.txt; \
    fi

# =============================================================================
# RUNTIME STAGE
# =============================================================================

FROM cuda-base AS runtime

# Copy virtual environment from builder
COPY --from=ml-builder /opt/venv /opt/venv

# Set PATH to use virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PATH=/usr/local/cuda/bin:$PATH

# Create non-root user
RUN groupadd -r quantum && \
    useradd -r -g quantum -u 1000 -m -s /bin/bash quantum && \
    mkdir -p \
        /app/logs/ml \
        /app/models \
        /app/checkpoints \
        /app/data/features \
        /app/data/training \
        /app/experiments \
        /app/tensorboard \
    && chown -R quantum:quantum /app

# Copy application code
COPY --chown=quantum:quantum src/ /app/src/
COPY --chown=quantum:quantum config/ /app/config/
COPY --chown=quantum:quantum pyproject.toml /app/

# Copy ML-specific configuration
COPY --chown=quantum:quantum config/ml/ /app/config/ml/
COPY --chown=quantum:quantum scripts/ml/ /app/scripts/ml/

# Install package
WORKDIR /app
RUN pip install -e /app

# Switch to non-root user
USER quantum

# Set environment variables for ML
ENV COMPONENT=ml \
    LOG_FILE=/app/logs/ml/ml.log \
    MODEL_DIR=/app/models \
    CHECKPOINT_DIR=/app/checkpoints \
    TENSORBOARD_DIR=/app/tensorboard \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8

# Health check
HEALTHCHECK --interval=60s --timeout=15s --start-period=120s --retries=3 \
    CMD python -c "import sys; \
                   import torch; \
                   cuda_available = torch.cuda.is_available(); \
                   print(f'CUDA available: {cuda_available}'); \
                   sys.exit(0)"

# Expose TensorBoard port
EXPOSE 6006

# Expose MLflow port
EXPOSE 5000

# Expose metrics port
EXPOSE 9093

# Set working directory
WORKDIR /app

# Entry point for graceful shutdown
COPY --chown=quantum:quantum docker/ml/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh || true

# Default command - run ML training/inference service
CMD ["python", "-m", "quantum_trader.ai.main"]

# =============================================================================
# LABELS
# =============================================================================

LABEL maintainer="Quantum Trader AI Team" \
      description="Machine Learning container for Quantum Trader AI platform" \
      component="ml" \
      version="1.0.0" \
      cuda.version="12.1.0" \
      pytorch.version="2.2.0"
