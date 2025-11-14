# =============================================================================
# QUANTUM TRADER AI - DASHBOARD/API DOCKER IMAGE
# =============================================================================
# FastAPI-based REST API and web dashboard for monitoring and control
# Optimized for high-throughput API requests
# =============================================================================

ARG BASE_IMAGE=quantum-trader-base:latest

FROM ${BASE_IMAGE}

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Switch to root for installations
USER root

# Install additional dependencies for dashboard
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Network tools
    netcat-openbsd \
    iputils-ping \
    # Process management
    procps \
    # Cleaning
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# Install dashboard-specific Python packages
RUN pip install --no-cache-dir \
    # Web framework
    fastapi==0.109.0 \
    uvicorn[standard]==0.25.0 \
    # ASGI server
    gunicorn==21.2.0 \
    # WebSocket support
    websockets==12.0 \
    python-socketio==5.11.0 \
    # Template engine
    jinja2==3.1.3 \
    # Static files
    aiofiles==23.2.1 \
    # Authentication
    python-jose[cryptography]==3.3.0 \
    passlib[bcrypt]==1.7.4 \
    python-multipart==0.0.6 \
    # CORS
    fastapi-cors==0.0.6 \
    # Rate limiting
    slowapi==0.1.9 \
    # Metrics
    prometheus-client==0.19.0 \
    prometheus-fastapi-instrumentator==6.1.0 \
    # Caching
    aiocache==0.12.2 \
    # Validation
    email-validator==2.1.0 \
    # Additional utilities
    python-dotenv==1.0.0

# Create dashboard-specific directories
RUN mkdir -p \
    /app/logs/api \
    /app/logs/dashboard \
    /app/static \
    /app/templates \
    /app/uploads \
    && chown -R quantum:quantum /app

# Copy dashboard configuration
COPY --chown=quantum:quantum config/api/ /app/config/api/
COPY --chown=quantum:quantum scripts/api/ /app/scripts/api/

# Switch back to non-root user
USER quantum

# Set environment variables for dashboard
ENV COMPONENT=dashboard \
    LOG_FILE=/app/logs/dashboard/api.log \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    WORKERS=4

# Health check for API
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; \
                   import sys; \
                   try: \
                       response = httpx.get('http://localhost:8000/health', timeout=5); \
                       sys.exit(0 if response.status_code == 200 else 1); \
                   except: \
                       sys.exit(1)"

# Expose API port
EXPOSE 8000

# Expose metrics port
EXPOSE 9092

# Set working directory
WORKDIR /app

# Entry point for graceful shutdown
COPY --chown=quantum:quantum docker/dashboard/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh || true

# Default command - run API server with Uvicorn
# Using Uvicorn with multiple workers for production
CMD ["uvicorn", \
     "quantum_trader.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--no-access-log", \
     "--log-config", "/app/config/api/logging.yaml"]

# Alternative with Gunicorn (uncomment to use)
# CMD ["gunicorn", \
#      "quantum_trader.api.main:app", \
#      "--bind", "0.0.0.0:8000", \
#      "--workers", "4", \
#      "--worker-class", "uvicorn.workers.UvicornWorker", \
#      "--timeout", "120", \
#      "--keepalive", "5", \
#      "--max-requests", "10000", \
#      "--max-requests-jitter", "1000", \
#      "--access-logfile", "-", \
#      "--error-logfile", "-", \
#      "--log-level", "info"]

# =============================================================================
# LABELS
# =============================================================================

LABEL maintainer="Quantum Trader AI Team" \
      description="Dashboard and API container for Quantum Trader AI platform" \
      component="dashboard" \
      version="1.0.0"
