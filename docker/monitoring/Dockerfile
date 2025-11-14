# Quantum Trader AI - Monitoring Stack Dockerfile
# Production-ready monitoring infrastructure for institutional trading platform
# Manages Prometheus, Grafana, and AlertManager

FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    gnupg \
    ca-certificates \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Create monitoring user and directories
RUN useradd -m -u 1000 -s /bin/bash monitoring && \
    mkdir -p /opt/monitoring/{prometheus,grafana,alertmanager} && \
    mkdir -p /var/lib/{prometheus,grafana} && \
    mkdir -p /var/log/monitoring && \
    chown -R monitoring:monitoring /opt/monitoring /var/lib/prometheus /var/lib/grafana /var/log/monitoring

# Install Python monitoring dependencies
COPY requirements-monitoring.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements-monitoring.txt && \
    rm /tmp/requirements-monitoring.txt

# Copy monitoring scripts
COPY --chown=monitoring:monitoring scripts/monitoring/ /opt/monitoring/scripts/
COPY --chown=monitoring:monitoring config/monitoring/ /opt/monitoring/config/

# Copy supervisor configuration
COPY docker/monitoring/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Health check script
COPY --chown=monitoring:monitoring scripts/monitoring/healthcheck.py /opt/monitoring/healthcheck.py
RUN chmod +x /opt/monitoring/healthcheck.py

# Set working directory
WORKDIR /opt/monitoring

# Switch to monitoring user
USER monitoring

# Expose ports (loaded from environment variables in runtime)
# Prometheus: ${PROMETHEUS_PORT}
# Grafana: ${GRAFANA_PORT}
# AlertManager: ${ALERTMANAGER_PORT}
EXPOSE 9090 3000 9093

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python /opt/monitoring/healthcheck.py

# Start supervisor to manage all monitoring services
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
