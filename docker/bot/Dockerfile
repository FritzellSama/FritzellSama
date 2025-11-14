# =============================================================================
# QUANTUM TRADER AI - TRADING BOT DOCKER IMAGE
# =============================================================================
# Optimized container for high-frequency trading operations
# Includes all trading, risk management, and order execution components
# =============================================================================

ARG BASE_IMAGE=quantum-trader-base:latest

FROM ${BASE_IMAGE}

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Switch to root for installations
USER root

# Install additional dependencies for trading bot
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Network tools for debugging
    netcat-openbsd \
    iputils-ping \
    dnsutils \
    # Process management
    procps \
    # Monitoring
    htop \
    # Cleaning
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# Install bot-specific Python packages
RUN pip install --no-cache-dir \
    # Exchange connectors (already in base, but ensure latest)
    ccxt==4.2.0 \
    # WebSocket for real-time data
    websockets==12.0 \
    # Advanced async
    aiofiles==23.2.1 \
    aiocache==0.12.2 \
    # Performance monitoring
    psutil==5.9.6 \
    py-cpuinfo==9.0.0 \
    # Additional utilities
    cachetools==5.3.2 \
    python-json-logger==2.0.7

# Create bot-specific directories
RUN mkdir -p \
    /app/logs/bot \
    /app/data/trades \
    /app/data/positions \
    /app/data/orders \
    /app/state \
    && chown -R quantum:quantum /app

# Copy bot-specific configuration
COPY --chown=quantum:quantum config/bot/ /app/config/bot/
COPY --chown=quantum:quantum config/strategies/ /app/config/strategies/
COPY --chown=quantum:quantum scripts/bot/ /app/scripts/bot/

# Switch back to non-root user
USER quantum

# Set environment variables for bot
ENV COMPONENT=bot \
    LOG_FILE=/app/logs/bot/trading_bot.log \
    STATE_DIR=/app/state \
    TRADES_DIR=/app/data/trades \
    POSITIONS_DIR=/app/data/positions \
    ORDERS_DIR=/app/data/orders

# Health check specific to trading bot
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; \
                   import os; \
                   from pathlib import Path; \
                   state_file = Path('/app/state/bot.state'); \
                   import time; \
                   if state_file.exists() and (time.time() - state_file.stat().st_mtime) < 120: \
                       sys.exit(0); \
                   else: \
                       sys.exit(1)"

# Expose metrics port
EXPOSE 9091

# Set working directory
WORKDIR /app

# Entry point for graceful shutdown handling
COPY --chown=quantum:quantum docker/bot/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh || true

# Default command - run trading bot
CMD ["python", "-m", "quantum_trader.engine.main"]

# =============================================================================
# LABELS
# =============================================================================

LABEL maintainer="Quantum Trader AI Team" \
      description="Trading bot container for Quantum Trader AI platform" \
      component="bot" \
      version="1.0.0"
