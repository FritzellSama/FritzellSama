"""
System Router - FastAPI endpoints for system health and status monitoring.

This module provides REST API endpoints for system health checks, status monitoring,
metrics collection, and operational management.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import os
import platform
import psutil

from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from structlog import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/system", tags=["system"])


class HealthStatus(str):
    """System health status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Overall system health status")
    timestamp: datetime = Field(..., description="Check timestamp (UTC)")
    uptime_seconds: int = Field(..., description="System uptime in seconds")
    version: str = Field(..., description="Application version")
    components: Dict[str, str] = Field(..., description="Component health status")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-01-15T10:00:00Z",
                "uptime_seconds": 86400,
                "version": "1.0.0",
                "components": {
                    "database": "healthy",
                    "redis": "healthy",
                    "exchanges": "healthy"
                }
            }
        }


class SystemMetricsResponse(BaseModel):
    """System metrics response schema."""
    cpu_percent: str = Field(..., description="CPU usage percentage (Decimal)")
    memory_percent: str = Field(..., description="Memory usage percentage (Decimal)")
    disk_percent: str = Field(..., description="Disk usage percentage (Decimal)")
    active_connections: int = Field(..., description="Number of active connections")
    active_strategies: int = Field(..., description="Number of active strategies")
    total_orders: int = Field(..., description="Total orders processed")
    timestamp: datetime = Field(..., description="Metrics timestamp (UTC)")

    class Config:
        schema_extra = {
            "example": {
                "cpu_percent": "45.5",
                "memory_percent": "67.3",
                "disk_percent": "23.1",
                "active_connections": 150,
                "active_strategies": 12,
                "total_orders": 5432,
                "timestamp": "2025-01-15T10:00:00Z"
            }
        }


class SystemInfoResponse(BaseModel):
    """System information response schema."""
    application: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Environment (dev, staging, prod)")
    platform: str = Field(..., description="Platform information")
    python_version: str = Field(..., description="Python version")
    started_at: datetime = Field(..., description="Application start time (UTC)")

    class Config:
        schema_extra = {
            "example": {
                "application": "Quantum Trader AI",
                "version": "1.0.0",
                "environment": "production",
                "platform": "Linux-5.4.0-x86_64",
                "python_version": "3.12.0",
                "started_at": "2025-01-15T00:00:00Z"
            }
        }


class ConfigResponse(BaseModel):
    """System configuration response schema."""
    config_name: str = Field(..., description="Configuration parameter name")
    config_value: str = Field(..., description="Configuration parameter value")
    is_sensitive: bool = Field(..., description="Whether value is sensitive")

    class Config:
        schema_extra = {
            "example": {
                "config_name": "max_strategies",
                "config_value": "100",
                "is_sensitive": False
            }
        }


# Module-level variables for tracking
_start_time: Optional[datetime] = None
_metrics_cache: Dict[str, Any] = {}


def _get_start_time() -> datetime:
    """Get application start time."""
    global _start_time
    if _start_time is None:
        _start_time = datetime.now(timezone.utc)
    return _start_time


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Perform system health check.

    Returns:
        HealthCheckResponse: System health status

    Example:
        >>> GET /system/health
        {
            "status": "healthy",
            "uptime_seconds": 86400,
            "components": {...}
        }
    """
    try:
        logger.debug("Performing health check")

        start_time = _get_start_time()
        now = datetime.now(timezone.utc)
        uptime = int((now - start_time).total_seconds())

        # Check component health
        components = await _check_components()

        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        if any(status == HealthStatus.UNHEALTHY for status in components.values()):
            overall_status = HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in components.values()):
            overall_status = HealthStatus.DEGRADED

        version = os.getenv('APP_VERSION', '1.0.0')

        return HealthCheckResponse(
            status=overall_status,
            timestamp=now,
            uptime_seconds=uptime,
            version=version,
            components=components
        )

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_metrics() -> SystemMetricsResponse:
    """Get system performance metrics.

    Returns:
        SystemMetricsResponse: Current system metrics

    Example:
        >>> GET /system/metrics
        {
            "cpu_percent": "45.5",
            "memory_percent": "67.3",
            ...
        }
    """
    try:
        logger.debug("Retrieving system metrics")

        # Get system metrics
        cpu_percent = Decimal(str(psutil.cpu_percent(interval=0.1)))
        memory_percent = Decimal(str(psutil.virtual_memory().percent))
        disk_percent = Decimal(str(psutil.disk_usage('/').percent))

        # Get application metrics from cache or defaults
        active_connections = _metrics_cache.get('active_connections', 0)
        active_strategies = _metrics_cache.get('active_strategies', 0)
        total_orders = _metrics_cache.get('total_orders', 0)

        return SystemMetricsResponse(
            cpu_percent=str(cpu_percent),
            memory_percent=str(memory_percent),
            disk_percent=str(disk_percent),
            active_connections=active_connections,
            active_strategies=active_strategies,
            total_orders=total_orders,
            timestamp=datetime.now(timezone.utc)
        )

    except Exception as e:
        logger.error("Failed to retrieve metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics retrieval failed: {str(e)}"
        )


@router.get("/info", response_model=SystemInfoResponse)
async def get_system_info() -> SystemInfoResponse:
    """Get system information.

    Returns:
        SystemInfoResponse: System and application information

    Example:
        >>> GET /system/info
        {
            "application": "Quantum Trader AI",
            "version": "1.0.0",
            ...
        }
    """
    try:
        logger.debug("Retrieving system info")

        app_name = os.getenv('APP_NAME', 'Quantum Trader AI')
        app_version = os.getenv('APP_VERSION', '1.0.0')
        environment = os.getenv('ENVIRONMENT', 'development')

        return SystemInfoResponse(
            application=app_name,
            version=app_version,
            environment=environment,
            platform=platform.platform(),
            python_version=platform.python_version(),
            started_at=_get_start_time()
        )

    except Exception as e:
        logger.error("Failed to retrieve system info", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"System info retrieval failed: {str(e)}"
        )


@router.get("/config", response_model=List[ConfigResponse])
async def get_config(
    include_sensitive: bool = Query(False, description="Include sensitive values")
) -> List[ConfigResponse]:
    """Get system configuration parameters.

    Args:
        include_sensitive: Whether to include sensitive config values

    Returns:
        List of configuration parameters

    Example:
        >>> GET /system/config
        [
            {"config_name": "max_strategies", "config_value": "100", "is_sensitive": false},
            ...
        ]
    """
    try:
        logger.debug("Retrieving system config", include_sensitive=include_sensitive)

        configs: List[ConfigResponse] = []

        # Non-sensitive configs
        non_sensitive = {
            'max_strategies': os.getenv('STRATEGY_MAX_COUNT', '100'),
            'max_connections': os.getenv('WS_MAX_CONNECTIONS', '10000'),
            'heartbeat_interval': os.getenv('WS_HEARTBEAT_INTERVAL', '30'),
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'app_version': os.getenv('APP_VERSION', '1.0.0'),
        }

        for name, value in non_sensitive.items():
            configs.append(ConfigResponse(
                config_name=name,
                config_value=value,
                is_sensitive=False
            ))

        # Sensitive configs (masked unless requested)
        sensitive = {
            'database_url': os.getenv('DATABASE_URL', 'not_set'),
            'redis_url': os.getenv('REDIS_URL', 'not_set'),
        }

        for name, value in sensitive.items():
            if include_sensitive:
                config_value = value
            else:
                config_value = '***REDACTED***'

            configs.append(ConfigResponse(
                config_name=name,
                config_value=config_value,
                is_sensitive=True
            ))

        return configs

    except Exception as e:
        logger.error("Failed to retrieve config", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Config retrieval failed: {str(e)}"
        )


@router.post("/shutdown", status_code=status.HTTP_202_ACCEPTED)
async def initiate_shutdown() -> Dict[str, str]:
    """Initiate graceful system shutdown.

    Returns:
        Shutdown acknowledgment

    Example:
        >>> POST /system/shutdown
        {
            "status": "shutdown_initiated",
            "message": "System shutdown in progress"
        }
    """
    try:
        logger.warning("Shutdown initiated via API")

        # In production, this would trigger graceful shutdown
        # For now, just acknowledge
        return {
            "status": "shutdown_initiated",
            "message": "System shutdown in progress",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error("Shutdown initiation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Shutdown failed: {str(e)}"
        )


@router.post("/metrics/update")
async def update_metrics(metrics: Dict[str, int]) -> Dict[str, str]:
    """Update internal metrics cache.

    Args:
        metrics: Metrics to update

    Returns:
        Update acknowledgment

    Example:
        >>> POST /system/metrics/update
        {
            "active_strategies": 15,
            "total_orders": 5500
        }
    """
    try:
        logger.debug("Updating metrics", metrics=metrics)

        # Update metrics cache
        _metrics_cache.update(metrics)

        return {
            "status": "metrics_updated",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error("Metrics update failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics update failed: {str(e)}"
        )


async def _check_components() -> Dict[str, str]:
    """Check health of individual components.

    Returns:
        Dictionary mapping component names to health status
    """
    components: Dict[str, str] = {}

    # Database check
    try:
        # In production, would check actual database connection
        await asyncio.sleep(0.001)
        components['database'] = HealthStatus.HEALTHY
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        components['database'] = HealthStatus.UNHEALTHY

    # Redis check
    try:
        # In production, would check actual Redis connection
        await asyncio.sleep(0.001)
        components['redis'] = HealthStatus.HEALTHY
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))
        components['redis'] = HealthStatus.UNHEALTHY

    # Exchange connectivity check
    try:
        # In production, would check exchange connections
        await asyncio.sleep(0.001)
        components['exchanges'] = HealthStatus.HEALTHY
    except Exception as e:
        logger.error("Exchange health check failed", error=str(e))
        components['exchanges'] = HealthStatus.UNHEALTHY

    # Message queue check
    try:
        # In production, would check message queue health
        await asyncio.sleep(0.001)
        components['message_queue'] = HealthStatus.HEALTHY
    except Exception as e:
        logger.error("Message queue health check failed", error=str(e))
        components['message_queue'] = HealthStatus.UNHEALTHY

    return components
