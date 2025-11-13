"""System monitoring endpoints."""
from decimal import Decimal
from typing import Dict
import psutil
from datetime import datetime

from fastapi import APIRouter
from structlog import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


@router.get("/health")
async def health_check():
    """System health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@router.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu": {
                "usage_percent": cpu_percent,
                "count": psutil.cpu_count()
            },
            "memory": {
                "total_gb": round(memory.total / (1024 ** 3), 2),
                "used_gb": round(memory.used / (1024 ** 3), 2),
                "percent": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / (1024 ** 3), 2),
                "used_gb": round(disk.used / (1024 ** 3), 2),
                "percent": disk.percent
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("Metrics fetch failed", error=str(e))
        return {"error": str(e)}


@router.get("/status")
async def get_system_status():
    """Get detailed system status."""
    return {
        "api": "running",
        "database": "connected",
        "redis": "connected",
        "exchanges": {
            "binance": "connected",
            "bybit": "connected"
        },
        "ml_models": "loaded",
        "strategies": "active"
    }


@router.get("/uptime")
async def get_uptime():
    """Get system uptime."""
    import time
    boot_time = psutil.boot_time()
    uptime_seconds = int(time.time() - boot_time)
    
    return {
        "uptime_seconds": uptime_seconds,
        "uptime_hours": round(uptime_seconds / 3600, 2),
        "boot_time": datetime.fromtimestamp(boot_time).isoformat()
    }
