"""
Monitoring router for Quantum Trader AI API.

Provides endpoints for system health, metrics, and monitoring.
"""

from decimal import Decimal
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from structlog import get_logger

from quantum_trader.api.dependencies import (
    get_current_user,
    get_optional_user,
    get_db_pool,
    get_redis_client,
    require_role
)

logger = get_logger(__name__)

router = APIRouter()


# Schemas

class HealthStatus(BaseModel):
    """System health status.

    Attributes:
        status: Overall status (healthy, degraded, unhealthy)
        timestamp: Check timestamp
        version: API version
        uptime_seconds: System uptime in seconds
        components: Component health status
    """
    status: str = Field(..., description="Overall status")
    timestamp: datetime = Field(..., description="Check timestamp (UTC)")
    version: str = Field(..., description="API version")
    uptime_seconds: int = Field(..., description="System uptime in seconds")
    components: Dict[str, str] = Field(..., description="Component health status")


class SystemMetrics(BaseModel):
    """System metrics.

    Attributes:
        request_count: Total request count
        error_count: Total error count
        error_rate: Error rate percentage
        avg_response_time_ms: Average response time
        active_connections: Active WebSocket connections
        active_users: Active users count
        timestamp: Metrics timestamp
    """
    request_count: int = Field(..., description="Total request count")
    error_count: int = Field(..., description="Total error count")
    error_rate: str = Field(..., description="Error rate percentage")
    avg_response_time_ms: Optional[str] = Field(None, description="Average response time")
    active_connections: int = Field(..., description="Active WebSocket connections")
    active_users: int = Field(..., description="Active users count")
    timestamp: datetime = Field(..., description="Metrics timestamp (UTC)")


class TradingMetrics(BaseModel):
    """Trading metrics.

    Attributes:
        total_orders_today: Total orders today
        filled_orders_today: Filled orders today
        cancelled_orders_today: Cancelled orders today
        total_volume_24h: Total trading volume 24h
        active_positions: Number of active positions
        total_pnl_today: Total P&L today
        timestamp: Metrics timestamp
    """
    total_orders_today: int = Field(..., description="Total orders today")
    filled_orders_today: int = Field(..., description="Filled orders today")
    cancelled_orders_today: int = Field(..., description="Cancelled orders today")
    total_volume_24h: str = Field(..., description="Total trading volume 24h")
    active_positions: int = Field(..., description="Active positions count")
    total_pnl_today: str = Field(..., description="Total P&L today")
    timestamp: datetime = Field(..., description="Metrics timestamp (UTC)")


class ExchangeStatus(BaseModel):
    """Exchange connection status.

    Attributes:
        exchange: Exchange name
        status: Connection status
        latency_ms: Connection latency
        last_update: Last update timestamp
    """
    exchange: str = Field(..., description="Exchange name")
    status: str = Field(..., description="Connection status")
    latency_ms: Optional[str] = Field(None, description="Connection latency")
    last_update: datetime = Field(..., description="Last update timestamp (UTC)")


class AlertLevel(str):
    """Alert level enumeration."""
    pass


class SystemAlert(BaseModel):
    """System alert.

    Attributes:
        alert_id: Alert identifier
        level: Alert level (info, warning, error, critical)
        title: Alert title
        message: Alert message
        component: Component that raised alert
        created_at: Alert creation timestamp
        resolved_at: Alert resolution timestamp
    """
    alert_id: str = Field(..., description="Alert identifier")
    level: str = Field(..., description="Alert level")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    component: str = Field(..., description="Component")
    created_at: datetime = Field(..., description="Creation timestamp (UTC)")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp (UTC)")


# Endpoints

@router.get(
    "/health",
    response_model=HealthStatus,
    summary="Health check",
    description="Get system health status"
)
async def health_check(
    db_pool = Depends(get_db_pool),
    redis_client = Depends(get_redis_client)
) -> HealthStatus:
    """Get system health status.

    Args:
        db_pool: Database pool
        redis_client: Redis client

    Returns:
        Health status

    Raises:
        HTTPException: If health check fails
    """
    try:
        components = {}

        # Check database
        try:
            async with db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            components["database"] = "healthy"
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            components["database"] = "unhealthy"

        # Check Redis
        try:
            await redis_client.ping()
            components["redis"] = "healthy"
        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
            components["redis"] = "unhealthy"

        # Determine overall status
        if all(status == "healthy" for status in components.values()):
            overall_status = "healthy"
        elif any(status == "unhealthy" for status in components.values()):
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        # TODO: Calculate actual uptime
        uptime_seconds = 0

        return HealthStatus(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            uptime_seconds=uptime_seconds,
            components=components
        )

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )


@router.get(
    "/metrics/system",
    response_model=SystemMetrics,
    summary="Get system metrics",
    description="Get system performance metrics",
    dependencies=[Depends(require_role("ADMIN"))]
)
async def get_system_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user),
    redis_client = Depends(get_redis_client)
) -> SystemMetrics:
    """Get system metrics.

    Args:
        current_user: Current user
        redis_client: Redis client

    Returns:
        System metrics

    Raises:
        HTTPException: If metrics retrieval fails
    """
    try:
        # Get metrics from Redis
        request_count = 0
        error_count = 0

        try:
            requests_data = await redis_client.hgetall("api:metrics:requests")
            if requests_data:
                request_count = sum(int(v) for v in requests_data.values())

            errors_data = await redis_client.hgetall("api:metrics:errors")
            if errors_data:
                error_count = sum(int(v) for v in errors_data.values())
        except Exception as e:
            logger.warning("Failed to get metrics from Redis", error=str(e))

        # Calculate error rate
        error_rate = Decimal("0")
        if request_count > 0:
            error_rate = (Decimal(str(error_count)) / Decimal(str(request_count))) * Decimal("100")

        return SystemMetrics(
            request_count=request_count,
            error_count=error_count,
            error_rate=str(error_rate),
            avg_response_time_ms=None,
            active_connections=0,
            active_users=0,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error("Get system metrics failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system metrics"
        )


@router.get(
    "/metrics/trading",
    response_model=TradingMetrics,
    summary="Get trading metrics",
    description="Get trading activity metrics"
)
async def get_trading_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_pool = Depends(get_db_pool)
) -> TradingMetrics:
    """Get trading metrics.

    Args:
        current_user: Current user
        db_pool: Database pool

    Returns:
        Trading metrics

    Raises:
        HTTPException: If metrics retrieval fails
    """
    try:
        today = datetime.utcnow().date()

        async with db_pool.acquire() as conn:
            # Get order counts
            order_metrics = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total_orders,
                    COUNT(*) FILTER (WHERE status = 'FILLED') as filled_orders,
                    COUNT(*) FILTER (WHERE status = 'CANCELLED') as cancelled_orders
                FROM orders
                WHERE DATE(created_at) = $1
                """,
                today
            )

            # Get active positions count
            active_positions = await conn.fetchval(
                "SELECT COUNT(*) FROM positions WHERE status = 'OPEN'"
            )

        return TradingMetrics(
            total_orders_today=order_metrics["total_orders"] or 0,
            filled_orders_today=order_metrics["filled_orders"] or 0,
            cancelled_orders_today=order_metrics["cancelled_orders"] or 0,
            total_volume_24h="0",
            active_positions=active_positions or 0,
            total_pnl_today="0",
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error("Get trading metrics failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get trading metrics"
        )


@router.get(
    "/exchanges",
    response_model=List[ExchangeStatus],
    summary="Get exchange status",
    description="Get exchange connection status"
)
async def get_exchange_status(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_pool = Depends(get_db_pool)
) -> List[ExchangeStatus]:
    """Get exchange connection status.

    Args:
        current_user: Current user
        db_pool: Database pool

    Returns:
        List of exchange statuses

    Raises:
        HTTPException: If status retrieval fails
    """
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    exchange, status, latency_ms, last_update
                FROM exchange_status
                ORDER BY exchange
                """
            )

        statuses = []
        for row in rows:
            statuses.append(ExchangeStatus(
                exchange=row["exchange"],
                status=row["status"],
                latency_ms=str(row["latency_ms"]) if row["latency_ms"] else None,
                last_update=row["last_update"]
            ))

        return statuses

    except Exception as e:
        logger.error("Get exchange status failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get exchange status"
        )


@router.get(
    "/alerts",
    response_model=List[SystemAlert],
    summary="Get system alerts",
    description="Get system alerts",
    dependencies=[Depends(require_role("ADMIN"))]
)
async def get_alerts(
    resolved: Optional[bool] = Query(None, description="Filter by resolved status"),
    level: Optional[str] = Query(None, description="Filter by alert level"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum results"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_pool = Depends(get_db_pool)
) -> List[SystemAlert]:
    """Get system alerts.

    Args:
        resolved: Filter by resolved status
        level: Filter by level
        limit: Maximum results
        current_user: Current user
        db_pool: Database pool

    Returns:
        List of system alerts

    Raises:
        HTTPException: If alerts retrieval fails
    """
    try:
        # Build query
        query = """
            SELECT
                alert_id, level, title, message, component,
                created_at, resolved_at
            FROM system_alerts
            WHERE 1=1
        """
        params: List[Any] = []
        param_count = 0

        if resolved is not None:
            param_count += 1
            if resolved:
                query += f" AND resolved_at IS NOT NULL"
            else:
                query += f" AND resolved_at IS NULL"

        if level:
            param_count += 1
            query += f" AND level = ${param_count}"
            params.append(level)

        query += f" ORDER BY created_at DESC LIMIT ${param_count + 1}"
        params.append(limit)

        async with db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        alerts = []
        for row in rows:
            alerts.append(SystemAlert(
                alert_id=row["alert_id"],
                level=row["level"],
                title=row["title"],
                message=row["message"],
                component=row["component"],
                created_at=row["created_at"],
                resolved_at=row["resolved_at"]
            ))

        return alerts

    except Exception as e:
        logger.error("Get alerts failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get alerts"
        )


@router.post(
    "/alerts/{alert_id}/resolve",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Resolve alert",
    description="Mark alert as resolved",
    dependencies=[Depends(require_role("ADMIN"))]
)
async def resolve_alert(
    alert_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_pool = Depends(get_db_pool)
) -> None:
    """Resolve system alert.

    Args:
        alert_id: Alert ID
        current_user: Current user
        db_pool: Database pool

    Raises:
        HTTPException: If resolution fails
    """
    try:
        logger.info("Resolve alert request", alert_id=alert_id)

        async with db_pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE system_alerts
                SET resolved_at = $1
                WHERE alert_id = $2 AND resolved_at IS NULL
                """,
                datetime.utcnow(),
                alert_id
            )

            if result == "UPDATE 0":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Alert not found or already resolved"
                )

        logger.info("Alert resolved", alert_id=alert_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Resolve alert failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resolve alert"
        )
