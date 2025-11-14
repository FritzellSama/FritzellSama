"""
Common response schemas for API endpoints.

Defines standard response formats for success, error,
and pagination across all API endpoints.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Any, Generic, TypeVar
from datetime import datetime

from pydantic import BaseModel, Field, validator
from pydantic.generics import GenericModel
from structlog import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class SuccessResponse(BaseModel):
    """Standard success response."""

    success: bool = Field(True, description="Success indicator")
    message: str = Field(..., description="Success message")
    timestamp: str = Field(..., description="Response timestamp (ISO 8601)")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "timestamp": "2025-01-15T12:00:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: str = Field(..., description="Error timestamp (ISO 8601)")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "VALIDATION_ERROR",
                "message": "Invalid parameter value",
                "details": {
                    "field": "quantity",
                    "reason": "Must be positive"
                },
                "timestamp": "2025-01-15T12:00:00Z"
            }
        }


class PaginatedResponse(GenericModel, Generic[T]):
    """Paginated response wrapper."""

    items: List[T] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    skip: int = Field(..., description="Number of items skipped")
    limit: int = Field(..., description="Maximum items per page")
    has_more: bool = Field(..., description="Whether more items are available")

    class Config:
        json_schema_extra = {
            "example": {
                "items": [],
                "total": 100,
                "skip": 0,
                "limit": 50,
                "has_more": True
            }
        }


class StatusResponse(BaseModel):
    """Generic status response."""

    status: str = Field(..., description="Status value")
    message: Optional[str] = Field(None, description="Status message")
    timestamp: str = Field(..., description="Status timestamp (ISO 8601)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "active",
                "message": "System operating normally",
                "timestamp": "2025-01-15T12:00:00Z",
                "metadata": {
                    "uptime_seconds": 86400.0
                }
            }
        }


class BatchOperationResponse(BaseModel):
    """Response for batch operations."""

    total_requested: int = Field(..., description="Total operations requested")
    successful: int = Field(..., description="Number of successful operations")
    failed: int = Field(..., description="Number of failed operations")
    errors: List[Dict[str, Any]] = Field(..., description="Details of failed operations")
    timestamp: str = Field(..., description="Operation timestamp (ISO 8601)")

    class Config:
        json_schema_extra = {
            "example": {
                "total_requested": 10,
                "successful": 8,
                "failed": 2,
                "errors": [
                    {
                        "index": 3,
                        "error": "VALIDATION_ERROR",
                        "message": "Invalid quantity"
                    },
                    {
                        "index": 7,
                        "error": "INSUFFICIENT_BALANCE",
                        "message": "Not enough balance"
                    }
                ],
                "timestamp": "2025-01-15T12:00:00Z"
            }
        }


class DataResponse(GenericModel, Generic[T]):
    """Generic data response wrapper."""

    data: T = Field(..., description="Response data")
    timestamp: str = Field(..., description="Response timestamp (ISO 8601)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "data": {},
                "timestamp": "2025-01-15T12:00:00Z",
                "metadata": {
                    "source": "cache"
                }
            }
        }


class ValidationErrorDetail(BaseModel):
    """Validation error detail."""

    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    value: Optional[Any] = Field(None, description="Invalid value")

    class Config:
        json_schema_extra = {
            "example": {
                "field": "quantity",
                "message": "Must be a positive number",
                "value": "-1.5"
            }
        }


class ValidationErrorResponse(BaseModel):
    """Validation error response with field details."""

    error: str = Field("VALIDATION_ERROR", description="Error code")
    message: str = Field(..., description="Error message")
    errors: List[ValidationErrorDetail] = Field(..., description="Field validation errors")
    timestamp: str = Field(..., description="Error timestamp (ISO 8601)")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "errors": [
                    {
                        "field": "quantity",
                        "message": "Must be positive",
                        "value": "-1.5"
                    },
                    {
                        "field": "price",
                        "message": "Required for LIMIT orders",
                        "value": None
                    }
                ],
                "timestamp": "2025-01-15T12:00:00Z"
            }
        }


class AggregatedMetrics(BaseModel):
    """Aggregated metrics response."""

    period_start: str = Field(..., description="Metrics period start (ISO 8601)")
    period_end: str = Field(..., description="Metrics period end (ISO 8601)")
    total_count: int = Field(..., description="Total count")
    sum_value: str = Field(..., description="Sum of values (as decimal string)")
    average_value: str = Field(..., description="Average value (as decimal string)")
    min_value: str = Field(..., description="Minimum value (as decimal string)")
    max_value: str = Field(..., description="Maximum value (as decimal string)")

    @validator("sum_value", "average_value", "min_value", "max_value")
    def validate_decimal(cls, v: str) -> str:
        """Validate decimal strings."""
        try:
            Decimal(v)
            return v
        except (ValueError, ArithmeticError) as e:
            raise ValueError(f"Invalid decimal value: {e}")

    class Config:
        json_schema_extra = {
            "example": {
                "period_start": "2025-01-15T00:00:00Z",
                "period_end": "2025-01-15T12:00:00Z",
                "total_count": 150,
                "sum_value": "15000.50",
                "average_value": "100.00",
                "min_value": "50.00",
                "max_value": "250.00"
            }
        }


class TimeSeriesDataPoint(BaseModel):
    """Time series data point."""

    timestamp: str = Field(..., description="Data point timestamp (ISO 8601)")
    value: str = Field(..., description="Data point value (as decimal string)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @validator("value")
    def validate_decimal(cls, v: str) -> str:
        """Validate decimal value."""
        try:
            Decimal(v)
            return v
        except (ValueError, ArithmeticError) as e:
            raise ValueError(f"Invalid decimal value: {e}")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-01-15T12:00:00Z",
                "value": "50000.00",
                "metadata": {
                    "source": "binance"
                }
            }
        }


class TimeSeriesResponse(BaseModel):
    """Time series data response."""

    metric_name: str = Field(..., description="Metric name")
    period_start: str = Field(..., description="Time series start (ISO 8601)")
    period_end: str = Field(..., description="Time series end (ISO 8601)")
    interval: str = Field(..., description="Data interval (e.g., '1m', '1h')")
    data_points: List[TimeSeriesDataPoint] = Field(..., description="Time series data points")
    count: int = Field(..., description="Number of data points")

    class Config:
        json_schema_extra = {
            "example": {
                "metric_name": "btc_price",
                "period_start": "2025-01-15T00:00:00Z",
                "period_end": "2025-01-15T12:00:00Z",
                "interval": "1h",
                "data_points": [
                    {
                        "timestamp": "2025-01-15T00:00:00Z",
                        "value": "49500.00"
                    },
                    {
                        "timestamp": "2025-01-15T01:00:00Z",
                        "value": "49750.00"
                    }
                ],
                "count": 2
            }
        }


class RateLimitInfo(BaseModel):
    """Rate limit information."""

    limit: int = Field(..., description="Maximum requests allowed")
    remaining: int = Field(..., description="Remaining requests")
    reset_at: str = Field(..., description="Rate limit reset time (ISO 8601)")
    retry_after: Optional[int] = Field(None, description="Seconds until retry allowed")

    class Config:
        json_schema_extra = {
            "example": {
                "limit": 100,
                "remaining": 45,
                "reset_at": "2025-01-15T12:01:00Z",
                "retry_after": None
            }
        }


class SystemInfoResponse(BaseModel):
    """System information response."""

    name: str = Field(..., description="System name")
    version: str = Field(..., description="System version")
    environment: str = Field(..., description="Environment (dev, staging, prod)")
    build_time: str = Field(..., description="Build timestamp (ISO 8601)")
    commit_hash: Optional[str] = Field(None, description="Git commit hash")
    uptime_seconds: float = Field(..., description="System uptime in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Quantum Trader AI",
                "version": "1.0.0",
                "environment": "production",
                "build_time": "2025-01-01T00:00:00Z",
                "commit_hash": "abc123def456",
                "uptime_seconds": 86400.0
            }
        }


class ActionResponse(BaseModel):
    """Response for action operations."""

    action: str = Field(..., description="Action performed")
    resource_id: str = Field(..., description="Affected resource ID")
    status: str = Field(..., description="Action status")
    message: str = Field(..., description="Action message")
    timestamp: str = Field(..., description="Action timestamp (ISO 8601)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional action metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "action": "create_order",
                "resource_id": "ord_1234567890",
                "status": "success",
                "message": "Order created successfully",
                "timestamp": "2025-01-15T12:00:00Z",
                "metadata": {
                    "symbol": "BTC/USDT",
                    "quantity": "0.1"
                }
            }
        }


class HealthStatus(BaseModel):
    """Component health status."""

    component: str = Field(..., description="Component name")
    status: str = Field(..., description="Health status (healthy, degraded, unhealthy)")
    latency_ms: Optional[float] = Field(None, description="Component latency in milliseconds")
    last_check: str = Field(..., description="Last health check timestamp (ISO 8601)")
    message: Optional[str] = Field(None, description="Health status message")

    class Config:
        json_schema_extra = {
            "example": {
                "component": "database",
                "status": "healthy",
                "latency_ms": 5.5,
                "last_check": "2025-01-15T12:00:00Z",
                "message": "Connection pool healthy"
            }
        }


class AuditLogEntry(BaseModel):
    """Audit log entry."""

    log_id: str = Field(..., description="Log entry ID")
    timestamp: str = Field(..., description="Event timestamp (ISO 8601)")
    user_id: Optional[str] = Field(None, description="User ID")
    action: str = Field(..., description="Action performed")
    resource_type: str = Field(..., description="Resource type")
    resource_id: Optional[str] = Field(None, description="Resource ID")
    status: str = Field(..., description="Action status")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "log_id": "log_1234567890",
                "timestamp": "2025-01-15T12:00:00Z",
                "user_id": "user_123",
                "action": "create_order",
                "resource_type": "order",
                "resource_id": "ord_1234567890",
                "status": "success",
                "ip_address": "192.168.1.1",
                "metadata": {
                    "symbol": "BTC/USDT"
                }
            }
        }


class ExportResponse(BaseModel):
    """Data export response."""

    export_id: str = Field(..., description="Export job ID")
    status: str = Field(..., description="Export status")
    format: str = Field(..., description="Export format (csv, json, parquet)")
    download_url: Optional[str] = Field(None, description="Download URL when ready")
    expires_at: Optional[str] = Field(None, description="Download expiration (ISO 8601)")
    created_at: str = Field(..., description="Export creation timestamp (ISO 8601)")
    completed_at: Optional[str] = Field(None, description="Export completion timestamp (ISO 8601)")
    record_count: Optional[int] = Field(None, description="Number of records exported")

    class Config:
        json_schema_extra = {
            "example": {
                "export_id": "exp_1234567890",
                "status": "completed",
                "format": "csv",
                "download_url": "https://api.example.com/exports/exp_1234567890/download",
                "expires_at": "2025-01-16T12:00:00Z",
                "created_at": "2025-01-15T12:00:00Z",
                "completed_at": "2025-01-15T12:05:00Z",
                "record_count": 1500
            }
        }
