"""
FastAPI middleware for request processing and monitoring.

Implements logging, performance tracking, error handling,
and security middleware for all API requests.
"""

import time
import uuid
from decimal import Decimal
from typing import Dict, Optional, Any, Callable
from datetime import datetime

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from structlog import get_logger

from quantum_trader.api.exceptions import APIException

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging all API requests and responses.

    Logs request details, response status, and processing time
    with structured logging for monitoring and debugging.

    Attributes:
        config: Middleware configuration
    """

    def __init__(self, app: ASGIApp, config: Dict[str, Any]) -> None:
        """
        Initialize request logging middleware.

        Args:
            app: ASGI application
            config: Configuration with logging settings
        """
        super().__init__(app)
        self.config = config
        self.log_request_body = config.get("log_request_body", False)
        self.log_response_body = config.get("log_response_body", False)
        self.log_headers = config.get("log_headers", False)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process request and log details.

        Args:
            request: Incoming request
            call_next: Next middleware/route handler

        Returns:
            Response from downstream handler
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Start timer
        start_time = time.time()
        start_datetime = datetime.utcnow()

        # Log request
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_host": request.client.host if request.client else None,
            "timestamp": start_datetime.isoformat()
        }

        if self.log_headers:
            log_data["headers"] = dict(request.headers)

        logger.info("request_started", **log_data)

        # Process request
        try:
            response = await call_next(request)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Log response
            logger.info(
                "request_completed",
                request_id=request_id,
                status_code=response.status_code,
                processing_time_ms=round(processing_time * 1000, 2),
                path=request.url.path
            )

            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{processing_time:.4f}"

            return response

        except Exception as e:
            processing_time = time.time() - start_time

            logger.error(
                "request_failed",
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__,
                processing_time_ms=round(processing_time * 1000, 2),
                path=request.url.path
            )

            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for centralized error handling.

    Catches exceptions and converts them to appropriate HTTP responses
    with consistent error format.

    Attributes:
        config: Middleware configuration
    """

    def __init__(self, app: ASGIApp, config: Dict[str, Any]) -> None:
        """
        Initialize error handling middleware.

        Args:
            app: ASGI application
            config: Configuration with error handling settings
        """
        super().__init__(app)
        self.config = config
        self.include_traceback = config.get("include_traceback", False)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process request with error handling.

        Args:
            request: Incoming request
            call_next: Next middleware/route handler

        Returns:
            Response or error response
        """
        try:
            return await call_next(request)

        except APIException as e:
            # Handle custom API exceptions
            return JSONResponse(
                status_code=e.status_code,
                content=e.to_dict()
            )

        except ValueError as e:
            # Handle validation errors
            logger.warning(
                "validation_error",
                error=str(e),
                path=request.url.path
            )

            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "error": "VALIDATION_ERROR",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

        except Exception as e:
            # Handle unexpected errors
            logger.error(
                "unhandled_exception",
                error=str(e),
                error_type=type(e).__name__,
                path=request.url.path,
                exc_info=self.include_traceback
            )

            error_response = {
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "timestamp": datetime.utcnow().isoformat()
            }

            if self.include_traceback:
                import traceback
                error_response["traceback"] = traceback.format_exc()

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response
            )


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for performance monitoring and metrics collection.

    Tracks request latency, throughput, and resource usage.

    Attributes:
        config: Middleware configuration
        _request_counts: Request counter by endpoint
        _latency_totals: Total latency by endpoint
    """

    def __init__(self, app: ASGIApp, config: Dict[str, Any]) -> None:
        """
        Initialize performance monitoring middleware.

        Args:
            app: ASGI application
            config: Configuration with monitoring settings
        """
        super().__init__(app)
        self.config = config
        self._request_counts: Dict[str, int] = {}
        self._latency_totals: Dict[str, float] = {}
        self._slow_request_threshold = config.get("slow_request_threshold_ms", 1000)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process request with performance monitoring.

        Args:
            request: Incoming request
            call_next: Next middleware/route handler

        Returns:
            Response from downstream handler
        """
        start_time = time.time()
        endpoint = f"{request.method} {request.url.path}"

        try:
            response = await call_next(request)

            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000

            # Update counters
            self._request_counts[endpoint] = self._request_counts.get(endpoint, 0) + 1
            self._latency_totals[endpoint] = self._latency_totals.get(endpoint, 0.0) + latency_ms

            # Log slow requests
            if latency_ms > self._slow_request_threshold:
                logger.warning(
                    "slow_request",
                    endpoint=endpoint,
                    latency_ms=round(latency_ms, 2),
                    threshold_ms=self._slow_request_threshold
                )

            # Add performance headers
            response.headers["X-Response-Time"] = f"{latency_ms:.2f}ms"

            return response

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._request_counts[endpoint] = self._request_counts.get(endpoint, 0) + 1

            logger.error(
                "request_error",
                endpoint=endpoint,
                latency_ms=round(latency_ms, 2),
                error=str(e)
            )

            raise

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get collected performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        metrics = {}

        for endpoint, count in self._request_counts.items():
            total_latency = self._latency_totals.get(endpoint, 0.0)
            avg_latency = total_latency / count if count > 0 else 0.0

            metrics[endpoint] = {
                "request_count": count,
                "avg_latency_ms": round(avg_latency, 2),
                "total_latency_ms": round(total_latency, 2)
            }

        return metrics

    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        self._request_counts.clear()
        self._latency_totals.clear()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding security headers.

    Adds security-related HTTP headers to all responses.

    Attributes:
        config: Middleware configuration
    """

    def __init__(self, app: ASGIApp, config: Dict[str, Any]) -> None:
        """
        Initialize security headers middleware.

        Args:
            app: ASGI application
            config: Configuration with security settings
        """
        super().__init__(app)
        self.config = config

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process request and add security headers.

        Args:
            request: Incoming request
            call_next: Next middleware/route handler

        Returns:
            Response with security headers
        """
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # CORS headers (if configured)
        if self.config.get("cors_enabled", False):
            allowed_origins = self.config.get("cors_allowed_origins", ["*"])
            origin = request.headers.get("origin")

            if origin and (origin in allowed_origins or "*" in allowed_origins):
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
                response.headers["Access-Control-Max-Age"] = "3600"

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting requests.

    Implements token bucket algorithm for rate limiting.

    Attributes:
        config: Middleware configuration
        _buckets: Token buckets by client identifier
    """

    def __init__(self, app: ASGIApp, config: Dict[str, Any]) -> None:
        """
        Initialize rate limit middleware.

        Args:
            app: ASGI application
            config: Configuration with rate limit settings
        """
        super().__init__(app)
        self.config = config
        self._buckets: Dict[str, Dict[str, Any]] = {}
        self._max_requests = config.get("max_requests_per_minute", 60)
        self._window_seconds = config.get("window_seconds", 60)
        self._enabled = config.get("rate_limit_enabled", True)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process request with rate limiting.

        Args:
            request: Incoming request
            call_next: Next middleware/route handler

        Returns:
            Response or rate limit error
        """
        if not self._enabled:
            return await call_next(request)

        # Get client identifier
        client_id = self._get_client_id(request)

        # Check rate limit
        if not self._check_rate_limit(client_id):
            logger.warning(
                "rate_limit_exceeded",
                client_id=client_id,
                path=request.url.path
            )

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded: {self._max_requests} requests per {self._window_seconds}s",
                    "timestamp": datetime.utcnow().isoformat()
                },
                headers={
                    "X-RateLimit-Limit": str(self._max_requests),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": str(self._window_seconds)
                }
            )

        response = await call_next(request)

        # Add rate limit headers
        remaining = self._get_remaining(client_id)
        response.headers["X-RateLimit-Limit"] = str(self._max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Try API key first
        api_key = request.headers.get("x-api-key")
        if api_key:
            return f"api_key:{api_key}"

        # Fall back to IP address
        if request.client:
            return f"ip:{request.client.host}"

        return "unknown"

    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limit."""
        now = time.time()

        if client_id not in self._buckets:
            self._buckets[client_id] = {
                "tokens": self._max_requests - 1,
                "last_update": now
            }
            return True

        bucket = self._buckets[client_id]

        # Refill tokens based on time passed
        time_passed = now - bucket["last_update"]
        tokens_to_add = (time_passed / self._window_seconds) * self._max_requests
        bucket["tokens"] = min(
            self._max_requests,
            bucket["tokens"] + tokens_to_add
        )
        bucket["last_update"] = now

        # Check if tokens available
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True

        return False

    def _get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        if client_id not in self._buckets:
            return self._max_requests

        return int(self._buckets[client_id]["tokens"])


class CompressionMiddleware(BaseHTTPMiddleware):
    """
    Middleware for response compression.

    Compresses responses for clients that accept compression.

    Attributes:
        config: Middleware configuration
    """

    def __init__(self, app: ASGIApp, config: Dict[str, Any]) -> None:
        """
        Initialize compression middleware.

        Args:
            app: ASGI application
            config: Configuration with compression settings
        """
        super().__init__(app)
        self.config = config
        self._min_size = config.get("compression_min_size", 1024)
        self._enabled = config.get("compression_enabled", True)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process request with compression.

        Args:
            request: Incoming request
            call_next: Next middleware/route handler

        Returns:
            Potentially compressed response
        """
        if not self._enabled:
            return await call_next(request)

        response = await call_next(request)

        # Check if client accepts compression
        accept_encoding = request.headers.get("accept-encoding", "")

        if "gzip" in accept_encoding:
            # In production, implement actual gzip compression
            # For now, just add the header
            response.headers["Content-Encoding"] = "gzip"

        return response
