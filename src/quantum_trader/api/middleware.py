"""
Custom middleware for Quantum Trader AI API.

Provides logging, rate limiting, and metrics collection.
"""

import time
import asyncio
from decimal import Decimal
from typing import Dict, Any, Callable
from datetime import datetime
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from structlog import get_logger
import redis.asyncio as redis

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware.

    Logs all HTTP requests and responses with timing information.

    Attributes:
        config: Configuration dictionary
    """

    def __init__(self, app: ASGIApp, config: Dict[str, Any]) -> None:
        """Initialize logging middleware.

        Args:
            app: ASGI application
            config: Configuration from config files
        """
        super().__init__(app)
        self.config = config

        # Load from config
        logging_config = config.get("api", {}).get("logging", {})
        self.log_request_body = logging_config.get("log_request_body", False)
        self.log_response_body = logging_config.get("log_response_body", False)
        self.exclude_paths = logging_config.get("exclude_paths", ["/health", "/metrics"])

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log.

        Args:
            request: HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response
        """
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Start timer
        start_time = time.time()

        # Get request details
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        # Log request
        logger.info(
            "HTTP request",
            method=request.method,
            path=request.url.path,
            client_ip=client_ip,
            user_agent=user_agent
        )

        # Process request
        try:
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log response
            logger.info(
                "HTTP response",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=f"{duration_ms:.2f}",
                client_ip=client_ip
            )

            # Add timing header
            response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            logger.error(
                "HTTP request failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                duration_ms=f"{duration_ms:.2f}",
                client_ip=client_ip
            )

            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware.

    Implements per-user and per-IP rate limiting using Redis.

    Attributes:
        config: Configuration dictionary
    """

    def __init__(self, app: ASGIApp, config: Dict[str, Any]) -> None:
        """Initialize rate limit middleware.

        Args:
            app: ASGI application
            config: Configuration from config files
        """
        super().__init__(app)
        self.config = config

        # Load from config
        rate_limit_config = config.get("api", {}).get("rate_limit", {})
        self.enabled = rate_limit_config.get("enabled", True)
        self.max_requests = rate_limit_config.get("max_requests_per_minute", 60)
        self.window_seconds = rate_limit_config.get("window_seconds", 60)
        self.exclude_paths = rate_limit_config.get("exclude_paths", ["/health", "/metrics"])

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting.

        Args:
            request: HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response

        Raises:
            HTTPException: If rate limit exceeded
        """
        # Skip if disabled or excluded
        if not self.enabled or request.url.path in self.exclude_paths:
            return await call_next(request)

        try:
            # Get rate limit key
            redis_client = request.app.state.redis_client
            client_ip = request.client.host if request.client else "unknown"
            rate_limit_key = f"rate_limit:ip:{client_ip}"

            # Check rate limit
            current = await redis_client.incr(rate_limit_key)

            # Set expiry on first request
            if current == 1:
                await redis_client.expire(rate_limit_key, self.window_seconds)

            # Add rate limit headers
            remaining = max(0, self.max_requests - current)

            response = await call_next(request)

            response.headers["X-RateLimit-Limit"] = str(self.max_requests)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(self.window_seconds)

            # Check if limit exceeded
            if current > self.max_requests:
                from fastapi import HTTPException, status

                logger.warning(
                    "Rate limit exceeded",
                    client_ip=client_ip,
                    current=current,
                    limit=self.max_requests
                )

                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )

            return response

        except Exception as e:
            # Don't block on rate limit errors
            if "rate limit exceeded" in str(e).lower():
                raise

            logger.error("Rate limit check failed", error=str(e))
            return await call_next(request)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Metrics collection middleware.

    Collects request/response metrics for monitoring.

    Attributes:
        config: Configuration dictionary
        request_count: Total request counter
        error_count: Error counter
    """

    def __init__(self, app: ASGIApp, config: Dict[str, Any]) -> None:
        """Initialize metrics middleware.

        Args:
            app: ASGI application
            config: Configuration from config files
        """
        super().__init__(app)
        self.config = config

        # Metrics storage
        self.request_count = 0
        self.error_count = 0
        self.request_durations: Dict[str, list] = {}

        # Load from config
        metrics_config = config.get("api", {}).get("metrics", {})
        self.enabled = metrics_config.get("enabled", True)
        self.collect_detailed = metrics_config.get("collect_detailed", True)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics.

        Args:
            request: HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response
        """
        if not self.enabled:
            return await call_next(request)

        start_time = time.time()

        try:
            response = await call_next(request)

            # Collect metrics
            duration_ms = (time.time() - start_time) * 1000

            self.request_count += 1

            if self.collect_detailed:
                path = request.url.path
                if path not in self.request_durations:
                    self.request_durations[path] = []

                self.request_durations[path].append(duration_ms)

                # Keep only last 1000 samples per path
                if len(self.request_durations[path]) > 1000:
                    self.request_durations[path] = self.request_durations[path][-1000:]

            # Store metrics in Redis
            try:
                redis_client = request.app.state.redis_client

                # Increment endpoint counter
                await redis_client.hincrby(
                    "api:metrics:requests",
                    f"{request.method}:{request.url.path}",
                    1
                )

                # Store response time
                await redis_client.lpush(
                    f"api:metrics:duration:{request.method}:{request.url.path}",
                    str(duration_ms)
                )

                # Trim to last 1000 samples
                await redis_client.ltrim(
                    f"api:metrics:duration:{request.method}:{request.url.path}",
                    0,
                    999
                )

            except Exception as e:
                logger.warning("Metrics storage failed", error=str(e))

            return response

        except Exception as e:
            self.error_count += 1

            # Store error metric
            try:
                redis_client = request.app.state.redis_client
                await redis_client.hincrby(
                    "api:metrics:errors",
                    f"{request.method}:{request.url.path}",
                    1
                )
            except Exception:
                pass

            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics.

        Returns:
            Dictionary with metrics data
        """
        metrics = {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0.0
        }

        if self.collect_detailed:
            # Calculate average durations
            avg_durations = {}
            for path, durations in self.request_durations.items():
                if durations:
                    avg_durations[path] = sum(durations) / len(durations)

            metrics["avg_durations_ms"] = avg_durations

        return metrics


class CORSMiddleware(BaseHTTPMiddleware):
    """CORS middleware with dynamic origin validation.

    Handles Cross-Origin Resource Sharing with configurable rules.

    Attributes:
        config: Configuration dictionary
    """

    def __init__(self, app: ASGIApp, config: Dict[str, Any]) -> None:
        """Initialize CORS middleware.

        Args:
            app: ASGI application
            config: Configuration from config files
        """
        super().__init__(app)
        self.config = config

        # Load from config
        cors_config = config.get("api", {}).get("cors", {})
        self.allowed_origins = cors_config.get("allowed_origins", ["*"])
        self.allow_credentials = cors_config.get("allow_credentials", True)
        self.allowed_methods = cors_config.get("allowed_methods", ["*"])
        self.allowed_headers = cors_config.get("allowed_headers", ["*"])
        self.max_age = cors_config.get("max_age", 600)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with CORS handling.

        Args:
            request: HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response with CORS headers
        """
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response(status_code=200)
        else:
            response = await call_next(request)

        # Add CORS headers
        origin = request.headers.get("origin")

        if origin and self._is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin

            if self.allow_credentials:
                response.headers["Access-Control-Allow-Credentials"] = "true"

            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
            response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
            response.headers["Access-Control-Max-Age"] = str(self.max_age)

        return response

    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed.

        Args:
            origin: Request origin

        Returns:
            True if allowed
        """
        if "*" in self.allowed_origins:
            return True

        return origin in self.allowed_origins


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers middleware.

    Adds security-related HTTP headers to responses.

    Attributes:
        config: Configuration dictionary
    """

    def __init__(self, app: ASGIApp, config: Dict[str, Any]) -> None:
        """Initialize security headers middleware.

        Args:
            app: ASGI application
            config: Configuration from config files
        """
        super().__init__(app)
        self.config = config

        # Load from config
        security_config = config.get("api", {}).get("security_headers", {})
        self.enabled = security_config.get("enabled", True)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and add security headers.

        Args:
            request: HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response with security headers
        """
        response = await call_next(request)

        if self.enabled:
            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Content-Security-Policy"] = "default-src 'self'"

        return response
