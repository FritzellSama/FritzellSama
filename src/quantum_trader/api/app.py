"""Main FastAPI application for Quantum Trader API.

This module provides the main FastAPI application with all routers,
middleware, exception handlers, and startup/shutdown event handlers.

Features:
- REST API endpoints for trading, analytics, and management
- WebSocket endpoints for real-time data streaming
- Authentication and authorization
- Rate limiting
- CORS support
- Health checks
- Prometheus metrics
- Structured logging
- OpenAPI documentation

Example:
    ```python
    # Run with uvicorn
    uvicorn quantum_trader.api.app:app --host 0.0.0.0 --port 8000

    # Or use the create_app factory
    from quantum_trader.api.app import create_app

    app = create_app(config)
    ```
"""

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
from typing import Any, AsyncIterator, Dict, Optional

import yaml
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from structlog import get_logger

from quantum_trader.api.routers.auth import router as auth_router
from quantum_trader.api.services.analytics_service import AnalyticsService
from quantum_trader.api.services.auth_service import AuthService
from quantum_trader.api.websocket.broadcasters import WebSocketBroadcaster

logger = get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"]
)

REQUEST_DURATION = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"]
)


class DecimalJSONResponse(JSONResponse):
    """JSON response that handles Decimal serialization."""

    def render(self, content: Any) -> bytes:
        """Render content to JSON bytes with Decimal support."""
        import json

        class DecimalEncoder(json.JSONEncoder):
            def default(self, obj: Any) -> Any:
                if isinstance(obj, Decimal):
                    return str(obj)
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return super().default(obj)

        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            cls=DecimalEncoder
        ).encode("utf-8")


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML files and environment variables.

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config files not found
    """
    # Determine environment
    environment = os.getenv("ENVIRONMENT", "development")

    # Load base bot config
    bot_config_path = os.getenv("BOT_CONFIG_PATH", "config/bot/bot.yaml")
    env_config_path = os.getenv("ENV_CONFIG_PATH", f"config/environments/{environment}.yaml")

    config = {}

    # Load bot config
    try:
        with open(bot_config_path, "r") as f:
            bot_config = yaml.safe_load(f)
            config.update(bot_config.get("bot", {}))
    except FileNotFoundError:
        logger.warning("bot_config_not_found", path=bot_config_path)

    # Load environment config
    try:
        with open(env_config_path, "r") as f:
            env_config = yaml.safe_load(f)
            config.update(env_config)
    except FileNotFoundError:
        logger.warning("env_config_not_found", path=env_config_path)

    # Override with environment variables
    _override_with_env(config)

    return config


def _override_with_env(config: Dict[str, Any]) -> None:
    """Override config values with environment variables."""

    def _recursive_override(obj: Any, prefix: str = "") -> Any:
        """Recursively override config with env vars."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                env_key = f"{prefix}{key}".upper()

                # Check if env var exists
                env_value = os.getenv(env_key)
                if env_value is not None:
                    # Try to preserve type
                    if isinstance(value, bool):
                        obj[key] = env_value.lower() in ("true", "1", "yes")
                    elif isinstance(value, int):
                        obj[key] = int(env_value)
                    elif isinstance(value, float):
                        obj[key] = float(env_value)
                    else:
                        obj[key] = env_value
                elif isinstance(value, dict):
                    _recursive_override(value, f"{env_key}_")

        return obj

    _recursive_override(config)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager for startup and shutdown events.

    Args:
        app: FastAPI application instance

    Yields:
        None during application lifetime
    """
    # Startup
    logger.info("api_starting", environment=app.state.config.get("environment"))

    try:
        # Initialize services
        logger.info("initializing_services")

        # Initialize auth service
        auth_config = {
            "jwt_secret": app.state.config.get("security", {}).get("jwt_secret", "default-secret-change-in-production-min-32-chars"),
            "encryption_key": app.state.config.get("security", {}).get("encryption_key", "default-key-change-in-production-base64"),
            "jwt_expiry_hours": app.state.config.get("security", {}).get("jwt_expiry_hours", 24),
            "max_login_attempts": app.state.config.get("security", {}).get("max_login_attempts", 5),
            "lockout_duration_minutes": app.state.config.get("security", {}).get("lockout_duration_minutes", 30),
            "enable_2fa": app.state.config.get("security", {}).get("enable_2fa", True),
        }
        app.state.auth_service = AuthService(auth_config)
        await app.state.auth_service.initialize()

        # Initialize analytics service
        analytics_config = {
            "risk_free_rate": app.state.config.get("analytics", {}).get("risk_free_rate", "0.02"),
            "var_confidence": app.state.config.get("analytics", {}).get("var_confidence", "0.95"),
            "trading_days_per_year": app.state.config.get("analytics", {}).get("trading_days_per_year", 365),
            "metrics_cache_seconds": app.state.config.get("analytics", {}).get("metrics_cache_seconds", 60),
        }
        app.state.analytics_service = AnalyticsService(analytics_config)
        await app.state.analytics_service.initialize()

        # Initialize WebSocket broadcaster
        ws_config = {
            "max_connections": app.state.config.get("websocket", {}).get("max_connections", 1000),
            "heartbeat_interval_seconds": app.state.config.get("websocket", {}).get("heartbeat_interval_seconds", 30),
            "message_queue_size": app.state.config.get("websocket", {}).get("message_queue_size", 10000),
        }
        app.state.broadcaster = WebSocketBroadcaster(ws_config)
        await app.state.broadcaster.initialize()

        logger.info("api_started")

        yield

    finally:
        # Shutdown
        logger.info("api_shutting_down")

        # Cleanup services
        if hasattr(app.state, "broadcaster"):
            await app.state.broadcaster.shutdown()

        if hasattr(app.state, "analytics_service"):
            await app.state.analytics_service.shutdown()

        if hasattr(app.state, "auth_service"):
            await app.state.auth_service.shutdown()

        logger.info("api_shutdown_complete")


def create_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured FastAPI application
    """
    # Load config if not provided
    if config is None:
        config = load_config()

    # Extract API config
    api_config = config.get("api", {})
    enable_docs = api_config.get("enable_docs", False)

    # Create FastAPI app
    app = FastAPI(
        title="Quantum Trader AI API",
        description="Enterprise-grade algorithmic trading platform API",
        version="1.0.0",
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
        openapi_url="/openapi.json" if enable_docs else None,
        lifespan=lifespan,
        default_response_class=DecimalJSONResponse
    )

    # Store config in app state
    app.state.config = config

    # Add middleware
    _configure_middleware(app, config)

    # Add exception handlers
    _configure_exception_handlers(app)

    # Add routers
    _configure_routers(app)

    # Add WebSocket endpoints
    _configure_websockets(app)

    # Add health check endpoints
    _configure_health_checks(app)

    return app


def _configure_middleware(app: FastAPI, config: Dict[str, Any]) -> None:
    """Configure middleware for the application."""

    # CORS middleware
    api_config = config.get("api", {})
    cors_origins = api_config.get("cors_origins", "*")

    if cors_origins == "*":
        cors_origins = ["*"]
    elif isinstance(cors_origins, str):
        cors_origins = [origin.strip() for origin in cors_origins.split(",")]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # GZip compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Request logging and metrics middleware
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next: Any) -> Response:
        """Middleware for request metrics and logging."""
        start_time = asyncio.get_event_loop().time()

        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            logger.error("request_error", path=request.url.path, error=str(e))
            status_code = 500
            raise

        # Record metrics
        duration = asyncio.get_event_loop().time() - start_time

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=status_code
        ).inc()

        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)

        # Log request
        logger.info(
            "api_request",
            method=request.method,
            path=request.url.path,
            status=status_code,
            duration_ms=int(duration * 1000)
        )

        return response


def _configure_exception_handlers(app: FastAPI) -> None:
    """Configure exception handlers."""

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError
    ) -> JSONResponse:
        """Handle validation errors."""
        logger.warning(
            "validation_error",
            path=request.url.path,
            errors=exc.errors()
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation error",
                "details": exc.errors()
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """Handle general exceptions."""
        logger.error(
            "unhandled_exception",
            path=request.url.path,
            error=str(exc),
            exc_info=True
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred"
            }
        )


def _configure_routers(app: FastAPI) -> None:
    """Configure API routers."""

    # Include auth router
    app.include_router(auth_router, prefix="/api/v1")

    # Additional routers would be added here:
    # app.include_router(trading_router, prefix="/api/v1")
    # app.include_router(analytics_router, prefix="/api/v1")
    # app.include_router(strategies_router, prefix="/api/v1")


def _configure_websockets(app: FastAPI) -> None:
    """Configure WebSocket endpoints."""

    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str) -> None:
        """WebSocket endpoint for real-time data streaming.

        Args:
            websocket: WebSocket connection
            client_id: Unique client identifier

        Example:
            ```javascript
            const ws = new WebSocket('ws://localhost:8000/ws/client123');

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log('Received:', data);
            };

            // Subscribe to topics
            ws.send(JSON.stringify({
                action: 'subscribe',
                topic: 'market_data'
            }));
            ```
        """
        broadcaster = app.state.broadcaster

        try:
            # Connect client
            await broadcaster.connect_client(client_id, websocket)

            # Send welcome message
            await broadcaster.send_to_client(
                client_id=client_id,
                message_type="welcome",
                data={
                    "message": "Connected to Quantum Trader WebSocket",
                    "client_id": client_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

            # Listen for messages
            while True:
                try:
                    # Receive message from client
                    data = await websocket.receive_json()

                    action = data.get("action")

                    if action == "subscribe":
                        topic = data.get("topic")
                        if topic:
                            await broadcaster.subscribe_to_topic(client_id, topic)
                            await broadcaster.send_to_client(
                                client_id=client_id,
                                message_type="subscribed",
                                data={"topic": topic}
                            )

                    elif action == "unsubscribe":
                        topic = data.get("topic")
                        if topic:
                            await broadcaster.unsubscribe_from_topic(client_id, topic)
                            await broadcaster.send_to_client(
                                client_id=client_id,
                                message_type="unsubscribed",
                                data={"topic": topic}
                            )

                    elif action == "ping":
                        await broadcaster.send_to_client(
                            client_id=client_id,
                            message_type="pong",
                            data={"timestamp": datetime.utcnow().isoformat()}
                        )

                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error("websocket_message_error", client_id=client_id, error=str(e))
                    break

        except Exception as e:
            logger.error("websocket_connection_error", client_id=client_id, error=str(e))

        finally:
            # Disconnect client
            await broadcaster.disconnect_client(client_id)


def _configure_health_checks(app: FastAPI) -> None:
    """Configure health check endpoints."""

    @app.get("/health", status_code=status.HTTP_200_OK)
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint.

        Returns:
            Health status information
        """
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }

    @app.get("/health/ready", status_code=status.HTTP_200_OK)
    async def readiness_check(request: Request) -> Dict[str, Any]:
        """Readiness check endpoint.

        Returns:
            Readiness status with service checks
        """
        services = {}

        # Check auth service
        if hasattr(request.app.state, "auth_service"):
            services["auth"] = "ready"
        else:
            services["auth"] = "not_ready"

        # Check analytics service
        if hasattr(request.app.state, "analytics_service"):
            services["analytics"] = "ready"
        else:
            services["analytics"] = "not_ready"

        # Check broadcaster
        if hasattr(request.app.state, "broadcaster"):
            services["websocket"] = "ready"
        else:
            services["websocket"] = "not_ready"

        all_ready = all(status == "ready" for status in services.values())

        return {
            "status": "ready" if all_ready else "not_ready",
            "services": services,
            "timestamp": datetime.utcnow().isoformat()
        }

    @app.get("/metrics", status_code=status.HTTP_200_OK)
    async def metrics() -> Response:
        """Prometheus metrics endpoint.

        Returns:
            Prometheus metrics in text format
        """
        return Response(
            content=generate_latest(),
            media_type="text/plain"
        )

    @app.get("/stats", status_code=status.HTTP_200_OK)
    async def stats(request: Request) -> Dict[str, Any]:
        """Statistics endpoint.

        Returns:
            Application statistics
        """
        stats_data = {
            "timestamp": datetime.utcnow().isoformat()
        }

        # Add broadcaster stats
        if hasattr(request.app.state, "broadcaster"):
            stats_data["websocket"] = request.app.state.broadcaster.get_stats()

        return stats_data


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    # Load config
    config = load_config()
    api_config = config.get("api", {})

    # Run server
    uvicorn.run(
        "quantum_trader.api.app:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        workers=api_config.get("workers", 1),
        log_level="info",
        reload=False
    )
