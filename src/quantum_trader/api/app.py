"""
Main FastAPI application for Quantum Trader AI.

Provides REST API and WebSocket endpoints for institutional trading platform.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
from structlog import get_logger

from quantum_trader.api.routers import auth, orders, market_data, monitoring, ml_models
from quantum_trader.api.middleware import LoggingMiddleware, RateLimitMiddleware, MetricsMiddleware
from quantum_trader.api.exceptions import QuantumTraderException
from quantum_trader.api.dependencies import get_config, get_db_pool, get_redis_client

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Handles startup and shutdown of application resources:
    - Database connection pools
    - Redis connections
    - WebSocket connection managers
    - Background tasks

    Args:
        app: FastAPI application instance

    Yields:
        None
    """
    # Startup
    logger.info("Starting Quantum Trader AI API")

    try:
        # Load configuration
        config = await get_config()
        app.state.config = config

        # Initialize database pool
        logger.info("Initializing database connection pool")
        db_pool = await get_db_pool(config)
        app.state.db_pool = db_pool

        # Initialize Redis client
        logger.info("Initializing Redis client")
        redis_client = await get_redis_client(config)
        app.state.redis_client = redis_client

        # Initialize services
        from quantum_trader.api.services.auth_service import AuthService
        from quantum_trader.api.services.data_service import DataService
        from quantum_trader.api.services.analytics_service import AnalyticsService

        app.state.auth_service = AuthService(config, db_pool, redis_client)
        app.state.data_service = DataService(config, db_pool, redis_client)
        app.state.analytics_service = AnalyticsService(config, db_pool, redis_client)

        # Initialize WebSocket connection manager
        from quantum_trader.api.websocket.connection_manager import ConnectionManager
        app.state.ws_manager = ConnectionManager(config)

        logger.info("Quantum Trader AI API started successfully")

        yield

    finally:
        # Shutdown
        logger.info("Shutting down Quantum Trader AI API")

        # Close WebSocket connections
        if hasattr(app.state, "ws_manager"):
            await app.state.ws_manager.disconnect_all()

        # Close Redis connection
        if hasattr(app.state, "redis_client"):
            await app.state.redis_client.close()
            logger.info("Redis connection closed")

        # Close database pool
        if hasattr(app.state, "db_pool"):
            await app.state.db_pool.close()
            logger.info("Database connection pool closed")

        logger.info("Quantum Trader AI API shutdown complete")


def create_app(config: Dict[str, Any]) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        config: Configuration dictionary from YAML files

    Returns:
        Configured FastAPI application instance

    Example:
        >>> config = load_config()
        >>> app = create_app(config)
        >>> # Run with: uvicorn app:app --host 0.0.0.0 --port 8000
    """
    # Create FastAPI app
    app = FastAPI(
        title=config.get("api", {}).get("title", "Quantum Trader AI API"),
        description=config.get("api", {}).get("description", "Institutional Trading Platform API"),
        version=config.get("api", {}).get("version", "1.0.0"),
        docs_url=config.get("api", {}).get("docs_url", "/docs"),
        redoc_url=config.get("api", {}).get("redoc_url", "/redoc"),
        openapi_url=config.get("api", {}).get("openapi_url", "/openapi.json"),
        lifespan=lifespan
    )

    # Configure CORS
    cors_config = config.get("api", {}).get("cors", {})
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.get("allow_origins", ["*"]),
        allow_credentials=cors_config.get("allow_credentials", True),
        allow_methods=cors_config.get("allow_methods", ["*"]),
        allow_headers=cors_config.get("allow_headers", ["*"]),
    )

    # Add custom middleware
    app.add_middleware(MetricsMiddleware, config=config)
    app.add_middleware(RateLimitMiddleware, config=config)
    app.add_middleware(LoggingMiddleware, config=config)

    # Include routers
    api_prefix = config.get("api", {}).get("prefix", "/api/v1")

    app.include_router(
        auth.router,
        prefix=f"{api_prefix}/auth",
        tags=["Authentication"]
    )

    app.include_router(
        orders.router,
        prefix=f"{api_prefix}/orders",
        tags=["Orders"]
    )

    app.include_router(
        market_data.router,
        prefix=f"{api_prefix}/market-data",
        tags=["Market Data"]
    )

    app.include_router(
        monitoring.router,
        prefix=f"{api_prefix}/monitoring",
        tags=["Monitoring"]
    )

    app.include_router(
        ml_models.router,
        prefix=f"{api_prefix}/ml-models",
        tags=["ML Models"]
    )

    # Exception handlers
    @app.exception_handler(QuantumTraderException)
    async def quantum_trader_exception_handler(
        request: Request,
        exc: QuantumTraderException
    ) -> JSONResponse:
        """Handle custom Quantum Trader exceptions.

        Args:
            request: FastAPI request
            exc: Custom exception

        Returns:
            JSON response with error details
        """
        logger.error(
            "Quantum Trader exception",
            error=str(exc),
            status_code=exc.status_code,
            path=request.url.path
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "details": exc.details
            }
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions.

        Args:
            request: FastAPI request
            exc: Exception

        Returns:
            JSON response with error details
        """
        logger.error(
            "Unhandled exception",
            error=str(exc),
            error_type=type(exc).__name__,
            path=request.url.path
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": {}
            }
        )

    # Health check endpoint
    @app.get(f"{api_prefix}/health")
    async def health_check() -> Dict[str, str]:
        """Health check endpoint.

        Returns:
            Dictionary with health status
        """
        return {
            "status": "healthy",
            "service": "quantum-trader-api",
            "version": config.get("api", {}).get("version", "1.0.0")
        }

    # Root endpoint
    @app.get("/")
    async def root() -> Dict[str, str]:
        """Root endpoint.

        Returns:
            Welcome message
        """
        return {
            "message": "Quantum Trader AI API",
            "docs": config.get("api", {}).get("docs_url", "/docs"),
            "version": config.get("api", {}).get("version", "1.0.0")
        }

    logger.info(
        "FastAPI application created",
        title=config.get("api", {}).get("title"),
        version=config.get("api", {}).get("version")
    )

    return app
