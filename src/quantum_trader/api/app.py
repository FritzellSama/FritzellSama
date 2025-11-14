"""
FastAPI Application

Production-ready FastAPI application for Quantum Trader AI.
Provides REST API and WebSocket endpoints for trading operations.
"""

import asyncio
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import os

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn
from structlog import get_logger

logger = get_logger(__name__)

# Set high precision
getcontext().prec = 28


def create_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        config: Configuration dictionary

    Returns:
        Configured FastAPI application

    Example:
        >>> app = create_app(config)
        >>> uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    # Load config from environment if not provided
    if config is None:
        config = {
            "api": {
                "title": os.getenv("API_TITLE", "Quantum Trader AI API"),
                "version": os.getenv("API_VERSION", "1.0.0"),
                "description": os.getenv("API_DESCRIPTION", "Institutional Trading Platform API"),
                "debug": os.getenv("API_DEBUG", "false").lower() == "true"
            }
        }

    api_config = config.get("api", {})

    # Create FastAPI app
    app = FastAPI(
        title=api_config.get("title", "Quantum Trader AI API"),
        version=api_config.get("version", "1.0.0"),
        description=api_config.get("description", "Institutional Trading Platform API"),
        debug=api_config.get("debug", False),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    # Configure CORS
    cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    # Add GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Exception handlers
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions."""
        logger.error(
            "http_exception",
            path=request.url.path,
            status_code=exc.status_code,
            detail=exc.detail
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors."""
        logger.error(
            "validation_error",
            path=request.url.path,
            errors=exc.errors()
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation error",
                "details": exc.errors(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(
            "unhandled_exception",
            path=request.url.path,
            error=str(exc),
            exc_type=type(exc).__name__
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "message": str(exc) if api_config.get("debug") else "An error occurred",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    # Lifecycle events
    @app.on_event("startup")
    async def startup_event():
        """Initialize application on startup."""
        logger.info(
            "api_starting",
            title=app.title,
            version=app.version
        )

        # Initialize database connections, redis, etc.
        # app.state.db = await init_database()
        # app.state.redis = await init_redis()

        logger.info("api_started")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        logger.info("api_shutting_down")

        # Close database connections, redis, etc.
        # await app.state.db.close()
        # await app.state.redis.close()

        logger.info("api_shutdown_complete")

    # Health check endpoint
    @app.get("/health", tags=["health"])
    async def health_check():
        """
        Health check endpoint.

        Returns:
            Health status of the application

        Example:
            GET /health
        """
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": app.version
        }

    @app.get("/", tags=["root"])
    async def root():
        """
        Root endpoint.

        Returns:
            API information
        """
        return {
            "name": app.title,
            "version": app.version,
            "description": app.description,
            "docs": "/docs",
            "health": "/health"
        }

    # Include routers
    # Note: Import routers here to avoid circular imports
    try:
        from quantum_trader.api.routers import analytics
        app.include_router(analytics.router)
        logger.info("analytics_router_included")
    except ImportError as e:
        logger.warning("analytics_router_not_found", error=str(e))

    try:
        from quantum_trader.api.routers import auth
        app.include_router(auth.router)
        logger.info("auth_router_included")
    except ImportError as e:
        logger.warning("auth_router_not_found", error=str(e))

    # Store config in app state
    app.state.config = config

    logger.info(
        "fastapi_app_created",
        title=app.title,
        version=app.version
    )

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1
) -> None:
    """
    Run the API server.

    Args:
        host: Host address
        port: Port number
        reload: Enable auto-reload
        workers: Number of worker processes

    Example:
        >>> run_server(host="0.0.0.0", port=8000, workers=4)
    """
    # Load config from environment
    host = os.getenv("API_HOST", host)
    port = int(os.getenv("API_PORT", str(port)))
    reload = os.getenv("API_RELOAD", str(reload)).lower() == "true"
    workers = int(os.getenv("API_WORKERS", str(workers)))

    logger.info(
        "starting_api_server",
        host=host,
        port=port,
        reload=reload,
        workers=workers
    )

    uvicorn.run(
        "quantum_trader.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=os.getenv("API_ACCESS_LOG", "true").lower() == "true"
    )


# Create default app instance
app = create_app()


if __name__ == "__main__":
    run_server()
