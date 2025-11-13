"""REST and GraphQL API for Quantum Trader.

This module provides the main API interface for the Quantum Trader application.
It includes REST endpoints, GraphQL API, WebSocket connections, and integrated
services for trading, analytics, and backtesting.

The API follows OpenAPI 3.0 specification for REST endpoints and supports
real-time communication via WebSocket for live market data and trading updates.

Typical usage:
    from quantum_trader.api import create_app

    app = create_app()
    # Use with Uvicorn or other ASGI server
"""

from typing import Any, Optional, Dict

__all__ = [
    "create_app",
    "APIConfig",
    "APIRouter",
    "APIService",
]

# Placeholder exports - import actual functions when available
# from .app import create_app
# from .config import APIConfig
# from .router import APIRouter
# from .service import APIService

__version__ = "1.0.0"
