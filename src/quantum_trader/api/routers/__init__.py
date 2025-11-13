"""API route definitions for Quantum Trader.

This module defines all REST API endpoints for the Quantum Trader application.
Routes are organized by functionality area: trading, portfolio, backtesting,
analytics, and market data.

Each router is a FastAPI APIRouter instance that can be included in the main
application. Routes follow RESTful conventions and include comprehensive
documentation, request validation, and error handling.

Route organization:
- /api/v1/trading - Trading operations
- /api/v1/portfolio - Portfolio management
- /api/v1/backtesting - Backtesting operations
- /api/v1/analytics - Analytics and reporting
- /api/v1/market - Market data access
"""

from typing import List

__all__ = [
    "trading_router",
    "portfolio_router",
    "backtesting_router",
    "analytics_router",
    "market_router",
    "create_routers",
]

# Placeholder exports - import actual routers when available
# from .trading import trading_router
# from .portfolio import portfolio_router
# from .backtesting import backtesting_router
# from .analytics import analytics_router
# from .market import market_router
# from .factory import create_routers

__version__ = "1.0.0"
