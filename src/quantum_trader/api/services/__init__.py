"""Business logic services for API operations.

This module contains service classes that implement the core business logic
for trading, portfolio management, analytics, and backtesting operations.
Services handle data processing, model execution, and interaction with
lower-level modules.

Services follow the separation of concerns principle:
- TradingService handles order creation, execution, and management
- PortfolioService manages positions, allocations, and valuation
- AnalyticsService computes metrics and generates reports
- BacktestingService orchestrates backtesting workflows
- DataService provides market data access and caching

Services use dependency injection for database connections, caches, and
external service clients.
"""

from typing import Optional, Dict, Any

__all__ = [
    "TradingService",
    "PortfolioService",
    "AnalyticsService",
    "BacktestingService",
    "DataService",
    "ServiceFactory",
]

# Placeholder exports - import actual services when available
# from .trading import TradingService
# from .portfolio import PortfolioService
# from .analytics import AnalyticsService
# from .backtesting import BacktestingService
# from .data import DataService
# from .factory import ServiceFactory

__version__ = "1.0.0"
