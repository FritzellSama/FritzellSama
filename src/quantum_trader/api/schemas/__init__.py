"""Request and response schemas for API endpoints.

This module defines Pydantic models for request validation and response
serialization. All API endpoints use these schemas to ensure data consistency,
type safety, and automatic OpenAPI documentation generation.

Schemas are organized by domain:
- Trading schemas for order and transaction data
- Portfolio schemas for position and asset information
- Backtesting schemas for simulation configuration and results
- Analytics schemas for metrics and reporting

Each schema includes validation rules, field descriptions, and example values
for API documentation.
"""

from typing import Optional, Dict, Any

__all__ = [
    "OrderSchema",
    "PortfolioSchema",
    "BacktestConfigSchema",
    "BacktestResultSchema",
    "AnalyticsMetricsSchema",
    "TradeSchema",
    "PositionSchema",
]

# Placeholder exports - import actual schemas when available
# from .trading import OrderSchema, TradeSchema
# from .portfolio import PortfolioSchema, PositionSchema
# from .backtesting import BacktestConfigSchema, BacktestResultSchema
# from .analytics import AnalyticsMetricsSchema

__version__ = "1.0.0"
