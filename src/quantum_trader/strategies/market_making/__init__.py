"""
Market making strategies module.

This module provides market making strategies designed to provide liquidity
while maintaining profitable bid-ask spreads. Strategies adapt to market volatility
and order flow patterns.

Features:
    - Dynamic spread management
    - Inventory risk management
    - Order flow prediction
    - Volatility-aware pricing
    - Execution optimization
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .classical import ClassicalMarketMaking
    from .inventory_based import InventoryBasedMarketMaking
    from .volatility_adjusted import VolatilityAdjustedMarketMaking

__version__: str = "1.0.0"

__all__: list[str] = [
    "ClassicalMarketMaking",
    "InventoryBasedMarketMaking",
    "VolatilityAdjustedMarketMaking",
]
