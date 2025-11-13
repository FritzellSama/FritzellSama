"""
Arbitrage trading strategy module.

This module implements arbitrage strategies that exploit price discrepancies
between correlated assets or markets. Strategies include statistical arbitrage,
cross-exchange arbitrage, and multi-leg arbitrage opportunities.

Features:
    - Real-time price monitoring across multiple markets
    - Statistical correlation analysis
    - Execution optimization for minimal slippage
    - Risk-adjusted position sizing
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .statistical import StatisticalArbitrage
    from .cross_exchange import CrossExchangeArbitrage
    from .multi_leg import MultiLegArbitrage

__version__: str = "1.0.0"

__all__: list[str] = [
    "StatisticalArbitrage",
    "CrossExchangeArbitrage",
    "MultiLegArbitrage",
]
