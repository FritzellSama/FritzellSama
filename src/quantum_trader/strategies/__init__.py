"""
Trading strategies module for quantum trader AI.

This module provides a collection of algorithmic trading strategies including
arbitrage, high-frequency trading, hybrid approaches, market making, mean reversion,
momentum, options, and order flow analysis strategies.

Strategies are designed to be modular, type-safe, and production-ready with
comprehensive error handling and performance optimization.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .arbitrage import ArbitrageStrategy
    from .high_frequency import HighFrequencyStrategy
    from .hybrid import HybridStrategy
    from .market_making import MarketMakingStrategy
    from .mean_reversion import MeanReversionStrategy
    from .momentum import MomentumStrategy
    from .options import OptionsStrategy
    from .order_flow import OrderFlowStrategy

__version__: str = "1.0.0"
__author__: str = "Quantum Trader AI Team"

__all__: list[str] = [
    "ArbitrageStrategy",
    "HighFrequencyStrategy",
    "HybridStrategy",
    "MarketMakingStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "OptionsStrategy",
    "OrderFlowStrategy",
]
