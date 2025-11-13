"""
Options trading strategies module.

This module implements sophisticated options trading strategies including
spreads, straddles, strangles, and volatility-based approaches. Strategies
incorporate Greeks management and advanced options analytics.

Features:
    - Options pricing models (Black-Scholes, Binomial)
    - Greeks calculation and hedging
    - Volatility smile analysis
    - Synthetic position creation
    - Risk-adjusted P&L management
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .spreads import SpreadsStrategy
    from .volatility_strategies import VolatilityStrategy
    from .greek_hedging import GreekHedging

__version__: str = "1.0.0"

__all__: list[str] = [
    "SpreadsStrategy",
    "VolatilityStrategy",
    "GreekHedging",
]
