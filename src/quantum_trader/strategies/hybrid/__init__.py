"""
Hybrid trading strategies module.

This module combines multiple trading approaches into unified strategies that
leverage the strengths of different methodologies. Hybrid strategies adapt
dynamically to changing market conditions.

Features:
    - Multi-strategy portfolio optimization
    - Dynamic strategy switching based on market regimes
    - Ensemble prediction methods
    - Adaptive risk management
    - Correlation-aware position sizing
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ensemble import EnsembleStrategy
    from .regime_adaptive import RegimeAdaptiveStrategy
    from .multi_timeframe import MultiTimeframeStrategy

__version__: str = "1.0.0"

__all__: list[str] = [
    "EnsembleStrategy",
    "RegimeAdaptiveStrategy",
    "MultiTimeframeStrategy",
]
