"""Technical analysis features module.

This module provides technical analysis-based features for trading signals,
including momentum indicators, trend indicators, and volatility measures.

Attributes:
    __all__: Public API exports for the technical analysis features module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__ = [
    "MomentumIndicators",
    "TrendIndicators",
    "VolatilityIndicators",
    "CandlestickPatterns",
    "MovingAverageCalculator",
    "OscillatorAnalyzer",
]

__version__ = "1.0.0"
__author__ = "Quantum Trader AI"


def __getattr__(name: str):
    """Lazy import for technical analysis features."""
    if name == "MomentumIndicators":
        from .momentum import MomentumIndicators
        return MomentumIndicators
    elif name == "TrendIndicators":
        from .trend import TrendIndicators
        return TrendIndicators
    elif name == "VolatilityIndicators":
        from .volatility import VolatilityIndicators
        return VolatilityIndicators
    elif name == "CandlestickPatterns":
        from .patterns import CandlestickPatterns
        return CandlestickPatterns
    elif name == "MovingAverageCalculator":
        from .moving_averages import MovingAverageCalculator
        return MovingAverageCalculator
    elif name == "OscillatorAnalyzer":
        from .oscillators import OscillatorAnalyzer
        return OscillatorAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
