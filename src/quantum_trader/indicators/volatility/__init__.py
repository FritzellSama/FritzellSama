"""Volatility indicators module for quantum trader.

This module provides implementations of volatility indicators that measure price
fluctuation intensity and identify low/high volatility environments. These
indicators help with position sizing, stop loss placement, and strategy selection.

Included Indicators:
    - Bollinger Bands
    - ATR (Average True Range)
    - Keltner Channels
    - Donchian Channels
    - Historical Volatility (Std Dev)
    - Parkinson Volatility
    - Garman-Klass Volatility
    - VIX (Volatility Index)
    - Nadaraya-Watson Volatility Estimator
    - TRIX Volatility
    - Volatility Oscillator

Features:
    - Multiple volatility calculation methods
    - Real-time volatility updates
    - Volatility regime detection
    - Expansion/Contraction identification
    - Squeeze detection (Bollinger + Keltner)
    - Adaptive indicator parameters based on volatility
    - Historical volatility analysis
    - Forward-looking volatility estimation
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__: list[str] = [
    # Add volatility indicator classes here as they are implemented
]

__version__ = "1.0.0"
__author__ = "QuantumTrader Team"
