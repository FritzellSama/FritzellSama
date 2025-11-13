"""Momentum indicators module for quantum trader.

This module provides implementations of momentum indicators that measure the rate
of price change and identify overbought/oversold conditions. These indicators help
identify trend strength and potential reversal points.

Included Indicators:
    - RSI (Relative Strength Index)
    - Stochastic Oscillator
    - Stochastic RSI
    - MACD (Moving Average Convergence Divergence)
    - CCI (Commodity Channel Index)
    - Williams %R
    - ROC (Rate of Change)
    - KDJ Indicator
    - TRIX (Triple Exponential Moving Average)
    - Momentum Oscillator

Features:
    - Fast numpy-based calculations
    - Configurable periods and parameters
    - Overbought/Oversold threshold detection
    - Divergence identification
    - Real-time streaming updates
    - Signal line generation and crossover detection
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__: list[str] = [
    # Add momentum indicator classes here as they are implemented
]

__version__ = "1.0.0"
__author__ = "QuantumTrader Team"
