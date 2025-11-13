"""Trend indicators module for quantum trader.

This module provides implementations of trend-following indicators that identify
and confirm the direction of price movement. These indicators help traders stay
aligned with the primary market trend.

Included Indicators:
    - SMA (Simple Moving Average)
    - EMA (Exponential Moving Average)
    - WMA (Weighted Moving Average)
    - DEMA (Double Exponential Moving Average)
    - TEMA (Triple Exponential Moving Average)
    - HMA (Hull Moving Average)
    - ADX (Average Directional Index)
    - ADXR (ADX Rating)
    - DI+ and DI- (Directional Indicators)
    - Ichimoku Cloud
    - Parabolic SAR (Stop and Reverse)
    - SuperTrend Indicator
    - Zigzag Pattern Detection

Features:
    - Multiple moving average types
    - Efficient recursive calculations
    - Trend strength quantification
    - Support/Resistance level identification
    - Trend change detection and confirmation
    - Real-time streaming support
    - Trend direction and slope analysis
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__: list[str] = [
    # Add trend indicator classes here as they are implemented
]

__version__ = "1.0.0"
__author__ = "QuantumTrader Team"
