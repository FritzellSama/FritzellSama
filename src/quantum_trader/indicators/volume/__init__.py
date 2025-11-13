"""Volume indicators module for quantum trader.

This module provides implementations of volume-based indicators that analyze
trading volume patterns and identify potential price movements. These indicators
are crucial for confirming trends and identifying strong support/resistance levels.

Included Indicators:
    - OBV (On-Balance Volume)
    - VWAP (Volume Weighted Average Price)
    - TWAP (Time Weighted Average Price)
    - Volume Profile
    - Market Profile
    - CMF (Chaikin Money Flow)
    - AD (Accumulation/Distribution Line)
    - MFI (Money Flow Index)
    - Volume Rate of Change
    - Klinger Volume Oscillator
    - Ease of Movement
    - Force Index

Features:
    - Multiple volume analysis methods
    - Volume profile generation
    - Market profile analysis
    - Volume-weighted price calculations
    - Money flow direction detection
    - Accumulation/Distribution tracking
    - Volume confirmation signals
    - Real-time volume stream processing
    - Historical volume analysis
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__: list[str] = [
    # Add volume indicator classes here as they are implemented
]

__version__ = "1.0.0"
__author__ = "QuantumTrader Team"
