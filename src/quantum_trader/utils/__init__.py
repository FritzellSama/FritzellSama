"""
Utilities module for quantum trader AI.

This module provides core utility functions and helpers for the trading platform,
including data processing, mathematical operations, technical indicators,
risk calculations, and performance analytics.

Submodules:
    - math: Mathematical and statistical operations
    - data: Data processing and manipulation utilities
    - indicators: Technical indicator calculations
    - risk: Risk metrics and calculations
    - performance: Performance and attribution analysis
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import math as math_utils
    from . import data as data_utils
    from . import indicators as indicators_utils
    from . import risk as risk_utils
    from . import performance as performance_utils

__version__: str = "1.0.0"
__author__: str = "Quantum Trader AI Team"

__all__: list[str] = [
    "math_utils",
    "data_utils",
    "indicators_utils",
    "risk_utils",
    "performance_utils",
]
