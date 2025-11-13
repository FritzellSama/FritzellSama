"""
Momentum trading strategies module.

This module provides momentum-based strategies that identify and capitalize on
sustained price trends. Strategies include classical momentum, machine learning
enhanced momentum, and multi-timeframe momentum approaches.

Features:
    - Trend identification and validation
    - Momentum indicator analysis
    - Multi-timeframe trend confirmation
    - Machine learning pattern recognition
    - Dynamic stop-loss management
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .classical import ClassicalMomentum
    from .machine_learning import MLMomentum
    from .multi_timeframe import MultiTimeframeMomentum

__version__: str = "1.0.0"

__all__: list[str] = [
    "ClassicalMomentum",
    "MLMomentum",
    "MultiTimeframeMomentum",
]
