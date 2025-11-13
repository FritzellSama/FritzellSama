"""
Mean reversion trading strategies module.

This module implements mean reversion strategies that capitalize on price movements
that deviate from historical averages. Strategies include statistical mean reversion,
Bollinger Band approaches, and machine learning enhanced variants.

Features:
    - Statistical mean reversion detection
    - Bollinger Band analysis
    - Z-score based signals
    - Machine learning enhancements
    - Risk-adjusted position sizing
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .statistical import StatisticalMeanReversion
    from .bollinger_bands import BollingerBandsMeanReversion
    from .machine_learning import MLMeanReversion

__version__: str = "1.0.0"

__all__: list[str] = [
    "StatisticalMeanReversion",
    "BollingerBandsMeanReversion",
    "MLMeanReversion",
]
