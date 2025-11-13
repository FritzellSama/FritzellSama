"""Statistical features module.

This module provides statistical analysis-based features for trading signals,
including correlation analysis, distribution features, and statistical indicators.

Attributes:
    __all__: Public API exports for the statistical features module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__ = [
    "CorrelationAnalyzer",
    "DistributionFeatures",
    "StatisticalIndicator",
    "SkewnessKurtosisCalculator",
    "AutocorrelationAnalyzer",
]

__version__ = "1.0.0"
__author__ = "Quantum Trader AI"


def __getattr__(name: str):
    """Lazy import for statistical features."""
    if name == "CorrelationAnalyzer":
        from .correlation import CorrelationAnalyzer
        return CorrelationAnalyzer
    elif name == "DistributionFeatures":
        from .distribution import DistributionFeatures
        return DistributionFeatures
    elif name == "StatisticalIndicator":
        from .indicators import StatisticalIndicator
        return StatisticalIndicator
    elif name == "SkewnessKurtosisCalculator":
        from .moments import SkewnessKurtosisCalculator
        return SkewnessKurtosisCalculator
    elif name == "AutocorrelationAnalyzer":
        from .autocorrelation import AutocorrelationAnalyzer
        return AutocorrelationAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
