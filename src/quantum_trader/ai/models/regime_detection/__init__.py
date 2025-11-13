"""Regime detection models module.

This module provides regime detection implementations for identifying market conditions
and trading regimes, including Hidden Markov Models, clustering-based methods,
and statistical regime detection.

Attributes:
    __all__: Public API exports for the regime detection module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hmm import HiddenMarkovModel
    from .clustering import ClusteringBasedRegimeDetector
    from .statistical import StatisticalRegimeDetector
    from .volatility import VolatilityRegimeDetector

__all__ = [
    "RegimeDetector",
    "HiddenMarkovModel",
    "ClusteringBasedRegimeDetector",
    "StatisticalRegimeDetector",
    "VolatilityRegimeDetector",
    "RegimeState",
]

__version__ = "1.0.0"
__author__ = "Quantum Trader AI"


def __getattr__(name: str):
    """Lazy import for regime detection models."""
    if name == "RegimeDetector":
        from .base import RegimeDetector
        return RegimeDetector
    elif name == "HiddenMarkovModel":
        from .hmm import HiddenMarkovModel
        return HiddenMarkovModel
    elif name == "ClusteringBasedRegimeDetector":
        from .clustering import ClusteringBasedRegimeDetector
        return ClusteringBasedRegimeDetector
    elif name == "StatisticalRegimeDetector":
        from .statistical import StatisticalRegimeDetector
        return StatisticalRegimeDetector
    elif name == "VolatilityRegimeDetector":
        from .volatility import VolatilityRegimeDetector
        return VolatilityRegimeDetector
    elif name == "RegimeState":
        from .state import RegimeState
        return RegimeState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
