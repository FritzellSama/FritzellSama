"""Classical machine learning models module.

This module provides classical machine learning model implementations,
including ensemble methods, linear models, tree-based models, and SVM variants.

Attributes:
    __all__: Public API exports for the classical ML models module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ensemble import RandomForestModel, GradientBoostingModel
    from .linear import LinearRegressionModel, LogisticRegressionModel
    from .svm import SVMModel
    from .neighbors import KNearestNeighborsModel

__all__ = [
    "ClassicalMLModel",
    "RandomForestModel",
    "GradientBoostingModel",
    "LinearRegressionModel",
    "LogisticRegressionModel",
    "SVMModel",
    "KNearestNeighborsModel",
    "TreeBasedModel",
]

__version__ = "1.0.0"
__author__ = "Quantum Trader AI"


def __getattr__(name: str):
    """Lazy import for classical ML models."""
    if name == "ClassicalMLModel":
        from .base import ClassicalMLModel
        return ClassicalMLModel
    elif name == "RandomForestModel":
        from .ensemble import RandomForestModel
        return RandomForestModel
    elif name == "GradientBoostingModel":
        from .ensemble import GradientBoostingModel
        return GradientBoostingModel
    elif name == "LinearRegressionModel":
        from .linear import LinearRegressionModel
        return LinearRegressionModel
    elif name == "LogisticRegressionModel":
        from .linear import LogisticRegressionModel
        return LogisticRegressionModel
    elif name == "SVMModel":
        from .svm import SVMModel
        return SVMModel
    elif name == "KNearestNeighborsModel":
        from .neighbors import KNearestNeighborsModel
        return KNearestNeighborsModel
    elif name == "TreeBasedModel":
        from .tree import TreeBasedModel
        return TreeBasedModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
