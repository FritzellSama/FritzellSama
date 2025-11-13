"""Optimization module.

This module provides optimization algorithms and techniques for hyperparameter tuning,
model optimization, and portfolio optimization, including Bayesian optimization and
genetic algorithms.

Attributes:
    __all__: Public API exports for the optimization module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bayesian import BayesianOptimizer
    from .genetic import GeneticAlgorithm
    from .hyperparameter import HyperparameterTuner
    from .portfolio import PortfolioOptimizer

__all__ = [
    "Optimizer",
    "BayesianOptimizer",
    "GeneticAlgorithm",
    "HyperparameterTuner",
    "PortfolioOptimizer",
    "ObjectiveFunction",
]

__version__ = "1.0.0"
__author__ = "Quantum Trader AI"


def __getattr__(name: str):
    """Lazy import for optimization components."""
    if name == "Optimizer":
        from .base import Optimizer
        return Optimizer
    elif name == "BayesianOptimizer":
        from .bayesian import BayesianOptimizer
        return BayesianOptimizer
    elif name == "GeneticAlgorithm":
        from .genetic import GeneticAlgorithm
        return GeneticAlgorithm
    elif name == "HyperparameterTuner":
        from .hyperparameter import HyperparameterTuner
        return HyperparameterTuner
    elif name == "PortfolioOptimizer":
        from .portfolio import PortfolioOptimizer
        return PortfolioOptimizer
    elif name == "ObjectiveFunction":
        from .objectives import ObjectiveFunction
        return ObjectiveFunction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
