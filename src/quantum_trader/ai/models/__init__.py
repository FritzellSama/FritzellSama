"""AI models module.

This module provides access to various machine learning and AI model implementations,
including classical ML models, deep learning networks, graph neural networks,
and reinforcement learning agents.

Attributes:
    __all__: Public API exports for the models module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .classical_ml import ClassicalMLModel
    from .deep_learning import DeepLearningModel
    from .graph_neural import GraphNeuralNetwork
    from .regime_detection import RegimeDetector

__all__ = [
    "ClassicalMLModel",
    "DeepLearningModel",
    "GraphNeuralNetwork",
    "RegimeDetector",
    "BaseModel",
    "ModelFactory",
]

__version__ = "1.0.0"
__author__ = "Quantum Trader AI"


def __getattr__(name: str):
    """Lazy import for model implementations."""
    if name == "ClassicalMLModel":
        from .classical_ml import ClassicalMLModel
        return ClassicalMLModel
    elif name == "DeepLearningModel":
        from .deep_learning import DeepLearningModel
        return DeepLearningModel
    elif name == "GraphNeuralNetwork":
        from .graph_neural import GraphNeuralNetwork
        return GraphNeuralNetwork
    elif name == "RegimeDetector":
        from .regime_detection import RegimeDetector
        return RegimeDetector
    elif name == "BaseModel":
        from .base import BaseModel
        return BaseModel
    elif name == "ModelFactory":
        from .factory import ModelFactory
        return ModelFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
