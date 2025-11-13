"""Deep learning models module.

This module provides deep learning model implementations using neural networks,
including CNNs, RNNs, LSTMs, Transformers, and attention mechanisms.

Attributes:
    __all__: Public API exports for the deep learning models module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cnn import ConvolutionalNeuralNetwork
    from .rnn import RecurrentNeuralNetwork
    from .lstm import LongShortTermMemory
    from .transformer import TransformerModel
    from .attention import AttentionLayer

__all__ = [
    "DeepLearningModel",
    "ConvolutionalNeuralNetwork",
    "RecurrentNeuralNetwork",
    "LongShortTermMemory",
    "TransformerModel",
    "AttentionLayer",
    "AutoEncoder",
]

__version__ = "1.0.0"
__author__ = "Quantum Trader AI"


def __getattr__(name: str):
    """Lazy import for deep learning models."""
    if name == "DeepLearningModel":
        from .base import DeepLearningModel
        return DeepLearningModel
    elif name == "ConvolutionalNeuralNetwork":
        from .cnn import ConvolutionalNeuralNetwork
        return ConvolutionalNeuralNetwork
    elif name == "RecurrentNeuralNetwork":
        from .rnn import RecurrentNeuralNetwork
        return RecurrentNeuralNetwork
    elif name == "LongShortTermMemory":
        from .lstm import LongShortTermMemory
        return LongShortTermMemory
    elif name == "TransformerModel":
        from .transformer import TransformerModel
        return TransformerModel
    elif name == "AttentionLayer":
        from .attention import AttentionLayer
        return AttentionLayer
    elif name == "AutoEncoder":
        from .autoencoder import AutoEncoder
        return AutoEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
