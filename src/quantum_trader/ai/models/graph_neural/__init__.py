"""Graph neural networks module.

This module provides graph neural network implementations for analyzing market structure
and relationships, including GCN, GAT, GraphSAGE, and message passing layers.

Attributes:
    __all__: Public API exports for the graph neural networks module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gcn import GraphConvolutionalNetwork
    from .gat import GraphAttentionNetwork
    from .graphsage import GraphSAGE
    from .message_passing import MessagePassingLayer

__all__ = [
    "GraphNeuralNetwork",
    "GraphConvolutionalNetwork",
    "GraphAttentionNetwork",
    "GraphSAGE",
    "MessagePassingLayer",
    "NodeEmbedding",
]

__version__ = "1.0.0"
__author__ = "Quantum Trader AI"


def __getattr__(name: str):
    """Lazy import for graph neural networks."""
    if name == "GraphNeuralNetwork":
        from .base import GraphNeuralNetwork
        return GraphNeuralNetwork
    elif name == "GraphConvolutionalNetwork":
        from .gcn import GraphConvolutionalNetwork
        return GraphConvolutionalNetwork
    elif name == "GraphAttentionNetwork":
        from .gat import GraphAttentionNetwork
        return GraphAttentionNetwork
    elif name == "GraphSAGE":
        from .graphsage import GraphSAGE
        return GraphSAGE
    elif name == "MessagePassingLayer":
        from .message_passing import MessagePassingLayer
        return MessagePassingLayer
    elif name == "NodeEmbedding":
        from .embeddings import NodeEmbedding
        return NodeEmbedding
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
