"""Online learning module.

This module provides online learning implementations for real-time model updates,
including streaming data processing, incremental learning, and adaptive models.

Attributes:
    __all__: Public API exports for the online learning module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .streaming import StreamingModel
    from .incremental import IncrementalLearner
    from .adaptive import AdaptiveModel
    from .concept_drift import ConceptDriftDetector

__all__ = [
    "OnlineLearner",
    "StreamingModel",
    "IncrementalLearner",
    "AdaptiveModel",
    "ConceptDriftDetector",
    "FeatureScaler",
]

__version__ = "1.0.0"
__author__ = "Quantum Trader AI"


def __getattr__(name: str):
    """Lazy import for online learning components."""
    if name == "OnlineLearner":
        from .base import OnlineLearner
        return OnlineLearner
    elif name == "StreamingModel":
        from .streaming import StreamingModel
        return StreamingModel
    elif name == "IncrementalLearner":
        from .incremental import IncrementalLearner
        return IncrementalLearner
    elif name == "AdaptiveModel":
        from .adaptive import AdaptiveModel
        return AdaptiveModel
    elif name == "ConceptDriftDetector":
        from .concept_drift import ConceptDriftDetector
        return ConceptDriftDetector
    elif name == "FeatureScaler":
        from .scaling import FeatureScaler
        return FeatureScaler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
