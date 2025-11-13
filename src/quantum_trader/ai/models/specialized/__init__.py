"""Specialized neural network models for financial analysis.

This module provides domain-specific neural network architectures optimized
for financial time series analysis and prediction. Includes models designed
for specific use cases like price forecasting, volatility estimation, and
market regime detection.

All models inherit from a common base class and support standard PyTorch
operations and frameworks like Lightning.
"""

from typing import Dict, Optional

__all__ = [
    "LSTMPredictor",
    "GRUPredictor",
    "CNNPredictor",
    "AttentionPredictor",
    "EnsemblePredictor",
    "VolatilityModel",
    "RegimeDetector",
]

# Placeholder exports - import actual classes when available
# from .lstm_predictor import LSTMPredictor
# from .gru_predictor import GRUPredictor
# from .cnn_predictor import CNNPredictor
# from .attention_predictor import AttentionPredictor
# from .ensemble_predictor import EnsemblePredictor
# from .volatility_model import VolatilityModel
# from .regime_detector import RegimeDetector

__version__ = "1.0.0"
