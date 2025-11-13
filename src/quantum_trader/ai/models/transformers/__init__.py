"""Transformer-based models for trading.

This module provides state-of-the-art transformer architectures adapted for
financial time series analysis. Includes Vision Transformers, Temporal Fusion
Transformers, and other attention-based architectures proven effective for
market analysis and price prediction.

Transformers excel at capturing long-range dependencies and are highly
parallelizable, making them suitable for large-scale financial data analysis.
"""

from typing import Optional, Dict

__all__ = [
    "TransformerEncoder",
    "TransformerDecoder",
    "TemporalFusionTransformer",
    "VisionTransformer",
    "InformerModel",
    "PatchTST",
]

# Placeholder exports - import actual classes when available
# from .transformer_encoder import TransformerEncoder
# from .transformer_decoder import TransformerDecoder
# from .tft_model import TemporalFusionTransformer
# from .vision_transformer import VisionTransformer
# from .informer_model import InformerModel
# from .patch_tst import PatchTST

__version__ = "1.0.0"
