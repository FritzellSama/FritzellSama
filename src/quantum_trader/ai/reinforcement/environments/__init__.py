"""Reinforcement learning trading environments.

This module provides OpenAI Gym-compatible environments for training
reinforcement learning agents on financial trading tasks.

Environments implement the standard Gym interface with reset() and step()
methods, enabling seamless integration with popular RL frameworks like
Stable-Baselines3, Ray RLlib, and PyTorch.
"""

from typing import Any

__all__ = [
    "TradingEnvironment",
    "MultiAssetEnvironment",
    "CryptoCurrencyEnvironment",
    "FuturesEnvironment",
]

# Placeholder exports - import actual classes when available
# from .trading_env import TradingEnvironment
# from .multi_asset_env import MultiAssetEnvironment
# from .crypto_env import CryptoCurrencyEnvironment
# from .futures_env import FuturesEnvironment

__version__ = "1.0.0"
