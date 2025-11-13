"""
High-frequency trading strategies module.

This module provides implementations of high-frequency trading (HFT) strategies
optimized for ultra-low latency execution and minimal market impact.

Features:
    - Microsecond-level order execution
    - Advanced latency optimization
    - Market microstructure analysis
    - Smart order routing
    - Real-time risk management
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .liquidity_provision import LiquidityProvision
    from .statistical_hft import StatisticalHFT
    from .latency_optimized import LatencyOptimized

__version__: str = "1.0.0"

__all__: list[str] = [
    "LiquidityProvision",
    "StatisticalHFT",
    "LatencyOptimized",
]
