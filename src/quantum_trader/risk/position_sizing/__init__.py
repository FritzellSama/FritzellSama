"""Position sizing and allocation strategies.

This submodule provides position sizing algorithms including:
- Fixed fractional sizing (Kelly criterion, fixed percentage)
- Volatility-adjusted sizing (inverse volatility weighting)
- Risk parity and equal-risk contribution methods
- Optimal f and portfolio optimization
- Dynamic sizing based on win rate and risk-reward ratios
- Leverage and margin management
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__: list[str] = []

__version__ = "1.0.0"
