"""Risk metrics calculation and analysis module.

This submodule provides computational functions for risk metrics including:
- Value at Risk (VaR) - parametric, historical, Monte Carlo methods
- Conditional Value at Risk (CVaR) / Expected Shortfall
- Performance ratios (Sharpe, Sortino, Calmar, Information ratio)
- Drawdown analysis and recovery metrics
- Volatility forecasting and GARCH models
- Correlation and covariance matrix analysis
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__: list[str] = []

__version__ = "1.0.0"
