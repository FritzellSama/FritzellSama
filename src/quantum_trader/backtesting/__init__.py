"""Backtesting framework for quantitative trading strategies.

This module provides a comprehensive backtesting engine for evaluating trading
strategies on historical market data. It supports multiple asset classes
(equities, futures, cryptocurrencies), handles realistic constraints
(slippage, commissions, margin), and generates detailed performance reports.

Features:
- High-performance OHLCV data processing
- Realistic transaction cost modeling
- Portfolio position tracking and P&L calculation
- Risk metrics computation (Sharpe, Sortino, max drawdown)
- Trade-by-trade analysis and equity curve visualization
- Portfolio rebalancing strategies
- Monte Carlo analysis and sensitivity testing

Typical usage:
    from quantum_trader.backtesting import Backtest, Strategy

    class MyStrategy(Strategy):
        def next(self):
            pass

    bt = Backtest(data, MyStrategy)
    stats = bt.run()

The backtesting engine is designed for speed and accuracy, supporting both
event-driven and bar-replay simulation modes.
"""

from typing import Optional, Dict, Any, List

__all__ = [
    "Backtest",
    "Strategy",
    "Portfolio",
    "PerformanceMetrics",
    "BacktestResult",
    "run_backtest",
]

# Placeholder exports - import actual classes when available
# from .engine import Backtest
# from .strategy import Strategy
# from .portfolio import Portfolio
# from .metrics import PerformanceMetrics
# from .result import BacktestResult
# from .runner import run_backtest

__version__ = "1.0.0"
