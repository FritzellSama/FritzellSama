"""
Dual Momentum Strategy.

Implements Gary Antonacci's dual momentum strategy that combines absolute
and relative momentum for enhanced risk-adjusted returns.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

import polars as pl
from structlog import get_logger

from quantum_trader.models import Signal, SignalAction
from quantum_trader.strategies.base_strategy import BaseStrategy
from quantum_trader.risk.manager import RiskManager

logger = get_logger(__name__)


class DualMomentumStrategy(BaseStrategy):
    """Dual momentum trading strategy.

    This strategy combines:
    1. Absolute Momentum: Compare asset returns to risk-free rate
    2. Relative Momentum: Compare asset returns across multiple assets

    Only invests in assets showing both positive absolute momentum AND
    outperforming relative momentum compared to peers.

    Features:
    - Multi-asset relative strength ranking
    - Absolute momentum filter
    - Dynamic risk management
    - Trend-following with defensive positioning

    Attributes:
        lookback_period: Momentum calculation period
        risk_free_rate: Risk-free rate for absolute momentum
        top_n_assets: Number of top assets to hold

    Example:
        >>> config = {
        ...     "lookback_period": "252",  # 1 year daily
        ...     "risk_free_rate": "0.05",
        ...     "top_n_assets": "1"
        ... }
        >>> strategy = DualMomentumStrategy(config, risk_manager)
    """

    def __init__(self, config: Dict[str, Any], risk_manager: RiskManager) -> None:
        """Initialize dual momentum strategy.

        Args:
            config: Strategy configuration
            risk_manager: Risk manager instance

        Raises:
            ValueError: If required parameters missing
        """
        super().__init__(config, risk_manager)

        # Momentum parameters
        self.lookback_period = int(config.get("lookback_period", 252))  # ~1 year
        self.risk_free_rate = Decimal(str(config.get("risk_free_rate", "0.05")))
        self.top_n_assets = int(config.get("top_n_assets", 1))

        # Ranking parameters
        self.rebalance_frequency = int(config.get("rebalance_frequency", 21))  # Monthly
        self.min_momentum = Decimal(str(config.get("min_momentum", "0.0")))

        # State tracking
        self.asset_returns: Dict[str, Decimal] = {}
        self.asset_rankings: List[Tuple[str, Decimal]] = []
        self.last_rebalance: Optional[datetime] = None
        self.current_holdings: List[str] = []

        logger.info(
            "Dual momentum strategy initialized",
            strategy=self.name,
            lookback_period=self.lookback_period,
            top_n_assets=self.top_n_assets
        )

    def _validate_config(self) -> None:
        """Validate strategy configuration.

        Raises:
            ValueError: If parameters invalid
        """
        super()._validate_config()

        lookback = int(self.config.get("lookback_period", 252))
        if lookback < 20 or lookback > 1000:
            raise ValueError(f"lookback_period must be between 20 and 1000: {lookback}")

        top_n = int(self.config.get("top_n_assets", 1))
        if top_n < 1:
            raise ValueError(f"top_n_assets must be at least 1: {top_n}")

        rebal_freq = int(self.config.get("rebalance_frequency", 21))
        if rebal_freq < 1:
            raise ValueError(f"rebalance_frequency must be positive: {rebal_freq}")

    def calculate_momentum(
        self,
        data: pl.DataFrame
    ) -> Decimal:
        """Calculate momentum score (total return over lookback period).

        Args:
            data: DataFrame with price data

        Returns:
            Momentum score as Decimal (total return)
        """
        try:
            if len(data) < self.lookback_period:
                logger.warning(
                    "Insufficient data for momentum calculation",
                    required=self.lookback_period,
                    available=len(data)
                )
                return Decimal("0")

            # Get prices
            closes = data.select(pl.col("close")).to_series().to_list()

            # Calculate total return over lookback period
            start_price = Decimal(str(closes[-self.lookback_period]))
            end_price = Decimal(str(closes[-1]))

            if start_price <= Decimal("0"):
                return Decimal("0")

            momentum = (end_price - start_price) / start_price

            logger.debug(
                "Momentum calculated",
                start_price=start_price,
                end_price=end_price,
                momentum=momentum
            )

            return momentum

        except Exception as e:
            logger.error("Error calculating momentum", error=str(e))
            return Decimal("0")

    def check_absolute_momentum(
        self,
        momentum: Decimal
    ) -> bool:
        """Check if asset passes absolute momentum filter.

        Args:
            momentum: Asset momentum score

        Returns:
            True if passes absolute momentum filter
        """
        try:
            # Annualize the risk-free rate for comparison
            # Assuming lookback is in days
            days_per_year = Decimal("365")
            annualized_lookback = Decimal(str(self.lookback_period)) / days_per_year

            # Expected risk-free return over lookback period
            risk_free_return = self.risk_free_rate * annualized_lookback

            # Asset must outperform risk-free rate
            passes = momentum > risk_free_return

            logger.debug(
                "Absolute momentum check",
                momentum=momentum,
                risk_free_return=risk_free_return,
                passes=passes
            )

            return passes

        except Exception as e:
            logger.error("Error checking absolute momentum", error=str(e))
            return False

    def rank_assets(
        self,
        market_data_dict: Dict[str, pl.DataFrame]
    ) -> List[Tuple[str, Decimal]]:
        """Rank assets by relative momentum.

        Args:
            market_data_dict: Dictionary mapping symbol to market data

        Returns:
            List of (symbol, momentum) tuples sorted by momentum (descending)
        """
        try:
            rankings = []

            for symbol, data in market_data_dict.items():
                momentum = self.calculate_momentum(data)
                self.asset_returns[symbol] = momentum

                # Check absolute momentum
                if self.check_absolute_momentum(momentum):
                    rankings.append((symbol, momentum))
                else:
                    logger.debug(
                        "Asset failed absolute momentum filter",
                        symbol=symbol,
                        momentum=momentum
                    )

            # Sort by momentum (descending)
            rankings.sort(key=lambda x: x[1], reverse=True)

            logger.info(
                "Assets ranked",
                total_assets=len(market_data_dict),
                passing_abs_momentum=len(rankings),
                top_asset=rankings[0][0] if rankings else None
            )

            self.asset_rankings = rankings

            return rankings

        except Exception as e:
            logger.error("Error ranking assets", error=str(e))
            return []

    def needs_rebalancing(self) -> bool:
        """Check if portfolio needs rebalancing.

        Returns:
            True if rebalancing needed
        """
        try:
            if self.last_rebalance is None:
                return True

            days_since_rebalance = (datetime.now(timezone.utc) - self.last_rebalance).days

            needs_rebal = days_since_rebalance >= self.rebalance_frequency

            if needs_rebal:
                logger.info(
                    "Rebalancing needed",
                    days_since_rebalance=days_since_rebalance,
                    frequency=self.rebalance_frequency
                )

            return needs_rebal

        except Exception as e:
            logger.error("Error checking rebalancing", error=str(e))
            return False

    def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """Generate dual momentum signals.

        Note: This strategy is designed for multi-asset portfolios.
        The market_data parameter represents data for a single asset.
        For full functionality, use with multiple assets.

        Args:
            market_data: DataFrame with OHLCV data for single asset

        Returns:
            List of Signal objects
        """
        signals: List[Signal] = []

        try:
            # Validate data
            is_valid, error = self.validate_market_data(market_data)
            if not is_valid:
                logger.warning("Invalid market data", error=error)
                return signals

            if not self.active:
                return signals

            # Check if rebalancing needed
            if not self.needs_rebalancing():
                logger.debug("Rebalancing not needed yet")
                return signals

            # Get symbols
            symbols = self.config.get("symbols", [])
            if not symbols:
                logger.warning("No symbols configured")
                return signals

            # For this implementation, we'll analyze the single provided asset
            # In production, this would analyze multiple assets
            symbol = symbols[0]

            # Calculate momentum
            momentum = self.calculate_momentum(market_data)
            self.asset_returns[symbol] = momentum

            # Check absolute momentum
            passes_absolute = self.check_absolute_momentum(momentum)

            if passes_absolute and momentum >= self.min_momentum:
                # Generate BUY signal
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    strength=min(Decimal("1"), momentum / Decimal("0.5")),  # Normalize to 0-1
                    confidence=Decimal("0.75"),
                    timestamp=datetime.now(timezone.utc),
                    strategy=self.name,
                    timeframe=f"{self.lookback_period}d",
                    indicators={
                        "momentum": momentum,
                        "absolute_momentum": passes_absolute,
                        "risk_free_rate": self.risk_free_rate
                    },
                    metadata={
                        "strategy_type": "momentum",
                        "momentum_type": "dual",
                        "lookback_days": self.lookback_period
                    }
                )

                if self.validate_signal(signal):
                    signals.append(signal)

                    logger.info(
                        "Dual momentum BUY signal",
                        symbol=symbol,
                        momentum=momentum,
                        absolute_pass=passes_absolute
                    )

                    # Update holdings
                    if symbol not in self.current_holdings:
                        self.current_holdings.append(symbol)

            elif symbol in self.current_holdings:
                # Generate CLOSE signal if currently holding
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.CLOSE,
                    strength=Decimal("1.0"),
                    confidence=Decimal("0.8"),
                    timestamp=datetime.now(timezone.utc),
                    strategy=self.name,
                    timeframe=f"{self.lookback_period}d",
                    indicators={
                        "momentum": momentum,
                        "absolute_momentum": passes_absolute,
                        "risk_free_rate": self.risk_free_rate
                    },
                    metadata={
                        "strategy_type": "momentum",
                        "reason": "failed_absolute_momentum" if not passes_absolute else "weak_momentum"
                    }
                )

                if self.validate_signal(signal):
                    signals.append(signal)

                    logger.info(
                        "Dual momentum CLOSE signal",
                        symbol=symbol,
                        momentum=momentum,
                        reason="Failed absolute momentum or weak momentum"
                    )

                    # Remove from holdings
                    self.current_holdings.remove(symbol)

            # Update rebalance time
            self.last_rebalance = datetime.now(timezone.utc)

            # Update state
            if signals:
                self.state["signals_generated"] = self.state.get("signals_generated", 0) + len(signals)
                self.state["last_signal_time"] = datetime.now(timezone.utc)

            return signals

        except Exception as e:
            logger.error("Error generating signals", error=str(e))
            return signals

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate dual momentum indicators.

        Args:
            data: Market data

        Returns:
            Dictionary of indicators
        """
        try:
            if len(data) < self.lookback_period:
                return {}

            momentum = self.calculate_momentum(data)
            passes_absolute = self.check_absolute_momentum(momentum)

            indicators = {
                "momentum": momentum,
                "absolute_momentum_pass": Decimal("1" if passes_absolute else "0"),
                "risk_free_rate": self.risk_free_rate,
                "lookback_period": Decimal(str(self.lookback_period))
            }

            # Add relative rankings if available
            if self.asset_rankings:
                indicators["num_ranked_assets"] = Decimal(str(len(self.asset_rankings)))
                if self.asset_rankings:
                    indicators["top_asset_momentum"] = self.asset_rankings[0][1]

            return indicators

        except Exception as e:
            logger.error("Error calculating indicators", error=str(e))
            return {}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        base_metrics = super().get_performance_metrics()

        dm_metrics = {
            "asset_returns": {symbol: float(ret) for symbol, ret in self.asset_returns.items()},
            "asset_rankings": [(symbol, float(momentum)) for symbol, momentum in self.asset_rankings],
            "current_holdings": self.current_holdings,
            "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None
        }

        return {**base_metrics, **dm_metrics}
