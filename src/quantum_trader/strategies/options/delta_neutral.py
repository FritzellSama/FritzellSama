"""
Delta Neutral Options Strategy.

Implements a delta-neutral options portfolio strategy that profits from volatility
while remaining directionally neutral to price movements.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import math

import polars as pl
from structlog import get_logger

from quantum_trader.models import Signal, SignalAction
from quantum_trader.strategies.base_strategy import BaseStrategy
from quantum_trader.risk.manager import RiskManager

logger = get_logger(__name__)


class DeltaNeutralStrategy(BaseStrategy):
    """Delta neutral options strategy.

    This strategy maintains a delta-neutral portfolio using options and underlying
    assets, profiting from volatility while hedging directional risk.

    Features:
    - Dynamic delta hedging
    - Gamma scalping
    - Volatility trading
    - Portfolio rebalancing

    Attributes:
        target_delta: Target portfolio delta (0 for perfect neutrality)
        rebalance_threshold: Delta threshold for rebalancing
        gamma_scalp_enabled: Whether to enable gamma scalping

    Example:
        >>> config = {
        ...     "target_delta": "0.0",
        ...     "rebalance_threshold": "0.1",
        ...     "gamma_scalp_enabled": True
        ... }
        >>> strategy = DeltaNeutralStrategy(config, risk_manager)
    """

    def __init__(self, config: Dict[str, Any], risk_manager: RiskManager) -> None:
        """Initialize delta neutral strategy.

        Args:
            config: Strategy configuration
            risk_manager: Risk manager instance

        Raises:
            ValueError: If required parameters missing
        """
        super().__init__(config, risk_manager)

        # Delta neutral parameters
        self.target_delta = Decimal(str(config.get("target_delta", "0.0")))
        self.rebalance_threshold = Decimal(str(config.get("rebalance_threshold", "0.1")))
        self.gamma_scalp_enabled = config.get("gamma_scalp_enabled", True)

        # Portfolio Greeks
        self.portfolio_delta: Dict[str, Decimal] = {}
        self.portfolio_gamma: Dict[str, Decimal] = {}
        self.portfolio_theta: Dict[str, Decimal] = {}
        self.portfolio_vega: Dict[str, Decimal] = {}

        # Position tracking
        self.option_positions: Dict[str, List[Dict]] = {}
        self.underlying_positions: Dict[str, Decimal] = {}
        self.hedge_ratios: Dict[str, Decimal] = {}

        # Rebalancing
        self.last_rebalance: Dict[str, datetime] = {}
        self.rebalance_count: Dict[str, int] = {}

        logger.info(
            "Delta neutral strategy initialized",
            strategy=self.name,
            target_delta=self.target_delta,
            rebalance_threshold=self.rebalance_threshold
        )

    def _validate_config(self) -> None:
        """Validate strategy configuration.

        Raises:
            ValueError: If parameters invalid
        """
        super()._validate_config()

        target = Decimal(str(self.config.get("target_delta", "0.0")))
        if abs(target) > Decimal("0.5"):
            raise ValueError(f"target_delta should be near 0 for neutrality: {target}")

        threshold = Decimal(str(self.config.get("rebalance_threshold", "0.1")))
        if threshold <= Decimal("0") or threshold > Decimal("1"):
            raise ValueError(f"rebalance_threshold must be between 0 and 1: {threshold}")

    def calculate_option_greeks(
        self,
        spot: Decimal,
        strike: Decimal,
        time_to_expiry: Decimal,
        volatility: Decimal,
        risk_free_rate: Decimal,
        option_type: str = "call"
    ) -> Dict[str, Decimal]:
        """Calculate option Greeks (delta, gamma, theta, vega).

        Args:
            spot: Current spot price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility
            risk_free_rate: Risk-free rate
            option_type: 'call' or 'put'

        Returns:
            Dictionary of Greeks
        """
        try:
            if time_to_expiry <= Decimal("0"):
                # At expiry
                if option_type == "call":
                    delta = Decimal("1") if spot > strike else Decimal("0")
                else:
                    delta = Decimal("-1") if spot < strike else Decimal("0")

                return {
                    "delta": delta,
                    "gamma": Decimal("0"),
                    "theta": Decimal("0"),
                    "vega": Decimal("0")
                }

            # Calculate d1 and d2
            sqrt_t = time_to_expiry.sqrt()
            d1 = (
                (spot / strike).ln() +
                (risk_free_rate + (volatility ** 2) / Decimal("2")) * time_to_expiry
            ) / (volatility * sqrt_t)

            d2 = d1 - volatility * sqrt_t

            # Normal PDF
            def norm_pdf(x: Decimal) -> Decimal:
                """Standard normal PDF."""
                return (Decimal("1") / (Decimal("2") * Decimal(str(math.pi))).sqrt()) * (-(x ** 2) / Decimal("2")).exp()

            # Normal CDF approximation
            def norm_cdf(x: Decimal) -> Decimal:
                """Standard normal CDF."""
                x_float = float(x)
                t = Decimal(str(1.0 / (1.0 + 0.2316419 * abs(x_float))))
                d = Decimal(str(0.3989423))
                prob = d * ((-float(x) ** 2) / 2).exp()

                b1, b2, b3, b4, b5 = [Decimal(str(v)) for v in
                    [0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429]]

                result = Decimal("1") - Decimal(str(prob)) * (
                    b1 * t + b2 * (t ** 2) + b3 * (t ** 3) + b4 * (t ** 4) + b5 * (t ** 5)
                )

                return result if x_float >= 0 else Decimal("1") - result

            # Calculate Greeks
            nd1 = norm_cdf(d1)
            nprime_d1 = norm_pdf(d1)

            # Delta
            if option_type == "call":
                delta = nd1
            else:
                delta = nd1 - Decimal("1")

            # Gamma (same for calls and puts)
            gamma = nprime_d1 / (spot * volatility * sqrt_t)

            # Theta
            term1 = -(spot * nprime_d1 * volatility) / (Decimal("2") * sqrt_t)
            term2 = risk_free_rate * strike * (-(risk_free_rate * time_to_expiry)).exp()

            if option_type == "call":
                theta = (term1 - term2 * norm_cdf(d2)) / Decimal("365")  # Daily theta
            else:
                theta = (term1 + term2 * norm_cdf(-d2)) / Decimal("365")

            # Vega (same for calls and puts)
            vega = (spot * nprime_d1 * sqrt_t) / Decimal("100")  # Per 1% volatility change

            greeks = {
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega
            }

            logger.debug("Greeks calculated", option_type=option_type, greeks={k: float(v) for k, v in greeks.items()})

            return greeks

        except Exception as e:
            logger.error("Error calculating Greeks", error=str(e))
            return {
                "delta": Decimal("0"),
                "gamma": Decimal("0"),
                "theta": Decimal("0"),
                "vega": Decimal("0")
            }

    def calculate_portfolio_greeks(
        self,
        symbol: str,
        spot: Decimal,
        volatility: Decimal
    ) -> Dict[str, Decimal]:
        """Calculate total portfolio Greeks.

        Args:
            symbol: Trading symbol
            spot: Current spot price
            volatility: Implied volatility

        Returns:
            Dictionary of portfolio Greeks
        """
        try:
            total_delta = Decimal("0")
            total_gamma = Decimal("0")
            total_theta = Decimal("0")
            total_vega = Decimal("0")

            # Add underlying position delta
            if symbol in self.underlying_positions:
                total_delta += self.underlying_positions[symbol]

            # Add option positions
            if symbol in self.option_positions:
                risk_free_rate = Decimal(str(self.config.get("risk_free_rate", "0.05")))

                for position in self.option_positions[symbol]:
                    strike = position["strike"]
                    expiry = position["expiry"]
                    option_type = position["type"]  # 'call' or 'put'
                    quantity = position["quantity"]

                    # Calculate time to expiry
                    now = datetime.now(timezone.utc)
                    days_to_expiry = (expiry - now).total_seconds() / 86400
                    time_to_expiry = max(Decimal("0"), Decimal(str(days_to_expiry)) / Decimal("365"))

                    # Calculate Greeks for this option
                    greeks = self.calculate_option_greeks(
                        spot, strike, time_to_expiry, volatility, risk_free_rate, option_type
                    )

                    # Add to portfolio totals
                    total_delta += greeks["delta"] * quantity
                    total_gamma += greeks["gamma"] * quantity
                    total_theta += greeks["theta"] * quantity
                    total_vega += greeks["vega"] * quantity

            portfolio_greeks = {
                "delta": total_delta,
                "gamma": total_gamma,
                "theta": total_theta,
                "vega": total_vega
            }

            # Update stored values
            self.portfolio_delta[symbol] = total_delta
            self.portfolio_gamma[symbol] = total_gamma
            self.portfolio_theta[symbol] = total_theta
            self.portfolio_vega[symbol] = total_vega

            logger.debug(
                "Portfolio Greeks calculated",
                symbol=symbol,
                greeks={k: float(v) for k, v in portfolio_greeks.items()}
            )

            return portfolio_greeks

        except Exception as e:
            logger.error("Error calculating portfolio Greeks", error=str(e))
            return {
                "delta": Decimal("0"),
                "gamma": Decimal("0"),
                "theta": Decimal("0"),
                "vega": Decimal("0")
            }

    def calculate_hedge_ratio(
        self,
        symbol: str,
        current_delta: Decimal
    ) -> Decimal:
        """Calculate hedge ratio to neutralize delta.

        Args:
            symbol: Trading symbol
            current_delta: Current portfolio delta

        Returns:
            Required position adjustment in underlying
        """
        try:
            # Calculate difference from target
            delta_diff = current_delta - self.target_delta

            # Hedge ratio is negative of delta difference
            # (if delta is +0.5, need to sell 0.5 underlying)
            hedge_ratio = -delta_diff

            logger.debug(
                "Hedge ratio calculated",
                symbol=symbol,
                current_delta=current_delta,
                target_delta=self.target_delta,
                hedge_ratio=hedge_ratio
            )

            return hedge_ratio

        except Exception as e:
            logger.error("Error calculating hedge ratio", error=str(e))
            return Decimal("0")

    def needs_rebalancing(
        self,
        symbol: str,
        current_delta: Decimal
    ) -> bool:
        """Check if portfolio needs rebalancing.

        Args:
            symbol: Trading symbol
            current_delta: Current portfolio delta

        Returns:
            True if rebalancing needed
        """
        try:
            delta_deviation = abs(current_delta - self.target_delta)

            needs_rebal = delta_deviation >= self.rebalance_threshold

            if needs_rebal:
                logger.info(
                    "Rebalancing needed",
                    symbol=symbol,
                    current_delta=current_delta,
                    target_delta=self.target_delta,
                    deviation=delta_deviation
                )

            return needs_rebal

        except Exception as e:
            logger.error("Error checking rebalancing", error=str(e))
            return False

    def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """Generate delta hedging signals.

        Args:
            market_data: DataFrame with OHLCV data

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

            # Get symbol
            symbols = self.config.get("symbols", [])
            if not symbols:
                logger.warning("No symbols configured")
                return signals

            symbol = symbols[0]

            # Get current spot price
            spot = Decimal(str(market_data["close"][-1]))

            # Calculate volatility
            volatility = self.calculate_volatility(market_data)

            # Calculate portfolio Greeks
            greeks = self.calculate_portfolio_greeks(symbol, spot, volatility)
            current_delta = greeks["delta"]

            # Check if rebalancing needed
            if self.needs_rebalancing(symbol, current_delta):
                # Calculate hedge ratio
                hedge_ratio = self.calculate_hedge_ratio(symbol, current_delta)

                # Determine action
                if hedge_ratio > Decimal("0"):
                    action = SignalAction.BUY
                else:
                    action = SignalAction.SELL

                signal = Signal(
                    symbol=symbol,
                    action=action,
                    strength=min(Decimal("1"), abs(hedge_ratio) / self.rebalance_threshold),
                    confidence=Decimal("0.9"),  # High confidence for delta hedging
                    timestamp=datetime.now(timezone.utc),
                    strategy=self.name,
                    timeframe="realtime",
                    indicators={
                        "portfolio_delta": current_delta,
                        "portfolio_gamma": greeks["gamma"],
                        "portfolio_theta": greeks["theta"],
                        "portfolio_vega": greeks["vega"],
                        "hedge_ratio": abs(hedge_ratio)
                    },
                    metadata={
                        "strategy_type": "options_hedging",
                        "hedge_type": "delta_neutral",
                        "current_spot": float(spot)
                    }
                )

                if self.validate_signal(signal):
                    signals.append(signal)

                    # Track rebalancing
                    self.last_rebalance[symbol] = datetime.now(timezone.utc)
                    self.rebalance_count[symbol] = self.rebalance_count.get(symbol, 0) + 1
                    self.hedge_ratios[symbol] = hedge_ratio

                    # Update state
                    self.state["signals_generated"] = self.state.get("signals_generated", 0) + 1
                    self.state["last_signal_time"] = datetime.now(timezone.utc)

            return signals

        except Exception as e:
            logger.error("Error generating signals", error=str(e))
            return signals

    def calculate_volatility(self, data: pl.DataFrame) -> Decimal:
        """Calculate historical volatility.

        Args:
            data: Historical price data

        Returns:
            Annualized volatility
        """
        try:
            if len(data) < 20:
                return Decimal(str(self.config.get("default_volatility", "0.5")))

            closes = data.select(pl.col("close")).to_series().to_list()[-20:]

            log_returns = []
            for i in range(1, len(closes)):
                if closes[i] > 0 and closes[i-1] > 0:
                    log_ret = (Decimal(str(closes[i])) / Decimal(str(closes[i-1]))).ln()
                    log_returns.append(float(log_ret))

            if not log_returns:
                return Decimal(str(self.config.get("default_volatility", "0.5")))

            mean = sum(log_returns) / len(log_returns)
            variance = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
            daily_vol = Decimal(str(math.sqrt(variance)))

            annual_vol = daily_vol * Decimal("365").sqrt()

            return annual_vol

        except Exception as e:
            logger.error("Error calculating volatility", error=str(e))
            return Decimal(str(self.config.get("default_volatility", "0.5")))

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate delta neutral indicators.

        Args:
            data: Market data

        Returns:
            Dictionary of indicators
        """
        try:
            indicators = {}

            for symbol in self.config.get("symbols", []):
                if symbol in self.portfolio_delta:
                    indicators[f"{symbol}_portfolio_delta"] = self.portfolio_delta[symbol]
                if symbol in self.portfolio_gamma:
                    indicators[f"{symbol}_portfolio_gamma"] = self.portfolio_gamma[symbol]
                if symbol in self.portfolio_theta:
                    indicators[f"{symbol}_portfolio_theta"] = self.portfolio_theta[symbol]
                if symbol in self.portfolio_vega:
                    indicators[f"{symbol}_portfolio_vega"] = self.portfolio_vega[symbol]

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

        dn_metrics = {
            "portfolio_delta": {symbol: float(delta) for symbol, delta in self.portfolio_delta.items()},
            "portfolio_gamma": {symbol: float(gamma) for symbol, gamma in self.portfolio_gamma.items()},
            "rebalance_count": self.rebalance_count,
            "hedge_ratios": {symbol: float(ratio) for symbol, ratio in self.hedge_ratios.items()}
        }

        return {**base_metrics, **dn_metrics}
