"""
Covered Calls Options Strategy.

Implements a covered call strategy that sells call options against long stock
positions to generate income from option premiums.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
import math

import polars as pl
from structlog import get_logger

from quantum_trader.models import Signal, SignalAction, Order, OrderSide, OrderType
from quantum_trader.strategies.base_strategy import BaseStrategy
from quantum_trader.risk.manager import RiskManager

logger = get_logger(__name__)


class CoveredCallsStrategy(BaseStrategy):
    """Covered calls options income strategy.

    This strategy generates income by selling out-of-the-money call options
    against long underlying positions. It's designed for neutral to bullish
    market conditions.

    Features:
    - Automatic strike selection based on delta
    - Expiry management and rolling
    - Premium target optimization
    - Risk management for assignment

    Attributes:
        target_delta: Target delta for sold calls (typically 0.2-0.3)
        min_premium: Minimum acceptable premium percentage
        days_to_expiry: Preferred days to expiration

    Example:
        >>> config = {
        ...     "target_delta": "0.25",
        ...     "min_premium": "0.01",
        ...     "days_to_expiry": "30"
        ... }
        >>> strategy = CoveredCallsStrategy(config, risk_manager)
    """

    def __init__(self, config: Dict[str, Any], risk_manager: RiskManager) -> None:
        """Initialize covered calls strategy.

        Args:
            config: Strategy configuration
            risk_manager: Risk manager instance

        Raises:
            ValueError: If required parameters missing
        """
        super().__init__(config, risk_manager)

        # Covered call parameters
        self.target_delta = Decimal(str(config.get("target_delta", "0.25")))  # 25 delta
        self.min_premium = Decimal(str(config.get("min_premium", "0.01")))  # 1% minimum
        self.days_to_expiry = int(config.get("days_to_expiry", 30))
        self.max_days_to_expiry = int(config.get("max_days_to_expiry", 45))

        # Strike selection parameters
        self.strike_otm_percent = Decimal(str(config.get("strike_otm_percent", "0.05")))  # 5% OTM
        self.roll_days_before_expiry = int(config.get("roll_days_before_expiry", 7))
        self.roll_profit_target = Decimal(str(config.get("roll_profit_target", "0.5")))  # 50% of premium

        # Position tracking
        self.underlying_positions: Dict[str, Decimal] = {}
        self.option_positions: Dict[str, List[Dict]] = {}
        self.covered_ratios: Dict[str, Decimal] = {}

        logger.info(
            "Covered calls strategy initialized",
            strategy=self.name,
            target_delta=self.target_delta,
            days_to_expiry=self.days_to_expiry
        )

    def _validate_config(self) -> None:
        """Validate strategy configuration.

        Raises:
            ValueError: If parameters invalid
        """
        super()._validate_config()

        delta = Decimal(str(self.config.get("target_delta", "0.25")))
        if delta <= Decimal("0") or delta >= Decimal("1"):
            raise ValueError(f"target_delta must be between 0 and 1: {delta}")

        min_prem = Decimal(str(self.config.get("min_premium", "0.01")))
        if min_prem <= Decimal("0"):
            raise ValueError(f"min_premium must be positive: {min_prem}")

        dte = int(self.config.get("days_to_expiry", 30))
        if dte < 1 or dte > 365:
            raise ValueError(f"days_to_expiry must be between 1 and 365: {dte}")

    def calculate_black_scholes_call(
        self,
        spot: Decimal,
        strike: Decimal,
        time_to_expiry: Decimal,
        volatility: Decimal,
        risk_free_rate: Decimal
    ) -> Tuple[Decimal, Decimal]:
        """Calculate Black-Scholes call option price and delta.

        Args:
            spot: Current spot price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility (annualized)
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Tuple of (call_price, delta)
        """
        try:
            if time_to_expiry <= Decimal("0"):
                # At expiry
                intrinsic = max(Decimal("0"), spot - strike)
                delta = Decimal("1") if spot > strike else Decimal("0")
                return intrinsic, delta

            # Calculate d1 and d2
            sqrt_t = time_to_expiry.sqrt()
            d1 = (
                (spot / strike).ln() +
                (risk_free_rate + (volatility ** 2) / Decimal("2")) * time_to_expiry
            ) / (volatility * sqrt_t)

            d2 = d1 - volatility * sqrt_t

            # Standard normal CDF approximation
            def norm_cdf(x: Decimal) -> Decimal:
                """Approximate standard normal CDF."""
                x_float = float(x)
                # Using error function approximation
                t = Decimal(str(1.0 / (1.0 + 0.2316419 * abs(x_float))))
                d = Decimal(str(0.3989423))
                prob = d * ((-float(x) ** 2) / 2).exp()

                b1 = Decimal("0.319381530")
                b2 = Decimal("-0.356563782")
                b3 = Decimal("1.781477937")
                b4 = Decimal("-1.821255978")
                b5 = Decimal("1.330274429")

                result = Decimal("1") - Decimal(str(prob)) * (
                    b1 * t + b2 * (t ** 2) + b3 * (t ** 3) + b4 * (t ** 4) + b5 * (t ** 5)
                )

                if x_float < 0:
                    result = Decimal("1") - result

                return result

            # Calculate call price
            nd1 = norm_cdf(d1)
            nd2 = norm_cdf(d2)

            call_price = spot * nd1 - strike * (-(risk_free_rate * time_to_expiry)).exp() * nd2

            # Delta is N(d1)
            delta = nd1

            logger.debug(
                "Black-Scholes calculated",
                spot=spot,
                strike=strike,
                call_price=call_price,
                delta=delta
            )

            return call_price, delta

        except Exception as e:
            logger.error("Error in Black-Scholes calculation", error=str(e))
            # Return conservative estimates
            intrinsic = max(Decimal("0"), spot - strike)
            return intrinsic, Decimal("0.5")

    def calculate_implied_volatility(self, data: pl.DataFrame) -> Decimal:
        """Calculate implied volatility from historical data.

        Args:
            data: Historical price data

        Returns:
            Annualized volatility as Decimal
        """
        try:
            if len(data) < 30:
                return Decimal(str(self.config.get("default_volatility", "0.5")))

            closes = data.select(pl.col("close")).to_series().to_list()[-30:]

            # Calculate log returns
            log_returns = []
            for i in range(1, len(closes)):
                if closes[i] > 0 and closes[i-1] > 0:
                    log_ret = (Decimal(str(closes[i])) / Decimal(str(closes[i-1]))).ln()
                    log_returns.append(float(log_ret))

            if not log_returns:
                return Decimal(str(self.config.get("default_volatility", "0.5")))

            # Calculate standard deviation
            mean = sum(log_returns) / len(log_returns)
            variance = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
            daily_vol = Decimal(str(math.sqrt(variance)))

            # Annualize (sqrt of trading days)
            annual_vol = daily_vol * Decimal("365").sqrt()

            logger.debug("Implied volatility calculated", volatility=annual_vol)

            return annual_vol

        except Exception as e:
            logger.error("Error calculating IV", error=str(e))
            return Decimal(str(self.config.get("default_volatility", "0.5")))

    def select_strike_price(
        self,
        spot: Decimal,
        volatility: Decimal,
        time_to_expiry: Decimal
    ) -> Decimal:
        """Select optimal strike price for covered call.

        Args:
            spot: Current spot price
            volatility: Implied volatility
            time_to_expiry: Time to expiry in years

        Returns:
            Selected strike price
        """
        try:
            risk_free_rate = Decimal(str(self.config.get("risk_free_rate", "0.05")))

            # Start with OTM target
            initial_strike = spot * (Decimal("1") + self.strike_otm_percent)

            # Search for strike with target delta
            strike = initial_strike
            best_strike = initial_strike
            min_delta_diff = Decimal("1")

            # Search range
            for i in range(-5, 6):
                test_strike = initial_strike * (Decimal("1") + Decimal(str(i)) * Decimal("0.01"))

                _, delta = self.calculate_black_scholes_call(
                    spot, test_strike, time_to_expiry, volatility, risk_free_rate
                )

                delta_diff = abs(delta - self.target_delta)
                if delta_diff < min_delta_diff:
                    min_delta_diff = delta_diff
                    best_strike = test_strike

            logger.debug(
                "Strike selected",
                spot=spot,
                strike=best_strike,
                otm_percent=((best_strike - spot) / spot)
            )

            return best_strike.quantize(Decimal("0.01"))

        except Exception as e:
            logger.error("Error selecting strike", error=str(e))
            return (spot * (Decimal("1") + self.strike_otm_percent)).quantize(Decimal("0.01"))

    def calculate_premium_yield(
        self,
        premium: Decimal,
        spot: Decimal,
        days_to_expiry: int
    ) -> Decimal:
        """Calculate annualized premium yield.

        Args:
            premium: Option premium
            spot: Spot price
            days_to_expiry: Days to expiration

        Returns:
            Annualized yield as Decimal
        """
        try:
            if spot <= Decimal("0") or days_to_expiry <= 0:
                return Decimal("0")

            # Calculate yield for this period
            period_yield = premium / spot

            # Annualize
            periods_per_year = Decimal("365") / Decimal(str(days_to_expiry))
            annual_yield = period_yield * periods_per_year

            return annual_yield

        except Exception as e:
            logger.error("Error calculating yield", error=str(e))
            return Decimal("0")

    def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """Generate covered call signals.

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

            # Calculate implied volatility
            iv = self.calculate_implied_volatility(market_data)

            # Time to expiry in years
            time_to_expiry = Decimal(str(self.days_to_expiry)) / Decimal("365")

            # Select strike price
            strike = self.select_strike_price(spot, iv, time_to_expiry)

            # Calculate option price and delta
            risk_free_rate = Decimal(str(self.config.get("risk_free_rate", "0.05")))
            call_price, delta = self.calculate_black_scholes_call(
                spot, strike, time_to_expiry, iv, risk_free_rate
            )

            # Calculate premium yield
            annual_yield = self.calculate_premium_yield(call_price, spot, self.days_to_expiry)

            # Check if premium meets minimum
            premium_pct = call_price / spot

            if premium_pct >= self.min_premium:
                # Generate signal to sell covered call
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.SELL,  # Sell call option
                    strength=min(Decimal("1"), premium_pct / self.min_premium),
                    confidence=Decimal("0.7"),
                    timestamp=datetime.now(timezone.utc),
                    strategy=self.name,
                    timeframe=f"{self.days_to_expiry}d",
                    indicators={
                        "spot": spot,
                        "strike": strike,
                        "call_price": call_price,
                        "delta": delta,
                        "iv": iv,
                        "premium_pct": premium_pct,
                        "annual_yield": annual_yield
                    },
                    metadata={
                        "strategy_type": "options",
                        "option_type": "covered_call",
                        "days_to_expiry": self.days_to_expiry,
                        "otm_percent": float((strike - spot) / spot)
                    }
                )

                if self.validate_signal(signal):
                    signals.append(signal)

                    logger.info(
                        "Covered call signal generated",
                        symbol=symbol,
                        spot=spot,
                        strike=strike,
                        premium=call_price,
                        yield_annual=annual_yield
                    )

                    # Update state
                    self.state["signals_generated"] = self.state.get("signals_generated", 0) + 1
                    self.state["last_signal_time"] = datetime.now(timezone.utc)

            else:
                logger.debug(
                    "Premium below minimum",
                    premium_pct=premium_pct,
                    min_premium=self.min_premium
                )

            return signals

        except Exception as e:
            logger.error("Error generating signals", error=str(e))
            return signals

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate covered call indicators.

        Args:
            data: Market data

        Returns:
            Dictionary of indicators
        """
        try:
            if data.is_empty():
                return {}

            spot = Decimal(str(data["close"][-1]))
            iv = self.calculate_implied_volatility(data)
            time_to_expiry = Decimal(str(self.days_to_expiry)) / Decimal("365")

            strike = self.select_strike_price(spot, iv, time_to_expiry)

            risk_free_rate = Decimal(str(self.config.get("risk_free_rate", "0.05")))
            call_price, delta = self.calculate_black_scholes_call(
                spot, strike, time_to_expiry, iv, risk_free_rate
            )

            annual_yield = self.calculate_premium_yield(call_price, spot, self.days_to_expiry)

            indicators = {
                "spot_price": spot,
                "strike_price": strike,
                "call_price": call_price,
                "delta": delta,
                "implied_volatility": iv,
                "premium_yield": call_price / spot,
                "annual_yield": annual_yield,
                "days_to_expiry": Decimal(str(self.days_to_expiry))
            }

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

        cc_metrics = {
            "underlying_positions": {symbol: float(pos) for symbol, pos in self.underlying_positions.items()},
            "option_positions": self.option_positions,
            "covered_ratios": {symbol: float(ratio) for symbol, ratio in self.covered_ratios.items()}
        }

        return {**base_metrics, **cc_metrics}
