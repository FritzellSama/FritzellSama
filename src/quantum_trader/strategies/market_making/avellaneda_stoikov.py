"""
Avellaneda-Stoikov Market Making Strategy.

Implements the Avellaneda-Stoikov optimal market making model, which dynamically
adjusts bid-ask spreads based on inventory risk and market conditions.

Reference: "High-frequency trading in a limit order book" by Avellaneda & Stoikov (2008)
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import math

import polars as pl
from structlog import get_logger

from quantum_trader.models import OrderBook, Ticker, OrderSide, Signal, SignalAction
from quantum_trader.strategies.market_making.base_market_maker import BaseMarketMaker
from quantum_trader.risk.manager import RiskManager

logger = get_logger(__name__)


class AvellanedaStoikovStrategy(BaseMarketMaker):
    """Avellaneda-Stoikov optimal market making strategy.

    This strategy uses a mathematical framework to optimize bid-ask quotes based on:
    - Inventory position and risk aversion
    - Time remaining until inventory target
    - Market volatility
    - Order arrival rates

    The model calculates reservation prices that account for inventory risk
    and adjusts spreads dynamically to manage exposure.

    Attributes:
        risk_aversion: Risk aversion parameter (gamma)
        volatility: Asset volatility (sigma)
        terminal_time: Time horizon for inventory management (T)
        order_intensity: Poisson intensity of order arrivals (k)

    Example:
        >>> config = {
        ...     "risk_aversion": "0.1",
        ...     "terminal_time": "3600",
        ...     "volatility_lookback": "100"
        ... }
        >>> strategy = AvellanedaStoikovStrategy(config, risk_manager)
    """

    def __init__(self, config: Dict[str, Any], risk_manager: RiskManager) -> None:
        """Initialize Avellaneda-Stoikov strategy.

        Args:
            config: Strategy configuration with AS-specific parameters
            risk_manager: Risk manager instance

        Raises:
            ValueError: If required parameters missing
        """
        super().__init__(config, risk_manager)

        # Avellaneda-Stoikov parameters
        self.risk_aversion = Decimal(str(config.get("risk_aversion", "0.1")))
        self.terminal_time = Decimal(str(config.get("terminal_time", "3600")))  # seconds
        self.order_intensity = Decimal(str(config.get("order_intensity", "0.5")))
        self.volatility_lookback = int(config.get("volatility_lookback", 100))

        # State tracking
        self.volatility: Dict[str, Decimal] = {}
        self.reservation_price: Dict[str, Decimal] = {}
        self.start_time = datetime.now(timezone.utc)

        logger.info(
            "Avellaneda-Stoikov strategy initialized",
            strategy=self.name,
            risk_aversion=self.risk_aversion,
            terminal_time=self.terminal_time
        )

    def _validate_config(self) -> None:
        """Validate strategy configuration.

        Raises:
            ValueError: If parameters invalid
        """
        super()._validate_config()

        # Validate AS-specific parameters
        risk_aversion = Decimal(str(self.config.get("risk_aversion", "0.1")))
        if risk_aversion <= Decimal("0") or risk_aversion > Decimal("10"):
            raise ValueError(f"risk_aversion must be in (0, 10]: {risk_aversion}")

        terminal_time = Decimal(str(self.config.get("terminal_time", "3600")))
        if terminal_time <= Decimal("0"):
            raise ValueError(f"terminal_time must be positive: {terminal_time}")

        order_intensity = Decimal(str(self.config.get("order_intensity", "0.5")))
        if order_intensity <= Decimal("0"):
            raise ValueError(f"order_intensity must be positive: {order_intensity}")

    def calculate_volatility(self, market_data: pl.DataFrame) -> Decimal:
        """Calculate asset volatility from market data.

        Uses log returns to estimate volatility (annualized).

        Args:
            market_data: DataFrame with OHLCV data

        Returns:
            Volatility as Decimal

        Raises:
            ValueError: If insufficient data
        """
        try:
            if len(market_data) < 2:
                logger.warning("Insufficient data for volatility calculation")
                return Decimal(str(self.config.get("default_volatility", "0.02")))

            # Use last N candles
            recent_data = market_data.tail(self.volatility_lookback)

            # Calculate log returns
            closes = recent_data.select(pl.col("close")).to_series().to_list()

            if len(closes) < 2:
                return Decimal(str(self.config.get("default_volatility", "0.02")))

            log_returns = []
            for i in range(1, len(closes)):
                if closes[i] > 0 and closes[i-1] > 0:
                    log_return = float(Decimal(str(closes[i])).ln() - Decimal(str(closes[i-1])).ln())
                    log_returns.append(log_return)

            if not log_returns:
                return Decimal(str(self.config.get("default_volatility", "0.02")))

            # Calculate standard deviation
            mean_return = sum(log_returns) / len(log_returns)
            variance = sum((r - mean_return) ** 2 for r in log_returns) / len(log_returns)
            std_dev = Decimal(str(math.sqrt(variance)))

            # Annualize (assuming 1h candles, 24*365 hours per year)
            periods_per_year = Decimal("8760")  # 24 * 365
            annualized_vol = std_dev * (periods_per_year.sqrt())

            logger.debug(
                "Volatility calculated",
                volatility=annualized_vol,
                samples=len(log_returns)
            )

            return annualized_vol

        except Exception as e:
            logger.error("Error calculating volatility", error=str(e))
            return Decimal(str(self.config.get("default_volatility", "0.02")))

    def calculate_time_remaining(self) -> Decimal:
        """Calculate time remaining until terminal time.

        Returns:
            Time remaining in seconds
        """
        try:
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            elapsed_decimal = Decimal(str(elapsed))

            remaining = max(Decimal("1"), self.terminal_time - elapsed_decimal)

            # Reset if we've exceeded terminal time
            if remaining <= Decimal("1"):
                self.start_time = datetime.now(timezone.utc)
                remaining = self.terminal_time

            return remaining

        except Exception as e:
            logger.error("Error calculating time remaining", error=str(e))
            return self.terminal_time

    def calculate_reservation_price(
        self,
        symbol: str,
        mid_price: Decimal,
        inventory: Decimal,
        volatility: Decimal,
        time_remaining: Decimal
    ) -> Decimal:
        """Calculate reservation price (optimal mid-price quote).

        The reservation price adjusts the mid price based on inventory risk:
        r = s - q * gamma * sigma^2 * (T - t)

        Where:
        - s: current mid price
        - q: inventory position
        - gamma: risk aversion
        - sigma: volatility
        - (T-t): time remaining

        Args:
            symbol: Trading symbol
            mid_price: Current mid price
            inventory: Current inventory position
            volatility: Asset volatility
            time_remaining: Time remaining in seconds

        Returns:
            Reservation price as Decimal
        """
        try:
            target = self.target_inventory.get(symbol, Decimal("0"))
            inventory_deviation = inventory - target

            # Calculate inventory risk adjustment
            # r = s - q * gamma * sigma^2 * (T - t)
            risk_adjustment = (
                inventory_deviation *
                self.risk_aversion *
                (volatility ** Decimal("2")) *
                time_remaining
            )

            reservation = mid_price - risk_adjustment

            logger.debug(
                "Reservation price calculated",
                symbol=symbol,
                mid_price=mid_price,
                inventory=inventory,
                reservation=reservation,
                adjustment=risk_adjustment
            )

            return reservation

        except Exception as e:
            logger.error("Error calculating reservation price", error=str(e))
            return mid_price

    def calculate_optimal_spread(
        self,
        volatility: Decimal,
        time_remaining: Decimal,
        order_intensity: Decimal
    ) -> Decimal:
        """Calculate optimal bid-ask spread.

        Based on the AS model:
        delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

        Where:
        - gamma: risk aversion
        - sigma: volatility
        - (T-t): time remaining
        - k: order intensity parameter

        Args:
            volatility: Asset volatility
            time_remaining: Time remaining
            order_intensity: Order arrival intensity

        Returns:
            Optimal half-spread as Decimal
        """
        try:
            # First term: inventory risk component
            inventory_component = (
                self.risk_aversion *
                (volatility ** Decimal("2")) *
                time_remaining
            )

            # Second term: adverse selection component
            # ln(1 + gamma/k)
            ratio = self.risk_aversion / order_intensity
            log_term = (Decimal("1") + ratio).ln()
            adverse_selection_component = (Decimal("2") / self.risk_aversion) * log_term

            optimal_half_spread = inventory_component + adverse_selection_component

            # Ensure spread is within reasonable bounds
            min_spread_half = self.min_spread / Decimal("2")
            max_spread_half = Decimal(str(self.config.get("max_spread", "0.01"))) / Decimal("2")

            optimal_half_spread = max(min_spread_half, min(optimal_half_spread, max_spread_half))

            logger.debug(
                "Optimal spread calculated",
                half_spread=optimal_half_spread,
                inventory_component=inventory_component,
                adverse_selection_component=adverse_selection_component
            )

            return optimal_half_spread * Decimal("2")  # Return full spread

        except Exception as e:
            logger.error("Error calculating optimal spread", error=str(e))
            return self.min_spread

    def calculate_spread(self, orderbook: OrderBook, ticker: Ticker) -> Decimal:
        """Calculate bid-ask spread using AS model.

        Args:
            orderbook: Current orderbook
            ticker: Current ticker

        Returns:
            Optimal spread as Decimal
        """
        try:
            symbol = orderbook.symbol

            # Get current market conditions
            mid_price = (ticker.bid + ticker.ask) / Decimal("2")
            inventory = self.inventory.get(symbol, Decimal("0"))

            # Get or calculate volatility
            if symbol not in self.volatility:
                # Use default if no historical data
                self.volatility[symbol] = Decimal(str(self.config.get("default_volatility", "0.02")))

            volatility = self.volatility[symbol]
            time_remaining = self.calculate_time_remaining()

            # Calculate reservation price
            reservation = self.calculate_reservation_price(
                symbol, mid_price, inventory, volatility, time_remaining
            )
            self.reservation_price[symbol] = reservation

            # Calculate optimal spread
            spread = self.calculate_optimal_spread(
                volatility, time_remaining, self.order_intensity
            )

            logger.debug(
                "AS spread calculated",
                symbol=symbol,
                spread=spread,
                reservation=reservation,
                mid_price=mid_price
            )

            return spread

        except Exception as e:
            logger.error("Error in calculate_spread", error=str(e))
            return self.min_spread

    def calculate_quote_size(self, side: OrderSide, inventory: Decimal) -> Decimal:
        """Calculate order size based on inventory and side.

        Reduces size when inventory is unfavorable.

        Args:
            side: Order side
            inventory: Current inventory

        Returns:
            Order size as Decimal
        """
        try:
            base_size = Decimal(str(self.config.get("base_order_size", "0.1")))

            # Calculate inventory ratio
            inventory_ratio = abs(inventory) / self.max_position

            # Reduce size when inventory is large and unfavorable
            if side == OrderSide.BUY and inventory > Decimal("0"):
                # Long inventory, reduce buy size
                size_multiplier = Decimal("1") - (inventory_ratio * Decimal("0.5"))
            elif side == OrderSide.SELL and inventory < Decimal("0"):
                # Short inventory, reduce sell size
                size_multiplier = Decimal("1") - (inventory_ratio * Decimal("0.5"))
            else:
                size_multiplier = Decimal("1")

            size = base_size * max(Decimal("0.1"), size_multiplier)

            return size.quantize(Decimal("0.00000001"))

        except Exception as e:
            logger.error("Error calculating quote size", error=str(e))
            return Decimal(str(self.config.get("base_order_size", "0.1")))

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate indicators including volatility.

        Args:
            data: Market data

        Returns:
            Dictionary of indicators
        """
        try:
            # Get base market making indicators
            indicators = super().calculate_indicators(data)

            # Calculate and store volatility
            if not data.is_empty() and "close" in data.columns:
                symbol = self.config.get("symbols", ["BTC/USDT"])[0]
                volatility = self.calculate_volatility(data)
                self.volatility[symbol] = volatility
                indicators["volatility"] = volatility

            # Add AS-specific metrics
            indicators["time_remaining"] = self.calculate_time_remaining()
            indicators["risk_aversion"] = self.risk_aversion

            if self.reservation_price:
                for symbol, price in self.reservation_price.items():
                    indicators[f"reservation_price_{symbol}"] = price

            return indicators

        except Exception as e:
            logger.error("Error calculating indicators", error=str(e))
            return {}

    async def warmup(self, historical_data: pl.DataFrame) -> None:
        """Warm up strategy with historical data to calculate volatility.

        Args:
            historical_data: Historical market data
        """
        try:
            logger.info("AS strategy warmup", data_points=len(historical_data))

            if not historical_data.is_empty():
                symbol = self.config.get("symbols", ["BTC/USDT"])[0]
                volatility = self.calculate_volatility(historical_data)
                self.volatility[symbol] = volatility

                logger.info(
                    "Volatility initialized from historical data",
                    symbol=symbol,
                    volatility=volatility
                )

        except Exception as e:
            logger.error("Error in warmup", error=str(e))

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        base_metrics = super().get_performance_metrics()

        as_metrics = {
            "volatility": {symbol: float(vol) for symbol, vol in self.volatility.items()},
            "reservation_price": {symbol: float(price) for symbol, price in self.reservation_price.items()},
            "time_remaining": float(self.calculate_time_remaining()),
            "risk_aversion": float(self.risk_aversion)
        }

        return {**base_metrics, **as_metrics}
