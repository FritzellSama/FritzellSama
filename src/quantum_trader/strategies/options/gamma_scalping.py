"""
Gamma Scalping Strategy for Options Trading.

This strategy dynamically hedges delta exposure from options positions
by trading the underlying asset, profiting from realized volatility.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import polars as pl
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LIMIT = "STOP_LIMIT"


class SignalAction(Enum):
    """Signal action enumeration."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


@dataclass
class Order:
    """Trading order - immutable after creation."""
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Optional[Decimal] = None
    order_type: OrderType = OrderType.MARKET
    exchange: str = ""
    strategy: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    """Trading signal from strategy."""
    symbol: str
    action: SignalAction
    strength: Decimal
    confidence: Decimal
    timestamp: datetime
    strategy: str
    timeframe: str
    indicators: Dict[str, Decimal] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GammaScalpingStrategy:
    """
    Gamma Scalping Strategy for Options Trading.

    Dynamically hedges delta exposure by trading the underlying asset,
    capturing profit from realized volatility exceeding implied volatility.

    Attributes:
        config: Strategy configuration from config files
        risk_manager: Risk management instance
        current_position: Current net delta position
        last_hedge_price: Price at last hedge execution
        pnl_tracker: Tracks realized PnL from scalping
    """

    def __init__(self, config: Dict[str, Any], risk_manager: Any) -> None:
        """
        Initialize Gamma Scalping Strategy.

        Args:
            config: Configuration dictionary with strategy parameters
            risk_manager: RiskManager instance for position validation

        Raises:
            ValueError: If required config parameters missing
        """
        self.config = config
        self.risk_manager = risk_manager
        self._validate_config()

        # Load parameters from config
        self.delta_threshold = Decimal(str(config["delta_threshold"]))
        self.gamma_sensitivity = Decimal(str(config["gamma_sensitivity"]))
        self.hedge_interval = Decimal(str(config["hedge_interval_seconds"]))
        self.min_profit_threshold = Decimal(str(config["min_profit_threshold"]))
        self.max_position_size = Decimal(str(config["max_position_size"]))
        self.rehedge_trigger = Decimal(str(config["rehedge_trigger_pct"]))

        # State tracking
        self.current_position = Decimal("0")
        self.last_hedge_price: Optional[Decimal] = None
        self.last_hedge_time: Optional[datetime] = None
        self.pnl_tracker = Decimal("0")
        self.options_positions: Dict[str, Dict] = {}

        logger.info(
            "gamma_scalping_initialized",
            delta_threshold=float(self.delta_threshold),
            gamma_sensitivity=float(self.gamma_sensitivity)
        )

    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required_params = [
            "delta_threshold",
            "gamma_sensitivity",
            "hedge_interval_seconds",
            "min_profit_threshold",
            "max_position_size",
            "rehedge_trigger_pct",
            "symbol",
            "exchange",
            "timeframe"
        ]

        missing = [p for p in required_params if p not in self.config]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")

    async def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """
        Generate hedging signals based on delta exposure and gamma.

        Args:
            market_data: Polars DataFrame with columns:
                - timestamp: UTC timestamp
                - symbol: Trading pair
                - close: Close price (Decimal)
                - volume: Trading volume

        Returns:
            List of Signal objects for delta hedging

        Raises:
            ValueError: If market_data format invalid
        """
        try:
            signals = []

            if market_data.is_empty():
                logger.warning("empty_market_data")
                return signals

            # Validate required columns
            required_cols = ["timestamp", "symbol", "close", "volume"]
            missing_cols = [c for c in required_cols if c not in market_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Get latest market price
            latest_row = market_data.sort("timestamp", descending=True).row(0, named=True)
            current_price = Decimal(str(latest_row["close"]))
            current_time = latest_row["timestamp"]
            symbol = latest_row["symbol"]

            # Calculate current delta exposure from options positions
            total_delta = await self._calculate_portfolio_delta(current_price)

            # Calculate required hedge adjustment
            hedge_adjustment = await self._calculate_hedge_adjustment(
                total_delta,
                current_price,
                current_time
            )

            if abs(hedge_adjustment) > Decimal("0"):
                # Determine signal action
                action = SignalAction.BUY if hedge_adjustment > 0 else SignalAction.SELL

                # Calculate gamma-adjusted strength
                gamma_value = await self._calculate_portfolio_gamma(current_price)
                strength = min(
                    Decimal("1.0"),
                    abs(hedge_adjustment) / self.max_position_size * gamma_value
                )

                # Confidence based on time since last hedge and price movement
                confidence = await self._calculate_confidence(current_price, current_time)

                indicators = await self.calculate_indicators(market_data)

                signal = Signal(
                    symbol=symbol,
                    action=action,
                    strength=strength,
                    confidence=confidence,
                    timestamp=current_time,
                    strategy="gamma_scalping",
                    timeframe=self.config["timeframe"],
                    indicators=indicators,
                    metadata={
                        "total_delta": str(total_delta),
                        "hedge_adjustment": str(hedge_adjustment),
                        "gamma": str(gamma_value),
                        "current_price": str(current_price)
                    }
                )

                if self.validate_signal(signal):
                    signals.append(signal)
                    logger.info(
                        "gamma_hedge_signal_generated",
                        symbol=symbol,
                        action=action.value,
                        adjustment=float(hedge_adjustment),
                        delta=float(total_delta)
                    )

            return signals

        except Exception as e:
            logger.error("signal_generation_failed", error=str(e), exc_info=True)
            raise

    async def _calculate_portfolio_delta(self, spot_price: Decimal) -> Decimal:
        """
        Calculate total delta exposure from all options positions.

        Args:
            spot_price: Current underlying asset price

        Returns:
            Net delta exposure (positive = long, negative = short)
        """
        total_delta = Decimal("0")

        for position_id, position in self.options_positions.items():
            try:
                # Extract position details
                strike = Decimal(str(position["strike"]))
                quantity = Decimal(str(position["quantity"]))
                is_call = position["is_call"]
                time_to_expiry = Decimal(str(position["time_to_expiry"]))

                # Calculate individual option delta (simplified Black-Scholes)
                moneyness = spot_price / strike
                option_delta = await self._calculate_option_delta(
                    moneyness=moneyness,
                    time_to_expiry=time_to_expiry,
                    is_call=is_call
                )

                position_delta = option_delta * quantity
                total_delta += position_delta

            except Exception as e:
                logger.error(
                    "delta_calculation_error",
                    position_id=position_id,
                    error=str(e)
                )
                continue

        return total_delta

    async def _calculate_option_delta(
        self,
        moneyness: Decimal,
        time_to_expiry: Decimal,
        is_call: bool
    ) -> Decimal:
        """
        Calculate option delta using simplified model.

        Args:
            moneyness: Spot / Strike ratio
            time_to_expiry: Time to expiration in years
            is_call: True for call option, False for put

        Returns:
            Option delta value
        """
        # Simplified delta calculation (for production use proper Black-Scholes)
        # This is a approximation based on moneyness

        if time_to_expiry <= Decimal("0"):
            # At expiration: delta is 0 or 1
            if is_call:
                return Decimal("1") if moneyness >= Decimal("1") else Decimal("0")
            else:
                return Decimal("-1") if moneyness <= Decimal("1") else Decimal("0")

        # Approximate delta using moneyness and time decay
        base_delta = (moneyness - Decimal("1")) / (Decimal("1") + time_to_expiry)
        base_delta = max(Decimal("-1"), min(Decimal("1"), base_delta))

        # Adjust for call vs put
        if is_call:
            delta = (Decimal("1") + base_delta) / Decimal("2")
        else:
            delta = (Decimal("1") + base_delta) / Decimal("2") - Decimal("1")

        return delta

    async def _calculate_portfolio_gamma(self, spot_price: Decimal) -> Decimal:
        """
        Calculate total gamma exposure from all options positions.

        Args:
            spot_price: Current underlying asset price

        Returns:
            Net gamma exposure
        """
        total_gamma = Decimal("0")

        for position_id, position in self.options_positions.items():
            try:
                strike = Decimal(str(position["strike"]))
                quantity = Decimal(str(position["quantity"]))
                time_to_expiry = Decimal(str(position["time_to_expiry"]))

                # Simplified gamma calculation
                # Gamma peaks at ATM and decays with time
                moneyness_diff = abs(spot_price - strike) / strike
                time_factor = Decimal("1") / (Decimal("1") + time_to_expiry)

                # Gamma is highest at ATM (moneyness_diff = 0)
                gamma = (Decimal("1") - moneyness_diff) * time_factor * quantity
                gamma = max(Decimal("0"), gamma)

                total_gamma += gamma

            except Exception as e:
                logger.error(
                    "gamma_calculation_error",
                    position_id=position_id,
                    error=str(e)
                )
                continue

        return total_gamma * self.gamma_sensitivity

    async def _calculate_hedge_adjustment(
        self,
        total_delta: Decimal,
        current_price: Decimal,
        current_time: datetime
    ) -> Decimal:
        """
        Calculate required hedge adjustment quantity.

        Args:
            total_delta: Current total delta exposure
            current_price: Current market price
            current_time: Current timestamp

        Returns:
            Quantity to hedge (positive = buy, negative = sell)
        """
        # If delta exposure below threshold, no hedge needed
        if abs(total_delta) < self.delta_threshold:
            return Decimal("0")

        # Check if enough time has passed since last hedge
        if self.last_hedge_time:
            time_delta = (current_time - self.last_hedge_time).total_seconds()
            if time_delta < float(self.hedge_interval):
                return Decimal("0")

        # Check if price has moved enough to trigger rehedge
        if self.last_hedge_price:
            price_change_pct = abs(
                (current_price - self.last_hedge_price) / self.last_hedge_price
            )
            if price_change_pct < self.rehedge_trigger:
                return Decimal("0")

        # Calculate hedge to neutralize delta
        # Negative total_delta means short delta -> need to buy
        # Positive total_delta means long delta -> need to sell
        hedge_quantity = -total_delta

        # Apply position size limits
        if abs(hedge_quantity) > self.max_position_size:
            hedge_quantity = (
                self.max_position_size if hedge_quantity > 0
                else -self.max_position_size
            )

        return hedge_quantity

    async def _calculate_confidence(
        self,
        current_price: Decimal,
        current_time: datetime
    ) -> Decimal:
        """
        Calculate signal confidence based on market conditions.

        Args:
            current_price: Current market price
            current_time: Current timestamp

        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = Decimal("0.5")  # Base confidence

        # Increase confidence if price has moved significantly
        if self.last_hedge_price:
            price_move = abs(
                (current_price - self.last_hedge_price) / self.last_hedge_price
            )
            # Higher price movement = higher confidence
            confidence += min(Decimal("0.3"), price_move * Decimal("10"))

        # Increase confidence if time since last hedge is longer
        if self.last_hedge_time:
            time_factor = (current_time - self.last_hedge_time).total_seconds()
            time_factor = Decimal(str(time_factor)) / (self.hedge_interval * Decimal("10"))
            confidence += min(Decimal("0.2"), time_factor)

        return min(Decimal("1.0"), confidence)

    async def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """
        Calculate technical indicators for options gamma scalping.

        Args:
            data: Polars DataFrame with OHLCV data

        Returns:
            Dictionary of indicator values
        """
        try:
            indicators = {}

            # Calculate realized volatility
            if len(data) >= 20:
                returns = data.select([
                    (pl.col("close").pct_change().alias("returns"))
                ])["returns"].to_list()[1:]

                # Convert to Decimal for calculations
                decimal_returns = [Decimal(str(r)) if r is not None else Decimal("0") for r in returns]

                # Calculate standard deviation
                mean_return = sum(decimal_returns) / len(decimal_returns)
                variance = sum((r - mean_return) ** 2 for r in decimal_returns) / len(decimal_returns)
                std_dev = Decimal(str(np.sqrt(float(variance))))

                # Annualize volatility (assuming 1m timeframe, 525600 minutes/year)
                annualization_factor = Decimal(str(np.sqrt(525600)))
                realized_vol = std_dev * annualization_factor

                indicators["realized_volatility"] = realized_vol

            # Calculate price momentum
            if len(data) >= 10:
                recent_prices = data.tail(10)["close"].to_list()
                price_change = (
                    Decimal(str(recent_prices[-1])) - Decimal(str(recent_prices[0]))
                ) / Decimal(str(recent_prices[0]))
                indicators["momentum_10"] = price_change

            # Volume trend
            if len(data) >= 5:
                recent_volume = data.tail(5)["volume"].to_list()
                avg_volume = Decimal(str(sum(recent_volume))) / Decimal("5")
                current_volume = Decimal(str(recent_volume[-1]))
                indicators["volume_ratio"] = current_volume / avg_volume if avg_volume > 0 else Decimal("1")

            return indicators

        except Exception as e:
            logger.error("indicator_calculation_failed", error=str(e))
            return {}

    def validate_signal(self, signal: Signal) -> bool:
        """
        Validate signal before execution.

        Args:
            signal: Signal to validate

        Returns:
            True if signal is valid, False otherwise
        """
        try:
            # Check signal strength
            if signal.strength <= Decimal("0") or signal.strength > Decimal("1"):
                logger.warning("invalid_signal_strength", strength=float(signal.strength))
                return False

            # Check confidence
            if signal.confidence < Decimal("0.3"):
                logger.warning("low_confidence_signal", confidence=float(signal.confidence))
                return False

            # Check timestamp is recent (within last minute)
            now = datetime.now(timezone.utc)
            age_seconds = (now - signal.timestamp).total_seconds()
            if age_seconds > 60:
                logger.warning("stale_signal", age_seconds=age_seconds)
                return False

            return True

        except Exception as e:
            logger.error("signal_validation_error", error=str(e))
            return False

    async def update_position(
        self,
        position_id: str,
        strike: Decimal,
        quantity: Decimal,
        is_call: bool,
        time_to_expiry: Decimal
    ) -> None:
        """
        Update or add an options position.

        Args:
            position_id: Unique position identifier
            strike: Strike price
            quantity: Position quantity (positive = long, negative = short)
            is_call: True for call option, False for put
            time_to_expiry: Time to expiration in years
        """
        self.options_positions[position_id] = {
            "strike": strike,
            "quantity": quantity,
            "is_call": is_call,
            "time_to_expiry": time_to_expiry,
            "last_updated": datetime.now(timezone.utc)
        }

        logger.info(
            "position_updated",
            position_id=position_id,
            strike=float(strike),
            quantity=float(quantity)
        )

    async def record_hedge_execution(
        self,
        price: Decimal,
        quantity: Decimal,
        timestamp: datetime
    ) -> None:
        """
        Record hedge execution for tracking.

        Args:
            price: Execution price
            quantity: Executed quantity
            timestamp: Execution timestamp
        """
        self.last_hedge_price = price
        self.last_hedge_time = timestamp
        self.current_position += quantity

        # Calculate PnL if we had previous hedge
        if quantity != Decimal("0"):
            # Simple PnL tracking (can be enhanced)
            trade_value = price * abs(quantity)
            self.pnl_tracker += trade_value * (Decimal("1") if quantity < 0 else Decimal("-1"))

        logger.info(
            "hedge_executed",
            price=float(price),
            quantity=float(quantity),
            total_pnl=float(self.pnl_tracker)
        )
