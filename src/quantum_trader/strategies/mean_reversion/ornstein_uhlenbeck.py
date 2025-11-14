"""
Ornstein-Uhlenbeck Mean Reversion Strategy.

Uses the Ornstein-Uhlenbeck process to model mean-reverting assets
and generate signals when price deviates from equilibrium level.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import deque
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


@dataclass
class OUParameters:
    """Ornstein-Uhlenbeck process parameters."""
    theta: Decimal  # Mean reversion speed
    mu: Decimal  # Long-term mean
    sigma: Decimal  # Volatility
    last_update: datetime


class OrnsteinUhlenbeckStrategy:
    """
    Ornstein-Uhlenbeck Mean Reversion Strategy.

    Models asset price as an OU process: dX = θ(μ - X)dt + σdW
    where θ is mean reversion speed, μ is equilibrium level, σ is volatility.

    Generates signals when price deviates significantly from equilibrium.

    Attributes:
        config: Strategy configuration
        risk_manager: Risk management instance
        ou_parameters: Estimated OU process parameters
        price_history: Historical price data for calibration
    """

    def __init__(self, config: Dict[str, Any], risk_manager: Any) -> None:
        """
        Initialize Ornstein-Uhlenbeck Strategy.

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
        self.calibration_window = int(config["calibration_window"])
        self.entry_threshold_std = Decimal(str(config["entry_threshold_std"]))
        self.exit_threshold_std = Decimal(str(config["exit_threshold_std"]))
        self.min_half_life = Decimal(str(config["min_half_life_periods"]))
        self.max_half_life = Decimal(str(config["max_half_life_periods"]))
        self.recalibration_interval = int(config.get("recalibration_interval_periods", 100))

        # State tracking
        self.ou_parameters: Optional[OUParameters] = None
        self.price_history: deque = deque(maxlen=self.calibration_window * 2)
        self.residuals_history: deque = deque(maxlen=100)
        self.periods_since_calibration = 0
        self.current_position_side: Optional[OrderSide] = None

        logger.info(
            "ou_strategy_initialized",
            calibration_window=self.calibration_window,
            entry_threshold=float(self.entry_threshold_std),
            exit_threshold=float(self.exit_threshold_std)
        )

    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required_params = [
            "calibration_window",
            "entry_threshold_std",
            "exit_threshold_std",
            "min_half_life_periods",
            "max_half_life_periods",
            "symbol",
            "exchange",
            "timeframe"
        ]

        missing = [p for p in required_params if p not in self.config]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")

    async def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """
        Generate mean reversion signals based on OU process.

        Args:
            market_data: Polars DataFrame with columns:
                - timestamp: UTC timestamp
                - symbol: Trading pair
                - close: Close price
                - volume: Trading volume

        Returns:
            List of Signal objects

        Raises:
            ValueError: If market_data format invalid
        """
        try:
            signals = []

            if market_data.is_empty():
                logger.warning("empty_market_data")
                return signals

            # Validate required columns
            required_cols = ["timestamp", "symbol", "close"]
            missing_cols = [c for c in required_cols if c not in market_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Get latest market data
            latest_row = market_data.sort("timestamp", descending=True).row(0, named=True)
            current_time = latest_row["timestamp"]
            symbol = latest_row["symbol"]
            current_price = Decimal(str(latest_row["close"]))

            # Store price in history
            self.price_history.append(current_price)

            # Check if we need (re)calibration
            if await self._needs_calibration():
                await self._calibrate_ou_parameters()

            # If parameters not available, cannot generate signals
            if self.ou_parameters is None:
                logger.debug("ou_parameters_not_calibrated")
                return signals

            # Calculate current deviation from equilibrium
            deviation = current_price - self.ou_parameters.mu
            z_score = deviation / self.ou_parameters.sigma if self.ou_parameters.sigma > 0 else Decimal("0")

            # Store residual
            self.residuals_history.append(z_score)

            # Determine signal action
            action, entry_exit = await self._determine_action(z_score)

            if action != SignalAction.HOLD:
                # Calculate signal strength from z-score magnitude
                strength = min(
                    Decimal("1.0"),
                    abs(z_score) / self.entry_threshold_std
                )

                # Calculate confidence based on OU parameter quality
                confidence = await self._calculate_confidence()

                indicators = await self.calculate_indicators(market_data)

                signal = Signal(
                    symbol=symbol,
                    action=action,
                    strength=strength,
                    confidence=confidence,
                    timestamp=current_time,
                    strategy="ornstein_uhlenbeck",
                    timeframe=self.config["timeframe"],
                    indicators=indicators,
                    metadata={
                        "z_score": str(z_score),
                        "equilibrium_price": str(self.ou_parameters.mu),
                        "deviation": str(deviation),
                        "mean_reversion_speed": str(self.ou_parameters.theta),
                        "half_life": str(await self._calculate_half_life()),
                        "entry_exit": entry_exit,
                        "current_price": str(current_price)
                    }
                )

                if self.validate_signal(signal):
                    signals.append(signal)

                    # Update position tracking
                    if entry_exit == "entry":
                        self.current_position_side = OrderSide.BUY if action == SignalAction.BUY else OrderSide.SELL
                    elif entry_exit == "exit":
                        self.current_position_side = None

                    logger.info(
                        "ou_signal_generated",
                        symbol=symbol,
                        action=action.value,
                        z_score=float(z_score),
                        equilibrium=float(self.ou_parameters.mu),
                        entry_exit=entry_exit
                    )

            self.periods_since_calibration += 1

            return signals

        except Exception as e:
            logger.error("signal_generation_failed", error=str(e), exc_info=True)
            raise

    async def _needs_calibration(self) -> bool:
        """
        Check if OU parameters need (re)calibration.

        Returns:
            True if calibration needed
        """
        # Initial calibration
        if self.ou_parameters is None:
            return len(self.price_history) >= self.calibration_window

        # Periodic recalibration
        if self.periods_since_calibration >= self.recalibration_interval:
            return True

        return False

    async def _calibrate_ou_parameters(self) -> None:
        """
        Calibrate Ornstein-Uhlenbeck process parameters from price history.

        Uses maximum likelihood estimation for OU parameters.
        """
        try:
            if len(self.price_history) < self.calibration_window:
                logger.warning(
                    "insufficient_data_for_calibration",
                    available=len(self.price_history),
                    required=self.calibration_window
                )
                return

            logger.info("calibrating_ou_parameters")

            prices = list(self.price_history)[-self.calibration_window:]

            # Calculate log prices for OU estimation
            log_prices = [Decimal(str(np.log(float(p)))) for p in prices]

            # Calculate price differences
            dX = [log_prices[i] - log_prices[i-1] for i in range(1, len(log_prices))]

            # Estimate parameters using discretized OU
            # For discretized OU: X(t+dt) - X(t) = θ(μ - X(t))dt + σ√dt * ε

            # Estimate μ (long-term mean) as average of log prices
            mu_log = sum(log_prices) / Decimal(str(len(log_prices)))

            # Estimate θ (mean reversion speed) using regression
            # dX = a + b*X(t) where b = -θ*dt
            X_lagged = log_prices[:-1]
            n = len(dX)

            mean_dX = sum(dX) / Decimal(str(n))
            mean_X = sum(X_lagged) / Decimal(str(n))

            # Calculate regression slope
            numerator = sum((X_lagged[i] - mean_X) * (dX[i] - mean_dX) for i in range(n))
            denominator = sum((X_lagged[i] - mean_X) ** 2 for i in range(n))

            if denominator > Decimal("0"):
                b = numerator / denominator
                theta = -b  # θ = -b/dt, assuming dt=1
            else:
                theta = Decimal("0.1")  # Default value

            # Ensure theta is positive (mean reverting)
            theta = abs(theta)

            # Estimate σ (volatility) from residuals
            fitted_dX = [theta * (mu_log - X_lagged[i]) for i in range(n)]
            residuals = [dX[i] - fitted_dX[i] for i in range(n)]

            variance = sum(r ** 2 for r in residuals) / Decimal(str(n))
            sigma = Decimal(str(np.sqrt(float(variance))))

            # Convert mu back to price space
            mu = Decimal(str(np.exp(float(mu_log))))

            # Validate parameters
            half_life = await self._calculate_half_life_from_theta(theta)

            if half_life < self.min_half_life or half_life > self.max_half_life:
                logger.warning(
                    "half_life_out_of_range",
                    half_life=float(half_life),
                    min=float(self.min_half_life),
                    max=float(self.max_half_life)
                )
                # Adjust theta to bring half-life into range
                if half_life < self.min_half_life:
                    theta = Decimal(str(np.log(2))) / self.min_half_life
                else:
                    theta = Decimal(str(np.log(2))) / self.max_half_life

            self.ou_parameters = OUParameters(
                theta=theta,
                mu=mu,
                sigma=sigma,
                last_update=datetime.now(timezone.utc)
            )

            self.periods_since_calibration = 0

            logger.info(
                "ou_parameters_calibrated",
                theta=float(theta),
                mu=float(mu),
                sigma=float(sigma),
                half_life=float(await self._calculate_half_life())
            )

        except Exception as e:
            logger.error("calibration_failed", error=str(e), exc_info=True)

    async def _calculate_half_life(self) -> Decimal:
        """
        Calculate half-life of mean reversion.

        Returns:
            Half-life in number of periods
        """
        if self.ou_parameters is None or self.ou_parameters.theta == Decimal("0"):
            return Decimal("0")

        return await self._calculate_half_life_from_theta(self.ou_parameters.theta)

    async def _calculate_half_life_from_theta(self, theta: Decimal) -> Decimal:
        """
        Calculate half-life from theta parameter.

        Args:
            theta: Mean reversion speed

        Returns:
            Half-life in periods
        """
        if theta <= Decimal("0"):
            return Decimal("999999")  # Very large number

        # Half-life = ln(2) / θ
        ln2 = Decimal(str(np.log(2)))
        half_life = ln2 / theta

        return half_life

    async def _determine_action(
        self,
        z_score: Decimal
    ) -> Tuple[SignalAction, str]:
        """
        Determine trading action from z-score.

        Args:
            z_score: Current z-score (deviation in standard deviations)

        Returns:
            Tuple of (SignalAction, entry_exit_flag)
        """
        # Entry signals (open new position)
        if self.current_position_side is None:
            if z_score <= -self.entry_threshold_std:
                # Price below equilibrium -> buy (expect reversion up)
                return SignalAction.BUY, "entry"
            elif z_score >= self.entry_threshold_std:
                # Price above equilibrium -> sell (expect reversion down)
                return SignalAction.SELL, "entry"

        # Exit signals (close existing position)
        else:
            if self.current_position_side == OrderSide.BUY:
                # We're long, exit if price reverted to/above equilibrium
                if z_score >= -self.exit_threshold_std:
                    return SignalAction.SELL, "exit"
            else:  # SHORT position
                # We're short, exit if price reverted to/below equilibrium
                if z_score <= self.exit_threshold_std:
                    return SignalAction.BUY, "exit"

        return SignalAction.HOLD, "none"

    async def _calculate_confidence(self) -> Decimal:
        """
        Calculate signal confidence based on OU model quality.

        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = Decimal("0.6")  # Base confidence

        if self.ou_parameters is None:
            return Decimal("0.3")

        # Higher confidence for stronger mean reversion (lower half-life)
        half_life = await self._calculate_half_life()
        if half_life > Decimal("0"):
            # Normalize half-life to confidence
            # Lower half-life = faster reversion = higher confidence
            hl_factor = self.min_half_life / max(half_life, self.min_half_life)
            confidence += hl_factor * Decimal("0.2")

        # Higher confidence if recent residuals are well-behaved
        if len(self.residuals_history) >= 10:
            recent_residuals = list(self.residuals_history)[-10:]
            # Check if residuals are mean-reverting (autocorrelation)
            has_mean_reversion = await self._check_residual_mean_reversion(recent_residuals)
            if has_mean_reversion:
                confidence += Decimal("0.2")

        return min(Decimal("1.0"), confidence)

    async def _check_residual_mean_reversion(
        self,
        residuals: List[Decimal]
    ) -> bool:
        """
        Check if residuals show mean-reverting behavior.

        Args:
            residuals: List of z-score residuals

        Returns:
            True if mean-reverting
        """
        if len(residuals) < 3:
            return False

        # Simple check: do residuals cross zero frequently?
        zero_crossings = 0
        for i in range(1, len(residuals)):
            if (residuals[i-1] > Decimal("0") and residuals[i] <= Decimal("0")) or \
               (residuals[i-1] <= Decimal("0") and residuals[i] > Decimal("0")):
                zero_crossings += 1

        # If more than 30% of periods have zero crossings, consider it mean-reverting
        crossing_rate = Decimal(str(zero_crossings)) / Decimal(str(len(residuals) - 1))

        return crossing_rate > Decimal("0.3")

    async def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """
        Calculate technical indicators.

        Args:
            data: Polars DataFrame with OHLCV data

        Returns:
            Dictionary of indicator values
        """
        try:
            indicators = {}

            if self.ou_parameters:
                indicators["ou_theta"] = self.ou_parameters.theta
                indicators["ou_mu"] = self.ou_parameters.mu
                indicators["ou_sigma"] = self.ou_parameters.sigma
                indicators["half_life"] = await self._calculate_half_life()

            # Current z-score
            if self.residuals_history:
                indicators["current_z_score"] = list(self.residuals_history)[-1]

            # Average absolute z-score (measure of deviation)
            if len(self.residuals_history) >= 10:
                recent_z = list(self.residuals_history)[-10:]
                avg_abs_z = sum(abs(z) for z in recent_z) / Decimal("10")
                indicators["avg_abs_z_score"] = avg_abs_z

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

            # Check confidence threshold
            min_confidence = Decimal(str(self.config.get("min_confidence", "0.4")))
            if signal.confidence < min_confidence:
                logger.warning(
                    "low_confidence_signal",
                    confidence=float(signal.confidence),
                    min_required=float(min_confidence)
                )
                return False

            # Verify z-score is reasonable
            if "z_score" in signal.metadata:
                z_score = abs(Decimal(str(signal.metadata["z_score"])))
                max_z_score = Decimal(str(self.config.get("max_z_score", "5.0")))

                if z_score > max_z_score:
                    logger.warning(
                        "excessive_z_score",
                        z_score=float(z_score),
                        max_allowed=float(max_z_score)
                    )
                    return False

            # Verify half-life is in acceptable range
            if "half_life" in signal.metadata:
                half_life = Decimal(str(signal.metadata["half_life"]))
                if half_life < self.min_half_life or half_life > self.max_half_life:
                    logger.debug(
                        "half_life_out_of_range",
                        half_life=float(half_life)
                    )
                    return False

            return True

        except Exception as e:
            logger.error("signal_validation_error", error=str(e))
            return False
