"""
Quantum Trader AI - Adaptive Hybrid Trading Strategy

Advanced hybrid strategy combining multiple approaches:
- Technical analysis indicators
- Machine learning predictions
- Market microstructure analysis
- Adaptive position sizing
- Dynamic risk management

This strategy adapts to changing market conditions and optimizes
parameters in real-time based on performance feedback.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import structlog
from pydantic import BaseModel, Field, validator

from quantum_trader.core.base_strategy import Strategy
from quantum_trader.core.enums import OrderSide, OrderType, TimeFrame
from quantum_trader.core.exceptions import StrategyError, ValidationError
from quantum_trader.indicators.technical import TechnicalIndicators
from quantum_trader.ml.prediction import PredictionEngine
from quantum_trader.risk.position_sizer import PositionSizer

logger = structlog.get_logger(__name__)


class AdaptiveStrategyConfig(BaseModel):
    """Configuration for adaptive strategy."""

    # General settings
    name: str = "adaptive_hybrid"
    version: str = "1.0.0"
    enabled: bool = True

    # Timeframes
    primary_timeframe: TimeFrame = TimeFrame.M15
    secondary_timeframes: List[TimeFrame] = Field(
        default_factory=lambda: [TimeFrame.M5, TimeFrame.H1]
    )

    # Technical indicators
    use_technical_indicators: bool = True
    rsi_period: int = Field(default=14, ge=5, le=50)
    rsi_overbought: Decimal = Field(default=Decimal("70"))
    rsi_oversold: Decimal = Field(default=Decimal("30"))
    ema_fast_period: int = Field(default=12, ge=5, le=50)
    ema_slow_period: int = Field(default=26, ge=10, le=100)
    macd_signal_period: int = Field(default=9, ge=5, le=20)
    bb_period: int = Field(default=20, ge=10, le=50)
    bb_std_dev: Decimal = Field(default=Decimal("2.0"))

    # Machine learning
    use_ml_predictions: bool = True
    ml_model_path: str = ""
    ml_confidence_threshold: Decimal = Field(default=Decimal("0.7"))
    ml_feature_window: int = Field(default=100, ge=50, le=500)

    # Position sizing
    base_position_size_usd: Decimal = Field(default=Decimal("1000"))
    max_position_size_usd: Decimal = Field(default=Decimal("10000"))
    kelly_fraction: Decimal = Field(default=Decimal("0.25"))
    use_dynamic_sizing: bool = True

    # Risk management
    max_risk_per_trade_pct: Decimal = Field(default=Decimal("1.0"))
    stop_loss_pct: Decimal = Field(default=Decimal("2.0"))
    take_profit_pct: Decimal = Field(default=Decimal("4.0"))
    trailing_stop_pct: Decimal = Field(default=Decimal("1.5"))
    use_trailing_stop: bool = True

    # Adaptive parameters
    adaptation_enabled: bool = True
    adaptation_window: int = Field(default=100, ge=50, le=500)
    performance_threshold: Decimal = Field(default=Decimal("0.5"))

    # Market conditions
    min_volume_usd: Decimal = Field(default=Decimal("100000"))
    max_spread_bps: Decimal = Field(default=Decimal("10"))

    @validator("ema_slow_period")
    def validate_ema_periods(cls, v: int, values: Dict[str, Any]) -> int:
        """Ensure slow EMA is greater than fast EMA."""
        fast_period = values.get("ema_fast_period", 12)
        if v <= fast_period:
            raise ValueError("ema_slow_period must be greater than ema_fast_period")
        return v


class SignalStrength(BaseModel):
    """Signal strength from different components."""

    technical: Decimal = Decimal("0")
    ml_prediction: Decimal = Decimal("0")
    microstructure: Decimal = Decimal("0")
    combined: Decimal = Decimal("0")
    confidence: Decimal = Decimal("0")

    def calculate_combined(
        self, tech_weight: Decimal, ml_weight: Decimal, micro_weight: Decimal
    ) -> None:
        """Calculate combined signal strength."""
        total_weight = tech_weight + ml_weight + micro_weight
        if total_weight == 0:
            self.combined = Decimal("0")
            self.confidence = Decimal("0")
            return

        self.combined = (
            self.technical * tech_weight
            + self.ml_prediction * ml_weight
            + self.microstructure * micro_weight
        ) / total_weight

        # Calculate confidence based on agreement
        signals = [self.technical, self.ml_prediction, self.microstructure]
        active_signals = [s for s in signals if s != 0]
        if len(active_signals) > 0:
            agreement = sum(
                Decimal("1") for s in active_signals if s * self.combined > 0
            )
            self.confidence = agreement / Decimal(str(len(active_signals)))
        else:
            self.confidence = Decimal("0")


class AdaptiveHybridStrategy(Strategy):
    """
    Adaptive hybrid trading strategy.

    Combines technical analysis, machine learning, and market microstructure
    analysis with dynamic adaptation to market conditions.

    Attributes:
        config: Strategy configuration
        technical_indicators: Technical indicator calculator
        ml_engine: Machine learning prediction engine
        position_sizer: Position sizing calculator
        performance_history: Recent performance metrics
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize adaptive hybrid strategy.

        Args:
            config: Strategy configuration dictionary

        Raises:
            ValidationError: If configuration is invalid
        """
        try:
            self.config = AdaptiveStrategyConfig(**config)
        except Exception as e:
            logger.error("Invalid strategy configuration", error=str(e))
            raise ValidationError(f"Configuration validation failed: {e}") from e

        super().__init__(config)

        # Initialize components
        self.technical_indicators = TechnicalIndicators(config)
        self.ml_engine: Optional[PredictionEngine] = None
        self.position_sizer = PositionSizer(config)

        # Performance tracking
        self.performance_history: List[Decimal] = []
        self.signal_weights = {
            "technical": Decimal("0.4"),
            "ml": Decimal("0.4"),
            "microstructure": Decimal("0.2"),
        }

        # State
        self.last_signal: Optional[SignalStrength] = None
        self.last_adaptation: Optional[datetime] = None

        logger.info(
            "Adaptive hybrid strategy initialized",
            strategy=self.config.name,
            version=self.config.version,
        )

    async def initialize(self) -> None:
        """
        Initialize strategy components.

        Raises:
            StrategyError: If initialization fails
        """
        try:
            logger.info("Initializing strategy components")

            # Initialize ML engine if enabled
            if self.config.use_ml_predictions and self.config.ml_model_path:
                self.ml_engine = PredictionEngine(
                    model_path=self.config.ml_model_path,
                    feature_window=self.config.ml_feature_window,
                )
                await self.ml_engine.load_model()
                logger.info("ML engine initialized")

            # Initialize technical indicators
            await self.technical_indicators.initialize()

            logger.info("Strategy initialization complete")

        except Exception as e:
            logger.error("Strategy initialization failed", error=str(e))
            raise StrategyError(f"Initialization failed: {e}") from e

    async def on_tick(
        self, symbol: str, price: Decimal, volume: Decimal, timestamp: datetime
    ) -> None:
        """
        Process market tick data.

        Args:
            symbol: Trading symbol
            price: Current price
            volume: Trade volume
            timestamp: Tick timestamp

        Raises:
            StrategyError: If processing fails
        """
        try:
            # Update internal state
            await self._update_state(symbol, price, volume, timestamp)

            # Check if we should adapt parameters
            if self.config.adaptation_enabled:
                await self._adapt_if_needed()

        except Exception as e:
            logger.error("Tick processing failed", error=str(e), symbol=symbol)
            raise StrategyError(f"Tick processing failed: {e}") from e

    async def on_orderbook(
        self, symbol: str, orderbook: Dict[str, Any], timestamp: datetime
    ) -> None:
        """
        Process orderbook update.

        Args:
            symbol: Trading symbol
            orderbook: Orderbook data
            timestamp: Update timestamp

        Raises:
            StrategyError: If processing fails
        """
        try:
            # Analyze market microstructure
            microstructure_signal = await self._analyze_microstructure(
                orderbook, timestamp
            )

            # Store for signal generation
            if not hasattr(self, "_microstructure_signals"):
                self._microstructure_signals = {}
            self._microstructure_signals[symbol] = microstructure_signal

        except Exception as e:
            logger.error(
                "Orderbook processing failed", error=str(e), symbol=symbol
            )
            raise StrategyError(f"Orderbook processing failed: {e}") from e

    async def calculate_signals(
        self, symbol: str, data: pl.DataFrame
    ) -> Tuple[OrderSide, Decimal]:
        """
        Calculate trading signals.

        Args:
            symbol: Trading symbol
            data: Market data (Polars DataFrame)

        Returns:
            Tuple of (signal side, signal strength)

        Raises:
            StrategyError: If calculation fails
        """
        try:
            logger.debug("Calculating signals", symbol=symbol)

            signal_strength = SignalStrength()

            # 1. Technical analysis signals
            if self.config.use_technical_indicators:
                tech_signal = await self._calculate_technical_signal(data)
                signal_strength.technical = tech_signal
                logger.debug("Technical signal", value=str(tech_signal))

            # 2. Machine learning predictions
            if self.config.use_ml_predictions and self.ml_engine:
                ml_signal = await self._calculate_ml_signal(data)
                signal_strength.ml_prediction = ml_signal
                logger.debug("ML signal", value=str(ml_signal))

            # 3. Market microstructure
            if hasattr(self, "_microstructure_signals") and symbol in self._microstructure_signals:
                signal_strength.microstructure = self._microstructure_signals[symbol]
                logger.debug(
                    "Microstructure signal",
                    value=str(signal_strength.microstructure),
                )

            # Combine signals
            signal_strength.calculate_combined(
                self.signal_weights["technical"],
                self.signal_weights["ml"],
                self.signal_weights["microstructure"],
            )

            self.last_signal = signal_strength

            # Determine side and strength
            if signal_strength.combined > 0:
                side = OrderSide.BUY
            elif signal_strength.combined < 0:
                side = OrderSide.SELL
            else:
                side = OrderSide.BUY  # Default, but strength will be 0
                signal_strength.combined = Decimal("0")

            logger.info(
                "Signals calculated",
                symbol=symbol,
                side=side.value,
                strength=str(abs(signal_strength.combined)),
                confidence=str(signal_strength.confidence),
            )

            return side, abs(signal_strength.combined)

        except Exception as e:
            logger.error("Signal calculation failed", error=str(e), symbol=symbol)
            raise StrategyError(f"Signal calculation failed: {e}") from e

    async def generate_orders(
        self, symbol: str, signal_side: OrderSide, signal_strength: Decimal
    ) -> List[Dict[str, Any]]:
        """
        Generate orders based on signals.

        Args:
            symbol: Trading symbol
            signal_side: Signal direction (BUY/SELL)
            signal_strength: Signal strength (0-1)

        Returns:
            List of order dictionaries

        Raises:
            StrategyError: If order generation fails
        """
        try:
            # Check if signal is strong enough
            min_signal_strength = Decimal("0.5")
            if signal_strength < min_signal_strength:
                logger.debug(
                    "Signal too weak",
                    strength=str(signal_strength),
                    min_required=str(min_signal_strength),
                )
                return []

            # Check confidence threshold
            if (
                self.last_signal
                and self.last_signal.confidence < self.config.ml_confidence_threshold
            ):
                logger.debug(
                    "Confidence too low",
                    confidence=str(self.last_signal.confidence),
                    threshold=str(self.config.ml_confidence_threshold),
                )
                return []

            # Calculate position size
            position_size = await self._calculate_position_size(
                symbol, signal_strength
            )

            if position_size <= Decimal("0"):
                logger.debug("Position size too small", size=str(position_size))
                return []

            # Get current price
            current_price = await self._get_current_price(symbol)

            # Calculate stop loss and take profit
            if signal_side == OrderSide.BUY:
                stop_loss_price = current_price * (
                    Decimal("1") - self.config.stop_loss_pct / Decimal("100")
                )
                take_profit_price = current_price * (
                    Decimal("1") + self.config.take_profit_pct / Decimal("100")
                )
            else:  # SELL
                stop_loss_price = current_price * (
                    Decimal("1") + self.config.stop_loss_pct / Decimal("100")
                )
                take_profit_price = current_price * (
                    Decimal("1") - self.config.take_profit_pct / Decimal("100")
                )

            # Create entry order
            entry_order = {
                "symbol": symbol,
                "side": signal_side.value,
                "type": OrderType.LIMIT.value,
                "quantity": str(position_size),
                "price": str(current_price),
                "stop_loss": str(stop_loss_price),
                "take_profit": str(take_profit_price),
                "use_trailing_stop": self.config.use_trailing_stop,
                "trailing_stop_pct": str(self.config.trailing_stop_pct),
                "metadata": {
                    "strategy": self.config.name,
                    "signal_strength": str(signal_strength),
                    "confidence": str(self.last_signal.confidence) if self.last_signal else "0",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }

            logger.info(
                "Order generated",
                symbol=symbol,
                side=signal_side.value,
                quantity=str(position_size),
                price=str(current_price),
            )

            return [entry_order]

        except Exception as e:
            logger.error("Order generation failed", error=str(e), symbol=symbol)
            raise StrategyError(f"Order generation failed: {e}") from e

    async def update_state(self, performance_metrics: Dict[str, Decimal]) -> None:
        """
        Update strategy state with performance metrics.

        Args:
            performance_metrics: Recent performance data

        Raises:
            StrategyError: If update fails
        """
        try:
            # Track performance
            if "pnl" in performance_metrics:
                self.performance_history.append(performance_metrics["pnl"])

                # Keep only recent history
                if len(self.performance_history) > self.config.adaptation_window:
                    self.performance_history = self.performance_history[
                        -self.config.adaptation_window :
                    ]

            logger.debug("State updated", metrics=performance_metrics)

        except Exception as e:
            logger.error("State update failed", error=str(e))
            raise StrategyError(f"State update failed: {e}") from e

    async def get_performance_metrics(self) -> Dict[str, Decimal]:
        """
        Get current performance metrics.

        Returns:
            Dictionary of performance metrics

        Raises:
            StrategyError: If calculation fails
        """
        try:
            if not self.performance_history:
                return {}

            total_pnl = sum(self.performance_history)
            avg_pnl = total_pnl / Decimal(str(len(self.performance_history)))
            winning_periods = sum(
                Decimal("1") for pnl in self.performance_history if pnl > 0
            )
            win_rate = winning_periods / Decimal(str(len(self.performance_history)))

            return {
                "total_pnl": total_pnl,
                "avg_pnl": avg_pnl,
                "win_rate": win_rate,
                "signal_count": Decimal(str(len(self.performance_history))),
            }

        except Exception as e:
            logger.error("Performance metrics calculation failed", error=str(e))
            raise StrategyError(f"Metrics calculation failed: {e}") from e

    # Private helper methods

    async def _calculate_technical_signal(self, data: pl.DataFrame) -> Decimal:
        """Calculate signal from technical indicators."""
        try:
            # Calculate indicators
            indicators = await self.technical_indicators.calculate_all(data)

            signal = Decimal("0")
            signal_count = 0

            # RSI signal
            if "rsi" in indicators:
                rsi = Decimal(str(indicators["rsi"]))
                if rsi < self.config.rsi_oversold:
                    signal += Decimal("1")  # Buy signal
                elif rsi > self.config.rsi_overbought:
                    signal -= Decimal("1")  # Sell signal
                signal_count += 1

            # MACD signal
            if "macd" in indicators and "macd_signal" in indicators:
                macd = Decimal(str(indicators["macd"]))
                macd_signal = Decimal(str(indicators["macd_signal"]))
                if macd > macd_signal:
                    signal += Decimal("1")
                elif macd < macd_signal:
                    signal -= Decimal("1")
                signal_count += 1

            # Bollinger Bands signal
            if all(k in indicators for k in ["bb_upper", "bb_lower", "close"]):
                price = Decimal(str(indicators["close"]))
                bb_upper = Decimal(str(indicators["bb_upper"]))
                bb_lower = Decimal(str(indicators["bb_lower"]))

                if price < bb_lower:
                    signal += Decimal("1")
                elif price > bb_upper:
                    signal -= Decimal("1")
                signal_count += 1

            # Average the signals
            if signal_count > 0:
                signal = signal / Decimal(str(signal_count))

            return signal

        except Exception as e:
            logger.error("Technical signal calculation failed", error=str(e))
            return Decimal("0")

    async def _calculate_ml_signal(self, data: pl.DataFrame) -> Decimal:
        """Calculate signal from ML predictions."""
        try:
            if not self.ml_engine:
                return Decimal("0")

            prediction = await self.ml_engine.predict(data)

            # prediction is expected to be between -1 and 1
            return Decimal(str(prediction))

        except Exception as e:
            logger.error("ML signal calculation failed", error=str(e))
            return Decimal("0")

    async def _analyze_microstructure(
        self, orderbook: Dict[str, Any], timestamp: datetime
    ) -> Decimal:
        """Analyze market microstructure."""
        try:
            # Calculate bid-ask imbalance
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])

            if not bids or not asks:
                return Decimal("0")

            # Calculate volume imbalance
            bid_volume = sum(Decimal(str(b[1])) for b in bids[:10])  # Top 10 levels
            ask_volume = sum(Decimal(str(a[1])) for a in asks[:10])

            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                return Decimal("0")

            imbalance = (bid_volume - ask_volume) / total_volume

            return imbalance

        except Exception as e:
            logger.error("Microstructure analysis failed", error=str(e))
            return Decimal("0")

    async def _calculate_position_size(
        self, symbol: str, signal_strength: Decimal
    ) -> Decimal:
        """Calculate position size based on signal and risk."""
        try:
            if self.config.use_dynamic_sizing:
                # Use Kelly criterion with fractional sizing
                win_rate = await self._get_recent_win_rate()
                avg_win = await self._get_avg_win()
                avg_loss = await self._get_avg_loss()

                position_size = await self.position_sizer.calculate_kelly_size(
                    win_rate=win_rate,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    fraction=self.config.kelly_fraction,
                )
            else:
                position_size = self.config.base_position_size_usd

            # Scale by signal strength
            position_size = position_size * signal_strength

            # Apply limits
            position_size = min(position_size, self.config.max_position_size_usd)
            position_size = max(position_size, Decimal("0"))

            return position_size

        except Exception as e:
            logger.error("Position size calculation failed", error=str(e))
            return self.config.base_position_size_usd

    async def _adapt_if_needed(self) -> None:
        """Adapt strategy parameters based on performance."""
        try:
            now = datetime.now(timezone.utc)

            # Check if adaptation is due
            if self.last_adaptation:
                time_since_adaptation = (now - self.last_adaptation).total_seconds()
                adaptation_interval = 3600  # 1 hour
                if time_since_adaptation < adaptation_interval:
                    return

            # Check if we have enough history
            if len(self.performance_history) < self.config.adaptation_window:
                return

            # Calculate recent performance
            metrics = await self.get_performance_metrics()
            win_rate = metrics.get("win_rate", Decimal("0"))

            # Adapt signal weights based on performance
            if win_rate > Decimal("0.6"):  # Performing well
                # Increase weight of best performing component
                # (simplified - in production would track component-specific performance)
                logger.info("Adapting weights - good performance", win_rate=str(win_rate))
            elif win_rate < Decimal("0.4"):  # Performing poorly
                # Revert to default weights or reduce exposure
                self.signal_weights = {
                    "technical": Decimal("0.4"),
                    "ml": Decimal("0.4"),
                    "microstructure": Decimal("0.2"),
                }
                logger.info("Adapting weights - poor performance", win_rate=str(win_rate))

            self.last_adaptation = now

        except Exception as e:
            logger.error("Adaptation failed", error=str(e))

    async def _update_state(
        self, symbol: str, price: Decimal, volume: Decimal, timestamp: datetime
    ) -> None:
        """Update internal state with new tick data."""
        # Implementation depends on specific state tracking needs
        pass

    async def _get_current_price(self, symbol: str) -> Decimal:
        """Get current market price."""
        # This would fetch from market data feed
        # Placeholder - actual implementation depends on market data source
        raise NotImplementedError("Market data integration required")

    async def _get_recent_win_rate(self) -> Decimal:
        """Get recent win rate."""
        if len(self.performance_history) < 10:
            return Decimal("0.5")  # Default

        winning = sum(Decimal("1") for p in self.performance_history[-50:] if p > 0)
        return winning / Decimal("50") if len(self.performance_history) >= 50 else Decimal("0.5")

    async def _get_avg_win(self) -> Decimal:
        """Get average winning trade."""
        wins = [p for p in self.performance_history if p > 0]
        if not wins:
            return Decimal("100")  # Default
        return sum(wins) / Decimal(str(len(wins)))

    async def _get_avg_loss(self) -> Decimal:
        """Get average losing trade."""
        losses = [abs(p) for p in self.performance_history if p < 0]
        if not losses:
            return Decimal("50")  # Default
        return sum(losses) / Decimal(str(len(losses)))
