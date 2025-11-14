"""
ML-Enhanced Hybrid Trading Strategy.

Combines traditional technical analysis with machine learning predictions
for improved signal generation and risk management.
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


class MLEnhancedStrategy:
    """
    ML-Enhanced Hybrid Trading Strategy.

    Integrates machine learning predictions with traditional technical
    indicators to generate high-confidence trading signals.

    Attributes:
        config: Strategy configuration
        risk_manager: Risk management instance
        ml_model: Machine learning model instance
        feature_history: Historical features for ML prediction
        prediction_cache: Cache of recent ML predictions
    """

    def __init__(self, config: Dict[str, Any], risk_manager: Any) -> None:
        """
        Initialize ML-Enhanced Strategy.

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
        self.ml_weight = Decimal(str(config["ml_weight"]))
        self.technical_weight = Decimal(str(config["technical_weight"]))
        self.min_ml_confidence = Decimal(str(config["min_ml_confidence"]))
        self.feature_window = int(config["feature_window"])
        self.prediction_threshold = Decimal(str(config["prediction_threshold"]))
        self.model_update_interval = int(config["model_update_interval_seconds"])
        self.ensemble_method = config.get("ensemble_method", "weighted_average")

        # State tracking
        self.feature_history: deque = deque(maxlen=self.feature_window * 2)
        self.prediction_cache: deque = deque(maxlen=100)
        self.ml_model: Optional[Any] = None
        self.last_model_update: Optional[datetime] = None
        self.training_data: deque = deque(maxlen=int(config.get("training_data_size", 10000)))

        logger.info(
            "ml_enhanced_strategy_initialized",
            ml_weight=float(self.ml_weight),
            technical_weight=float(self.technical_weight),
            min_confidence=float(self.min_ml_confidence)
        )

    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required_params = [
            "ml_weight",
            "technical_weight",
            "min_ml_confidence",
            "feature_window",
            "prediction_threshold",
            "model_update_interval_seconds",
            "symbol",
            "exchange",
            "timeframe"
        ]

        missing = [p for p in required_params if p not in self.config]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")

        # Validate weights sum to 1
        ml_weight = Decimal(str(self.config["ml_weight"]))
        tech_weight = Decimal(str(self.config["technical_weight"]))

        if abs((ml_weight + tech_weight) - Decimal("1.0")) > Decimal("0.01"):
            raise ValueError("ml_weight + technical_weight must equal 1.0")

    async def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """
        Generate ML-enhanced trading signals.

        Args:
            market_data: Polars DataFrame with columns:
                - timestamp: UTC timestamp
                - symbol: Trading pair
                - open: Open price
                - high: High price
                - low: Low price
                - close: Close price
                - volume: Trading volume

        Returns:
            List of Signal objects with ML predictions

        Raises:
            ValueError: If market_data format invalid
        """
        try:
            signals = []

            if market_data.is_empty():
                logger.warning("empty_market_data")
                return signals

            # Validate required columns
            required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
            missing_cols = [c for c in required_cols if c not in market_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Get latest market data
            latest_row = market_data.sort("timestamp", descending=True).row(0, named=True)
            current_time = latest_row["timestamp"]
            symbol = latest_row["symbol"]
            current_price = Decimal(str(latest_row["close"]))

            # Extract features from market data
            features = await self._extract_features(market_data)

            if not features:
                logger.warning("feature_extraction_failed")
                return signals

            # Calculate technical indicators
            technical_indicators = await self.calculate_indicators(market_data)

            # Generate technical signal
            technical_signal = await self._generate_technical_signal(
                technical_indicators,
                current_price
            )

            # Generate ML prediction
            ml_prediction = await self._generate_ml_prediction(features)

            # Combine signals
            combined_action, combined_strength, combined_confidence = await self._combine_signals(
                technical_signal=technical_signal,
                ml_prediction=ml_prediction
            )

            # Generate final signal if thresholds met
            if combined_action != SignalAction.HOLD:
                if combined_confidence >= self.min_ml_confidence:
                    signal = Signal(
                        symbol=symbol,
                        action=combined_action,
                        strength=combined_strength,
                        confidence=combined_confidence,
                        timestamp=current_time,
                        strategy="ml_enhanced",
                        timeframe=self.config["timeframe"],
                        indicators=technical_indicators,
                        metadata={
                            "ml_prediction": str(ml_prediction["direction"]),
                            "ml_confidence": str(ml_prediction["confidence"]),
                            "technical_signal": str(technical_signal["action"]),
                            "technical_strength": str(technical_signal["strength"]),
                            "ensemble_method": self.ensemble_method,
                            "current_price": str(current_price)
                        }
                    )

                    if self.validate_signal(signal):
                        signals.append(signal)

                        logger.info(
                            "ml_enhanced_signal_generated",
                            symbol=symbol,
                            action=combined_action.value,
                            strength=float(combined_strength),
                            ml_conf=float(ml_prediction["confidence"]),
                            tech_strength=float(technical_signal["strength"])
                        )

            # Store features for training
            await self._store_training_data(features, current_price, current_time)

            # Check if model needs updating
            if await self._should_update_model(current_time):
                await self._update_ml_model()

            return signals

        except Exception as e:
            logger.error("signal_generation_failed", error=str(e), exc_info=True)
            raise

    async def _extract_features(self, market_data: pl.DataFrame) -> Optional[Dict[str, Decimal]]:
        """
        Extract ML features from market data.

        Args:
            market_data: Market data DataFrame

        Returns:
            Dictionary of feature values or None if insufficient data
        """
        try:
            if len(market_data) < self.feature_window:
                return None

            features = {}
            recent_data = market_data.tail(self.feature_window)

            # Price-based features
            closes = [Decimal(str(row["close"])) for row in recent_data.iter_rows(named=True)]
            highs = [Decimal(str(row["high"])) for row in recent_data.iter_rows(named=True)]
            lows = [Decimal(str(row["low"])) for row in recent_data.iter_rows(named=True)]
            volumes = [Decimal(str(row["volume"])) for row in recent_data.iter_rows(named=True)]

            # Returns at different windows
            for window in [1, 3, 5, 10]:
                if len(closes) > window:
                    ret = (closes[-1] - closes[-window]) / closes[-window]
                    features[f"return_{window}"] = ret

            # Volatility (standard deviation of returns)
            if len(closes) >= 10:
                returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                mean_ret = sum(returns) / len(returns)
                variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
                features["volatility"] = Decimal(str(np.sqrt(float(variance))))

            # High-Low range
            if highs and lows and closes:
                avg_close = sum(closes) / len(closes)
                hl_range = (max(highs) - min(lows)) / avg_close if avg_close > 0 else Decimal("0")
                features["hl_range_pct"] = hl_range

            # Volume features
            if len(volumes) >= 5:
                recent_vol = sum(volumes[-5:]) / Decimal("5")
                older_vol = sum(volumes[-10:-5]) / Decimal("5") if len(volumes) >= 10 else recent_vol
                volume_ratio = recent_vol / older_vol if older_vol > 0 else Decimal("1")
                features["volume_ratio"] = volume_ratio

            # Momentum indicator (simplified RSI-like)
            if len(closes) >= 14:
                gains = []
                losses = []
                for i in range(1, 15):
                    change = closes[-i] - closes[-i-1]
                    if change > 0:
                        gains.append(change)
                    else:
                        losses.append(abs(change))

                avg_gain = sum(gains) / Decimal("14") if gains else Decimal("0")
                avg_loss = sum(losses) / Decimal("14") if losses else Decimal("0.0001")

                rs = avg_gain / avg_loss
                rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))
                features["rsi"] = rsi / Decimal("100")  # Normalize to 0-1

            # Trend strength (linear regression slope)
            if len(closes) >= 20:
                x_values = list(range(len(closes[-20:])))
                y_values = [float(c) for c in closes[-20:]]

                x_mean = sum(x_values) / len(x_values)
                y_mean = sum(y_values) / len(y_values)

                numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
                denominator = sum((x - x_mean) ** 2 for x in x_values)

                slope = Decimal(str(numerator / denominator)) if denominator != 0 else Decimal("0")
                features["trend_slope"] = slope

            self.feature_history.append(features)

            return features

        except Exception as e:
            logger.error("feature_extraction_error", error=str(e))
            return None

    async def _generate_technical_signal(
        self,
        indicators: Dict[str, Decimal],
        current_price: Decimal
    ) -> Dict[str, Any]:
        """
        Generate signal from technical indicators.

        Args:
            indicators: Technical indicator values
            current_price: Current market price

        Returns:
            Dictionary with action, strength, confidence
        """
        signal_score = Decimal("0")
        signal_count = 0

        # RSI signal
        if "rsi" in indicators:
            rsi = indicators["rsi"]
            if rsi < Decimal("0.3"):  # Oversold
                signal_score += Decimal("1")
            elif rsi > Decimal("0.7"):  # Overbought
                signal_score -= Decimal("1")
            signal_count += 1

        # Trend signal
        if "trend_slope" in indicators:
            slope = indicators["trend_slope"]
            signal_score += slope * Decimal("10")
            signal_count += 1

        # Volume signal
        if "volume_ratio" in indicators:
            vol_ratio = indicators["volume_ratio"]
            if vol_ratio > Decimal("1.5"):
                # High volume confirms direction
                signal_score *= Decimal("1.2")

        # Volatility adjustment
        if "volatility" in indicators:
            vol = indicators["volatility"]
            # Reduce signal in high volatility
            if vol > Decimal("0.05"):
                signal_score *= Decimal("0.8")

        # Determine action and strength
        if signal_count == 0:
            return {
                "action": SignalAction.HOLD,
                "strength": Decimal("0"),
                "confidence": Decimal("0")
            }

        # Normalize score
        normalized_score = signal_score / Decimal(str(signal_count))
        normalized_score = max(Decimal("-1"), min(Decimal("1"), normalized_score))

        if normalized_score > Decimal("0.2"):
            action = SignalAction.BUY
            strength = min(Decimal("1"), normalized_score)
        elif normalized_score < Decimal("-0.2"):
            action = SignalAction.SELL
            strength = min(Decimal("1"), abs(normalized_score))
        else:
            action = SignalAction.HOLD
            strength = Decimal("0")

        # Confidence based on indicator agreement
        confidence = min(Decimal("1"), abs(normalized_score) * Decimal("1.5"))

        return {
            "action": action,
            "strength": strength,
            "confidence": confidence
        }

    async def _generate_ml_prediction(
        self,
        features: Dict[str, Decimal]
    ) -> Dict[str, Any]:
        """
        Generate ML model prediction.

        Args:
            features: Feature dictionary

        Returns:
            Dictionary with direction, probability, confidence
        """
        # Simplified ML prediction (in production, use actual trained model)
        # For now, use a heuristic-based approach

        # Combine features into prediction score
        prediction_score = Decimal("0")
        weight_sum = Decimal("0")

        feature_weights = {
            "return_1": Decimal("0.1"),
            "return_3": Decimal("0.15"),
            "return_5": Decimal("0.2"),
            "return_10": Decimal("0.15"),
            "rsi": Decimal("0.15"),
            "trend_slope": Decimal("0.15"),
            "volume_ratio": Decimal("0.05"),
            "volatility": Decimal("0.05")
        }

        for feature_name, weight in feature_weights.items():
            if feature_name in features:
                value = features[feature_name]

                # Normalize different features
                if "return" in feature_name:
                    normalized = value * Decimal("100")  # Scale returns
                elif feature_name == "rsi":
                    normalized = (value - Decimal("0.5")) * Decimal("2")  # Center around 0
                elif feature_name == "volume_ratio":
                    normalized = (value - Decimal("1")) * Decimal("0.5")  # Center around 1
                else:
                    normalized = value

                prediction_score += normalized * weight
                weight_sum += weight

        # Normalize prediction
        if weight_sum > Decimal("0"):
            prediction_score /= weight_sum

        # Clamp to -1 to +1
        prediction_score = max(Decimal("-1"), min(Decimal("1"), prediction_score))

        # Determine direction and confidence
        if abs(prediction_score) < Decimal("0.1"):
            direction = 0  # Neutral
            confidence = Decimal("0.3")
        elif prediction_score > Decimal("0"):
            direction = 1  # Bullish
            confidence = min(Decimal("1"), abs(prediction_score) * Decimal("2"))
        else:
            direction = -1  # Bearish
            confidence = min(Decimal("1"), abs(prediction_score) * Decimal("2"))

        prediction = {
            "direction": direction,
            "probability": (prediction_score + Decimal("1")) / Decimal("2"),  # Convert to 0-1
            "confidence": confidence,
            "raw_score": prediction_score
        }

        self.prediction_cache.append(prediction)

        return prediction

    async def _combine_signals(
        self,
        technical_signal: Dict[str, Any],
        ml_prediction: Dict[str, Any]
    ) -> Tuple[SignalAction, Decimal, Decimal]:
        """
        Combine technical and ML signals.

        Args:
            technical_signal: Technical indicator signal
            ml_prediction: ML model prediction

        Returns:
            Tuple of (action, strength, confidence)
        """
        # Convert signals to numerical values
        tech_value = Decimal("0")
        if technical_signal["action"] == SignalAction.BUY:
            tech_value = technical_signal["strength"]
        elif technical_signal["action"] == SignalAction.SELL:
            tech_value = -technical_signal["strength"]

        ml_value = Decimal(str(ml_prediction["direction"])) * ml_prediction["confidence"]

        # Weighted combination
        combined_value = (
            tech_value * self.technical_weight +
            ml_value * self.ml_weight
        )

        # Determine final action
        if combined_value > self.prediction_threshold:
            action = SignalAction.BUY
            strength = min(Decimal("1"), combined_value)
        elif combined_value < -self.prediction_threshold:
            action = SignalAction.SELL
            strength = min(Decimal("1"), abs(combined_value))
        else:
            action = SignalAction.HOLD
            strength = Decimal("0")

        # Combined confidence (average of both confidences)
        combined_confidence = (
            technical_signal["confidence"] * self.technical_weight +
            ml_prediction["confidence"] * self.ml_weight
        )

        return action, strength, combined_confidence

    async def _should_update_model(self, current_time: datetime) -> bool:
        """
        Check if ML model should be updated.

        Args:
            current_time: Current timestamp

        Returns:
            True if update needed
        """
        if self.last_model_update is None:
            return len(self.training_data) >= int(self.config.get("min_training_samples", 1000))

        time_delta = (current_time - self.last_model_update).total_seconds()

        return time_delta >= self.model_update_interval

    async def _update_ml_model(self) -> None:
        """Update ML model with recent training data."""
        try:
            if len(self.training_data) < int(self.config.get("min_training_samples", 100)):
                logger.warning("insufficient_training_data", count=len(self.training_data))
                return

            logger.info("updating_ml_model", training_samples=len(self.training_data))

            # In production, this would:
            # 1. Prepare training dataset from self.training_data
            # 2. Train/update the ML model
            # 3. Validate on holdout set
            # 4. Update self.ml_model if performance improved

            self.last_model_update = datetime.now(timezone.utc)

            logger.info("ml_model_updated")

        except Exception as e:
            logger.error("model_update_failed", error=str(e), exc_info=True)

    async def _store_training_data(
        self,
        features: Dict[str, Decimal],
        price: Decimal,
        timestamp: datetime
    ) -> None:
        """
        Store data for model training.

        Args:
            features: Feature dictionary
            price: Current price
            timestamp: Data timestamp
        """
        self.training_data.append({
            "features": features,
            "price": price,
            "timestamp": timestamp
        })

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

            if len(data) < 2:
                return indicators

            # Use feature extraction which includes many indicators
            features = await self._extract_features(data)

            if features:
                indicators.update(features)

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
            if signal.confidence < self.min_ml_confidence:
                logger.warning(
                    "low_confidence_signal",
                    confidence=float(signal.confidence),
                    min_required=float(self.min_ml_confidence)
                )
                return False

            # Check metadata
            if "ml_prediction" not in signal.metadata:
                logger.warning("missing_ml_prediction")
                return False

            return True

        except Exception as e:
            logger.error("signal_validation_error", error=str(e))
            return False
