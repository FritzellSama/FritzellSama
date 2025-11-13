"""
Market Regime Detection and State Classification.

This module provides tools for detecting and classifying market regimes
(trending, ranging, volatile) using Hidden Markov Models and other techniques.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import numpy as np
import polars as pl
from structlog import get_logger
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from quantum_trader.exceptions import ModelError, ValidationError

logger = get_logger(__name__)


class MarketRegime(Enum):
    """Market regime types."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeState:
    """Market regime state at a point in time.

    Attributes:
        timestamp: Timestamp of the state
        regime: Detected market regime
        probability: Confidence probability
        features: Feature values used for detection
        metadata: Additional metadata
    """
    timestamp: datetime
    regime: MarketRegime
    probability: Decimal
    features: Dict[str, Decimal]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimeConfig:
    """Configuration for regime detection.

    Attributes:
        num_states: Number of hidden states
        lookback_window: Window size for feature calculation
        volatility_threshold: Threshold for volatility classification
        trend_threshold: Threshold for trend classification
        update_interval: How often to update the model
        use_gmm: Whether to use Gaussian Mixture Model
        random_state: Random seed for reproducibility
    """
    num_states: int = 3
    lookback_window: int = 20
    volatility_threshold: Decimal = Decimal("0.02")
    trend_threshold: Decimal = Decimal("0.01")
    update_interval: int = 100
    use_gmm: bool = True
    random_state: int = 42


class MarketRegimeDetector:
    """Detect and classify market regimes.

    Uses statistical methods and machine learning to identify different
    market states and transitions between them.

    Attributes:
        config: Detector configuration
        model: Regime detection model
        scaler: Feature scaler
        regime_history: Historical regime detections

    Example:
        >>> config = RegimeConfig(
        ...     num_states=3,
        ...     lookback_window=20,
        ...     volatility_threshold=Decimal("0.02"),
        ...     trend_threshold=Decimal("0.01")
        ... )
        >>> detector = MarketRegimeDetector(config)
        >>> detector.fit(market_data)
        >>> regime = detector.detect_regime(current_data)
    """

    def __init__(self, config: RegimeConfig) -> None:
        """Initialize regime detector.

        Args:
            config: Detector configuration

        Raises:
            ValidationError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        # Initialize components
        if config.use_gmm:
            self.model = GaussianMixture(
                n_components=config.num_states,
                covariance_type='full',
                random_state=config.random_state,
                max_iter=100
            )
        else:
            self.model = None

        self.scaler = StandardScaler()
        self.regime_history: List[RegimeState] = []
        self._fitted = False
        self._update_counter = 0

        logger.info(
            "Market regime detector initialized",
            num_states=config.num_states,
            lookback_window=config.lookback_window,
            use_gmm=config.use_gmm
        )

    def _validate_config(self) -> None:
        """Validate detector configuration.

        Raises:
            ValidationError: If configuration is invalid
        """
        if self.config.num_states < 2:
            raise ValidationError("num_states must be >= 2")

        if self.config.lookback_window < 2:
            raise ValidationError("lookback_window must be >= 2")

        if self.config.volatility_threshold <= 0:
            raise ValidationError("volatility_threshold must be > 0")

        if self.config.trend_threshold <= 0:
            raise ValidationError("trend_threshold must be > 0")

    def fit(
        self,
        data: pl.DataFrame,
        price_col: str = "close"
    ) -> None:
        """Fit regime detection model.

        Args:
            data: Market data
            price_col: Column name for price data

        Raises:
            ModelError: If fitting fails
            ValidationError: If data is invalid
        """
        try:
            self._validate_data(data, price_col)

            logger.info("Fitting regime detection model", num_rows=len(data))

            # Extract features
            features = self._extract_features(data, price_col)

            # Scale features
            features_scaled = self.scaler.fit_transform(features)

            # Fit model
            if self.config.use_gmm:
                self.model.fit(features_scaled)

            self._fitted = True

            logger.info("Regime detection model fitted successfully")

        except Exception as e:
            logger.error("Model fitting failed", error=str(e))
            raise ModelError(f"Model fitting failed: {e}") from e

    def detect_regime(
        self,
        data: pl.DataFrame,
        price_col: str = "close",
        timestamp: Optional[datetime] = None
    ) -> RegimeState:
        """Detect current market regime.

        Args:
            data: Recent market data
            price_col: Column name for price data
            timestamp: Optional timestamp for the detection

        Returns:
            Regime state

        Raises:
            ModelError: If detection fails
            ValidationError: If data is invalid
        """
        try:
            if not self._fitted:
                raise ModelError("Model not fitted")

            self._validate_data(data, price_col)

            if timestamp is None:
                timestamp = datetime.now(timezone.utc)

            # Extract features
            features = self._extract_features(data, price_col)

            # Scale features
            features_scaled = self.scaler.transform(features[-1:])

            # Predict regime
            if self.config.use_gmm:
                state_idx = self.model.predict(features_scaled)[0]
                probabilities = self.model.predict_proba(features_scaled)[0]
                probability = Decimal(str(probabilities[state_idx]))
            else:
                # Rule-based detection
                state_idx, probability = self._rule_based_detection(features[-1])

            # Map state index to regime
            regime = self._map_state_to_regime(state_idx, features[-1])

            # Create regime state
            regime_state = RegimeState(
                timestamp=timestamp,
                regime=regime,
                probability=probability,
                features=self._features_to_dict(features[-1]),
                metadata={
                    "state_index": state_idx,
                    "num_observations": len(data)
                }
            )

            # Add to history
            self.regime_history.append(regime_state)

            logger.debug(
                "Regime detected",
                regime=regime.value,
                probability=float(probability)
            )

            return regime_state

        except Exception as e:
            logger.error("Regime detection failed", error=str(e))
            raise ModelError(f"Regime detection failed: {e}") from e

    def detect_batch(
        self,
        data: pl.DataFrame,
        price_col: str = "close"
    ) -> List[RegimeState]:
        """Detect regimes for multiple time points.

        Args:
            data: Market data
            price_col: Column name for price data

        Returns:
            List of regime states

        Raises:
            ModelError: If detection fails
        """
        try:
            if not self._fitted:
                raise ModelError("Model not fitted")

            logger.info("Detecting batch regimes", num_rows=len(data))

            regime_states = []

            # Process in windows
            for i in range(self.config.lookback_window, len(data)):
                window_data = data[i - self.config.lookback_window:i + 1]
                regime_state = self.detect_regime(window_data, price_col)
                regime_states.append(regime_state)

            logger.info("Batch detection completed", num_regimes=len(regime_states))

            return regime_states

        except Exception as e:
            logger.error("Batch detection failed", error=str(e))
            raise ModelError(f"Batch detection failed: {e}") from e

    def _extract_features(
        self,
        data: pl.DataFrame,
        price_col: str
    ) -> np.ndarray:
        """Extract features for regime detection.

        Args:
            data: Market data
            price_col: Price column name

        Returns:
            Feature array [N, num_features]
        """
        features_list = []

        # Calculate returns
        returns = data[price_col].pct_change().fill_null(0)

        for i in range(self.config.lookback_window, len(data)):
            window_returns = returns[i - self.config.lookback_window:i].to_numpy()

            # Feature 1: Mean return
            mean_return = np.mean(window_returns)

            # Feature 2: Volatility (std of returns)
            volatility = np.std(window_returns)

            # Feature 3: Trend strength (linear regression slope)
            x = np.arange(len(window_returns))
            trend = np.polyfit(x, window_returns, 1)[0]

            # Feature 4: Skewness
            skewness = self._calculate_skewness(window_returns)

            # Feature 5: Kurtosis
            kurtosis = self._calculate_kurtosis(window_returns)

            # Feature 6: Max drawdown
            prices = data[price_col][i - self.config.lookback_window:i].to_numpy()
            max_drawdown = self._calculate_max_drawdown(prices)

            features = np.array([
                mean_return,
                volatility,
                trend,
                skewness,
                kurtosis,
                max_drawdown
            ])

            features_list.append(features)

        return np.array(features_list)

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data.

        Args:
            data: Input data

        Returns:
            Skewness value
        """
        if len(data) == 0:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0

        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data.

        Args:
            data: Input data

        Returns:
            Kurtosis value
        """
        if len(data) == 0:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0

        kurtosis = np.mean(((data - mean) / std) ** 4) - 3
        return kurtosis

    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown.

        Args:
            prices: Price array

        Returns:
            Max drawdown value
        """
        if len(prices) == 0:
            return 0.0

        cummax = np.maximum.accumulate(prices)
        drawdown = (prices - cummax) / cummax
        max_drawdown = np.min(drawdown)

        return max_drawdown

    def _rule_based_detection(
        self,
        features: np.ndarray
    ) -> Tuple[int, Decimal]:
        """Rule-based regime detection.

        Args:
            features: Feature array

        Returns:
            Tuple of (state_index, probability)
        """
        mean_return = features[0]
        volatility = features[1]
        trend = features[2]

        # State 0: Trending up
        if trend > float(self.config.trend_threshold) and mean_return > 0:
            return 0, Decimal("0.8")

        # State 1: Trending down
        elif trend < -float(self.config.trend_threshold) and mean_return < 0:
            return 1, Decimal("0.8")

        # State 2: Ranging / low volatility
        elif volatility < float(self.config.volatility_threshold):
            return 2, Decimal("0.7")

        # State 3: High volatility
        elif volatility > float(self.config.volatility_threshold) * 2:
            return 3, Decimal("0.7")

        # Default: ranging
        return 2, Decimal("0.5")

    def _map_state_to_regime(
        self,
        state_idx: int,
        features: np.ndarray
    ) -> MarketRegime:
        """Map state index to market regime.

        Args:
            state_idx: State index
            features: Feature array

        Returns:
            Market regime
        """
        mean_return = features[0]
        volatility = features[1]
        trend = features[2]

        # High volatility
        if volatility > float(self.config.volatility_threshold) * 2:
            return MarketRegime.HIGH_VOLATILITY

        # Low volatility
        if volatility < float(self.config.volatility_threshold) * 0.5:
            return MarketRegime.LOW_VOLATILITY

        # Trending up
        if trend > float(self.config.trend_threshold) and mean_return > 0:
            return MarketRegime.TRENDING_UP

        # Trending down
        if trend < -float(self.config.trend_threshold) and mean_return < 0:
            return MarketRegime.TRENDING_DOWN

        # Ranging
        return MarketRegime.RANGING

    def _features_to_dict(self, features: np.ndarray) -> Dict[str, Decimal]:
        """Convert feature array to dictionary.

        Args:
            features: Feature array

        Returns:
            Feature dictionary
        """
        feature_names = [
            "mean_return",
            "volatility",
            "trend",
            "skewness",
            "kurtosis",
            "max_drawdown"
        ]

        return {
            name: Decimal(str(value))
            for name, value in zip(feature_names, features)
        }

    def _validate_data(
        self,
        data: pl.DataFrame,
        price_col: str
    ) -> None:
        """Validate input data.

        Args:
            data: Market data
            price_col: Price column name

        Raises:
            ValidationError: If data is invalid
        """
        if data is None or len(data) == 0:
            raise ValidationError("Data cannot be empty")

        if price_col not in data.columns:
            raise ValidationError(f"Price column not found: {price_col}")

        if len(data) < self.config.lookback_window + 1:
            raise ValidationError(
                f"Data must have at least {self.config.lookback_window + 1} rows"
            )

    async def detect_regime_async(
        self,
        data: pl.DataFrame,
        price_col: str = "close",
        timestamp: Optional[datetime] = None
    ) -> RegimeState:
        """Detect regime asynchronously.

        Args:
            data: Recent market data
            price_col: Column name for price data
            timestamp: Optional timestamp for the detection

        Returns:
            Regime state
        """
        return await asyncio.to_thread(
            self.detect_regime,
            data,
            price_col,
            timestamp
        )

    def get_regime_distribution(self) -> Dict[MarketRegime, int]:
        """Get distribution of detected regimes.

        Returns:
            Dictionary of regime counts
        """
        distribution = {regime: 0 for regime in MarketRegime}

        for state in self.regime_history:
            distribution[state.regime] += 1

        return distribution

    def get_regime_transitions(self) -> Dict[Tuple[MarketRegime, MarketRegime], int]:
        """Get regime transition counts.

        Returns:
            Dictionary of transition counts
        """
        transitions: Dict[Tuple[MarketRegime, MarketRegime], int] = {}

        for i in range(1, len(self.regime_history)):
            prev_regime = self.regime_history[i - 1].regime
            curr_regime = self.regime_history[i].regime

            transition = (prev_regime, curr_regime)
            transitions[transition] = transitions.get(transition, 0) + 1

        return transitions

    def clear_history(self) -> None:
        """Clear regime history."""
        self.regime_history.clear()
        logger.info("Regime history cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "fitted": self._fitted,
            "num_states": self.config.num_states,
            "history_length": len(self.regime_history),
            "current_regime": self.regime_history[-1].regime.value if self.regime_history else None,
            "use_gmm": self.config.use_gmm
        }
