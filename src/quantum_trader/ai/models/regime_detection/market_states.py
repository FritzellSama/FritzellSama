"""
Market State Detection and Classification.

Production implementation for identifying market regimes using statistical methods.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class MarketState(Enum):
    """Market regime states."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    CALM = "CALM"


@dataclass
class RegimeDetectionResult:
    """Result of regime detection."""
    current_state: MarketState
    state_probability: Decimal
    timestamp: datetime
    features: Dict[str, Decimal]


class MarketStateDetector:
    """
    Detect and classify market regimes.

    Example:
        >>> config = load_config('config/ai/regime_detection.yaml')
        >>> detector = MarketStateDetector(config)
        >>> state = detector.detect_state(market_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._validate_config()

        self.window_size = config['window_size']
        self.vol_window = config.get('volatility_window', 20)
        self.trend_threshold = Decimal(str(config.get('trend_threshold', 0.02)))
        self.vol_threshold_high = Decimal(str(config.get('vol_threshold_high', 0.03)))
        self.vol_threshold_low = Decimal(str(config.get('vol_threshold_low', 0.005)))

        logger.info("Market state detector initialized")

    def _validate_config(self) -> None:
        if 'window_size' not in self.config:
            raise ValueError("Missing required config: window_size")

    def detect_state(self, data: pl.DataFrame) -> RegimeDetectionResult:
        """Detect current market state."""
        if len(data) < self.window_size:
            raise ValueError(f"Insufficient data: {len(data)} < {self.window_size}")

        features = self._calculate_features(data)

        state = self._classify_state(features)

        confidence = self._calculate_confidence(features, state)

        return RegimeDetectionResult(
            current_state=state,
            state_probability=confidence,
            timestamp=datetime.utcnow(),
            features=features
        )

    def _calculate_features(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate regime detection features."""
        recent_data = data.tail(self.window_size)

        closes = recent_data['close'].to_numpy()

        # Calculate returns
        returns = np.diff(closes) / closes[:-1]
        avg_return = Decimal(str(np.mean(returns)))

        # Calculate volatility
        volatility = Decimal(str(np.std(returns)))

        # Calculate trend strength
        if len(closes) > 1:
            trend_slope = Decimal(str((closes[-1] - closes[0]) / closes[0]))
        else:
            trend_slope = Decimal('0')

        # Volume analysis
        if 'volume' in recent_data.columns:
            volumes = recent_data['volume'].to_numpy()
            avg_volume = Decimal(str(np.mean(volumes)))
            volume_std = Decimal(str(np.std(volumes)))
        else:
            avg_volume = Decimal('0')
            volume_std = Decimal('0')

        return {
            'avg_return': avg_return,
            'volatility': volatility,
            'trend_slope': trend_slope,
            'avg_volume': avg_volume,
            'volume_std': volume_std
        }

    def _classify_state(self, features: Dict[str, Decimal]) -> MarketState:
        """Classify market state based on features."""
        volatility = features['volatility']
        trend_slope = features['trend_slope']

        # High volatility
        if volatility > self.vol_threshold_high:
            return MarketState.VOLATILE

        # Low volatility
        if volatility < self.vol_threshold_low:
            return MarketState.CALM

        # Strong uptrend
        if trend_slope > self.trend_threshold:
            return MarketState.TRENDING_UP

        # Strong downtrend
        if trend_slope < -self.trend_threshold:
            return MarketState.TRENDING_DOWN

        # Default to ranging
        return MarketState.RANGING

    def _calculate_confidence(self, features: Dict[str, Decimal], state: MarketState) -> Decimal:
        """Calculate confidence in state classification."""
        volatility = features['volatility']
        trend_slope = abs(features['trend_slope'])

        if state == MarketState.VOLATILE:
            return min(volatility / self.vol_threshold_high, Decimal('1'))
        elif state == MarketState.CALM:
            return min(self.vol_threshold_low / (volatility + Decimal('0.001')), Decimal('1'))
        elif state in [MarketState.TRENDING_UP, MarketState.TRENDING_DOWN]:
            return min(trend_slope / self.trend_threshold, Decimal('1'))
        else:
            return Decimal('0.5')
