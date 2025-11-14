"""Market regime detection using Hidden Markov Models and clustering.

This module implements sophisticated market state detection to identify
different market regimes (trending, ranging, volatile) for adaptive trading.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel

logger = get_logger(__name__)


class RegimeType(Enum):
    """Market regime types."""

    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    CRISIS = "CRISIS"
    RECOVERY = "RECOVERY"


@dataclass
class RegimeFeatures:
    """Features for regime detection.

    Attributes:
        returns: Price returns
        volatility: Rolling volatility
        volume: Trading volume
        trend_strength: Trend strength indicator
        momentum: Price momentum
        range_ratio: High-low range ratio
        autocorrelation: Price autocorrelation
        skewness: Returns skewness
        kurtosis: Returns kurtosis
    """

    returns: Decimal
    volatility: Decimal
    volume: Decimal
    trend_strength: Decimal
    momentum: Decimal
    range_ratio: Decimal
    autocorrelation: Decimal
    skewness: Decimal
    kurtosis: Decimal


@dataclass
class RegimeState:
    """Current market regime state.

    Attributes:
        regime: Current regime type
        confidence: Confidence in regime classification
        duration: Duration in current regime (bars)
        transition_probability: Probability of regime transition
        features: Current regime features
        timestamp: Timestamp of regime detection
    """

    regime: RegimeType
    confidence: Decimal
    duration: int
    transition_probability: Decimal
    features: RegimeFeatures
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class HiddenMarkovRegimeDetector:
    """Hidden Markov Model for regime detection.

    Uses HMM to model regime transitions and identify current market state.
    """

    def __init__(self, n_states: int, config: Dict[str, Any]) -> None:
        """Initialize HMM regime detector.

        Args:
            n_states: Number of hidden states (regimes)
            config: Configuration dictionary
        """
        self.n_states = n_states
        self.config = config

        # Transition matrix (n_states x n_states)
        self.transition_matrix = self._initialize_transition_matrix()

        # Emission parameters (mean and covariance for each state)
        self.emission_means: Optional[np.ndarray] = None
        self.emission_covs: Optional[np.ndarray] = None

        # Initial state distribution
        self.initial_state = np.ones(n_states) / n_states

        # Current state
        self.current_state = 0
        self.state_duration = 0

        logger.info("hmm_detector_initialized", n_states=n_states)

    def _initialize_transition_matrix(self) -> np.ndarray:
        """Initialize transition matrix.

        Returns:
            Transition matrix
        """
        # Initialize with slight bias towards staying in same state
        matrix = np.ones((self.n_states, self.n_states)) * 0.1
        np.fill_diagonal(matrix, 0.7)

        # Normalize rows
        matrix = matrix / matrix.sum(axis=1, keepdims=True)

        return matrix

    def fit(self, features: np.ndarray) -> None:
        """Fit HMM to observed features.

        Args:
            features: Feature matrix (n_samples, n_features)
        """
        try:
            # Use Gaussian Mixture Model for initial parameter estimation
            gmm = GaussianMixture(
                n_components=self.n_states,
                covariance_type='full',
                max_iter=100,
                random_state=42
            )

            gmm.fit(features)

            # Extract parameters
            self.emission_means = gmm.means_
            self.emission_covs = gmm.covariances_

            # Use GMM predictions to estimate transition probabilities
            states = gmm.predict(features)

            # Count transitions
            transition_counts = np.zeros((self.n_states, self.n_states))

            for i in range(len(states) - 1):
                transition_counts[states[i], states[i + 1]] += 1

            # Normalize to get probabilities
            row_sums = transition_counts.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero

            self.transition_matrix = transition_counts / row_sums

            logger.info("hmm_fitted", n_samples=len(features))

        except Exception as e:
            logger.error("hmm_fit_failed", error=str(e))
            raise

    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """Predict current regime.

        Args:
            features: Current feature vector

        Returns:
            Tuple of (regime_index, confidence)
        """
        try:
            if self.emission_means is None:
                return 0, 0.0

            # Calculate likelihood for each state
            likelihoods = np.zeros(self.n_states)

            for state in range(self.n_states):
                mean = self.emission_means[state]
                cov = self.emission_covs[state]

                # Multivariate Gaussian likelihood
                diff = features - mean
                inv_cov = np.linalg.inv(cov + np.eye(len(cov)) * 1e-6)

                likelihood = np.exp(
                    -0.5 * diff @ inv_cov @ diff.T
                ) / np.sqrt(np.linalg.det(2 * np.pi * cov))

                likelihoods[state] = likelihood

            # Combine with transition probabilities
            posteriors = likelihoods * self.transition_matrix[self.current_state]
            posteriors = posteriors / (posteriors.sum() + 1e-10)

            # Get most likely state
            predicted_state = int(np.argmax(posteriors))
            confidence = float(posteriors[predicted_state])

            # Update state tracking
            if predicted_state == self.current_state:
                self.state_duration += 1
            else:
                self.current_state = predicted_state
                self.state_duration = 1

            return predicted_state, confidence

        except Exception as e:
            logger.error("prediction_failed", error=str(e))
            return 0, 0.0

    def get_transition_probability(self, from_state: int, to_state: int) -> float:
        """Get transition probability between states.

        Args:
            from_state: Source state
            to_state: Target state

        Returns:
            Transition probability
        """
        return float(self.transition_matrix[from_state, to_state])


class MarketStateDetector(BaseMLModel):
    """Advanced market regime detection system.

    Combines HMM, clustering, and statistical methods to identify market regimes
    and provide actionable trading signals based on regime state.

    Example:
        >>> config = {
        ...     'n_regimes': 5,
        ...     'feature_window': 100,
        ...     'volatility_window': 20,
        ...     'trend_window': 50,
        ...     'update_frequency': 100,
        ...     'min_regime_duration': 10,
        ...     'transition_threshold': Decimal('0.7')
        ... }
        >>> detector = MarketStateDetector(config)
        >>> features, labels = prepare_data(market_data)
        >>> detector.train(features, labels)
        >>> regime_state = detector.detect_regime(current_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize market state detector.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        self.n_regimes = config.get('n_regimes', 5)
        self.feature_window = config.get('feature_window', 100)
        self.volatility_window = config.get('volatility_window', 20)
        self.trend_window = config.get('trend_window', 50)

        # Models
        self.hmm_detector = HiddenMarkovRegimeDetector(self.n_regimes, config)
        self.scaler = StandardScaler()

        # Regime mapping
        self.regime_mapping: Dict[int, RegimeType] = {}

        # State tracking
        self.current_regime: Optional[RegimeType] = None
        self.regime_history: List[RegimeType] = []
        self.feature_history: List[np.ndarray] = []

        logger.info(
            "market_state_detector_initialized",
            n_regimes=self.n_regimes
        )

    def extract_features(self, data: pl.DataFrame) -> np.ndarray:
        """Extract regime detection features from market data.

        Args:
            data: Market data DataFrame with OHLCV columns

        Returns:
            Feature array (n_samples, n_features)
        """
        try:
            features_list = []

            # Calculate returns
            closes = data.select('close').to_numpy().flatten()
            returns = np.diff(closes) / closes[:-1]

            # Rolling volatility
            volatility = self._rolling_std(returns, self.volatility_window)

            # Volume metrics
            volumes = data.select('volume').to_numpy().flatten()
            volume_ma = self._rolling_mean(volumes, self.volatility_window)
            volume_ratio = volumes[self.volatility_window:] / (volume_ma + 1e-10)

            # Trend strength (ADX-like)
            trend_strength = self._calculate_trend_strength(data)

            # Momentum
            momentum = self._calculate_momentum(closes, self.trend_window)

            # Range ratio
            highs = data.select('high').to_numpy().flatten()
            lows = data.select('low').to_numpy().flatten()
            range_ratio = (highs - lows) / (closes + 1e-10)

            # Statistical moments
            autocorr = self._calculate_autocorrelation(returns, lag=5)
            skewness = self._rolling_skewness(returns, self.volatility_window)
            kurtosis = self._rolling_kurtosis(returns, self.volatility_window)

            # Align lengths (take minimum common length)
            min_length = min(
                len(volatility),
                len(volume_ratio),
                len(trend_strength),
                len(momentum),
                len(range_ratio[self.volatility_window:]),
                len(autocorr),
                len(skewness),
                len(kurtosis)
            )

            # Stack features
            features = np.column_stack([
                volatility[-min_length:],
                volume_ratio[-min_length:],
                trend_strength[-min_length:],
                momentum[-min_length:],
                range_ratio[-min_length:],
                autocorr[-min_length:],
                skewness[-min_length:],
                kurtosis[-min_length:]
            ])

            return features

        except Exception as e:
            logger.error("feature_extraction_failed", error=str(e))
            return np.array([])

    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean.

        Args:
            data: Input data
            window: Window size

        Returns:
            Rolling mean
        """
        return np.convolve(data, np.ones(window) / window, mode='valid')

    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling standard deviation.

        Args:
            data: Input data
            window: Window size

        Returns:
            Rolling standard deviation
        """
        result = []

        for i in range(window, len(data) + 1):
            result.append(np.std(data[i - window:i]))

        return np.array(result)

    def _calculate_trend_strength(self, data: pl.DataFrame) -> np.ndarray:
        """Calculate trend strength indicator.

        Args:
            data: Market data

        Returns:
            Trend strength array
        """
        closes = data.select('close').to_numpy().flatten()
        highs = data.select('high').to_numpy().flatten()
        lows = data.select('low').to_numpy().flatten()

        # Calculate directional movement
        dm_plus = np.maximum(highs[1:] - highs[:-1], 0)
        dm_minus = np.maximum(lows[:-1] - lows[1:], 0)

        # True range
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                abs(highs[1:] - closes[:-1]),
                abs(lows[1:] - closes[:-1])
            )
        )

        # Smooth and normalize
        window = min(14, len(dm_plus) // 2)

        if window < 2:
            return np.zeros(len(closes) - 1)

        dm_plus_smooth = self._rolling_mean(dm_plus, window)
        dm_minus_smooth = self._rolling_mean(dm_minus, window)
        tr_smooth = self._rolling_mean(tr, window)

        # Directional indicators
        di_plus = 100 * dm_plus_smooth / (tr_smooth + 1e-10)
        di_minus = 100 * dm_minus_smooth / (tr_smooth + 1e-10)

        # ADX-like trend strength
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)

        return dx

    def _calculate_momentum(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate price momentum.

        Args:
            prices: Price array
            window: Window size

        Returns:
            Momentum array
        """
        if len(prices) <= window:
            return np.zeros(len(prices))

        momentum = (prices[window:] - prices[:-window]) / (prices[:-window] + 1e-10)

        return momentum

    def _calculate_autocorrelation(self, data: np.ndarray, lag: int) -> np.ndarray:
        """Calculate autocorrelation.

        Args:
            data: Input data
            lag: Lag for autocorrelation

        Returns:
            Autocorrelation array
        """
        result = []

        window = 30

        for i in range(window, len(data)):
            series = data[i - window:i]
            if len(series) > lag:
                corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                result.append(corr if not np.isnan(corr) else 0.0)

        return np.array(result)

    def _rolling_skewness(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling skewness.

        Args:
            data: Input data
            window: Window size

        Returns:
            Rolling skewness
        """
        result = []

        for i in range(window, len(data) + 1):
            window_data = data[i - window:i]
            mean = np.mean(window_data)
            std = np.std(window_data)

            if std > 0:
                skew = np.mean(((window_data - mean) / std) ** 3)
                result.append(skew)
            else:
                result.append(0.0)

        return np.array(result)

    def _rolling_kurtosis(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling kurtosis.

        Args:
            data: Input data
            window: Window size

        Returns:
            Rolling kurtosis
        """
        result = []

        for i in range(window, len(data) + 1):
            window_data = data[i - window:i]
            mean = np.mean(window_data)
            std = np.std(window_data)

            if std > 0:
                kurt = np.mean(((window_data - mean) / std) ** 4) - 3
                result.append(kurt)
            else:
                result.append(0.0)

        return np.array(result)

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train regime detection models.

        Args:
            features: Training features
            labels: Training labels (not used for unsupervised learning)
        """
        try:
            logger.info("training_started", n_samples=len(features))

            # Scale features
            scaled_features = self.scaler.fit_transform(features)

            # Fit HMM
            self.hmm_detector.fit(scaled_features)

            # Cluster regimes to create mapping
            kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_features)

            # Map clusters to regime types based on characteristics
            self._create_regime_mapping(kmeans.cluster_centers_)

            logger.info("training_completed", n_regimes=self.n_regimes)

        except Exception as e:
            logger.error("training_failed", error=str(e))
            raise

    def _create_regime_mapping(self, cluster_centers: np.ndarray) -> None:
        """Create mapping from cluster indices to regime types.

        Args:
            cluster_centers: Cluster center features
        """
        # Map based on feature characteristics
        # Feature order: volatility, volume_ratio, trend_strength, momentum, etc.

        for idx, center in enumerate(cluster_centers):
            volatility = center[0]
            trend_strength = center[2]
            momentum = center[3]

            # High volatility
            if volatility > 1.0:
                if abs(momentum) < 0.5:
                    self.regime_mapping[idx] = RegimeType.CRISIS
                else:
                    self.regime_mapping[idx] = RegimeType.HIGH_VOLATILITY

            # Low volatility
            elif volatility < 0.3:
                self.regime_mapping[idx] = RegimeType.LOW_VOLATILITY

            # Trending
            elif trend_strength > 0.5:
                if momentum > 0:
                    self.regime_mapping[idx] = RegimeType.BULL_TREND
                else:
                    self.regime_mapping[idx] = RegimeType.BEAR_TREND

            # Sideways
            else:
                self.regime_mapping[idx] = RegimeType.SIDEWAYS

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict regime for given features.

        Args:
            features: Input features

        Returns:
            Regime predictions
        """
        try:
            scaled_features = self.scaler.transform(features)

            predictions = []

            for feature_vec in scaled_features:
                regime_idx, _ = self.hmm_detector.predict(feature_vec)
                predictions.append(regime_idx)

            return np.array(predictions)

        except Exception as e:
            logger.error("prediction_failed", error=str(e))
            return np.zeros(len(features))

    def detect_regime(self, data: pl.DataFrame) -> RegimeState:
        """Detect current market regime.

        Args:
            data: Recent market data

        Returns:
            Current regime state
        """
        try:
            # Extract features
            features = self.extract_features(data)

            if len(features) == 0:
                return self._get_default_regime_state()

            # Get latest features
            latest_features = features[-1]

            # Scale features
            scaled_features = self.scaler.transform([latest_features])[0]

            # Predict regime
            regime_idx, confidence = self.hmm_detector.predict(scaled_features)

            # Map to regime type
            regime_type = self.regime_mapping.get(regime_idx, RegimeType.SIDEWAYS)

            # Calculate transition probability
            transition_probs = [
                self.hmm_detector.get_transition_probability(regime_idx, i)
                for i in range(self.n_regimes)
                if i != regime_idx
            ]

            max_transition_prob = max(transition_probs) if transition_probs else 0.0

            # Create regime features
            regime_features = RegimeFeatures(
                returns=Decimal(str(latest_features[0])),
                volatility=Decimal(str(latest_features[0])),
                volume=Decimal(str(latest_features[1])),
                trend_strength=Decimal(str(latest_features[2])),
                momentum=Decimal(str(latest_features[3])),
                range_ratio=Decimal(str(latest_features[4])),
                autocorrelation=Decimal(str(latest_features[5])),
                skewness=Decimal(str(latest_features[6])),
                kurtosis=Decimal(str(latest_features[7]))
            )

            # Create regime state
            regime_state = RegimeState(
                regime=regime_type,
                confidence=Decimal(str(confidence)),
                duration=self.hmm_detector.state_duration,
                transition_probability=Decimal(str(max_transition_prob)),
                features=regime_features
            )

            # Update history
            self.current_regime = regime_type
            self.regime_history.append(regime_type)
            self.feature_history.append(latest_features)

            return regime_state

        except Exception as e:
            logger.error("regime_detection_failed", error=str(e))
            return self._get_default_regime_state()

    def _get_default_regime_state(self) -> RegimeState:
        """Get default regime state.

        Returns:
            Default regime state
        """
        return RegimeState(
            regime=RegimeType.SIDEWAYS,
            confidence=Decimal("0.0"),
            duration=0,
            transition_probability=Decimal("0.0"),
            features=RegimeFeatures(
                returns=Decimal("0"),
                volatility=Decimal("0"),
                volume=Decimal("0"),
                trend_strength=Decimal("0"),
                momentum=Decimal("0"),
                range_ratio=Decimal("0"),
                autocorrelation=Decimal("0"),
                skewness=Decimal("0"),
                kurtosis=Decimal("0")
            )
        )

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate regime detection performance.

        Args:
            features: Test features
            labels: True labels

        Returns:
            Evaluation metrics
        """
        try:
            predictions = self.predict(features)

            # Calculate stability (how often regime stays the same)
            stability = np.mean(predictions[1:] == predictions[:-1])

            # Calculate diversity (how many different regimes detected)
            diversity = len(np.unique(predictions)) / self.n_regimes

            return {
                'stability': float(stability),
                'diversity': float(diversity),
                'n_transitions': int(np.sum(predictions[1:] != predictions[:-1]))
            }

        except Exception as e:
            logger.error("evaluation_failed", error=str(e))
            return {}

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Save path
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            state = {
                'config': self.config,
                'scaler': self.scaler,
                'hmm_transition_matrix': self.hmm_detector.transition_matrix,
                'hmm_emission_means': self.hmm_detector.emission_means,
                'hmm_emission_covs': self.hmm_detector.emission_covs,
                'regime_mapping': {k: v.value for k, v in self.regime_mapping.items()},
                'regime_history': [r.value for r in self.regime_history]
            }

            import pickle

            with open(save_path, 'wb') as f:
                pickle.dump(state, f)

            logger.info("model_saved", path=path)

        except Exception as e:
            logger.error("save_failed", error=str(e))
            raise

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Load path
        """
        try:
            import pickle

            with open(path, 'rb') as f:
                state = pickle.load(f)

            self.config = state['config']
            self.scaler = state['scaler']

            self.hmm_detector.transition_matrix = state['hmm_transition_matrix']
            self.hmm_detector.emission_means = state['hmm_emission_means']
            self.hmm_detector.emission_covs = state['hmm_emission_covs']

            self.regime_mapping = {
                k: RegimeType(v) for k, v in state['regime_mapping'].items()
            }

            self.regime_history = [
                RegimeType(r) for r in state['regime_history']
            ]

            logger.info("model_loaded", path=path)

        except Exception as e:
            logger.error("load_failed", error=str(e))
            raise
