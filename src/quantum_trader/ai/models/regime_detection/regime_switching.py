"""Regime switching model for market state detection.

This module implements Hidden Markov Models (HMM) and Markov Switching models
for detecting different market regimes (trending, ranging, volatile, etc.).
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import polars as pl
import structlog
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm

logger = structlog.get_logger(__name__)


class MarketRegime(Enum):
    """Market regime types."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    CRISIS = "CRISIS"
    RECOVERY = "RECOVERY"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeState:
    """Current regime state.

    Attributes:
        regime: Current regime type
        probability: Probability of current regime (Decimal)
        transition_probs: Probabilities of transitioning to other regimes
        features: Feature values that determined regime
        timestamp: UTC timestamp
        duration: Duration in current regime (seconds)
    """
    regime: MarketRegime
    probability: Decimal
    transition_probs: Dict[MarketRegime, Decimal]
    features: Dict[str, Decimal]
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())
    duration: Optional[float] = None


class RegimeSwitchingModel:
    """Regime switching model using Hidden Markov Models.

    Detects different market regimes based on price action, volatility,
    and other market features using HMM.

    Attributes:
        config: Configuration dictionary
        n_regimes: Number of regimes to detect
        model: HMM model
        regime_labels: Mapping from state indices to regime types

    Example:
        >>> config = {
        ...     'n_regimes': 4,
        ...     'covariance_type': 'full',
        ...     'n_iter': 100,
        ...     'random_state': 42,
        ...     'min_probability': 0.6,
        ...     'feature_columns': ['returns', 'volatility', 'volume']
        ... }
        >>> model = RegimeSwitchingModel(config)
        >>> model.fit(market_data)
        >>> current_regime = model.predict_current_regime(latest_features)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize regime switching model.

        Args:
            config: Configuration with keys:
                - n_regimes: Number of regimes
                - covariance_type: 'spherical', 'diag', 'full', 'tied'
                - n_iter: Maximum iterations
                - random_state: Random seed
                - min_probability: Minimum probability for regime assignment
                - feature_columns: List of feature column names
        """
        self.config = config
        self._validate_config()

        self.n_regimes = config['n_regimes']
        self.covariance_type = config['covariance_type']
        self.min_probability = Decimal(str(config['min_probability']))
        self.feature_columns = config['feature_columns']

        # Initialize HMM model
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=config['n_iter'],
            random_state=config.get('random_state'),
            verbose=False
        )

        # State tracking
        self.is_fitted = False
        self.regime_labels: Dict[int, MarketRegime] = {}
        self.regime_characteristics: Dict[int, Dict[str, Decimal]] = {}
        self.current_regime: Optional[RegimeState] = None
        self.regime_history: List[RegimeState] = []

        logger.info(
            "initialized_regime_switching_model",
            n_regimes=self.n_regimes,
            covariance_type=self.covariance_type,
            features=self.feature_columns
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config is invalid
        """
        required_keys = [
            'n_regimes', 'covariance_type', 'n_iter',
            'min_probability', 'feature_columns'
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if self.config['n_regimes'] < 2:
            raise ValueError("n_regimes must be at least 2")

        if self.config['covariance_type'] not in ['spherical', 'diag', 'full', 'tied']:
            raise ValueError("Invalid covariance_type")

        if not (0 < self.config['min_probability'] <= 1):
            raise ValueError("min_probability must be in (0, 1]")

        if not self.config['feature_columns']:
            raise ValueError("feature_columns cannot be empty")

    def _prepare_features(self, data: pl.DataFrame) -> np.ndarray:
        """Prepare feature matrix from dataframe.

        Args:
            data: Polars dataframe with feature columns

        Returns:
            Feature matrix (n_samples, n_features)

        Raises:
            ValueError: If required columns missing
        """
        try:
            # Validate columns
            missing_cols = set(self.feature_columns) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing feature columns: {missing_cols}")

            # Select and convert features
            features = data.select(self.feature_columns).to_numpy()

            # Check for NaN or Inf
            if np.any(~np.isfinite(features)):
                logger.warning("non_finite_values_detected", action="imputing")
                # Simple forward fill and backward fill
                for col_idx in range(features.shape[1]):
                    col = features[:, col_idx]
                    mask = ~np.isfinite(col)
                    if np.any(mask):
                        # Forward fill
                        idx = np.where(~mask, np.arange(len(col)), 0)
                        np.maximum.accumulate(idx, axis=0, out=idx)
                        features[:, col_idx] = col[idx]

            return features

        except Exception as e:
            logger.error("failed_to_prepare_features", error=str(e))
            raise

    def fit(self, data: pl.DataFrame) -> None:
        """Fit regime switching model to historical data.

        Args:
            data: Historical market data with feature columns

        Raises:
            ValueError: If data is invalid
        """
        try:
            logger.info("starting_model_fitting", n_samples=len(data))

            # Prepare features
            features = self._prepare_features(data)

            if len(features) < self.n_regimes * 10:
                raise ValueError(
                    f"Need at least {self.n_regimes * 10} samples, got {len(features)}"
                )

            # Fit HMM
            self.model.fit(features)
            self.is_fitted = True

            # Predict states for all data
            states = self.model.predict(features)

            # Characterize each regime
            self._characterize_regimes(features, states)

            # Assign regime labels based on characteristics
            self._assign_regime_labels()

            logger.info(
                "model_fitting_complete",
                converged=self.model.monitor_.converged,
                n_iter=self.model.monitor_.iter,
                regime_labels={k: v.value for k, v in self.regime_labels.items()}
            )

        except Exception as e:
            logger.error("failed_to_fit_model", error=str(e))
            raise

    def _characterize_regimes(
        self,
        features: np.ndarray,
        states: np.ndarray
    ) -> None:
        """Characterize each regime based on feature statistics.

        Args:
            features: Feature matrix
            states: Predicted states
        """
        for state_idx in range(self.n_regimes):
            mask = states == state_idx
            if np.sum(mask) == 0:
                continue

            state_features = features[mask]

            characteristics = {}
            for i, col_name in enumerate(self.feature_columns):
                col_data = state_features[:, i]
                characteristics[f'{col_name}_mean'] = Decimal(str(np.mean(col_data)))
                characteristics[f'{col_name}_std'] = Decimal(str(np.std(col_data)))
                characteristics[f'{col_name}_min'] = Decimal(str(np.min(col_data)))
                characteristics[f'{col_name}_max'] = Decimal(str(np.max(col_data)))

            self.regime_characteristics[state_idx] = characteristics

            logger.debug(
                "characterized_regime",
                state_idx=state_idx,
                n_samples=int(np.sum(mask)),
                characteristics={k: str(v) for k, v in characteristics.items()}
            )

    def _assign_regime_labels(self) -> None:
        """Assign regime labels based on characteristics."""
        for state_idx in range(self.n_regimes):
            if state_idx not in self.regime_characteristics:
                self.regime_labels[state_idx] = MarketRegime.UNKNOWN
                continue

            char = self.regime_characteristics[state_idx]

            # Determine regime type based on characteristics
            # This is a simplified heuristic - adjust based on your features

            # Check for trending regimes (if returns feature available)
            if 'returns_mean' in char:
                returns_mean = char['returns_mean']
                returns_std = char.get('returns_std', Decimal('0'))

                # High positive returns with moderate volatility
                if returns_mean > Decimal('0.001') and returns_std < Decimal('0.02'):
                    self.regime_labels[state_idx] = MarketRegime.TRENDING_UP
                    continue

                # High negative returns
                elif returns_mean < Decimal('-0.001'):
                    if returns_std > Decimal('0.03'):
                        self.regime_labels[state_idx] = MarketRegime.CRISIS
                    else:
                        self.regime_labels[state_idx] = MarketRegime.TRENDING_DOWN
                    continue

            # Check for volatility regimes
            if 'volatility_mean' in char:
                vol_mean = char['volatility_mean']

                if vol_mean > Decimal('0.03'):
                    self.regime_labels[state_idx] = MarketRegime.HIGH_VOLATILITY
                    continue
                elif vol_mean < Decimal('0.01'):
                    self.regime_labels[state_idx] = MarketRegime.LOW_VOLATILITY
                    continue

            # Check for ranging regime (low returns, low volatility)
            if 'returns_mean' in char and 'volatility_mean' in char:
                if (abs(char['returns_mean']) < Decimal('0.0005') and
                    char['volatility_mean'] < Decimal('0.015')):
                    self.regime_labels[state_idx] = MarketRegime.RANGING
                    continue

            # Default to unknown
            self.regime_labels[state_idx] = MarketRegime.UNKNOWN

    def predict_current_regime(
        self,
        features: Dict[str, Decimal]
    ) -> RegimeState:
        """Predict current market regime.

        Args:
            features: Current feature values

        Returns:
            RegimeState object

        Raises:
            ValueError: If model not fitted
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before prediction")

            # Convert features to array
            feature_array = np.array([
                [float(features[col]) for col in self.feature_columns]
            ])

            # Predict state probabilities
            state_probs = self.model.predict_proba(feature_array)[0]

            # Get most likely state
            state_idx = int(np.argmax(state_probs))
            probability = Decimal(str(state_probs[state_idx]))

            # Get regime label
            regime = self.regime_labels.get(state_idx, MarketRegime.UNKNOWN)

            # Calculate transition probabilities
            transition_probs = {}
            for next_state_idx in range(self.n_regimes):
                next_regime = self.regime_labels.get(next_state_idx, MarketRegime.UNKNOWN)
                trans_prob = Decimal(str(self.model.transmat_[state_idx, next_state_idx]))
                transition_probs[next_regime] = trans_prob

            # Calculate duration if we have history
            duration = None
            if self.current_regime is not None:
                if self.current_regime.regime == regime:
                    prev_duration = self.current_regime.duration or 0
                    duration = prev_duration + (datetime.utcnow() - self.current_regime.timestamp).total_seconds()
                else:
                    duration = 0.0

            regime_state = RegimeState(
                regime=regime,
                probability=probability,
                transition_probs=transition_probs,
                features=features,
                duration=duration
            )

            # Update current regime
            self.current_regime = regime_state
            self.regime_history.append(regime_state)

            # Keep history limited
            max_history = self.config.get('max_history_size', 1000)
            if len(self.regime_history) > max_history:
                self.regime_history = self.regime_history[-max_history:]

            logger.debug(
                "predicted_regime",
                regime=regime.value,
                probability=str(probability),
                duration=duration
            )

            return regime_state

        except Exception as e:
            logger.error("failed_to_predict_regime", error=str(e))
            raise

    def predict_regimes_batch(
        self,
        data: pl.DataFrame
    ) -> List[MarketRegime]:
        """Predict regimes for batch of data.

        Args:
            data: Dataframe with feature columns

        Returns:
            List of predicted regimes

        Raises:
            ValueError: If model not fitted
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before prediction")

            features = self._prepare_features(data)
            states = self.model.predict(features)

            regimes = [
                self.regime_labels.get(state, MarketRegime.UNKNOWN)
                for state in states
            ]

            logger.debug("predicted_batch_regimes", n_samples=len(regimes))

            return regimes

        except Exception as e:
            logger.error("failed_to_predict_batch", error=str(e))
            raise

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime distribution and transitions.

        Returns:
            Dictionary of regime statistics
        """
        if not self.regime_history:
            return {'error': 'No regime history available'}

        # Count regime occurrences
        regime_counts = {}
        for state in self.regime_history:
            regime_name = state.regime.value
            regime_counts[regime_name] = regime_counts.get(regime_name, 0) + 1

        # Calculate average durations
        regime_durations = {}
        for state in self.regime_history:
            if state.duration is not None:
                regime_name = state.regime.value
                if regime_name not in regime_durations:
                    regime_durations[regime_name] = []
                regime_durations[regime_name].append(state.duration)

        avg_durations = {
            regime: float(np.mean(durations))
            for regime, durations in regime_durations.items()
        }

        # Transition matrix from history
        transitions = np.zeros((len(MarketRegime), len(MarketRegime)))
        regime_to_idx = {r: i for i, r in enumerate(MarketRegime)}

        for i in range(len(self.regime_history) - 1):
            current = regime_to_idx[self.regime_history[i].regime]
            next_regime = regime_to_idx[self.regime_history[i + 1].regime]
            transitions[current, next_regime] += 1

        # Normalize
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_probs = transitions / row_sums

        stats = {
            'total_states': len(self.regime_history),
            'regime_counts': regime_counts,
            'regime_probabilities': {
                k: v / len(self.regime_history)
                for k, v in regime_counts.items()
            },
            'average_durations': avg_durations,
            'current_regime': self.current_regime.regime.value if self.current_regime else None,
            'transition_matrix': transition_probs.tolist(),
            'regime_characteristics': {
                self.regime_labels[k].value: {
                    feat: str(val) for feat, val in v.items()
                }
                for k, v in self.regime_characteristics.items()
            }
        }

        return stats

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Save path
        """
        try:
            import pickle

            model_data = {
                'model': self.model,
                'config': self.config,
                'is_fitted': self.is_fitted,
                'regime_labels': {k: v.value for k, v in self.regime_labels.items()},
                'regime_characteristics': {
                    k: {feat: str(val) for feat, val in v.items()}
                    for k, v in self.regime_characteristics.items()
                }
            }

            with open(path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info("saved_model", path=path)

        except Exception as e:
            logger.error("failed_to_save_model", error=str(e), path=path)
            raise

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Load path
        """
        try:
            import pickle

            with open(path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.config = model_data['config']
            self.is_fitted = model_data['is_fitted']

            # Reconstruct regime labels
            self.regime_labels = {
                k: MarketRegime(v)
                for k, v in model_data['regime_labels'].items()
            }

            # Reconstruct characteristics
            self.regime_characteristics = {
                k: {feat: Decimal(val) for feat, val in v.items()}
                for k, v in model_data['regime_characteristics'].items()
            }

            logger.info("loaded_model", path=path)

        except Exception as e:
            logger.error("failed_to_load_model", error=str(e), path=path)
            raise
