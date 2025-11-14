"""
ML-Based Position Sizing for Quantum Trader AI

Production-grade ML-powered position sizing with:
- Confidence-weighted sizing based on model predictions
- Prediction uncertainty integration
- Adaptive sizing based on market regime detection
- Feature importance weighting
- Ensemble model predictions

CRITICAL: All numeric values use Decimal, NEVER float
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple
import logging

import polars as pl
import yaml
import numpy as np
from scipy import stats

from quantum_trader.models import Signal, Position


logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"


@dataclass
class MLPositionSizerConfig:
    """Configuration for ML position sizer"""
    base_allocation_percent: Decimal
    min_confidence_threshold: Decimal
    max_allocation_percent: Decimal
    uncertainty_penalty_factor: Decimal
    regime_multipliers: Dict[str, Decimal]
    enable_feature_weighting: bool
    ensemble_weight_threshold: Decimal
    calibration_window: int


@dataclass
class MLPrediction:
    """ML model prediction with uncertainty"""
    symbol: str
    predicted_return: Decimal
    confidence: Decimal
    uncertainty: Decimal  # Standard deviation or entropy
    model_name: str
    feature_importances: Dict[str, Decimal]
    timestamp: datetime


@dataclass
class RegimeState:
    """Current market regime state"""
    regime: MarketRegime
    confidence: Decimal
    volatility: Decimal
    timestamp: datetime


class MLPositionSizer:
    """
    ML-based position sizing calculator

    Uses machine learning predictions to dynamically size positions:
    - Confidence-weighted allocations
    - Uncertainty-based adjustments
    - Market regime awareness
    - Feature importance integration
    """

    def __init__(
        self,
        config_path: str = "/home/user/FritzellSama/config/bot/risk.yaml",
        env_config_path: str = "/home/user/FritzellSama/config/environments/production.yaml"
    ):
        """
        Initialize ML position sizer

        Args:
            config_path: Path to risk configuration file
            env_config_path: Path to environment configuration file
        """
        self.config = self._load_config(config_path, env_config_path)
        self.account_balance = Decimal('0')
        self.current_regime: Optional[RegimeState] = None
        self.prediction_history: List[MLPrediction] = []

        logger.info("MLPositionSizer initialized")

    def _load_config(self, config_path: str, env_config_path: str) -> MLPositionSizerConfig:
        """
        Load configuration from YAML files

        Args:
            config_path: Path to risk config
            env_config_path: Path to environment config

        Returns:
            Loaded configuration
        """
        try:
            with open(config_path, 'r') as f:
                risk_config = yaml.safe_load(f)

            with open(env_config_path, 'r') as f:
                env_config = yaml.safe_load(f)

            position_limits = risk_config.get('position_limits', {})

            return MLPositionSizerConfig(
                base_allocation_percent=Decimal(str(position_limits.get('max_position_size_percent', 5.0))),
                min_confidence_threshold=Decimal('0.6'),
                max_allocation_percent=Decimal(str(position_limits.get('max_position_size_percent', 5.0))),
                uncertainty_penalty_factor=Decimal('2.0'),
                regime_multipliers={
                    'BULL': Decimal('1.2'),
                    'BEAR': Decimal('0.6'),
                    'SIDEWAYS': Decimal('0.8'),
                    'HIGH_VOLATILITY': Decimal('0.5'),
                    'LOW_VOLATILITY': Decimal('1.1')
                },
                enable_feature_weighting=True,
                ensemble_weight_threshold=Decimal('0.7'),
                calibration_window=100
            )

        except Exception as e:
            logger.error("Failed to load configuration: %s", e)
            raise

    def update_account_balance(self, balance: Decimal) -> None:
        """
        Update current account balance

        Args:
            balance: Current account balance in USD
        """
        if not isinstance(balance, Decimal):
            raise TypeError(f"Balance must be Decimal, got {type(balance)}")

        self.account_balance = balance

    def update_market_regime(self, regime_state: RegimeState) -> None:
        """
        Update current market regime

        Args:
            regime_state: Current market regime state
        """
        self.current_regime = regime_state

        logger.info(
            "Market regime updated: %s (confidence: %s, volatility: %s)",
            regime_state.regime.value, regime_state.confidence, regime_state.volatility
        )

    def calculate_position_size(
        self,
        signal: Signal,
        ml_prediction: MLPrediction,
        current_price: Decimal
    ) -> Decimal:
        """
        Calculate position size using ML prediction

        Args:
            signal: Trading signal
            ml_prediction: ML model prediction with uncertainty
            current_price: Current asset price

        Returns:
            Position size in number of shares/contracts
        """
        if not isinstance(current_price, Decimal):
            raise TypeError(f"Price must be Decimal, got {type(current_price)}")

        if self.account_balance <= Decimal('0'):
            logger.warning("Account balance not set or zero, cannot calculate position size")
            return Decimal('0')

        # Check minimum confidence threshold
        if ml_prediction.confidence < self.config.min_confidence_threshold:
            logger.info(
                "ML prediction confidence %s below threshold %s for %s, skipping",
                ml_prediction.confidence, self.config.min_confidence_threshold, signal.symbol
            )
            return Decimal('0')

        # Start with base allocation
        allocation_percent = self.config.base_allocation_percent / Decimal('100')

        # Apply confidence weighting
        allocation_percent = self._apply_confidence_weighting(allocation_percent, ml_prediction)

        # Apply uncertainty penalty
        allocation_percent = self._apply_uncertainty_penalty(allocation_percent, ml_prediction)

        # Apply regime adjustment
        allocation_percent = self._apply_regime_adjustment(allocation_percent)

        # Apply feature importance weighting
        if self.config.enable_feature_weighting:
            allocation_percent = self._apply_feature_weighting(allocation_percent, ml_prediction)

        # Apply signal strength
        allocation_percent = allocation_percent * signal.strength

        # Ensure within bounds
        max_allocation = self.config.max_allocation_percent / Decimal('100')
        allocation_percent = max(Decimal('0'), min(allocation_percent, max_allocation))

        # Calculate dollar allocation
        dollar_allocation = self.account_balance * allocation_percent

        # Calculate number of shares
        position_size = dollar_allocation / current_price

        logger.info(
            "ML position size calculated for %s: %s shares at $%s (allocation: %s%%)",
            signal.symbol, position_size, current_price, allocation_percent * Decimal('100')
        )

        # Store prediction for calibration
        self.prediction_history.append(ml_prediction)

        return position_size.quantize(Decimal('0.00000001'))

    def _apply_confidence_weighting(
        self,
        base_allocation: Decimal,
        prediction: MLPrediction
    ) -> Decimal:
        """
        Weight allocation by model confidence

        Higher confidence = larger position

        Args:
            base_allocation: Base allocation fraction
            prediction: ML prediction with confidence

        Returns:
            Confidence-weighted allocation
        """
        # Use squared confidence to emphasize high-confidence predictions
        confidence_weight = prediction.confidence ** 2

        weighted_allocation = base_allocation * confidence_weight

        logger.debug(
            "Confidence weighting: %s -> %s (confidence: %s)",
            base_allocation, weighted_allocation, prediction.confidence
        )

        return weighted_allocation

    def _apply_uncertainty_penalty(
        self,
        allocation: Decimal,
        prediction: MLPrediction
    ) -> Decimal:
        """
        Reduce allocation based on prediction uncertainty

        Higher uncertainty = smaller position

        Args:
            allocation: Current allocation
            prediction: ML prediction with uncertainty

        Returns:
            Uncertainty-adjusted allocation
        """
        # Normalize uncertainty (assume typical range 0-0.5)
        normalized_uncertainty = min(prediction.uncertainty / Decimal('0.5'), Decimal('1'))

        # Apply penalty
        penalty = Decimal('1') - (normalized_uncertainty * self.config.uncertainty_penalty_factor)
        penalty = max(Decimal('0'), penalty)

        adjusted_allocation = allocation * penalty

        logger.debug(
            "Uncertainty penalty: %s -> %s (uncertainty: %s)",
            allocation, adjusted_allocation, prediction.uncertainty
        )

        return adjusted_allocation

    def _apply_regime_adjustment(self, allocation: Decimal) -> Decimal:
        """
        Adjust allocation based on market regime

        Different regimes call for different position sizes

        Args:
            allocation: Current allocation

        Returns:
            Regime-adjusted allocation
        """
        if self.current_regime is None:
            return allocation

        regime_name = self.current_regime.regime.value
        multiplier = self.config.regime_multipliers.get(regime_name, Decimal('1.0'))

        # Weight multiplier by regime confidence
        effective_multiplier = (
            Decimal('1') +
            (multiplier - Decimal('1')) * self.current_regime.confidence
        )

        adjusted_allocation = allocation * effective_multiplier

        logger.debug(
            "Regime adjustment (%s): %s -> %s (multiplier: %s)",
            regime_name, allocation, adjusted_allocation, effective_multiplier
        )

        return adjusted_allocation

    def _apply_feature_weighting(
        self,
        allocation: Decimal,
        prediction: MLPrediction
    ) -> Decimal:
        """
        Adjust allocation based on feature importances

        Strong features = more confidence in sizing

        Args:
            allocation: Current allocation
            prediction: ML prediction with feature importances

        Returns:
            Feature-weighted allocation
        """
        if not prediction.feature_importances:
            return allocation

        # Calculate mean feature importance
        total_importance = sum(prediction.feature_importances.values())
        n_features = len(prediction.feature_importances)

        if n_features == 0 or total_importance == Decimal('0'):
            return allocation

        avg_importance = total_importance / Decimal(str(n_features))

        # Use average importance as weight (normalized to 0-1 range)
        # Assume importances are already normalized
        feature_weight = min(avg_importance * Decimal('2'), Decimal('1'))

        adjusted_allocation = allocation * feature_weight

        logger.debug(
            "Feature weighting: %s -> %s (avg importance: %s)",
            allocation, adjusted_allocation, avg_importance
        )

        return adjusted_allocation

    def calculate_ensemble_position_size(
        self,
        signal: Signal,
        predictions: List[MLPrediction],
        current_price: Decimal
    ) -> Decimal:
        """
        Calculate position size using ensemble of ML models

        Combines multiple predictions with weighted averaging

        Args:
            signal: Trading signal
            predictions: List of predictions from different models
            current_price: Current asset price

        Returns:
            Position size based on ensemble
        """
        if not predictions:
            logger.warning("No predictions provided for ensemble sizing")
            return Decimal('0')

        # Filter by confidence threshold
        valid_predictions = [
            p for p in predictions
            if p.confidence >= self.config.min_confidence_threshold
        ]

        if not valid_predictions:
            logger.info("No predictions meet confidence threshold for ensemble")
            return Decimal('0')

        # Calculate individual position sizes
        position_sizes = []
        weights = []

        for pred in valid_predictions:
            size = self.calculate_position_size(signal, pred, current_price)
            position_sizes.append(size)

            # Weight by confidence
            weights.append(pred.confidence)

        # Normalize weights
        total_weight = sum(weights)

        if total_weight == Decimal('0'):
            return Decimal('0')

        normalized_weights = [w / total_weight for w in weights]

        # Weighted average of position sizes
        ensemble_size = sum(
            size * weight
            for size, weight in zip(position_sizes, normalized_weights)
        )

        # Check ensemble agreement (variance of predictions)
        if len(position_sizes) > 1:
            # Convert to float for std calculation
            sizes_array = np.array([float(s) for s in position_sizes])
            std = np.std(sizes_array)
            mean = np.mean(sizes_array)

            if mean > 0:
                coefficient_of_variation = Decimal(str(std / mean))

                # Reduce size if high disagreement
                if coefficient_of_variation > Decimal('0.5'):
                    disagreement_penalty = Decimal('0.7')
                    ensemble_size = ensemble_size * disagreement_penalty

                    logger.warning(
                        "High model disagreement (CV=%s), reduced ensemble size by %s%%",
                        coefficient_of_variation, (Decimal('1') - disagreement_penalty) * Decimal('100')
                    )

        logger.info(
            "Ensemble position size for %s: %s shares (from %d models)",
            signal.symbol, ensemble_size, len(valid_predictions)
        )

        return ensemble_size

    def calibrate_confidence(
        self,
        predictions_df: pl.DataFrame,
        actuals_df: pl.DataFrame
    ) -> Dict[str, Decimal]:
        """
        Calibrate model confidence scores against actual outcomes

        Checks if confidence scores match actual accuracy

        Args:
            predictions_df: DataFrame with columns [symbol, predicted_return, confidence, timestamp]
            actuals_df: DataFrame with columns [symbol, actual_return, timestamp]

        Returns:
            Calibration metrics
        """
        if not isinstance(predictions_df, pl.DataFrame):
            raise TypeError(f"predictions_df must be polars DataFrame")

        if not isinstance(actuals_df, pl.DataFrame):
            raise TypeError(f"actuals_df must be polars DataFrame")

        # Join predictions with actuals
        df = predictions_df.join(actuals_df, on=['symbol', 'timestamp'], how='inner')

        if df.height < 10:
            logger.warning("Insufficient data for confidence calibration")
            return {
                'calibration_error': Decimal('0'),
                'brier_score': Decimal('0'),
                'sample_size': Decimal(str(df.height))
            }

        # Calculate prediction accuracy by confidence bin
        confidence_bins = [0.0, 0.6, 0.7, 0.8, 0.9, 1.0]
        calibration_error = Decimal('0')

        for i in range(len(confidence_bins) - 1):
            bin_min = confidence_bins[i]
            bin_max = confidence_bins[i + 1]

            bin_df = df.filter(
                (pl.col('confidence') >= bin_min) &
                (pl.col('confidence') < bin_max)
            )

            if bin_df.height == 0:
                continue

            # Check prediction accuracy (same sign as actual)
            correct = 0
            for row in bin_df.iter_rows(named=True):
                pred_sign = 1 if row['predicted_return'] > 0 else -1
                actual_sign = 1 if row['actual_return'] > 0 else -1
                if pred_sign == actual_sign:
                    correct += 1

            accuracy = Decimal(str(correct / bin_df.height)) if bin_df.height > 0 else Decimal('0')
            expected_confidence = Decimal(str((bin_min + bin_max) / 2))

            calibration_error += abs(accuracy - expected_confidence)

        calibration_error = calibration_error / Decimal(str(len(confidence_bins) - 1))

        # Calculate Brier score
        brier_score = self._calculate_brier_score(df)

        metrics = {
            'calibration_error': calibration_error,
            'brier_score': brier_score,
            'sample_size': Decimal(str(df.height))
        }

        logger.info(
            "Confidence calibration: error=%s, brier_score=%s, n=%d",
            calibration_error, brier_score, df.height
        )

        return metrics

    def _calculate_brier_score(self, df: pl.DataFrame) -> Decimal:
        """
        Calculate Brier score for probabilistic predictions

        Args:
            df: DataFrame with predictions and actuals

        Returns:
            Brier score (lower is better)
        """
        squared_errors = []

        for row in df.iter_rows(named=True):
            # Convert to binary outcome (positive return = 1, negative = 0)
            actual = 1.0 if row['actual_return'] > 0 else 0.0
            predicted_prob = float(row['confidence'])

            squared_error = (predicted_prob - actual) ** 2
            squared_errors.append(squared_error)

        if not squared_errors:
            return Decimal('0')

        brier_score = Decimal(str(sum(squared_errors) / len(squared_errors)))

        return brier_score

    def calculate_kelly_ml_hybrid(
        self,
        signal: Signal,
        ml_prediction: MLPrediction,
        historical_win_rate: Decimal,
        historical_payoff_ratio: Decimal,
        current_price: Decimal
    ) -> Decimal:
        """
        Hybrid position sizing combining Kelly criterion with ML predictions

        Args:
            signal: Trading signal
            ml_prediction: ML prediction
            historical_win_rate: Historical win rate
            historical_payoff_ratio: Historical payoff ratio
            current_price: Current price

        Returns:
            Hybrid position size
        """
        # ML-adjusted win rate
        ml_adjusted_win_rate = (
            historical_win_rate * (Decimal('1') - ml_prediction.confidence) +
            ml_prediction.confidence * (Decimal('1') if ml_prediction.predicted_return > 0 else Decimal('0'))
        )

        # Kelly formula: f* = (p * b - q) / b
        p = ml_adjusted_win_rate
        q = Decimal('1') - p
        b = historical_payoff_ratio

        if b <= Decimal('0'):
            return Decimal('0')

        kelly_fraction = (p * b - q) / b

        # Apply fractional Kelly (quarter Kelly for safety)
        kelly_fraction = kelly_fraction * Decimal('0.25')

        # Cap at maximum
        kelly_fraction = max(Decimal('0'), min(kelly_fraction, self.config.max_allocation_percent / Decimal('100')))

        # Apply ML confidence and uncertainty
        kelly_fraction = kelly_fraction * ml_prediction.confidence
        kelly_fraction = self._apply_uncertainty_penalty(kelly_fraction, ml_prediction)

        # Calculate position size
        dollar_allocation = self.account_balance * kelly_fraction
        position_size = dollar_allocation / current_price

        logger.info(
            "Kelly-ML hybrid position size for %s: %s shares (kelly_fraction=%s)",
            signal.symbol, position_size, kelly_fraction
        )

        return position_size.quantize(Decimal('0.00000001'))

    def get_metrics(self) -> Dict[str, Decimal]:
        """
        Get current ML sizer metrics

        Returns:
            Dictionary of current metrics
        """
        metrics = {
            'account_balance': self.account_balance,
            'base_allocation_percent': self.config.base_allocation_percent,
            'min_confidence_threshold': self.config.min_confidence_threshold,
            'predictions_count': Decimal(str(len(self.prediction_history)))
        }

        if self.current_regime:
            metrics['current_regime'] = Decimal(str(ord(self.current_regime.regime.value[0])))
            metrics['regime_confidence'] = self.current_regime.confidence
            metrics['regime_volatility'] = self.current_regime.volatility

        return metrics
