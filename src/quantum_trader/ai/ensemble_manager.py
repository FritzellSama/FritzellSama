"""
Ensemble Manager for ML Model Aggregation

CRITICAL: Production ensemble management with multiple aggregation strategies
- Model weighting (equal, performance-based, dynamic)
- Prediction aggregation (voting, averaging, stacking)
- Model health monitoring
- Dynamic model selection
"""

from decimal import Decimal
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import os
import logging
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Performance metrics for a single model"""
    model_id: str
    accuracy: Decimal
    sharpe_ratio: Decimal
    win_rate: Decimal
    avg_prediction_time_ms: Decimal
    predictions_count: int
    last_updated: datetime
    rolling_accuracy: Decimal  # Last N predictions
    is_healthy: bool


@dataclass
class EnsemblePrediction:
    """Ensemble prediction result"""
    prediction: np.ndarray
    confidence: Decimal
    model_contributions: Dict[str, Decimal]
    aggregation_method: str
    timestamp: datetime
    models_used: List[str]


class BaseMLModel(ABC):
    """Abstract base for ML models (from API contract)"""

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        pass


class EnsembleManager:
    """
    Ensemble Model Manager

    Manages multiple ML models and aggregates their predictions using
    various strategies (voting, weighted averaging, stacking).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize ensemble manager

        Args:
            config: Configuration dictionary loaded from config files
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Configuration
        self.aggregation_method = config.get('aggregation_method', os.getenv('ENSEMBLE_AGG_METHOD', 'weighted_avg'))
        self.min_models = int(config.get('min_models', os.getenv('ENSEMBLE_MIN_MODELS', '3')))
        self.performance_window = int(config.get('performance_window', os.getenv('ENSEMBLE_PERF_WINDOW', '100')))
        self.health_check_interval = int(config.get('health_check_interval', os.getenv('ENSEMBLE_HEALTH_INTERVAL', '300')))
        self.min_confidence = Decimal(str(config.get('min_confidence', os.getenv('ENSEMBLE_MIN_CONFIDENCE', '0.6'))))

        # Model registry
        self.models: Dict[str, BaseMLModel] = {}
        self.model_weights: Dict[str, Decimal] = {}
        self.model_performance: Dict[str, ModelPerformance] = {}

        # Performance tracking
        self.prediction_history: Dict[str, deque] = {}  # model_id -> deque of (prediction, actual, correct)

        # Monitoring
        self.total_predictions = 0
        self.failed_predictions = 0
        self.last_health_check = datetime.utcnow()

        self.logger.info(f"EnsembleManager initialized with {self.aggregation_method} aggregation")

    def register_model(
        self,
        model_id: str,
        model: BaseMLModel,
        initial_weight: Optional[Decimal] = None
    ) -> None:
        """
        Register a model in the ensemble

        Args:
            model_id: Unique identifier for the model
            model: Model instance implementing BaseMLModel
            initial_weight: Initial weight (default: equal weight)
        """
        try:
            if model_id in self.models:
                self.logger.warning(f"Model {model_id} already registered, replacing")

            self.models[model_id] = model

            # Set initial weight
            if initial_weight is None:
                initial_weight = Decimal('1.0') / Decimal(str(len(self.models)))

            self.model_weights[model_id] = initial_weight

            # Initialize performance tracking
            self.model_performance[model_id] = ModelPerformance(
                model_id=model_id,
                accuracy=Decimal('0.0'),
                sharpe_ratio=Decimal('0.0'),
                win_rate=Decimal('0.0'),
                avg_prediction_time_ms=Decimal('0.0'),
                predictions_count=0,
                last_updated=datetime.utcnow(),
                rolling_accuracy=Decimal('0.0'),
                is_healthy=True
            )

            self.prediction_history[model_id] = deque(maxlen=self.performance_window)

            self.logger.info(f"Registered model {model_id} with weight {initial_weight}")

        except Exception as e:
            self.logger.error(f"Failed to register model {model_id}: {e}")
            raise ValueError(f"Model registration error: {e}")

    def unregister_model(self, model_id: str) -> None:
        """Remove a model from the ensemble"""
        try:
            if model_id not in self.models:
                self.logger.warning(f"Model {model_id} not found")
                return

            del self.models[model_id]
            del self.model_weights[model_id]
            del self.model_performance[model_id]
            del self.prediction_history[model_id]

            # Renormalize weights
            self._normalize_weights()

            self.logger.info(f"Unregistered model {model_id}")

        except Exception as e:
            self.logger.error(f"Failed to unregister model {model_id}: {e}")

    def predict(self, features: np.ndarray) -> EnsemblePrediction:
        """
        Generate ensemble prediction

        Args:
            features: Input features for prediction

        Returns:
            EnsemblePrediction with aggregated result
        """
        start_time = datetime.utcnow()

        try:
            if len(self.models) < self.min_models:
                raise ValueError(f"Insufficient models: {len(self.models)} < {self.min_models}")

            # Get predictions from all healthy models
            model_predictions: Dict[str, np.ndarray] = {}
            model_times: Dict[str, Decimal] = {}

            for model_id, model in self.models.items():
                if not self.model_performance[model_id].is_healthy:
                    self.logger.debug(f"Skipping unhealthy model {model_id}")
                    continue

                try:
                    pred_start = datetime.utcnow()
                    prediction = model.predict(features)
                    pred_time = (datetime.utcnow() - pred_start).total_seconds() * 1000

                    model_predictions[model_id] = prediction
                    model_times[model_id] = Decimal(str(pred_time))

                except Exception as e:
                    self.logger.error(f"Model {model_id} prediction failed: {e}")
                    self._mark_model_unhealthy(model_id)
                    continue

            if not model_predictions:
                raise RuntimeError("No healthy models available for prediction")

            # Aggregate predictions
            if self.aggregation_method == 'voting':
                aggregated, contributions = self._voting_aggregation(model_predictions)
            elif self.aggregation_method == 'weighted_avg':
                aggregated, contributions = self._weighted_average_aggregation(model_predictions)
            elif self.aggregation_method == 'stacking':
                aggregated, contributions = self._stacking_aggregation(model_predictions, features)
            else:
                aggregated, contributions = self._simple_average_aggregation(model_predictions)

            # Calculate confidence
            confidence = self._calculate_confidence(model_predictions, contributions)

            # Update metrics
            self.total_predictions += 1

            # Update prediction times
            for model_id, pred_time in model_times.items():
                perf = self.model_performance[model_id]
                total_time = perf.avg_prediction_time_ms * Decimal(str(perf.predictions_count)) + pred_time
                perf.predictions_count += 1
                perf.avg_prediction_time_ms = total_time / Decimal(str(perf.predictions_count))

            self.logger.debug(
                f"Ensemble prediction completed in {(datetime.utcnow() - start_time).total_seconds()*1000:.2f}ms, "
                f"confidence={confidence}, models_used={len(model_predictions)}"
            )

            return EnsemblePrediction(
                prediction=aggregated,
                confidence=confidence,
                model_contributions=contributions,
                aggregation_method=self.aggregation_method,
                timestamp=datetime.utcnow(),
                models_used=list(model_predictions.keys())
            )

        except Exception as e:
            self.failed_predictions += 1
            self.logger.error(f"Ensemble prediction failed: {e}", exc_info=True)
            raise RuntimeError(f"Ensemble prediction error: {e}")

    def update_performance(
        self,
        model_id: str,
        prediction: np.ndarray,
        actual: np.ndarray
    ) -> None:
        """
        Update model performance metrics

        Args:
            model_id: Model identifier
            prediction: Model's prediction
            actual: Actual outcome
        """
        try:
            if model_id not in self.models:
                self.logger.warning(f"Unknown model {model_id}")
                return

            # Calculate correctness (for classification)
            correct = bool(np.argmax(prediction) == np.argmax(actual)) if prediction.ndim > 1 else bool(prediction == actual)

            # Update history
            self.prediction_history[model_id].append((prediction, actual, correct))

            # Calculate rolling accuracy
            recent_predictions = list(self.prediction_history[model_id])
            if recent_predictions:
                rolling_accuracy = Decimal(str(sum(1 for _, _, c in recent_predictions if c))) / Decimal(str(len(recent_predictions)))
            else:
                rolling_accuracy = Decimal('0.0')

            # Update performance
            perf = self.model_performance[model_id]
            perf.rolling_accuracy = rolling_accuracy
            perf.last_updated = datetime.utcnow()

            # Update weights based on performance
            if self.aggregation_method == 'weighted_avg':
                self._update_weights()

            self.logger.debug(f"Updated performance for {model_id}: rolling_acc={rolling_accuracy}")

        except Exception as e:
            self.logger.error(f"Failed to update performance for {model_id}: {e}")

    def _voting_aggregation(
        self,
        model_predictions: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, Decimal]]:
        """Majority voting aggregation"""
        # For classification: vote for most common class
        predictions = list(model_predictions.values())
        votes = np.array([np.argmax(p) if p.ndim > 1 else p for p in predictions])

        # Count votes
        unique, counts = np.unique(votes, return_counts=True)
        winner = unique[np.argmax(counts)]

        # Create one-hot result
        result = np.zeros(predictions[0].shape)
        if result.ndim > 1:
            result[winner] = 1.0
        else:
            result = winner

        # Calculate contributions (models that voted for winner)
        contributions = {}
        for model_id, pred in model_predictions.items():
            pred_class = np.argmax(pred) if pred.ndim > 1 else pred
            contributions[model_id] = Decimal('1.0') if pred_class == winner else Decimal('0.0')

        # Normalize
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}

        return result, contributions

    def _weighted_average_aggregation(
        self,
        model_predictions: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, Decimal]]:
        """Weighted average aggregation"""
        # Get weights for active models
        active_weights = {
            model_id: self.model_weights.get(model_id, Decimal('0.0'))
            for model_id in model_predictions.keys()
        }

        # Normalize weights
        total_weight = sum(active_weights.values())
        if total_weight == 0:
            total_weight = Decimal('1.0')

        normalized_weights = {k: v / total_weight for k, v in active_weights.items()}

        # Weighted sum
        result = None
        for model_id, pred in model_predictions.items():
            weight = float(normalized_weights[model_id])
            if result is None:
                result = weight * pred
            else:
                result += weight * pred

        return result, normalized_weights

    def _simple_average_aggregation(
        self,
        model_predictions: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, Decimal]]:
        """Simple average aggregation"""
        predictions = list(model_predictions.values())
        result = np.mean(predictions, axis=0)

        # Equal contributions
        n_models = Decimal(str(len(predictions)))
        contributions = {model_id: Decimal('1.0') / n_models for model_id in model_predictions.keys()}

        return result, contributions

    def _stacking_aggregation(
        self,
        model_predictions: Dict[str, np.ndarray],
        features: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Decimal]]:
        """Stacking aggregation (meta-learner)"""
        # Simplified stacking: use weighted average with performance-based weights
        # In production, this would use a trained meta-model
        return self._weighted_average_aggregation(model_predictions)

    def _calculate_confidence(
        self,
        model_predictions: Dict[str, np.ndarray],
        contributions: Dict[str, Decimal]
    ) -> Decimal:
        """Calculate prediction confidence"""
        try:
            # Agreement-based confidence
            predictions = list(model_predictions.values())

            if not predictions:
                return Decimal('0.0')

            # For classification: measure agreement
            if predictions[0].ndim > 1:
                pred_classes = [np.argmax(p) for p in predictions]
                unique, counts = np.unique(pred_classes, return_counts=True)
                max_agreement = max(counts) / len(pred_classes)
                confidence = Decimal(str(max_agreement))
            else:
                # For regression: use inverse of std deviation
                std = float(np.std(predictions))
                confidence = Decimal('1.0') / (Decimal('1.0') + Decimal(str(std)))

            # Weight by model health
            avg_health = sum(
                Decimal('1.0') if self.model_performance[mid].is_healthy else Decimal('0.5')
                for mid in model_predictions.keys()
            ) / Decimal(str(len(model_predictions)))

            return confidence * avg_health

        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return Decimal('0.5')

    def _update_weights(self) -> None:
        """Update model weights based on recent performance"""
        try:
            # Performance-based weighting
            for model_id in self.models.keys():
                perf = self.model_performance[model_id]

                if perf.predictions_count < 10:
                    # Not enough data, use equal weight
                    continue

                # Weight based on rolling accuracy
                self.model_weights[model_id] = max(Decimal('0.01'), perf.rolling_accuracy)

            # Normalize
            self._normalize_weights()

        except Exception as e:
            self.logger.error(f"Failed to update weights: {e}")

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0"""
        try:
            total = sum(self.model_weights.values())
            if total > 0:
                self.model_weights = {
                    k: v / total for k, v in self.model_weights.items()
                }
        except Exception as e:
            self.logger.error(f"Weight normalization failed: {e}")

    def _mark_model_unhealthy(self, model_id: str) -> None:
        """Mark a model as unhealthy"""
        if model_id in self.model_performance:
            self.model_performance[model_id].is_healthy = False
            self.logger.warning(f"Model {model_id} marked as unhealthy")

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all models

        Returns:
            Health status report
        """
        try:
            healthy_models = []
            unhealthy_models = []

            for model_id, perf in self.model_performance.items():
                if perf.is_healthy and perf.rolling_accuracy > Decimal('0.5'):
                    healthy_models.append(model_id)
                else:
                    unhealthy_models.append(model_id)

            self.last_health_check = datetime.utcnow()

            report = {
                'total_models': len(self.models),
                'healthy_models': len(healthy_models),
                'unhealthy_models': len(unhealthy_models),
                'total_predictions': self.total_predictions,
                'failed_predictions': self.failed_predictions,
                'success_rate': Decimal(str(self.total_predictions - self.failed_predictions)) / Decimal(str(max(1, self.total_predictions))),
                'last_check': self.last_health_check,
                'model_details': {
                    mid: {
                        'healthy': perf.is_healthy,
                        'accuracy': perf.rolling_accuracy,
                        'predictions': perf.predictions_count,
                        'avg_time_ms': perf.avg_prediction_time_ms
                    }
                    for mid, perf in self.model_performance.items()
                }
            }

            self.logger.info(
                f"Health check: {len(healthy_models)}/{len(self.models)} healthy, "
                f"success_rate={report['success_rate']:.2%}"
            )

            return report

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {'error': str(e)}

    def get_model_rankings(self) -> pl.DataFrame:
        """Get ranked list of models by performance"""
        try:
            data = []
            for model_id, perf in self.model_performance.items():
                data.append({
                    'model_id': model_id,
                    'rolling_accuracy': perf.rolling_accuracy,
                    'predictions_count': perf.predictions_count,
                    'avg_time_ms': perf.avg_prediction_time_ms,
                    'weight': self.model_weights.get(model_id, Decimal('0.0')),
                    'is_healthy': perf.is_healthy
                })

            df = pl.DataFrame(data)
            return df.sort('rolling_accuracy', descending=True)

        except Exception as e:
            self.logger.error(f"Failed to get model rankings: {e}")
            return pl.DataFrame()
