"""
AI Orchestrator - Central coordination of all AI/ML models.

This module orchestrates multiple AI models, manages model lifecycle,
coordinates predictions, and handles model ensemble strategies.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import torch
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class ModelType(Enum):
    """Types of ML models."""

    LSTM = "lstm"
    TRANSFORMER = "transformer"
    ATTENTION = "attention"
    AUTOENCODER = "autoencoder"
    CNN = "cnn"
    GRU = "gru"
    REINFORCEMENT = "reinforcement"
    ENSEMBLE = "ensemble"


class EnsembleStrategy(Enum):
    """Ensemble combination strategies."""

    AVERAGE = "average"  # Simple average
    WEIGHTED = "weighted"  # Weighted by performance
    VOTING = "voting"  # Majority voting
    STACKING = "stacking"  # Stacked generalization
    BOOSTING = "boosting"  # Gradient boosting
    DYNAMIC = "dynamic"  # Dynamic weight adjustment


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""

    model_id: str
    model_type: ModelType
    version: str
    created_at: datetime
    last_updated: datetime
    performance_metrics: Dict[str, Decimal] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    weight: Decimal = Decimal("1.0")
    prediction_count: int = 0
    error_count: int = 0


@dataclass
class PredictionResult:
    """Result from model prediction."""

    model_id: str
    prediction: Any
    confidence: Decimal
    timestamp: datetime
    latency_ms: Decimal
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsemblePrediction:
    """Combined prediction from ensemble."""

    prediction: Any
    confidence: Decimal
    contributing_models: List[str]
    individual_predictions: List[PredictionResult]
    ensemble_strategy: str
    timestamp: datetime


class ModelRegistry:
    """Registry for managing multiple models."""

    def __init__(self) -> None:
        """Initialize model registry."""
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, ModelMetadata] = {}
        logger.info("Model registry initialized")

    def register_model(
        self,
        model_id: str,
        model: Any,
        model_type: ModelType,
        config: Dict[str, Any]
    ) -> None:
        """Register a model.

        Args:
            model_id: Unique model identifier
            model: Model instance
            model_type: Type of model
            config: Model configuration
        """
        try:
            if model_id in self.models:
                logger.warning("Overwriting existing model", model_id=model_id)

            self.models[model_id] = model
            self.metadata[model_id] = ModelMetadata(
                model_id=model_id,
                model_type=model_type,
                version=config.get("version", "1.0.0"),
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                config=config
            )

            logger.info(
                "Model registered",
                model_id=model_id,
                model_type=model_type.value
            )

        except Exception as e:
            logger.error(
                "Model registration failed",
                model_id=model_id,
                error=str(e)
            )
            raise

    def unregister_model(self, model_id: str) -> None:
        """Unregister a model.

        Args:
            model_id: Model identifier
        """
        try:
            if model_id in self.models:
                del self.models[model_id]
                del self.metadata[model_id]
                logger.info("Model unregistered", model_id=model_id)
            else:
                logger.warning("Model not found", model_id=model_id)

        except Exception as e:
            logger.error(
                "Model unregistration failed",
                model_id=model_id,
                error=str(e)
            )
            raise

    def get_model(self, model_id: str) -> Optional[Any]:
        """Get model by ID.

        Args:
            model_id: Model identifier

        Returns:
            Model instance or None
        """
        return self.models.get(model_id)

    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata.

        Args:
            model_id: Model identifier

        Returns:
            Model metadata or None
        """
        return self.metadata.get(model_id)

    def get_active_models(self) -> List[str]:
        """Get list of active model IDs.

        Returns:
            List of active model IDs
        """
        return [
            model_id
            for model_id, meta in self.metadata.items()
            if meta.is_active
        ]

    def update_performance(
        self,
        model_id: str,
        metrics: Dict[str, Decimal]
    ) -> None:
        """Update model performance metrics.

        Args:
            model_id: Model identifier
            metrics: Performance metrics
        """
        try:
            if model_id in self.metadata:
                self.metadata[model_id].performance_metrics.update(metrics)
                self.metadata[model_id].last_updated = datetime.utcnow()
                logger.debug(
                    "Model performance updated",
                    model_id=model_id,
                    metrics=metrics
                )
        except Exception as e:
            logger.error(
                "Performance update failed",
                model_id=model_id,
                error=str(e)
            )


class EnsembleManager:
    """Manages ensemble prediction strategies."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize ensemble manager.

        Args:
            config: Ensemble configuration
        """
        self.config = config
        self.strategy = EnsembleStrategy(
            config.get("strategy", "weighted")
        )
        self.model_weights: Dict[str, Decimal] = {}
        logger.info("Ensemble manager initialized", strategy=self.strategy.value)

    async def combine_predictions(
        self,
        predictions: List[PredictionResult]
    ) -> EnsemblePrediction:
        """Combine predictions using ensemble strategy.

        Args:
            predictions: Individual model predictions

        Returns:
            Combined ensemble prediction
        """
        try:
            if not predictions:
                raise ValueError("No predictions to combine")

            if self.strategy == EnsembleStrategy.AVERAGE:
                result = await self._average_ensemble(predictions)
            elif self.strategy == EnsembleStrategy.WEIGHTED:
                result = await self._weighted_ensemble(predictions)
            elif self.strategy == EnsembleStrategy.VOTING:
                result = await self._voting_ensemble(predictions)
            elif self.strategy == EnsembleStrategy.DYNAMIC:
                result = await self._dynamic_ensemble(predictions)
            else:
                logger.warning(
                    "Unknown strategy, using average",
                    strategy=self.strategy
                )
                result = await self._average_ensemble(predictions)

            logger.debug(
                "Predictions combined",
                strategy=self.strategy.value,
                num_models=len(predictions)
            )

            return result

        except Exception as e:
            logger.error("Ensemble combination failed", error=str(e))
            raise

    async def _average_ensemble(
        self,
        predictions: List[PredictionResult]
    ) -> EnsemblePrediction:
        """Simple average ensemble.

        Args:
            predictions: Individual predictions

        Returns:
            Averaged prediction
        """
        # Extract numeric predictions
        pred_values = []
        for pred in predictions:
            if isinstance(pred.prediction, (int, float, Decimal)):
                pred_values.append(float(pred.prediction))
            elif isinstance(pred.prediction, np.ndarray):
                pred_values.append(pred.prediction.flatten())

        if pred_values:
            avg_pred = np.mean(pred_values, axis=0)
            avg_confidence = Decimal(str(np.mean([float(p.confidence) for p in predictions])))

            return EnsemblePrediction(
                prediction=avg_pred,
                confidence=avg_confidence,
                contributing_models=[p.model_id for p in predictions],
                individual_predictions=predictions,
                ensemble_strategy="average",
                timestamp=datetime.utcnow()
            )

        raise ValueError("No valid predictions to average")

    async def _weighted_ensemble(
        self,
        predictions: List[PredictionResult]
    ) -> EnsemblePrediction:
        """Weighted average ensemble.

        Args:
            predictions: Individual predictions

        Returns:
            Weighted prediction
        """
        pred_values = []
        weights = []

        for pred in predictions:
            weight = self.model_weights.get(pred.model_id, Decimal("1.0"))
            weights.append(float(weight))

            if isinstance(pred.prediction, (int, float, Decimal)):
                pred_values.append(float(pred.prediction))
            elif isinstance(pred.prediction, np.ndarray):
                pred_values.append(pred.prediction.flatten())

        if pred_values and weights:
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()

            # Weighted average
            weighted_pred = np.average(pred_values, axis=0, weights=weights)
            weighted_confidence = Decimal(str(
                np.average([float(p.confidence) for p in predictions], weights=weights)
            ))

            return EnsemblePrediction(
                prediction=weighted_pred,
                confidence=weighted_confidence,
                contributing_models=[p.model_id for p in predictions],
                individual_predictions=predictions,
                ensemble_strategy="weighted",
                timestamp=datetime.utcnow()
            )

        raise ValueError("No valid predictions for weighted ensemble")

    async def _voting_ensemble(
        self,
        predictions: List[PredictionResult]
    ) -> EnsemblePrediction:
        """Majority voting ensemble.

        Args:
            predictions: Individual predictions

        Returns:
            Voted prediction
        """
        # Extract class predictions
        votes = {}
        for pred in predictions:
            pred_class = pred.prediction
            if isinstance(pred_class, (int, str)):
                votes[pred_class] = votes.get(pred_class, 0) + 1

        if votes:
            # Get majority vote
            majority_class = max(votes, key=votes.get)
            confidence = Decimal(str(votes[majority_class] / len(predictions)))

            return EnsemblePrediction(
                prediction=majority_class,
                confidence=confidence,
                contributing_models=[p.model_id for p in predictions],
                individual_predictions=predictions,
                ensemble_strategy="voting",
                timestamp=datetime.utcnow()
            )

        raise ValueError("No valid votes")

    async def _dynamic_ensemble(
        self,
        predictions: List[PredictionResult]
    ) -> EnsemblePrediction:
        """Dynamic weight adjustment based on recent performance.

        Args:
            predictions: Individual predictions

        Returns:
            Dynamically weighted prediction
        """
        # Use confidence as dynamic weights
        weights = [float(p.confidence) for p in predictions]
        return await self._weighted_ensemble(predictions)

    def update_weights(self, model_weights: Dict[str, Decimal]) -> None:
        """Update model weights for ensemble.

        Args:
            model_weights: Dictionary of model_id -> weight
        """
        self.model_weights.update(model_weights)
        logger.info("Ensemble weights updated", weights=model_weights)


class AIOrchestrator:
    """Central orchestrator for all AI/ML models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize AI orchestrator.

        Args:
            config: Orchestrator configuration

        Example:
            >>> config = {
            ...     "ensemble_strategy": "weighted",
            ...     "prediction_timeout": 5.0,
            ...     "enable_caching": True
            ... }
            >>> orchestrator = AIOrchestrator(config)
        """
        self.config = config
        self.registry = ModelRegistry()
        self.ensemble_manager = EnsembleManager(config.get("ensemble", {}))

        # Prediction cache
        self.cache_enabled = config.get("enable_caching", True)
        self.prediction_cache: Dict[str, EnsemblePrediction] = {}
        self.cache_ttl = timedelta(seconds=config.get("cache_ttl_seconds", 60))

        # Performance tracking
        self.stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "cache_hits": 0,
            "avg_latency_ms": Decimal("0"),
            "models_registered": 0
        }

        logger.info("AI orchestrator initialized")

    async def register_model(
        self,
        model_id: str,
        model: Any,
        model_type: str,
        config: Dict[str, Any]
    ) -> None:
        """Register a model with the orchestrator.

        Args:
            model_id: Unique model identifier
            model: Model instance
            model_type: Type of model
            config: Model configuration

        Example:
            >>> await orchestrator.register_model(
            ...     "lstm_v1",
            ...     lstm_model,
            ...     "lstm",
            ...     {"version": "1.0.0"}
            ... )
        """
        try:
            model_type_enum = ModelType(model_type)
            self.registry.register_model(model_id, model, model_type_enum, config)
            self.stats["models_registered"] += 1

            logger.info(
                "Model registered with orchestrator",
                model_id=model_id,
                model_type=model_type
            )

        except Exception as e:
            logger.error(
                "Model registration failed",
                model_id=model_id,
                error=str(e)
            )
            raise

    async def predict(
        self,
        features: np.ndarray,
        model_ids: Optional[List[str]] = None,
        use_ensemble: bool = True
    ) -> EnsemblePrediction:
        """Make prediction using registered models.

        Args:
            features: Input features
            model_ids: Specific models to use (None = all active)
            use_ensemble: Whether to combine predictions

        Returns:
            Ensemble prediction result

        Example:
            >>> features = np.random.randn(1, 50)
            >>> prediction = await orchestrator.predict(features)
            >>> print(f"Prediction: {prediction.prediction}")
        """
        try:
            start_time = datetime.utcnow()

            # Check cache
            cache_key = self._generate_cache_key(features, model_ids)
            if self.cache_enabled and cache_key in self.prediction_cache:
                cached_pred = self.prediction_cache[cache_key]
                if datetime.utcnow() - cached_pred.timestamp < self.cache_ttl:
                    self.stats["cache_hits"] += 1
                    logger.debug("Cache hit", cache_key=cache_key)
                    return cached_pred

            # Get models to use
            if model_ids is None:
                model_ids = self.registry.get_active_models()

            if not model_ids:
                raise ValueError("No active models available")

            # Get predictions from all models
            predictions = await self._get_model_predictions(features, model_ids)

            if not predictions:
                raise ValueError("No predictions returned")

            # Combine predictions
            if use_ensemble and len(predictions) > 1:
                result = await self.ensemble_manager.combine_predictions(predictions)
            else:
                # Use single model prediction
                pred = predictions[0]
                result = EnsemblePrediction(
                    prediction=pred.prediction,
                    confidence=pred.confidence,
                    contributing_models=[pred.model_id],
                    individual_predictions=predictions,
                    ensemble_strategy="single",
                    timestamp=datetime.utcnow()
                )

            # Update cache
            if self.cache_enabled:
                self.prediction_cache[cache_key] = result

            # Update statistics
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.stats["total_predictions"] += 1
            self.stats["successful_predictions"] += 1
            self._update_avg_latency(Decimal(str(latency)))

            logger.info(
                "Prediction completed",
                num_models=len(predictions),
                latency_ms=latency,
                confidence=float(result.confidence)
            )

            return result

        except Exception as e:
            self.stats["total_predictions"] += 1
            self.stats["failed_predictions"] += 1
            logger.error("Prediction failed", error=str(e))
            raise

    async def _get_model_predictions(
        self,
        features: np.ndarray,
        model_ids: List[str]
    ) -> List[PredictionResult]:
        """Get predictions from multiple models concurrently.

        Args:
            features: Input features
            model_ids: List of model IDs

        Returns:
            List of prediction results
        """
        try:
            # Create prediction tasks
            tasks = []
            for model_id in model_ids:
                task = self._predict_single_model(model_id, features)
                tasks.append(task)

            # Execute concurrently with timeout
            timeout = self.config.get("prediction_timeout", 5.0)
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )

            # Filter successful predictions
            predictions = []
            for result in results:
                if isinstance(result, PredictionResult):
                    predictions.append(result)
                elif isinstance(result, Exception):
                    logger.error("Model prediction failed", error=str(result))

            return predictions

        except asyncio.TimeoutError:
            logger.error("Prediction timeout exceeded")
            raise
        except Exception as e:
            logger.error("Batch prediction failed", error=str(e))
            raise

    async def _predict_single_model(
        self,
        model_id: str,
        features: np.ndarray
    ) -> PredictionResult:
        """Get prediction from a single model.

        Args:
            model_id: Model identifier
            features: Input features

        Returns:
            Prediction result
        """
        try:
            start_time = datetime.utcnow()

            model = self.registry.get_model(model_id)
            if model is None:
                raise ValueError(f"Model not found: {model_id}")

            metadata = self.registry.get_metadata(model_id)

            # Make prediction
            if hasattr(model, 'predict') and asyncio.iscoroutinefunction(model.predict):
                prediction, confidence = await model.predict(features)
            elif hasattr(model, 'predict'):
                prediction, confidence = model.predict(features)
            else:
                # Assume PyTorch model
                model.eval()
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features)
                    output = model(features_tensor)
                    prediction = output.numpy()
                    confidence = Decimal("0.5")  # Default confidence

            # Calculate latency
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Update model metadata
            if metadata:
                metadata.prediction_count += 1

            return PredictionResult(
                model_id=model_id,
                prediction=prediction,
                confidence=confidence if isinstance(confidence, Decimal) else Decimal(str(confidence)),
                timestamp=datetime.utcnow(),
                latency_ms=Decimal(str(latency))
            )

        except Exception as e:
            metadata = self.registry.get_metadata(model_id)
            if metadata:
                metadata.error_count += 1

            logger.error(
                "Single model prediction failed",
                model_id=model_id,
                error=str(e)
            )
            raise

    def _generate_cache_key(
        self,
        features: np.ndarray,
        model_ids: Optional[List[str]]
    ) -> str:
        """Generate cache key for prediction.

        Args:
            features: Input features
            model_ids: Model IDs used

        Returns:
            Cache key string
        """
        import hashlib

        feature_hash = hashlib.md5(features.tobytes()).hexdigest()
        model_str = "_".join(sorted(model_ids or []))
        return f"{feature_hash}_{model_str}"

    def _update_avg_latency(self, latency: Decimal) -> None:
        """Update average latency statistic.

        Args:
            latency: New latency measurement
        """
        n = self.stats["successful_predictions"]
        current_avg = self.stats["avg_latency_ms"]
        self.stats["avg_latency_ms"] = (
            (current_avg * (n - 1) + latency) / n
        )

    async def update_model_weights(
        self,
        performance_metrics: pl.DataFrame
    ) -> None:
        """Update ensemble weights based on performance.

        Args:
            performance_metrics: DataFrame with model performance
        """
        try:
            # Calculate weights based on performance
            weights = {}
            for row in performance_metrics.iter_rows(named=True):
                model_id = row["model_id"]
                accuracy = Decimal(str(row.get("accuracy", 0.5)))
                weights[model_id] = accuracy

            # Normalize weights
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

            self.ensemble_manager.update_weights(weights)

            logger.info("Model weights updated", weights=weights)

        except Exception as e:
            logger.error("Weight update failed", error=str(e))
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            **self.stats,
            "active_models": len(self.registry.get_active_models()),
            "total_models": len(self.registry.models),
            "cache_size": len(self.prediction_cache)
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all models.

        Returns:
            Health status dictionary
        """
        try:
            health_status = {
                "orchestrator": "healthy",
                "models": {},
                "timestamp": datetime.utcnow().isoformat()
            }

            for model_id in self.registry.get_active_models():
                metadata = self.registry.get_metadata(model_id)
                if metadata:
                    error_rate = (
                        metadata.error_count / max(metadata.prediction_count, 1)
                    )
                    health_status["models"][model_id] = {
                        "status": "healthy" if error_rate < 0.1 else "degraded",
                        "error_rate": error_rate,
                        "predictions": metadata.prediction_count
                    }

            return health_status

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {"orchestrator": "unhealthy", "error": str(e)}

    def clear_cache(self) -> None:
        """Clear prediction cache."""
        self.prediction_cache.clear()
        logger.info("Prediction cache cleared")
