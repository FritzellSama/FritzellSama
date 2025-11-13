"""
AI Orchestrator for coordinating all AI/ML components.

Central coordinator that manages model lifecycle, predictions, training,
and integration across the trading system.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import polars as pl
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Status of AI models."""

    INITIALIZING = "initializing"
    READY = "ready"
    TRAINING = "training"
    PREDICTING = "predicting"
    ERROR = "error"
    DISABLED = "disabled"


class ModelType(str, Enum):
    """Types of AI models."""

    PRICE_PREDICTION = "price_prediction"
    SIGNAL_GENERATION = "signal_generation"
    RISK_ASSESSMENT = "risk_assessment"
    REGIME_DETECTION = "regime_detection"
    EXECUTION_OPTIMIZATION = "execution_optimization"
    SENTIMENT_ANALYSIS = "sentiment_analysis"


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    model_id: str
    model_type: ModelType
    model_class: str
    config: Dict[str, Any]
    enabled: bool = True
    priority: int = 1
    update_frequency_seconds: int = 300
    min_confidence: Decimal = Decimal("0.7")


@dataclass
class ModelRegistration:
    """Registration info for a model."""

    config: ModelConfig
    model: nn.Module
    status: ModelStatus = ModelStatus.INITIALIZING
    last_update: Optional[datetime] = None
    last_prediction: Optional[datetime] = None
    total_predictions: int = 0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionRequest:
    """Request for model prediction."""

    model_id: str
    features: pl.DataFrame
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result from model prediction."""

    model_id: str
    predictions: pl.DataFrame
    confidence: Decimal
    latency_ms: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsemblePrediction:
    """Aggregated prediction from multiple models."""

    predictions: pl.DataFrame
    individual_results: List[PredictionResult]
    weights: Dict[str, Decimal]
    combined_confidence: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AIOrchestrator:
    """Central orchestrator for all AI/ML components."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        max_concurrent_predictions: int = 10
    ):
        """
        Initialize AI orchestrator.

        Args:
            device: Computing device for models
            max_concurrent_predictions: Max concurrent prediction tasks
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_concurrent_predictions = max_concurrent_predictions

        # Model registry
        self.models: Dict[str, ModelRegistration] = {}
        self.model_types: Dict[ModelType, List[str]] = {}

        # Prediction semaphore
        self.prediction_semaphore = asyncio.Semaphore(max_concurrent_predictions)

        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.running = False

        # Metrics
        self.total_predictions = 0
        self.total_errors = 0

        logger.info(
            "Initialized AI orchestrator",
            extra={
                "device": str(self.device),
                "max_concurrent": max_concurrent_predictions,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    async def register_model(
        self,
        config: ModelConfig,
        model: nn.Module
    ) -> None:
        """
        Register a new model with the orchestrator.

        Args:
            config: Model configuration
            model: PyTorch model instance
        """
        if config.model_id in self.models:
            logger.warning(
                "Model already registered, replacing",
                extra={
                    "model_id": config.model_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )

        # Move model to device
        model = model.to(self.device)
        model.eval()

        # Create registration
        registration = ModelRegistration(
            config=config,
            model=model,
            status=ModelStatus.READY if config.enabled else ModelStatus.DISABLED
        )

        self.models[config.model_id] = registration

        # Update type index
        if config.model_type not in self.model_types:
            self.model_types[config.model_type] = []
        if config.model_id not in self.model_types[config.model_type]:
            self.model_types[config.model_type].append(config.model_id)

        logger.info(
            "Registered model",
            extra={
                "model_id": config.model_id,
                "model_type": config.model_type.value,
                "status": registration.status.value,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    async def unregister_model(self, model_id: str) -> None:
        """
        Unregister a model.

        Args:
            model_id: Model identifier
        """
        if model_id not in self.models:
            logger.warning(
                "Model not found for unregistration",
                extra={
                    "model_id": model_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            return

        registration = self.models[model_id]

        # Remove from type index
        model_type = registration.config.model_type
        if model_type in self.model_types:
            if model_id in self.model_types[model_type]:
                self.model_types[model_type].remove(model_id)

        # Remove registration
        del self.models[model_id]

        logger.info(
            "Unregistered model",
            extra={
                "model_id": model_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    async def predict(
        self,
        request: PredictionRequest
    ) -> PredictionResult:
        """
        Get prediction from a specific model.

        Args:
            request: Prediction request

        Returns:
            Prediction result

        Raises:
            ValueError: If model not found or disabled
        """
        if request.model_id not in self.models:
            raise ValueError(f"Model not found: {request.model_id}")

        registration = self.models[request.model_id]

        if registration.status == ModelStatus.DISABLED:
            raise ValueError(f"Model disabled: {request.model_id}")

        if not registration.config.enabled:
            raise ValueError(f"Model not enabled: {request.model_id}")

        # Acquire semaphore for rate limiting
        async with self.prediction_semaphore:
            start_time = datetime.now(timezone.utc)

            try:
                # Update status
                registration.status = ModelStatus.PREDICTING

                # Prepare features
                feature_array = request.features.to_numpy()
                features_tensor = torch.tensor(
                    feature_array,
                    dtype=torch.float32,
                    device=self.device
                )

                # Make prediction
                registration.model.eval()
                with torch.no_grad():
                    output = registration.model(features_tensor)

                    # Convert to probabilities if needed
                    if output.dim() > 1 and output.size(1) > 1:
                        probs = torch.softmax(output, dim=-1)
                    else:
                        probs = torch.sigmoid(output)

                    # Get confidence
                    max_probs, predictions = torch.max(probs, dim=-1)
                    confidence = max_probs.mean()

                # Convert to DataFrame
                predictions_np = predictions.cpu().numpy()
                probs_np = probs.cpu().numpy()

                pred_data = {
                    "prediction": predictions_np.tolist(),
                    "confidence": probs_np.max(axis=1).tolist(),
                }

                predictions_df = pl.DataFrame(pred_data)

                # Calculate latency
                end_time = datetime.now(timezone.utc)
                latency_ms = Decimal(
                    str((end_time - start_time).total_seconds() * 1000)
                )

                # Create result
                result = PredictionResult(
                    model_id=request.model_id,
                    predictions=predictions_df,
                    confidence=Decimal(str(confidence.item())),
                    latency_ms=latency_ms,
                    timestamp=end_time,
                    metadata=request.metadata
                )

                # Update registration metrics
                registration.status = ModelStatus.READY
                registration.last_prediction = end_time
                registration.total_predictions += 1
                self.total_predictions += 1

                logger.info(
                    "Prediction completed",
                    extra={
                        "model_id": request.model_id,
                        "num_samples": len(predictions_df),
                        "confidence": str(result.confidence),
                        "latency_ms": str(latency_ms),
                        "timestamp": end_time.isoformat()
                    }
                )

                return result

            except Exception as e:
                registration.status = ModelStatus.ERROR
                registration.error_count += 1
                self.total_errors += 1

                logger.error(
                    "Prediction failed",
                    extra={
                        "model_id": request.model_id,
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    },
                    exc_info=True
                )
                raise

    async def predict_ensemble(
        self,
        model_type: ModelType,
        features: pl.DataFrame,
        aggregation: str = "weighted_average"
    ) -> EnsemblePrediction:
        """
        Get ensemble prediction from all models of a type.

        Args:
            model_type: Type of models to use
            features: Input features
            aggregation: Aggregation method

        Returns:
            Ensemble prediction result
        """
        if model_type not in self.model_types:
            raise ValueError(f"No models registered for type: {model_type.value}")

        model_ids = self.model_types[model_type]
        enabled_models = [
            mid for mid in model_ids
            if self.models[mid].config.enabled
        ]

        if not enabled_models:
            raise ValueError(
                f"No enabled models for type: {model_type.value}"
            )

        # Get predictions from all models
        tasks = [
            self.predict(PredictionRequest(
                model_id=model_id,
                features=features
            ))
            for model_id in enabled_models
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        successful_results = [
            r for r in results
            if isinstance(r, PredictionResult)
        ]

        if not successful_results:
            raise RuntimeError(
                f"All predictions failed for type: {model_type.value}"
            )

        # Calculate weights based on confidence and priority
        weights = {}
        total_weight = Decimal("0")

        for result in successful_results:
            model_priority = self.models[result.model_id].config.priority
            weight = result.confidence * Decimal(str(model_priority))
            weights[result.model_id] = weight
            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            weights = {
                mid: w / total_weight
                for mid, w in weights.items()
            }

        # Aggregate predictions
        if aggregation == "weighted_average":
            predictions_df = self._weighted_average_predictions(
                successful_results,
                weights
            )
        elif aggregation == "majority_vote":
            predictions_df = self._majority_vote_predictions(successful_results)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        # Calculate combined confidence
        combined_confidence = sum(
            result.confidence * weights[result.model_id]
            for result in successful_results
        )

        ensemble = EnsemblePrediction(
            predictions=predictions_df,
            individual_results=successful_results,
            weights=weights,
            combined_confidence=combined_confidence,
            timestamp=datetime.now(timezone.utc)
        )

        logger.info(
            "Ensemble prediction completed",
            extra={
                "model_type": model_type.value,
                "num_models": len(successful_results),
                "confidence": str(combined_confidence),
                "aggregation": aggregation,
                "timestamp": ensemble.timestamp.isoformat()
            }
        )

        return ensemble

    def _weighted_average_predictions(
        self,
        results: List[PredictionResult],
        weights: Dict[str, Decimal]
    ) -> pl.DataFrame:
        """Aggregate predictions using weighted average."""
        # Initialize weighted sum
        weighted_sum = None
        num_samples = len(results[0].predictions)

        for result in results:
            weight = float(weights[result.model_id])
            pred_values = result.predictions["prediction"].to_numpy()

            if weighted_sum is None:
                weighted_sum = pred_values * weight
            else:
                weighted_sum += pred_values * weight

        # Create DataFrame
        pred_data = {
            "prediction": weighted_sum.tolist(),
        }

        return pl.DataFrame(pred_data)

    def _majority_vote_predictions(
        self,
        results: List[PredictionResult]
    ) -> pl.DataFrame:
        """Aggregate predictions using majority vote."""
        import numpy as np

        # Stack all predictions
        all_preds = [
            result.predictions["prediction"].to_numpy()
            for result in results
        ]
        stacked = np.stack(all_preds, axis=0)

        # Get majority vote
        from scipy import stats
        majority, _ = stats.mode(stacked, axis=0, keepdims=False)

        pred_data = {
            "prediction": majority.tolist(),
        }

        return pl.DataFrame(pred_data)

    async def update_model(
        self,
        model_id: str,
        train_df: pl.DataFrame,
        feature_cols: List[str],
        label_col: str,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epochs: int = 10
    ) -> Dict[str, Decimal]:
        """
        Update a model with new training data.

        Args:
            model_id: Model identifier
            train_df: Training DataFrame
            feature_cols: Feature column names
            label_col: Label column name
            optimizer: Optimizer
            criterion: Loss criterion
            epochs: Number of epochs

        Returns:
            Training metrics
        """
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")

        registration = self.models[model_id]
        registration.status = ModelStatus.TRAINING

        try:
            # Prepare data
            X = torch.tensor(
                train_df.select(feature_cols).to_numpy(),
                dtype=torch.float32,
                device=self.device
            )
            y = torch.tensor(
                train_df.select(label_col).to_numpy().flatten(),
                dtype=torch.long,
                device=self.device
            )

            # Training loop
            registration.model.train()
            total_loss = Decimal("0")

            for epoch in range(epochs):
                optimizer.zero_grad()

                outputs = registration.model(X)
                loss = criterion(outputs, y)

                loss.backward()
                optimizer.step()

                total_loss += Decimal(str(loss.item()))

            avg_loss = total_loss / epochs

            # Update registration
            registration.status = ModelStatus.READY
            registration.last_update = datetime.now(timezone.utc)

            logger.info(
                "Model updated",
                extra={
                    "model_id": model_id,
                    "num_samples": len(train_df),
                    "epochs": epochs,
                    "avg_loss": str(avg_loss),
                    "timestamp": registration.last_update.isoformat()
                }
            )

            return {
                "avg_loss": avg_loss,
                "num_samples": Decimal(str(len(train_df))),
                "epochs": Decimal(str(epochs))
            }

        except Exception as e:
            registration.status = ModelStatus.ERROR
            registration.error_count += 1

            logger.error(
                "Model update failed",
                extra={
                    "model_id": model_id,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                exc_info=True
            )
            raise

    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get status of a specific model."""
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")

        registration = self.models[model_id]

        return {
            "model_id": model_id,
            "model_type": registration.config.model_type.value,
            "status": registration.status.value,
            "enabled": registration.config.enabled,
            "last_update": registration.last_update.isoformat() if registration.last_update else None,
            "last_prediction": registration.last_prediction.isoformat() if registration.last_prediction else None,
            "total_predictions": registration.total_predictions,
            "error_count": registration.error_count,
        }

    def get_all_status(self) -> pl.DataFrame:
        """Get status of all registered models."""
        status_data = []

        for model_id in self.models:
            status = self.get_model_status(model_id)
            status_data.append(status)

        return pl.DataFrame(status_data)

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on orchestrator."""
        total_models = len(self.models)
        ready_models = sum(
            1 for r in self.models.values()
            if r.status == ModelStatus.READY
        )
        error_models = sum(
            1 for r in self.models.values()
            if r.status == ModelStatus.ERROR
        )

        return {
            "healthy": error_models == 0 and ready_models > 0,
            "total_models": total_models,
            "ready_models": ready_models,
            "error_models": error_models,
            "total_predictions": self.total_predictions,
            "total_errors": self.total_errors,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
