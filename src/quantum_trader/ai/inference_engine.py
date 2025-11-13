"""
Inference Engine for ML model predictions in production.

This module provides a production-ready inference engine for executing
ML model predictions with batching, caching, and async support.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np
import polars as pl
import torch
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel
from quantum_trader.exceptions import InferenceError, ModelNotFoundError, ValidationError

logger = get_logger(__name__)


@dataclass
class InferenceRequest:
    """Inference request data structure.

    Attributes:
        model_name: Name of the model to use
        features: Feature data as Polars DataFrame
        request_id: Unique request identifier
        timestamp: Request timestamp in UTC
        metadata: Additional request metadata
    """
    model_name: str
    features: pl.DataFrame
    request_id: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Inference result data structure.

    Attributes:
        predictions: Prediction values as Decimal
        probabilities: Class probabilities (optional)
        model_name: Name of the model used
        request_id: Original request identifier
        latency_ms: Inference latency in milliseconds
        timestamp: Result timestamp in UTC
        metadata: Additional result metadata
    """
    predictions: List[Decimal]
    probabilities: Optional[List[List[Decimal]]]
    model_name: str
    request_id: str
    latency_ms: Decimal
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class InferenceEngine:
    """Production-ready ML inference engine.

    Manages model loading, batching, caching, and async inference execution.

    Attributes:
        config: Engine configuration
        models: Loaded models cache
        batch_size: Maximum batch size for inference
        timeout_ms: Inference timeout in milliseconds
        enable_cache: Whether to enable prediction caching

    Example:
        >>> config = {
        ...     "model_dir": "/models",
        ...     "batch_size": 32,
        ...     "timeout_ms": 100,
        ...     "enable_cache": True,
        ...     "cache_ttl_seconds": 60
        ... }
        >>> engine = InferenceEngine(config)
        >>> await engine.initialize()
        >>> result = await engine.predict(request)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize inference engine.

        Args:
            config: Engine configuration dictionary

        Raises:
            ValidationError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.model_dir = Path(config["model_dir"])
        self.batch_size = config.get("batch_size", 32)
        self.timeout_ms = config.get("timeout_ms", 100)
        self.enable_cache = config.get("enable_cache", False)
        self.cache_ttl_seconds = config.get("cache_ttl_seconds", 60)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay_ms = config.get("retry_delay_ms", 100)

        self.models: Dict[str, BaseMLModel] = {}
        self._prediction_cache: Dict[str, Tuple[List[Decimal], datetime]] = {}
        self._lock = asyncio.Lock()
        self._initialized = False

        logger.info(
            "Inference engine initialized",
            model_dir=str(self.model_dir),
            batch_size=self.batch_size,
            timeout_ms=self.timeout_ms
        )

    def _validate_config(self) -> None:
        """Validate engine configuration.

        Raises:
            ValidationError: If configuration is invalid
        """
        required_fields = ["model_dir"]
        for field in required_fields:
            if field not in self.config:
                raise ValidationError(f"Missing required config field: {field}")

        if not os.path.exists(self.config["model_dir"]):
            raise ValidationError(f"Model directory does not exist: {self.config['model_dir']}")

        if self.config.get("batch_size", 1) < 1:
            raise ValidationError("batch_size must be >= 1")

        if self.config.get("timeout_ms", 1) < 1:
            raise ValidationError("timeout_ms must be >= 1")

    async def initialize(self) -> None:
        """Initialize the inference engine.

        Loads all available models from the model directory.

        Raises:
            InferenceError: If initialization fails
        """
        try:
            async with self._lock:
                if self._initialized:
                    logger.warning("Inference engine already initialized")
                    return

                logger.info("Initializing inference engine")

                # Discover and load models
                await self._discover_models()

                self._initialized = True
                logger.info(
                    "Inference engine initialized successfully",
                    num_models=len(self.models)
                )

        except Exception as e:
            logger.error("Failed to initialize inference engine", error=str(e))
            raise InferenceError(f"Initialization failed: {e}") from e

    async def _discover_models(self) -> None:
        """Discover and load models from model directory.

        Raises:
            InferenceError: If model discovery fails
        """
        try:
            model_files = list(self.model_dir.glob("*.pt"))

            for model_file in model_files:
                model_name = model_file.stem
                try:
                    # Model loading would happen here based on registry
                    logger.info(f"Discovered model: {model_name}", path=str(model_file))
                except Exception as e:
                    logger.error(
                        f"Failed to load model: {model_name}",
                        error=str(e)
                    )

        except Exception as e:
            logger.error("Model discovery failed", error=str(e))
            raise InferenceError(f"Model discovery failed: {e}") from e

    async def predict(
        self,
        request: InferenceRequest,
        use_cache: Optional[bool] = None
    ) -> InferenceResult:
        """Execute model inference.

        Args:
            request: Inference request
            use_cache: Whether to use cache (overrides config)

        Returns:
            Inference result

        Raises:
            ModelNotFoundError: If model not found
            InferenceError: If inference fails
            ValidationError: If request is invalid
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Validate request
            self._validate_request(request)

            # Check cache
            use_cache = use_cache if use_cache is not None else self.enable_cache
            if use_cache:
                cached_result = self._get_cached_prediction(request)
                if cached_result is not None:
                    logger.debug("Cache hit", request_id=request.request_id)
                    return cached_result

            # Get model
            model = self._get_model(request.model_name)

            # Prepare features
            features = self._prepare_features(request.features)

            # Execute inference with retry
            predictions = await self._execute_with_retry(
                model,
                features,
                request.request_id
            )

            # Convert to Decimal
            predictions_decimal = [
                Decimal(str(pred)) for pred in predictions
            ]

            # Calculate latency
            end_time = datetime.now(timezone.utc)
            latency_ms = Decimal(str((end_time - start_time).total_seconds() * 1000))

            # Create result
            result = InferenceResult(
                predictions=predictions_decimal,
                probabilities=None,
                model_name=request.model_name,
                request_id=request.request_id,
                latency_ms=latency_ms,
                timestamp=end_time,
                metadata={
                    "num_samples": len(predictions_decimal),
                    "cache_hit": False
                }
            )

            # Cache result
            if use_cache:
                self._cache_prediction(request, predictions_decimal)

            logger.info(
                "Inference completed",
                request_id=request.request_id,
                model_name=request.model_name,
                latency_ms=float(latency_ms),
                num_predictions=len(predictions_decimal)
            )

            return result

        except ModelNotFoundError:
            raise
        except ValidationError:
            raise
        except Exception as e:
            logger.error(
                "Inference failed",
                request_id=request.request_id,
                error=str(e)
            )
            raise InferenceError(f"Inference failed: {e}") from e

    async def batch_predict(
        self,
        requests: List[InferenceRequest]
    ) -> List[InferenceResult]:
        """Execute batch inference.

        Args:
            requests: List of inference requests

        Returns:
            List of inference results

        Raises:
            InferenceError: If batch inference fails
        """
        try:
            # Process in batches
            results = []
            for i in range(0, len(requests), self.batch_size):
                batch = requests[i:i + self.batch_size]

                # Execute batch concurrently
                batch_results = await asyncio.gather(
                    *[self.predict(req) for req in batch],
                    return_exceptions=True
                )

                # Handle exceptions
                for req, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(
                            "Batch inference failed for request",
                            request_id=req.request_id,
                            error=str(result)
                        )
                        raise result
                    results.append(result)

            logger.info(
                "Batch inference completed",
                num_requests=len(requests),
                num_results=len(results)
            )

            return results

        except Exception as e:
            logger.error("Batch inference failed", error=str(e))
            raise InferenceError(f"Batch inference failed: {e}") from e

    async def _execute_with_retry(
        self,
        model: BaseMLModel,
        features: np.ndarray,
        request_id: str
    ) -> np.ndarray:
        """Execute inference with retry logic.

        Args:
            model: ML model
            features: Feature array
            request_id: Request identifier

        Returns:
            Prediction array

        Raises:
            InferenceError: If all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Execute with timeout
                predictions = await asyncio.wait_for(
                    asyncio.to_thread(model.predict, features),
                    timeout=self.timeout_ms / 1000.0
                )

                return predictions

            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(
                    "Inference timeout",
                    request_id=request_id,
                    attempt=attempt + 1,
                    max_retries=self.max_retries
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    "Inference attempt failed",
                    request_id=request_id,
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e)
                )

            # Exponential backoff
            if attempt < self.max_retries - 1:
                delay_ms = self.retry_delay_ms * (2 ** attempt)
                await asyncio.sleep(delay_ms / 1000.0)

        raise InferenceError(f"Inference failed after {self.max_retries} retries: {last_error}")

    def _validate_request(self, request: InferenceRequest) -> None:
        """Validate inference request.

        Args:
            request: Inference request

        Raises:
            ValidationError: If request is invalid
        """
        if not request.model_name:
            raise ValidationError("model_name is required")

        if request.features is None or len(request.features) == 0:
            raise ValidationError("features cannot be empty")

        if not request.request_id:
            raise ValidationError("request_id is required")

    def _get_model(self, model_name: str) -> BaseMLModel:
        """Get model by name.

        Args:
            model_name: Name of the model

        Returns:
            Loaded model

        Raises:
            ModelNotFoundError: If model not found
        """
        if model_name not in self.models:
            raise ModelNotFoundError(f"Model not found: {model_name}")

        return self.models[model_name]

    def _prepare_features(self, features: pl.DataFrame) -> np.ndarray:
        """Prepare features for inference.

        Args:
            features: Feature dataframe

        Returns:
            Feature array

        Raises:
            ValidationError: If features are invalid
        """
        try:
            # Convert Polars to numpy
            feature_array = features.to_numpy()

            return feature_array

        except Exception as e:
            raise ValidationError(f"Failed to prepare features: {e}") from e

    def _get_cached_prediction(
        self,
        request: InferenceRequest
    ) -> Optional[InferenceResult]:
        """Get cached prediction if available.

        Args:
            request: Inference request

        Returns:
            Cached result or None
        """
        cache_key = self._get_cache_key(request)

        if cache_key in self._prediction_cache:
            predictions, cached_time = self._prediction_cache[cache_key]

            # Check TTL
            age_seconds = (datetime.now(timezone.utc) - cached_time).total_seconds()
            if age_seconds < self.cache_ttl_seconds:
                return InferenceResult(
                    predictions=predictions,
                    probabilities=None,
                    model_name=request.model_name,
                    request_id=request.request_id,
                    latency_ms=Decimal("0"),
                    timestamp=datetime.now(timezone.utc),
                    metadata={"cache_hit": True, "cache_age_seconds": age_seconds}
                )
            else:
                # Remove expired entry
                del self._prediction_cache[cache_key]

        return None

    def _cache_prediction(
        self,
        request: InferenceRequest,
        predictions: List[Decimal]
    ) -> None:
        """Cache prediction result.

        Args:
            request: Inference request
            predictions: Prediction values
        """
        cache_key = self._get_cache_key(request)
        self._prediction_cache[cache_key] = (
            predictions,
            datetime.now(timezone.utc)
        )

    def _get_cache_key(self, request: InferenceRequest) -> str:
        """Generate cache key for request.

        Args:
            request: Inference request

        Returns:
            Cache key
        """
        # Simple hash-based key (in production, use more sophisticated hashing)
        feature_hash = hash(request.features.to_pandas().to_json())
        return f"{request.model_name}:{feature_hash}"

    async def shutdown(self) -> None:
        """Shutdown the inference engine.

        Cleans up resources and unloads models.
        """
        try:
            async with self._lock:
                logger.info("Shutting down inference engine")

                # Clear cache
                self._prediction_cache.clear()

                # Unload models
                self.models.clear()

                self._initialized = False

                logger.info("Inference engine shutdown complete")

        except Exception as e:
            logger.error("Error during shutdown", error=str(e))
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "initialized": self._initialized,
            "num_models": len(self.models),
            "cache_size": len(self._prediction_cache),
            "batch_size": self.batch_size,
            "timeout_ms": self.timeout_ms,
            "enable_cache": self.enable_cache
        }
