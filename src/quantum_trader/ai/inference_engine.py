"""Inference engine for real-time model predictions.

This module provides a high-performance inference engine that manages
model loading, caching, batching, and distributed inference.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """High-performance inference engine.

    Manages model lifecycle, request batching, caching, and
    distributed inference for low-latency predictions.

    Attributes:
        config: Configuration dictionary
        models: Dictionary of loaded models
        cache: Prediction cache
        batch_queue: Queue for batching requests

    Example:
        >>> config = {
        ...     "batch_size": 32,
        ...     "batch_timeout_ms": 100,
        ...     "cache_ttl_seconds": 60,
        ...     "max_models": 10
        ... }
        >>> engine = InferenceEngine(config)
        >>> await engine.load_model("price_predictor", model_path)
        >>> predictions = await engine.predict("price_predictor", features)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize inference engine.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        # Batching parameters
        self.batch_size = config.get("batch_size", 32)
        self.batch_timeout_ms = config.get("batch_timeout_ms", 100)

        # Caching parameters
        self.cache_enabled = config.get("cache_enabled", True)
        self.cache_ttl_seconds = config.get("cache_ttl_seconds", 60)

        # Model management
        self.max_models = config.get("max_models", 10)
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}

        # Request batching
        self.batch_queues: Dict[str, List] = defaultdict(list)
        self.batch_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.batch_tasks: Dict[str, Optional[asyncio.Task]] = {}

        # Prediction cache
        self.cache: Dict[str, Tuple[np.ndarray, datetime]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Statistics
        self.total_predictions = 0
        self.total_latency_ms = Decimal("0")
        self.model_load_count = 0

        logger.info(
            "Inference engine initialized",
            batch_size=self.batch_size,
            cache_enabled=self.cache_enabled,
            max_models=self.max_models
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If required config missing
        """
        required_keys = ["batch_size"]

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    async def load_model(
        self,
        model_id: str,
        model: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Load model into engine.

        Args:
            model_id: Unique model identifier
            model: Model object with predict() method
            metadata: Optional model metadata

        Raises:
            ValueError: If model already loaded or max models exceeded
        """
        try:
            if model_id in self.models:
                raise ValueError(f"Model {model_id} already loaded")

            if len(self.models) >= self.max_models:
                # Evict least recently used model
                await self._evict_lru_model()

            # Store model
            self.models[model_id] = model
            self.model_metadata[model_id] = {
                "loaded_at": datetime.utcnow(),
                "last_used": datetime.utcnow(),
                "prediction_count": 0,
                "average_latency_ms": Decimal("0"),
                **(metadata or {})
            }

            self.model_load_count += 1

            logger.info(
                "Model loaded",
                model_id=model_id,
                total_models=len(self.models)
            )

        except Exception as e:
            logger.error("Failed to load model", model_id=model_id, error=str(e))
            raise

    async def unload_model(self, model_id: str) -> None:
        """Unload model from engine.

        Args:
            model_id: Model identifier

        Raises:
            ValueError: If model not found
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")

            # Clear pending batches
            if model_id in self.batch_queues:
                del self.batch_queues[model_id]

            # Cancel batch task
            if model_id in self.batch_tasks and self.batch_tasks[model_id]:
                self.batch_tasks[model_id].cancel()

            # Remove model
            del self.models[model_id]
            del self.model_metadata[model_id]

            # Clear cache entries for this model
            self._clear_model_cache(model_id)

            logger.info("Model unloaded", model_id=model_id)

        except Exception as e:
            logger.error("Failed to unload model", model_id=model_id, error=str(e))
            raise

    async def predict(
        self,
        model_id: str,
        features: np.ndarray,
        use_cache: bool = True,
        use_batching: bool = True
    ) -> np.ndarray:
        """Make predictions using specified model.

        Args:
            model_id: Model identifier
            features: Input features
            use_cache: Whether to use cache
            use_batching: Whether to use request batching

        Returns:
            Predictions

        Raises:
            ValueError: If model not found
        """
        try:
            start_time = datetime.utcnow()

            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")

            # Check cache
            if use_cache and self.cache_enabled:
                cached = self._get_from_cache(model_id, features)

                if cached is not None:
                    self.cache_hits += 1
                    return cached

                self.cache_misses += 1

            # Get prediction
            if use_batching:
                predictions = await self._batched_predict(model_id, features)
            else:
                predictions = await self._direct_predict(model_id, features)

            # Update cache
            if use_cache and self.cache_enabled:
                self._add_to_cache(model_id, features, predictions)

            # Update statistics
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._update_statistics(model_id, latency_ms)

            return predictions

        except Exception as e:
            logger.error(
                "Prediction failed",
                model_id=model_id,
                error=str(e)
            )
            raise

    async def _direct_predict(
        self,
        model_id: str,
        features: np.ndarray
    ) -> np.ndarray:
        """Direct prediction without batching.

        Args:
            model_id: Model identifier
            features: Input features

        Returns:
            Predictions
        """
        model = self.models[model_id]

        if asyncio.iscoroutinefunction(model.predict):
            predictions = await model.predict(features)
        else:
            predictions = model.predict(features)

        return predictions

    async def _batched_predict(
        self,
        model_id: str,
        features: np.ndarray
    ) -> np.ndarray:
        """Batched prediction for efficiency.

        Args:
            model_id: Model identifier
            features: Input features

        Returns:
            Predictions
        """
        # Create future for this request
        future = asyncio.Future()

        # Add to batch queue
        async with self.batch_locks[model_id]:
            self.batch_queues[model_id].append((features, future))

            # Start batch processor if not running
            if model_id not in self.batch_tasks or self.batch_tasks[model_id] is None:
                self.batch_tasks[model_id] = asyncio.create_task(
                    self._process_batch(model_id)
                )

        # Wait for result
        predictions = await future

        return predictions

    async def _process_batch(self, model_id: str) -> None:
        """Process batched requests for a model.

        Args:
            model_id: Model identifier
        """
        try:
            # Wait for batch to fill or timeout
            await asyncio.sleep(self.batch_timeout_ms / 1000.0)

            async with self.batch_locks[model_id]:
                if not self.batch_queues[model_id]:
                    self.batch_tasks[model_id] = None
                    return

                # Get batch
                batch = self.batch_queues[model_id][:self.batch_size]
                self.batch_queues[model_id] = self.batch_queues[model_id][self.batch_size:]

            # Prepare batch
            batch_features = np.vstack([item[0] for item in batch])

            # Predict
            batch_predictions = await self._direct_predict(model_id, batch_features)

            # Distribute results
            for i, (_, future) in enumerate(batch):
                if not future.done():
                    future.set_result(batch_predictions[i:i+1])

            # Continue processing if queue not empty
            if self.batch_queues[model_id]:
                self.batch_tasks[model_id] = asyncio.create_task(
                    self._process_batch(model_id)
                )
            else:
                self.batch_tasks[model_id] = None

        except Exception as e:
            logger.error(
                "Batch processing failed",
                model_id=model_id,
                error=str(e)
            )

            # Fail all futures in batch
            async with self.batch_locks[model_id]:
                for _, future in self.batch_queues[model_id]:
                    if not future.done():
                        future.set_exception(e)

                self.batch_queues[model_id] = []
                self.batch_tasks[model_id] = None

    def _get_from_cache(
        self,
        model_id: str,
        features: np.ndarray
    ) -> Optional[np.ndarray]:
        """Get prediction from cache.

        Args:
            model_id: Model identifier
            features: Input features

        Returns:
            Cached predictions or None
        """
        cache_key = self._generate_cache_key(model_id, features)

        if cache_key in self.cache:
            predictions, timestamp = self.cache[cache_key]

            # Check TTL
            age_seconds = (datetime.utcnow() - timestamp).total_seconds()

            if age_seconds <= self.cache_ttl_seconds:
                return predictions
            else:
                # Expired
                del self.cache[cache_key]

        return None

    def _add_to_cache(
        self,
        model_id: str,
        features: np.ndarray,
        predictions: np.ndarray
    ) -> None:
        """Add prediction to cache.

        Args:
            model_id: Model identifier
            features: Input features
            predictions: Predictions to cache
        """
        cache_key = self._generate_cache_key(model_id, features)
        self.cache[cache_key] = (predictions.copy(), datetime.utcnow())

        # Limit cache size
        max_cache_size = self.config.get("max_cache_size", 10000)

        if len(self.cache) > max_cache_size:
            # Remove oldest entries
            sorted_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k][1]
            )

            for key in sorted_keys[:len(self.cache) - max_cache_size]:
                del self.cache[key]

    def _generate_cache_key(
        self,
        model_id: str,
        features: np.ndarray
    ) -> str:
        """Generate cache key for features.

        Args:
            model_id: Model identifier
            features: Input features

        Returns:
            Cache key string
        """
        # Use hash of features for key
        feature_hash = hash(features.tobytes())
        return f"{model_id}:{feature_hash}"

    def _clear_model_cache(self, model_id: str) -> None:
        """Clear cache entries for a model.

        Args:
            model_id: Model identifier
        """
        keys_to_remove = [
            key for key in self.cache.keys()
            if key.startswith(f"{model_id}:")
        ]

        for key in keys_to_remove:
            del self.cache[key]

    async def _evict_lru_model(self) -> None:
        """Evict least recently used model."""
        if not self.models:
            return

        # Find LRU model
        lru_model_id = min(
            self.model_metadata.keys(),
            key=lambda k: self.model_metadata[k]["last_used"]
        )

        logger.info("Evicting LRU model", model_id=lru_model_id)

        await self.unload_model(lru_model_id)

    async def _update_statistics(
        self,
        model_id: str,
        latency_ms: float
    ) -> None:
        """Update prediction statistics.

        Args:
            model_id: Model identifier
            latency_ms: Prediction latency in milliseconds
        """
        metadata = self.model_metadata[model_id]

        metadata["last_used"] = datetime.utcnow()
        metadata["prediction_count"] += 1

        # Update rolling average latency
        count = metadata["prediction_count"]
        avg_latency = metadata["average_latency_ms"]

        metadata["average_latency_ms"] = (
            (avg_latency * Decimal(str(count - 1)) + Decimal(str(latency_ms))) /
            Decimal(str(count))
        )

        self.total_predictions += 1
        self.total_latency_ms += Decimal(str(latency_ms))

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "total_predictions": self.total_predictions,
            "average_latency_ms": (
                float(self.total_latency_ms / max(self.total_predictions, 1))
            ),
            "models_loaded": len(self.models),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            ),
            "cache_size": len(self.cache),
            "model_stats": {
                model_id: {
                    "prediction_count": meta["prediction_count"],
                    "average_latency_ms": float(meta["average_latency_ms"]),
                    "last_used": meta["last_used"]
                }
                for model_id, meta in self.model_metadata.items()
            }
        }

    async def warmup(self, model_id: str, sample_features: np.ndarray) -> None:
        """Warm up model with sample predictions.

        Args:
            model_id: Model identifier
            sample_features: Sample features for warmup
        """
        try:
            logger.info("Warming up model", model_id=model_id)

            # Run several predictions
            for _ in range(self.config.get("warmup_iterations", 10)):
                await self.predict(
                    model_id,
                    sample_features,
                    use_cache=False,
                    use_batching=False
                )

            logger.info("Model warmup completed", model_id=model_id)

        except Exception as e:
            logger.error("Model warmup failed", model_id=model_id, error=str(e))
            raise

    def clear_cache(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info("Cache cleared")

    async def shutdown(self) -> None:
        """Shutdown engine gracefully."""
        try:
            logger.info("Shutting down inference engine")

            # Cancel all batch tasks
            for task in self.batch_tasks.values():
                if task and not task.done():
                    task.cancel()

            # Wait for tasks to complete
            await asyncio.gather(*self.batch_tasks.values(), return_exceptions=True)

            # Unload all models
            model_ids = list(self.models.keys())
            for model_id in model_ids:
                await self.unload_model(model_id)

            # Clear cache
            self.clear_cache()

            logger.info("Inference engine shutdown completed")

        except Exception as e:
            logger.error("Shutdown failed", error=str(e))
            raise
