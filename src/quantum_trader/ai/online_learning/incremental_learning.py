"""Incremental (online) learning system for continuous model updates.

This module implements online learning algorithms that update models
incrementally as new data arrives, enabling adaptation to changing markets.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from collections import deque
import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class IncrementalLearner:
    """Incremental learning system for online model updates.

    Implements various incremental learning strategies including
    mini-batch SGD, exponentially weighted averaging, and adaptive learning rates.

    Attributes:
        config: Configuration dictionary
        model: Base model to update incrementally
        buffer: Circular buffer for recent samples
        learning_rate: Current learning rate

    Example:
        >>> config = {
        ...     "buffer_size": 10000,
        ...     "mini_batch_size": 32,
        ...     "initial_learning_rate": "0.001",
        ...     "learning_rate_decay": "0.99",
        ...     "update_frequency": 100
        ... }
        >>> learner = IncrementalLearner(config, base_model)
        >>> await learner.update(new_features, new_labels)
        >>> predictions = learner.predict(test_features)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        base_model: Any
    ) -> None:
        """Initialize incremental learner.

        Args:
            config: Configuration dictionary
            base_model: Base ML model to update incrementally

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self.base_model = base_model
        self._validate_config()

        # Buffer parameters
        self.buffer_size = config.get("buffer_size", 10000)
        self.mini_batch_size = config.get("mini_batch_size", 32)

        # Learning parameters
        self.initial_learning_rate = Decimal(
            str(config.get("initial_learning_rate", "0.001"))
        )
        self.learning_rate_decay = Decimal(
            str(config.get("learning_rate_decay", "0.99"))
        )
        self.min_learning_rate = Decimal(
            str(config.get("min_learning_rate", "0.00001"))
        )
        self.current_learning_rate = self.initial_learning_rate

        # Update parameters
        self.update_frequency = config.get("update_frequency", 100)
        self.warmup_samples = config.get("warmup_samples", 1000)

        # Exponential weighting
        self.ema_alpha = Decimal(str(config.get("ema_alpha", "0.1")))

        # Data storage
        self.feature_buffer: deque = deque(maxlen=self.buffer_size)
        self.label_buffer: deque = deque(maxlen=self.buffer_size)

        # Statistics
        self.num_updates = 0
        self.num_samples_seen = 0
        self.update_history: List[Dict[str, Any]] = []

        # Performance tracking
        self.performance_window = deque(
            maxlen=config.get("performance_window_size", 1000)
        )

        logger.info(
            "Incremental learner initialized",
            buffer_size=self.buffer_size,
            mini_batch_size=self.mini_batch_size,
            initial_lr=float(self.initial_learning_rate)
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If required config missing
        """
        required_keys = ["buffer_size", "mini_batch_size"]

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    async def update(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Update model with new data incrementally.

        Args:
            features: New features [num_samples, feature_dim]
            labels: New labels [num_samples, output_dim]
            sample_weights: Optional sample weights

        Returns:
            Dictionary with update statistics

        Raises:
            ValueError: If input shapes don't match
        """
        try:
            if len(features) != len(labels):
                raise ValueError(
                    f"Features and labels length mismatch: "
                    f"{len(features)} != {len(labels)}"
                )

            logger.info(
                "Updating model incrementally",
                num_samples=len(features),
                buffer_size=len(self.feature_buffer)
            )

            # Add to buffer
            for i in range(len(features)):
                self.feature_buffer.append(features[i])
                self.label_buffer.append(labels[i])
                self.num_samples_seen += 1

            # Check if should perform update
            should_update = (
                len(self.feature_buffer) >= self.warmup_samples and
                self.num_samples_seen % self.update_frequency == 0
            )

            update_stats = {}

            if should_update:
                update_stats = await self._perform_update(sample_weights)

                # Decay learning rate
                self._decay_learning_rate()

            return {
                "updated": should_update,
                "num_samples_seen": self.num_samples_seen,
                "buffer_size": len(self.feature_buffer),
                "learning_rate": float(self.current_learning_rate),
                **update_stats
            }

        except Exception as e:
            logger.error("Failed to update model", error=str(e))
            raise

    async def _perform_update(
        self,
        sample_weights: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Perform actual model update.

        Args:
            sample_weights: Optional sample weights

        Returns:
            Update statistics
        """
        try:
            # Sample mini-batch from buffer
            batch_features, batch_labels = self._sample_mini_batch()

            # Update model based on type
            if hasattr(self.base_model, 'partial_fit'):
                # Scikit-learn style incremental learning
                self.base_model.partial_fit(batch_features, batch_labels)
                loss = None

            elif hasattr(self.base_model, 'train_step'):
                # Custom train_step method
                loss = self.base_model.train_step(
                    batch_features,
                    batch_labels,
                    float(self.current_learning_rate)
                )

            else:
                # Full retrain on buffer (less efficient)
                buffer_features = np.array(list(self.feature_buffer))
                buffer_labels = np.array(list(self.label_buffer))

                self.base_model.train(buffer_features, buffer_labels)
                loss = None

            self.num_updates += 1

            # Track performance
            performance = await self._evaluate_performance(
                batch_features,
                batch_labels
            )

            update_record = {
                "update_num": self.num_updates,
                "timestamp": datetime.utcnow(),
                "learning_rate": float(self.current_learning_rate),
                "batch_size": len(batch_features),
                "loss": float(loss) if loss is not None else None,
                **performance
            }

            self.update_history.append(update_record)

            logger.info(
                "Model update completed",
                update_num=self.num_updates,
                learning_rate=float(self.current_learning_rate),
                performance=performance
            )

            return update_record

        except Exception as e:
            logger.error("Failed to perform update", error=str(e))
            raise

    def _sample_mini_batch(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample mini-batch from buffer.

        Returns:
            Tuple of (batch_features, batch_labels)
        """
        # Sample indices
        buffer_size = len(self.feature_buffer)
        batch_size = min(self.mini_batch_size, buffer_size)

        # Use recent data bias
        if self.config.get("recent_bias", False):
            # Exponentially weighted sampling (favor recent)
            weights = np.exp(np.linspace(
                -2,
                0,
                buffer_size
            ))
            weights /= weights.sum()

            indices = np.random.choice(
                buffer_size,
                size=batch_size,
                replace=False,
                p=weights
            )
        else:
            # Uniform sampling
            indices = np.random.choice(
                buffer_size,
                size=batch_size,
                replace=False
            )

        # Gather samples
        batch_features = np.array([self.feature_buffer[i] for i in indices])
        batch_labels = np.array([self.label_buffer[i] for i in indices])

        return batch_features, batch_labels

    async def _evaluate_performance(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance on batch.

        Args:
            features: Evaluation features
            labels: True labels

        Returns:
            Performance metrics
        """
        try:
            if hasattr(self.base_model, 'evaluate'):
                metrics = self.base_model.evaluate(features, labels)
            else:
                # Basic MSE evaluation
                predictions = self.base_model.predict(features)
                mse = float(np.mean((predictions - labels) ** 2))
                metrics = {"mse": mse}

            # Store in performance window
            self.performance_window.append(metrics)

            return metrics

        except Exception as e:
            logger.error("Failed to evaluate performance", error=str(e))
            return {}

    def _decay_learning_rate(self) -> None:
        """Decay learning rate."""
        self.current_learning_rate *= self.learning_rate_decay
        self.current_learning_rate = max(
            self.current_learning_rate,
            self.min_learning_rate
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions using current model.

        Args:
            features: Input features

        Returns:
            Predictions
        """
        try:
            predictions = self.base_model.predict(features)

            logger.debug("Generated predictions", num_samples=len(predictions))

            return predictions

        except Exception as e:
            logger.error("Failed to generate predictions", error=str(e))
            raise

    def get_current_performance(self) -> Dict[str, Decimal]:
        """Get current rolling performance metrics.

        Returns:
            Dictionary of aggregated performance metrics
        """
        try:
            if not self.performance_window:
                return {}

            # Aggregate metrics across window
            metrics = {}

            # Get all metric keys
            all_keys = set()
            for perf in self.performance_window:
                all_keys.update(perf.keys())

            # Average each metric
            for key in all_keys:
                values = [
                    perf[key]
                    for perf in self.performance_window
                    if key in perf
                ]

                if values:
                    metrics[f"{key}_mean"] = Decimal(str(np.mean(values)))
                    metrics[f"{key}_std"] = Decimal(str(np.std(values)))

            return metrics

        except Exception as e:
            logger.error("Failed to get current performance", error=str(e))
            return {}

    def reset_learning_rate(self) -> None:
        """Reset learning rate to initial value."""
        self.current_learning_rate = self.initial_learning_rate

        logger.info(
            "Learning rate reset",
            learning_rate=float(self.current_learning_rate)
        )

    def clear_buffer(self) -> None:
        """Clear data buffer."""
        self.feature_buffer.clear()
        self.label_buffer.clear()

        logger.info("Buffer cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Get learner statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "num_samples_seen": self.num_samples_seen,
            "num_updates": self.num_updates,
            "buffer_size": len(self.feature_buffer),
            "buffer_capacity": self.buffer_size,
            "current_learning_rate": float(self.current_learning_rate),
            "current_performance": {
                k: float(v)
                for k, v in self.get_current_performance().items()
            }
        }

    def get_update_history(self) -> pl.DataFrame:
        """Get update history as DataFrame.

        Returns:
            Polars DataFrame with update history
        """
        try:
            if not self.update_history:
                return pl.DataFrame()

            # Flatten history
            records = []
            for entry in self.update_history:
                record = {
                    "update_num": entry["update_num"],
                    "timestamp": entry["timestamp"],
                    "learning_rate": Decimal(str(entry["learning_rate"])),
                    "batch_size": entry["batch_size"]
                }

                # Add loss if available
                if entry["loss"] is not None:
                    record["loss"] = Decimal(str(entry["loss"]))

                # Add performance metrics
                for key, value in entry.items():
                    if key not in ["update_num", "timestamp", "learning_rate", "batch_size", "loss"]:
                        record[key] = Decimal(str(value)) if value is not None else None

                records.append(record)

            df = pl.DataFrame(records)

            logger.info("Generated update history DataFrame", rows=len(df))

            return df

        except Exception as e:
            logger.error("Failed to generate update history DataFrame", error=str(e))
            raise

    async def adaptive_update(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        performance_threshold: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """Adaptive update based on performance.

        Only updates model if performance degrades below threshold.

        Args:
            features: New features
            labels: New labels
            performance_threshold: Optional performance threshold

        Returns:
            Update statistics
        """
        try:
            # Add to buffer
            for i in range(len(features)):
                self.feature_buffer.append(features[i])
                self.label_buffer.append(labels[i])
                self.num_samples_seen += 1

            # Evaluate current performance
            current_perf = self.get_current_performance()

            # Check if update needed
            if performance_threshold is not None:
                # Use primary metric
                metric_key = self.config.get("primary_metric", "mse_mean")

                if metric_key in current_perf:
                    current_value = current_perf[metric_key]

                    # Check if performance degraded
                    needs_update = current_value > performance_threshold
                else:
                    needs_update = True
            else:
                needs_update = True

            update_stats = {}

            if needs_update:
                logger.info(
                    "Performance degraded, triggering update",
                    current_perf=current_perf
                )

                update_stats = await self._perform_update(None)

                # Increase learning rate temporarily for faster adaptation
                self.current_learning_rate = min(
                    self.current_learning_rate * Decimal("1.5"),
                    self.initial_learning_rate
                )

            return {
                "updated": needs_update,
                "num_samples_seen": self.num_samples_seen,
                "current_performance": {
                    k: float(v) for k, v in current_perf.items()
                },
                **update_stats
            }

        except Exception as e:
            logger.error("Failed to perform adaptive update", error=str(e))
            raise

    def save_state(self, path: str) -> None:
        """Save learner state to disk.

        Args:
            path: File path to save state
        """
        try:
            state = {
                "feature_buffer": list(self.feature_buffer),
                "label_buffer": list(self.label_buffer),
                "num_updates": self.num_updates,
                "num_samples_seen": self.num_samples_seen,
                "current_learning_rate": float(self.current_learning_rate),
                "update_history": self.update_history,
                "config": self.config
            }

            np.savez_compressed(path, **state)

            logger.info("Learner state saved", path=path)

        except Exception as e:
            logger.error("Failed to save learner state", error=str(e))
            raise

    def load_state(self, path: str) -> None:
        """Load learner state from disk.

        Args:
            path: File path to load state from
        """
        try:
            data = np.load(path, allow_pickle=True)

            self.feature_buffer = deque(
                data["feature_buffer"],
                maxlen=self.buffer_size
            )
            self.label_buffer = deque(
                data["label_buffer"],
                maxlen=self.buffer_size
            )
            self.num_updates = int(data["num_updates"])
            self.num_samples_seen = int(data["num_samples_seen"])
            self.current_learning_rate = Decimal(
                str(data["current_learning_rate"])
            )
            self.update_history = data["update_history"].tolist()

            logger.info("Learner state loaded", path=path)

        except Exception as e:
            logger.error("Failed to load learner state", error=str(e))
            raise
