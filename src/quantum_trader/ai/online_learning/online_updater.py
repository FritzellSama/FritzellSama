"""
Online Learning Model Updater.

This module implements online learning capabilities for continuous
model adaptation in production trading environments.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class UpdateMetrics:
    """Metrics from model update.

    Attributes:
        timestamp: Update timestamp
        samples_processed: Number of samples used
        loss_before: Loss before update
        loss_after: Loss after update
        update_time_seconds: Time taken for update
        model_version: Model version after update
    """
    timestamp: datetime
    samples_processed: int
    loss_before: Decimal
    loss_after: Decimal
    update_time_seconds: Decimal
    model_version: int


class OnlineUpdater:
    """Production-ready online learning model updater.

    Implements incremental model updates in production without
    requiring full retraining. Supports concept drift detection
    and adaptive learning rates.

    Attributes:
        config: Configuration dictionary
        model: PyTorch model to update
        optimizer: Optimizer for updates
        buffer: Experience replay buffer
        device: PyTorch device
        model_version: Current model version
        update_history: History of updates
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module
    ) -> None:
        """Initialize online updater.

        Args:
            config: Configuration dictionary containing:
                - online_learning.buffer_size
                - online_learning.min_samples_per_update
                - online_learning.update_frequency_minutes
                - online_learning.learning_rate
                - online_learning.drift_detection_threshold
                - online_learning.checkpoint_dir
            model: PyTorch model to update

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.model = model
        self._validate_config()

        online_config = self.config["online_learning"]
        self.buffer_size = online_config["buffer_size"]
        self.min_samples = online_config["min_samples_per_update"]
        self.update_frequency = online_config["update_frequency_minutes"]
        self.drift_threshold = Decimal(str(online_config["drift_detection_threshold"]))
        self.checkpoint_dir = Path(online_config["checkpoint_dir"])

        # Initialize device
        self.device = torch.device(
            self.config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self.model.to(self.device)

        # Initialize optimizer with online learning rate
        lr = float(online_config["learning_rate"])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=float(online_config.get("weight_decay", 0.0001))
        )

        # Experience replay buffer
        self.buffer: deque = deque(maxlen=self.buffer_size)

        # Loss function
        self.criterion = nn.MSELoss()

        # State tracking
        self.model_version = 0
        self.last_update_time = datetime.utcnow()
        self.update_history: List[UpdateMetrics] = []
        self.recent_losses: deque = deque(maxlen=100)

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("online_updater_initialized",
                   buffer_size=self.buffer_size,
                   min_samples=self.min_samples,
                   device=str(self.device))

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config parameters are missing
        """
        required_keys = [
            "online_learning.buffer_size",
            "online_learning.min_samples_per_update",
            "online_learning.update_frequency_minutes",
            "online_learning.learning_rate",
            "online_learning.drift_detection_threshold",
            "online_learning.checkpoint_dir"
        ]

        for key in required_keys:
            parts = key.split(".")
            current = self.config
            for part in parts:
                if part not in current:
                    raise ValueError(f"Missing required config key: {key}")
                current = current[part]

    async def add_experience(
        self,
        features: np.ndarray,
        target: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add new experience to buffer.

        Args:
            features: Feature array
            target: Target value
            metadata: Optional metadata about the experience
        """
        try:
            experience = {
                "features": features,
                "target": target,
                "timestamp": datetime.utcnow(),
                "metadata": metadata or {}
            }

            self.buffer.append(experience)

            logger.debug("experience_added",
                        buffer_size=len(self.buffer),
                        max_size=self.buffer_size)

            # Check if update is needed
            if await self._should_update():
                await self.update_model()

        except Exception as e:
            logger.error("add_experience_failed", error=str(e))
            raise

    async def _should_update(self) -> bool:
        """Check if model should be updated.

        Returns:
            True if update should be performed
        """
        # Check buffer size
        if len(self.buffer) < self.min_samples:
            return False

        # Check time since last update
        time_since_update = (datetime.utcnow() - self.last_update_time).total_seconds() / 60

        if time_since_update < self.update_frequency:
            return False

        # Check for concept drift
        drift_detected = await self._detect_concept_drift()

        return drift_detected or time_since_update >= self.update_frequency

    async def _detect_concept_drift(self) -> bool:
        """Detect concept drift in recent data.

        Returns:
            True if concept drift is detected
        """
        if len(self.recent_losses) < 50:
            return False

        try:
            # Calculate recent loss trend
            recent_losses_list = list(self.recent_losses)
            first_half = recent_losses_list[:len(recent_losses_list)//2]
            second_half = recent_losses_list[len(recent_losses_list)//2:]

            avg_loss_first = Decimal(str(np.mean(first_half)))
            avg_loss_second = Decimal(str(np.mean(second_half)))

            # Check if loss is increasing significantly
            if avg_loss_second > avg_loss_first * (Decimal("1") + self.drift_threshold):
                logger.warning("concept_drift_detected",
                             avg_loss_first=str(avg_loss_first),
                             avg_loss_second=str(avg_loss_second))
                return True

            return False

        except Exception as e:
            logger.error("drift_detection_failed", error=str(e))
            return False

    async def update_model(self) -> UpdateMetrics:
        """Perform online model update.

        Returns:
            Update metrics

        Raises:
            RuntimeError: If update fails
        """
        update_start = datetime.utcnow()

        try:
            logger.info("starting_online_update",
                       buffer_size=len(self.buffer),
                       model_version=self.model_version)

            # Prepare batch from buffer
            batch_size = min(
                len(self.buffer),
                self.config["online_learning"].get("batch_size", 32)
            )

            # Sample from buffer (experience replay)
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            experiences = [self.buffer[i] for i in indices]

            features_list = [exp["features"] for exp in experiences]
            targets_list = [exp["target"] for exp in experiences]

            X_batch = np.array(features_list)
            y_batch = np.array(targets_list)

            # Measure loss before update
            loss_before = await self._calculate_loss(X_batch, y_batch)

            # Perform update steps
            num_update_steps = self.config["online_learning"].get("update_steps", 5)

            for step in range(num_update_steps):
                await self._update_step(X_batch, y_batch)

            # Measure loss after update
            loss_after = await self._calculate_loss(X_batch, y_batch)

            # Update state
            self.model_version += 1
            self.last_update_time = datetime.utcnow()

            # Calculate metrics
            update_time = (datetime.utcnow() - update_start).total_seconds()

            metrics = UpdateMetrics(
                timestamp=datetime.utcnow(),
                samples_processed=batch_size,
                loss_before=loss_before,
                loss_after=loss_after,
                update_time_seconds=Decimal(str(update_time)),
                model_version=self.model_version
            )

            self.update_history.append(metrics)

            # Save checkpoint periodically
            checkpoint_interval = self.config["online_learning"].get("checkpoint_interval", 10)
            if self.model_version % checkpoint_interval == 0:
                await self._save_checkpoint()

            logger.info("online_update_completed",
                       model_version=self.model_version,
                       loss_before=str(loss_before),
                       loss_after=str(loss_after),
                       update_time=str(update_time))

            return metrics

        except Exception as e:
            logger.error("online_update_failed", error=str(e))
            raise

    async def _update_step(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray
    ) -> Decimal:
        """Perform single update step.

        Args:
            X_batch: Feature batch
            y_batch: Target batch

        Returns:
            Loss value
        """
        self.model.train()

        try:
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_batch).to(self.device)
            y_tensor = torch.FloatTensor(y_batch).to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(X_tensor)

            # Calculate loss
            loss = self.criterion(predictions, y_tensor)

            # Backward pass
            loss.backward()

            # Gradient clipping
            max_grad_norm = self.config["training"].get("max_grad_norm", 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            self.optimizer.step()

            loss_value = Decimal(str(loss.item()))
            self.recent_losses.append(float(loss_value))

            return loss_value

        except Exception as e:
            logger.error("update_step_failed", error=str(e))
            raise

    async def _calculate_loss(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray
    ) -> Decimal:
        """Calculate loss on batch.

        Args:
            X_batch: Feature batch
            y_batch: Target batch

        Returns:
            Loss value
        """
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_batch).to(self.device)
            y_tensor = torch.FloatTensor(y_batch).to(self.device)

            predictions = self.model(X_tensor)
            loss = self.criterion(predictions, y_tensor)

            return Decimal(str(loss.item()))

    async def _save_checkpoint(self) -> None:
        """Save model checkpoint."""
        try:
            checkpoint_path = self.checkpoint_dir / f"online_model_v{self.model_version}.pt"

            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_version': self.model_version,
                'timestamp': datetime.utcnow().isoformat(),
                'update_history': [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'samples_processed': m.samples_processed,
                        'loss_before': str(m.loss_before),
                        'loss_after': str(m.loss_after),
                        'model_version': m.model_version
                    }
                    for m in self.update_history[-10:]  # Keep last 10
                ]
            }

            torch.save(checkpoint, checkpoint_path)

            logger.info("checkpoint_saved",
                       path=str(checkpoint_path),
                       version=self.model_version)

        except Exception as e:
            logger.error("checkpoint_save_failed", error=str(e))
            raise

    async def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model_version = checkpoint['model_version']

            logger.info("checkpoint_loaded",
                       path=checkpoint_path,
                       version=self.model_version)

        except Exception as e:
            logger.error("checkpoint_load_failed", error=str(e))
            raise

    def get_update_statistics(self) -> Dict[str, Any]:
        """Get update statistics.

        Returns:
            Dictionary of update statistics
        """
        if not self.update_history:
            return {
                "total_updates": 0,
                "avg_loss_improvement": "0",
                "avg_update_time": "0",
                "current_version": self.model_version
            }

        improvements = [
            float(m.loss_before - m.loss_after) for m in self.update_history
        ]

        update_times = [
            float(m.update_time_seconds) for m in self.update_history
        ]

        return {
            "total_updates": len(self.update_history),
            "avg_loss_improvement": str(Decimal(str(np.mean(improvements)))),
            "avg_update_time": str(Decimal(str(np.mean(update_times)))),
            "current_version": self.model_version,
            "buffer_utilization": f"{len(self.buffer)}/{self.buffer_size}",
            "last_update": self.last_update_time.isoformat()
        }
