"""Continual learning system for adaptive model updates.

This module provides infrastructure for continuously updating ML models
with new data while preventing catastrophic forgetting.
"""

from __future__ import annotations

from collections import deque
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel

logger = get_logger(__name__)


class ExperienceReplay:
    """Experience replay buffer for continual learning.

    Stores past experiences to prevent catastrophic forgetting by
    replaying old data during training on new data.

    Attributes:
        config: Configuration dictionary for replay buffer
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize experience replay buffer.

        Args:
            config: Configuration dictionary with parameters:
                - max_size: Maximum buffer size (default: 10000)
                - sample_size: Number of samples to replay (default: 32)
                - priority_alpha: Priority exponent for prioritized replay (default: 0.6)

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.max_size = int(config.get("max_size", 10000))
        self.sample_size = int(config.get("sample_size", 32))
        self.priority_alpha = Decimal(str(config.get("priority_alpha", "0.6")))

        # Storage
        self.buffer: deque = deque(maxlen=self.max_size)
        self.priorities: deque = deque(maxlen=self.max_size)

        logger.info(
            "experience_replay_initialized",
            max_size=self.max_size,
            sample_size=self.sample_size,
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(self.config)}")

    def add(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        priority: Optional[Decimal] = None,
    ) -> None:
        """Add experience to replay buffer.

        Args:
            features: Feature array
            labels: Label array
            priority: Optional priority value (higher = more important)

        Raises:
            ValueError: If inputs are invalid
        """
        try:
            if not isinstance(features, np.ndarray) or not isinstance(labels, np.ndarray):
                raise ValueError("Features and labels must be numpy arrays")

            # Default priority
            if priority is None:
                priority = Decimal("1.0")
            elif not isinstance(priority, Decimal):
                priority = Decimal(str(priority))

            # Store experience
            self.buffer.append({"features": features, "labels": labels})
            self.priorities.append(priority)

            logger.debug(
                "experience_added",
                buffer_size=len(self.buffer),
                priority=str(priority),
            )

        except Exception as e:
            logger.error("add_experience_failed", error=str(e))
            raise

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample batch from replay buffer.

        Returns:
            Tuple of (features, labels) arrays

        Raises:
            ValueError: If buffer is empty or sampling fails
        """
        try:
            if len(self.buffer) == 0:
                raise ValueError("Cannot sample from empty buffer")

            # Calculate sampling probabilities based on priorities
            priorities_array = np.array([float(p) for p in self.priorities])
            priorities_raised = priorities_array ** float(self.priority_alpha)
            probabilities = priorities_raised / np.sum(priorities_raised)

            # Sample indices
            sample_size = min(self.sample_size, len(self.buffer))
            indices = np.random.choice(
                len(self.buffer),
                size=sample_size,
                replace=False,
                p=probabilities,
            )

            # Gather samples
            features_list = []
            labels_list = []

            for idx in indices:
                experience = self.buffer[idx]
                features_list.append(experience["features"])
                labels_list.append(experience["labels"])

            features_batch = np.array(features_list)
            labels_batch = np.array(labels_list)

            logger.debug("experience_sampled", sample_size=sample_size)

            return features_batch, labels_batch

        except Exception as e:
            logger.error("sample_failed", error=str(e))
            raise

    def update_priority(self, index: int, priority: Decimal) -> None:
        """Update priority for an experience.

        Args:
            index: Index of experience to update
            priority: New priority value

        Raises:
            ValueError: If index is invalid
        """
        try:
            if index < 0 or index >= len(self.priorities):
                raise ValueError(f"Invalid index: {index}")

            if not isinstance(priority, Decimal):
                priority = Decimal(str(priority))

            self.priorities[index] = priority

        except Exception as e:
            logger.error("update_priority_failed", error=str(e))
            raise

    def clear(self) -> None:
        """Clear the replay buffer."""
        self.buffer.clear()
        self.priorities.clear()
        logger.info("replay_buffer_cleared")


class ElasticWeightConsolidation:
    """Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting.

    EWC protects important weights by adding a penalty term to the loss function
    based on Fisher information.

    Attributes:
        config: Configuration dictionary for EWC
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize EWC.

        Args:
            config: Configuration dictionary with parameters:
                - lambda_ewc: EWC regularization strength (default: 0.4)
                - fisher_sample_size: Number of samples for Fisher computation (default: 200)

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.lambda_ewc = Decimal(str(config.get("lambda_ewc", "0.4")))
        self.fisher_sample_size = int(config.get("fisher_sample_size", 200))

        # Storage for old parameters and Fisher information
        self.old_params: Dict[str, torch.Tensor] = {}
        self.fisher_info: Dict[str, torch.Tensor] = {}

        logger.info(
            "ewc_initialized",
            lambda_ewc=str(self.lambda_ewc),
            fisher_sample_size=self.fisher_sample_size,
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(self.config)}")

    def compute_fisher_information(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
    ) -> None:
        """Compute Fisher information matrix for model parameters.

        Args:
            model: PyTorch model
            data_loader: DataLoader with training data

        Raises:
            ValueError: If computation fails
        """
        try:
            logger.info("computing_fisher_information")

            # Initialize Fisher information
            for name, param in model.named_parameters():
                self.fisher_info[name] = torch.zeros_like(param.data)

            model.eval()
            sample_count = 0

            # Compute Fisher information
            for features, labels in data_loader:
                if sample_count >= self.fisher_sample_size:
                    break

                model.zero_grad()
                output = model(features)
                loss = nn.functional.cross_entropy(output, labels)
                loss.backward()

                # Accumulate squared gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        self.fisher_info[name] += param.grad.data ** 2

                sample_count += len(features)

            # Average Fisher information
            for name in self.fisher_info:
                self.fisher_info[name] /= sample_count

            # Store current parameters
            for name, param in model.named_parameters():
                self.old_params[name] = param.data.clone()

            logger.info(
                "fisher_information_computed",
                num_params=len(self.fisher_info),
                samples=sample_count,
            )

        except Exception as e:
            logger.error("fisher_computation_failed", error=str(e))
            raise

    def compute_ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC penalty for current model parameters.

        Args:
            model: PyTorch model

        Returns:
            EWC penalty term

        Raises:
            ValueError: If computation fails
        """
        try:
            if len(self.old_params) == 0:
                return torch.tensor(0.0)

            ewc_loss = torch.tensor(0.0)

            for name, param in model.named_parameters():
                if name in self.fisher_info:
                    # EWC penalty: λ/2 * F * (θ - θ*)²
                    penalty = (
                        self.fisher_info[name]
                        * (param - self.old_params[name]) ** 2
                    )
                    ewc_loss += penalty.sum()

            ewc_loss *= float(self.lambda_ewc) / 2.0

            return ewc_loss

        except Exception as e:
            logger.error("ewc_loss_computation_failed", error=str(e))
            raise


class ContinualLearner:
    """Main continual learning system.

    Combines experience replay and EWC for robust continual learning
    without catastrophic forgetting.

    Attributes:
        config: Configuration dictionary for continual learning
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize continual learning system.

        Args:
            config: Configuration dictionary with subsystem configs

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        # Initialize subsystems
        self.replay_enabled = config.get("replay_enabled", True)
        self.ewc_enabled = config.get("ewc_enabled", True)

        if self.replay_enabled:
            self.replay_buffer = ExperienceReplay(config.get("replay", {}))

        if self.ewc_enabled:
            self.ewc = ElasticWeightConsolidation(config.get("ewc", {}))

        self.learning_rate = Decimal(str(config.get("learning_rate", "0.001")))
        self.batch_size = int(config.get("batch_size", 32))
        self.update_frequency = int(config.get("update_frequency", 100))

        self.update_count = 0

        logger.info(
            "continual_learner_initialized",
            replay_enabled=self.replay_enabled,
            ewc_enabled=self.ewc_enabled,
            learning_rate=str(self.learning_rate),
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(self.config)}")

    def update(
        self,
        model: nn.Module,
        features: np.ndarray,
        labels: np.ndarray,
        optimizer: optim.Optimizer,
    ) -> Dict[str, Any]:
        """Update model with new data using continual learning.

        Args:
            model: PyTorch model to update
            features: New training features
            labels: New training labels
            optimizer: PyTorch optimizer

        Returns:
            Dictionary with update metrics

        Raises:
            ValueError: If update fails
        """
        try:
            self.update_count += 1

            # Add to replay buffer
            if self.replay_enabled:
                self.replay_buffer.add(features, labels)

            # Convert to tensors
            features_tensor = torch.FloatTensor(features)
            labels_tensor = torch.LongTensor(labels)

            # Training mode
            model.train()
            optimizer.zero_grad()

            # Forward pass on new data
            output = model(features_tensor)
            loss = nn.functional.cross_entropy(output, labels_tensor)

            # Add EWC penalty
            if self.ewc_enabled and len(self.ewc.old_params) > 0:
                ewc_loss = self.ewc.compute_ewc_loss(model)
                total_loss = loss + ewc_loss
            else:
                ewc_loss = torch.tensor(0.0)
                total_loss = loss

            # Replay old experiences
            replay_loss = torch.tensor(0.0)
            if self.replay_enabled and len(self.replay_buffer.buffer) > 0:
                replay_features, replay_labels = self.replay_buffer.sample()
                replay_features_tensor = torch.FloatTensor(replay_features)
                replay_labels_tensor = torch.LongTensor(replay_labels)

                replay_output = model(replay_features_tensor)
                replay_loss = nn.functional.cross_entropy(
                    replay_output, replay_labels_tensor
                )
                total_loss += replay_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Periodic Fisher information update
            if (
                self.ewc_enabled
                and self.update_count % self.update_frequency == 0
            ):
                logger.info("updating_fisher_information", update_count=self.update_count)
                # Create simple dataloader for Fisher computation
                dataset = torch.utils.data.TensorDataset(features_tensor, labels_tensor)
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=self.batch_size
                )
                self.ewc.compute_fisher_information(model, dataloader)

            metrics = {
                "total_loss": Decimal(str(total_loss.item())),
                "new_data_loss": Decimal(str(loss.item())),
                "ewc_loss": Decimal(str(ewc_loss.item())),
                "replay_loss": Decimal(str(replay_loss.item())),
                "update_count": self.update_count,
                "buffer_size": len(self.replay_buffer.buffer) if self.replay_enabled else 0,
            }

            logger.info(
                "model_updated",
                update_count=self.update_count,
                total_loss=str(metrics["total_loss"]),
            )

            return metrics

        except Exception as e:
            logger.error("continual_learning_update_failed", error=str(e))
            raise

    def reset(self) -> None:
        """Reset continual learning state."""
        try:
            if self.replay_enabled:
                self.replay_buffer.clear()

            if self.ewc_enabled:
                self.ewc.old_params.clear()
                self.ewc.fisher_info.clear()

            self.update_count = 0

            logger.info("continual_learner_reset")

        except Exception as e:
            logger.error("reset_failed", error=str(e))
            raise


class IncrementalLearner:
    """Incremental learning for online model updates.

    Provides simple incremental learning without replay or EWC,
    suitable for scenarios where catastrophic forgetting is less of a concern.

    Attributes:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize incremental learner.

        Args:
            config: Configuration dictionary with parameters:
                - learning_rate: Learning rate for updates (default: 0.001)
                - momentum: Momentum for gradient updates (default: 0.9)
                - window_size: Size of sliding window for statistics (default: 1000)

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.learning_rate = Decimal(str(config.get("learning_rate", "0.001")))
        self.momentum = Decimal(str(config.get("momentum", "0.9")))
        self.window_size = int(config.get("window_size", 1000))

        # Statistics tracking
        self.loss_history: deque = deque(maxlen=self.window_size)
        self.accuracy_history: deque = deque(maxlen=self.window_size)

        logger.info(
            "incremental_learner_initialized",
            learning_rate=str(self.learning_rate),
            window_size=self.window_size,
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(self.config)}")

    def partial_fit(
        self,
        model: BaseMLModel,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, Any]:
        """Incrementally update model with new data.

        Args:
            model: Model to update (must support partial_fit)
            features: New training features
            labels: New training labels

        Returns:
            Dictionary with update metrics

        Raises:
            ValueError: If update fails
        """
        try:
            # Train on new data
            if hasattr(model, "partial_fit"):
                model.partial_fit(features, labels)
            else:
                model.train(features, labels)

            # Evaluate
            predictions = model.predict(features)
            accuracy = np.mean(predictions == labels)

            # Track statistics
            self.accuracy_history.append(Decimal(str(accuracy)))

            metrics = {
                "accuracy": Decimal(str(accuracy)),
                "avg_accuracy": (
                    sum(self.accuracy_history) / Decimal(str(len(self.accuracy_history)))
                    if self.accuracy_history
                    else Decimal("0.0")
                ),
                "samples_processed": len(features),
            }

            logger.info(
                "incremental_update_complete",
                accuracy=str(metrics["accuracy"]),
                samples=len(features),
            )

            return metrics

        except Exception as e:
            logger.error("partial_fit_failed", error=str(e))
            raise


__all__ = [
    "ExperienceReplay",
    "ElasticWeightConsolidation",
    "ContinualLearner",
    "IncrementalLearner",
]
