"""
Meta-Learning for Rapid Model Adaptation.

This module implements Model-Agnostic Meta-Learning (MAML) for quick
adaptation to new market conditions with minimal training data.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from copy import deepcopy

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from structlog import get_logger

from quantum_trader.exceptions import ModelError, ValidationError

logger = get_logger(__name__)


class MetaNetwork(nn.Module):
    """Base neural network for meta-learning.

    Simple feedforward network that can be quickly adapted.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3
    ) -> None:
        """Initialize meta network.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of hidden layers
        """
        super(MetaNetwork, self).__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.network(x)


@dataclass
class Task:
    """Meta-learning task.

    Attributes:
        task_id: Task identifier
        support_x: Support set features
        support_y: Support set labels
        query_x: Query set features
        query_y: Query set labels
        metadata: Additional task metadata
    """
    task_id: str
    support_x: np.ndarray
    support_y: np.ndarray
    query_x: np.ndarray
    query_y: np.ndarray
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        """Initialize metadata if None."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MAMLConfig:
    """Configuration for MAML.

    Attributes:
        input_dim: Input feature dimension
        output_dim: Output dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of network layers
        meta_lr: Meta learning rate
        inner_lr: Inner loop learning rate
        num_inner_steps: Number of inner loop gradient steps
        first_order: Whether to use first-order MAML
        num_tasks_per_batch: Number of tasks per meta-batch
        device: Computation device
    """
    input_dim: int
    output_dim: int
    hidden_dim: int = 128
    num_layers: int = 3
    meta_lr: Decimal = Decimal("0.001")
    inner_lr: Decimal = Decimal("0.01")
    num_inner_steps: int = 5
    first_order: bool = False
    num_tasks_per_batch: int = 4
    device: str = "cuda"


class MAMLModel:
    """Model-Agnostic Meta-Learning for trading.

    Implements MAML algorithm for quick adaptation to new market
    conditions with few-shot learning capabilities.

    Attributes:
        config: MAML configuration
        model: Meta network
        meta_optimizer: Meta optimizer
        loss_fn: Loss function

    Example:
        >>> config = MAMLConfig(
        ...     input_dim=50,
        ...     output_dim=1,
        ...     hidden_dim=128,
        ...     num_layers=3,
        ...     meta_lr=Decimal("0.001"),
        ...     inner_lr=Decimal("0.01"),
        ...     num_inner_steps=5
        ... )
        >>> maml = MAMLModel(config)
        >>> maml.meta_train(tasks)
        >>> adapted_model = maml.adapt(new_task)
    """

    def __init__(self, config: MAMLConfig) -> None:
        """Initialize MAML model.

        Args:
            config: MAML configuration

        Raises:
            ValidationError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        # Initialize meta model
        self.model = MetaNetwork(
            config.input_dim,
            config.hidden_dim,
            config.output_dim,
            config.num_layers
        ).to(self.device)

        # Meta optimizer
        self.meta_optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(config.meta_lr)
        )

        # Loss function
        self.loss_fn = nn.MSELoss()

        self._meta_step = 0

        logger.info(
            "MAML model initialized",
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
            device=str(self.device)
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValidationError: If configuration is invalid
        """
        if self.config.input_dim < 1:
            raise ValidationError("input_dim must be >= 1")

        if self.config.output_dim < 1:
            raise ValidationError("output_dim must be >= 1")

        if self.config.hidden_dim < 1:
            raise ValidationError("hidden_dim must be >= 1")

        if self.config.num_layers < 1:
            raise ValidationError("num_layers must be >= 1")

        if self.config.meta_lr <= 0:
            raise ValidationError("meta_lr must be > 0")

        if self.config.inner_lr <= 0:
            raise ValidationError("inner_lr must be > 0")

        if self.config.num_inner_steps < 1:
            raise ValidationError("num_inner_steps must be >= 1")

    def meta_train(
        self,
        tasks: List[Task],
        num_epochs: int = 100
    ) -> Dict[str, List[Decimal]]:
        """Meta-train the model on multiple tasks.

        Args:
            tasks: List of training tasks
            num_epochs: Number of meta-training epochs

        Returns:
            Training history

        Raises:
            ModelError: If training fails
        """
        try:
            logger.info(
                "Starting meta-training",
                num_tasks=len(tasks),
                num_epochs=num_epochs
            )

            history = {
                "meta_loss": [],
                "support_loss": [],
                "query_loss": []
            }

            for epoch in range(num_epochs):
                # Sample batch of tasks
                batch_tasks = np.random.choice(
                    tasks,
                    size=min(self.config.num_tasks_per_batch, len(tasks)),
                    replace=False
                )

                # Meta-update
                meta_loss = self._meta_update(batch_tasks)

                history["meta_loss"].append(Decimal(str(meta_loss)))

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        "Meta-training progress",
                        epoch=epoch + 1,
                        meta_loss=float(meta_loss)
                    )

                self._meta_step += 1

            logger.info("Meta-training completed")

            return history

        except Exception as e:
            logger.error("Meta-training failed", error=str(e))
            raise ModelError(f"Meta-training failed: {e}") from e

    def _meta_update(self, tasks: List[Task]) -> float:
        """Perform one meta-update step.

        Args:
            tasks: Batch of tasks

        Returns:
            Meta loss value
        """
        self.meta_optimizer.zero_grad()

        meta_losses = []

        for task in tasks:
            # Clone model for inner loop
            task_model = deepcopy(self.model)

            # Inner loop: adapt to task
            adapted_params = self._inner_loop(
                task_model,
                task.support_x,
                task.support_y
            )

            # Compute query loss with adapted parameters
            query_x = torch.FloatTensor(task.query_x).to(self.device)
            query_y = torch.FloatTensor(task.query_y).to(self.device)

            # Apply adapted parameters
            self._set_parameters(task_model, adapted_params)

            # Forward pass on query set
            query_pred = task_model(query_x)
            query_loss = self.loss_fn(query_pred, query_y)

            meta_losses.append(query_loss)

        # Meta loss is average of query losses
        meta_loss = torch.stack(meta_losses).mean()

        # Meta backward
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def _inner_loop(
        self,
        model: nn.Module,
        support_x: np.ndarray,
        support_y: np.ndarray
    ) -> List[torch.Tensor]:
        """Perform inner loop adaptation.

        Args:
            model: Model to adapt
            support_x: Support set features
            support_y: Support set labels

        Returns:
            Adapted parameters
        """
        # Convert to tensors
        support_x = torch.FloatTensor(support_x).to(self.device)
        support_y = torch.FloatTensor(support_y).to(self.device)

        # Get initial parameters
        params = list(model.parameters())

        # Inner loop gradient descent
        for _ in range(self.config.num_inner_steps):
            # Forward pass
            support_pred = model(support_x)
            support_loss = self.loss_fn(support_pred, support_y)

            # Compute gradients
            grads = torch.autograd.grad(
                support_loss,
                params,
                create_graph=not self.config.first_order
            )

            # Update parameters
            params = [
                p - float(self.config.inner_lr) * g
                for p, g in zip(params, grads)
            ]

            # Set new parameters
            self._set_parameters(model, params)

        return params

    def _set_parameters(
        self,
        model: nn.Module,
        params: List[torch.Tensor]
    ) -> None:
        """Set model parameters.

        Args:
            model: Model to update
            params: New parameters
        """
        for model_param, new_param in zip(model.parameters(), params):
            model_param.data = new_param.data

    def adapt(
        self,
        task: Task,
        num_steps: Optional[int] = None
    ) -> nn.Module:
        """Adapt model to a new task.

        Args:
            task: Task to adapt to
            num_steps: Number of adaptation steps (uses config if None)

        Returns:
            Adapted model

        Raises:
            ModelError: If adaptation fails
        """
        try:
            if num_steps is None:
                num_steps = self.config.num_inner_steps

            logger.debug("Adapting model to task", task_id=task.task_id)

            # Clone model
            adapted_model = deepcopy(self.model)

            # Convert to tensors
            support_x = torch.FloatTensor(task.support_x).to(self.device)
            support_y = torch.FloatTensor(task.support_y).to(self.device)

            # Create optimizer for adapted model
            optimizer = optim.SGD(
                adapted_model.parameters(),
                lr=float(self.config.inner_lr)
            )

            # Adaptation steps
            for step in range(num_steps):
                optimizer.zero_grad()

                # Forward pass
                pred = adapted_model(support_x)
                loss = self.loss_fn(pred, support_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                if (step + 1) % 10 == 0:
                    logger.debug(
                        "Adaptation progress",
                        task_id=task.task_id,
                        step=step + 1,
                        loss=float(loss.item())
                    )

            logger.debug("Adaptation completed", task_id=task.task_id)

            return adapted_model

        except Exception as e:
            logger.error("Adaptation failed", error=str(e))
            raise ModelError(f"Adaptation failed: {e}") from e

    def predict(
        self,
        model: nn.Module,
        features: np.ndarray
    ) -> np.ndarray:
        """Generate predictions.

        Args:
            model: Model to use (base or adapted)
            features: Input features

        Returns:
            Predictions

        Raises:
            ModelError: If prediction fails
        """
        try:
            model.eval()

            with torch.no_grad():
                x = torch.FloatTensor(features).to(self.device)
                predictions = model(x).cpu().numpy()

            return predictions

        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            raise ModelError(f"Prediction failed: {e}") from e

    def evaluate(
        self,
        model: nn.Module,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            model: Model to evaluate
            features: Test features
            labels: Test labels

        Returns:
            Evaluation metrics

        Raises:
            ModelError: If evaluation fails
        """
        try:
            predictions = self.predict(model, features)

            # Calculate metrics
            mse = np.mean((predictions.flatten() - labels.flatten()) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions.flatten() - labels.flatten()))

            metrics = {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae)
            }

            logger.info("Model evaluation completed", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("Evaluation failed", error=str(e))
            raise ModelError(f"Evaluation failed: {e}") from e

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save model

        Raises:
            ModelError: If save fails
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'model_state_dict': self.model.state_dict(),
                'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
                'config': {
                    'input_dim': self.config.input_dim,
                    'output_dim': self.config.output_dim,
                    'hidden_dim': self.config.hidden_dim,
                    'num_layers': self.config.num_layers,
                    'meta_lr': float(self.config.meta_lr),
                    'inner_lr': float(self.config.inner_lr),
                    'num_inner_steps': self.config.num_inner_steps,
                    'first_order': self.config.first_order
                },
                'meta_step': self._meta_step
            }, path)

            logger.info("Model saved", path=path)

        except Exception as e:
            logger.error("Failed to save model", error=str(e))
            raise ModelError(f"Failed to save model: {e}") from e

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to load model from

        Raises:
            ModelError: If load fails
        """
        try:
            if not os.path.exists(path):
                raise ModelError(f"Model file not found: {path}")

            checkpoint = torch.load(path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
            self._meta_step = checkpoint.get('meta_step', 0)

            logger.info("Model loaded", path=path)

        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise ModelError(f"Failed to load model: {e}") from e

    async def meta_train_async(
        self,
        tasks: List[Task],
        num_epochs: int = 100
    ) -> Dict[str, List[Decimal]]:
        """Meta-train asynchronously.

        Args:
            tasks: List of training tasks
            num_epochs: Number of meta-training epochs

        Returns:
            Training history
        """
        return await asyncio.to_thread(self.meta_train, tasks, num_epochs)

    async def adapt_async(
        self,
        task: Task,
        num_steps: Optional[int] = None
    ) -> nn.Module:
        """Adapt model asynchronously.

        Args:
            task: Task to adapt to
            num_steps: Number of adaptation steps

        Returns:
            Adapted model
        """
        return await asyncio.to_thread(self.adapt, task, num_steps)

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "meta_step": self._meta_step,
            "input_dim": self.config.input_dim,
            "output_dim": self.config.output_dim,
            "hidden_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "meta_lr": float(self.config.meta_lr),
            "inner_lr": float(self.config.inner_lr),
            "num_inner_steps": self.config.num_inner_steps
        }
