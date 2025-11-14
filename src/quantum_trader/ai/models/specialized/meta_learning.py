"""Meta-learning models for rapid adaptation to new market conditions.

This module implements Model-Agnostic Meta-Learning (MAML) and related algorithms
for few-shot learning in trading, enabling quick adaptation to regime changes.
"""

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel

logger = get_logger(__name__)


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning models.

    Attributes:
        inner_lr: Learning rate for inner loop adaptation
        outer_lr: Learning rate for meta-optimizer
        n_inner_steps: Number of gradient steps in inner loop
        meta_batch_size: Number of tasks per meta-batch
        hidden_dims: Hidden layer dimensions
        dropout: Dropout rate
        task_embedding_dim: Dimension of task embeddings
        support_size: Number of samples in support set
        query_size: Number of samples in query set
        adapt_steps: Steps for adaptation at test time
        checkpoint_dir: Directory for checkpoints
    """

    inner_lr: Decimal
    outer_lr: Decimal
    n_inner_steps: int
    meta_batch_size: int
    hidden_dims: List[int]
    dropout: Decimal
    task_embedding_dim: int
    support_size: int
    query_size: int
    adapt_steps: int
    checkpoint_dir: str


class TaskEncoder(nn.Module):
    """Encoder for task embeddings.

    Learns to encode task-specific information for rapid adaptation.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dims: List[int]
    ) -> None:
        """Initialize task encoder.

        Args:
            input_dim: Input feature dimension
            embedding_dim: Task embedding dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, embedding_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode task features.

        Args:
            x: Input features

        Returns:
            Task embedding
        """
        return self.encoder(x)


class PredictionNetwork(nn.Module):
    """Base prediction network for meta-learning.

    Neural network that can be quickly adapted to new tasks.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1
    ) -> None:
        """Initialize prediction network.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features

        Returns:
            Predictions
        """
        return self.network(x)


class MAMLModel(BaseMLModel):
    """Model-Agnostic Meta-Learning (MAML) for trading.

    Implements MAML algorithm for few-shot learning, enabling rapid adaptation
    to new market regimes with minimal data.

    Example:
        >>> config = MetaLearningConfig(
        ...     inner_lr=Decimal("0.01"),
        ...     outer_lr=Decimal("0.001"),
        ...     n_inner_steps=5,
        ...     meta_batch_size=16,
        ...     hidden_dims=[128, 64],
        ...     dropout=Decimal("0.1"),
        ...     task_embedding_dim=32,
        ...     support_size=10,
        ...     query_size=20,
        ...     adapt_steps=10,
        ...     checkpoint_dir="./checkpoints/maml"
        ... )
        >>> model = MAMLModel(
        ...     input_dim=50,
        ...     output_dim=1,
        ...     config=config
        ... )
        >>> model.meta_train(tasks)
        >>> adapted_predictions = model.adapt_and_predict(support_set, query_set)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: MetaLearningConfig,
        device: Optional[str] = None
    ) -> None:
        """Initialize MAML model.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (1 for regression)
            config: Meta-learning configuration
            device: Device for computation
        """
        super().__init__(config.__dict__ if hasattr(config, '__dict__') else config)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Base model
        self.model = PredictionNetwork(
            input_dim,
            output_dim,
            config.hidden_dims,
            float(config.dropout)
        ).to(self.device)

        # Task encoder
        self.task_encoder = TaskEncoder(
            input_dim,
            config.task_embedding_dim,
            [64, 32]
        ).to(self.device)

        # Meta-optimizer
        self.meta_optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.task_encoder.parameters()),
            lr=float(config.outer_lr)
        )

        # Training state
        self.meta_iteration = 0
        self.adaptation_cache: Dict[str, Any] = {}

        # Metrics
        self.meta_train_losses: List[float] = []
        self.meta_val_losses: List[float] = []
        self.adaptation_metrics: List[Dict[str, float]] = []

        logger.info(
            "maml_initialized",
            input_dim=input_dim,
            output_dim=output_dim,
            device=self.device
        )

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train is replaced by meta_train for MAML.

        Args:
            features: Not used (meta-learning uses tasks)
            labels: Not used (meta-learning uses tasks)
        """
        logger.warning(
            "standard_train_called",
            message="Use meta_train() for MAML training"
        )

    def meta_train_step(
        self,
        tasks: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]
    ) -> float:
        """Perform one meta-training step.

        Args:
            tasks: List of tasks, each containing support and query sets

        Returns:
            Meta-training loss
        """
        try:
            self.model.train()
            self.task_encoder.train()

            meta_loss = 0.0
            batch_size = min(self.config.meta_batch_size, len(tasks))

            # Sample batch of tasks
            task_indices = np.random.choice(len(tasks), batch_size, replace=False)

            for task_idx in task_indices:
                task = tasks[task_idx]

                support_x, support_y = task['support']
                query_x, query_y = task['query']

                support_x = support_x.to(self.device)
                support_y = support_y.to(self.device)
                query_x = query_x.to(self.device)
                query_y = query_y.to(self.device)

                # Clone model for inner loop
                adapted_params = {
                    name: param.clone()
                    for name, param in self.model.named_parameters()
                }

                # Inner loop: adapt to task
                for _ in range(self.config.n_inner_steps):
                    # Forward pass with adapted parameters
                    support_pred = self._forward_with_params(
                        support_x,
                        adapted_params
                    )

                    # Inner loss
                    inner_loss = F.mse_loss(support_pred, support_y)

                    # Compute gradients
                    grads = torch.autograd.grad(
                        inner_loss,
                        adapted_params.values(),
                        create_graph=True
                    )

                    # Update adapted parameters
                    adapted_params = {
                        name: param - float(self.config.inner_lr) * grad
                        for (name, param), grad in zip(
                            adapted_params.items(),
                            grads
                        )
                    }

                # Outer loop: evaluate on query set
                query_pred = self._forward_with_params(query_x, adapted_params)
                outer_loss = F.mse_loss(query_pred, query_y)

                meta_loss += outer_loss

            # Average meta-loss
            meta_loss = meta_loss / batch_size

            # Meta-optimization step
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.task_encoder.parameters()),
                1.0
            )
            self.meta_optimizer.step()

            self.meta_iteration += 1
            self.meta_train_losses.append(float(meta_loss.item()))

            return float(meta_loss.item())

        except Exception as e:
            logger.error("meta_train_step_failed", error=str(e))
            return 0.0

    def _forward_with_params(
        self,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass with custom parameters.

        Args:
            x: Input tensor
            params: Model parameters

        Returns:
            Model output
        """
        # Manual forward pass through network layers
        # This is needed for meta-learning to compute higher-order gradients

        param_list = list(params.values())
        param_idx = 0

        out = x

        for layer in self.model.network:
            if isinstance(layer, nn.Linear):
                weight = param_list[param_idx]
                bias = param_list[param_idx + 1]
                out = F.linear(out, weight, bias)
                param_idx += 2
            elif isinstance(layer, nn.LayerNorm):
                out = F.layer_norm(
                    out,
                    layer.normalized_shape,
                    layer.weight,
                    layer.bias
                )
            elif isinstance(layer, nn.ReLU):
                out = F.relu(out)
            elif isinstance(layer, nn.Dropout):
                out = F.dropout(out, p=layer.p, training=self.model.training)

        return out

    def adapt(
        self,
        support_x: np.ndarray,
        support_y: np.ndarray,
        n_steps: Optional[int] = None
    ) -> None:
        """Adapt model to new task using support set.

        Args:
            support_x: Support set features
            support_y: Support set labels
            n_steps: Number of adaptation steps (uses config default if None)
        """
        try:
            self.model.train()

            n_steps = n_steps or self.config.adapt_steps

            support_x_tensor = torch.FloatTensor(support_x).to(self.device)
            support_y_tensor = torch.FloatTensor(support_y).to(self.device)

            # Create temporary optimizer for adaptation
            adapt_optimizer = optim.SGD(
                self.model.parameters(),
                lr=float(self.config.inner_lr)
            )

            losses = []

            for step in range(n_steps):
                adapt_optimizer.zero_grad()

                predictions = self.model(support_x_tensor)
                loss = F.mse_loss(predictions, support_y_tensor)

                loss.backward()
                adapt_optimizer.step()

                losses.append(float(loss.item()))

            # Cache adaptation metrics
            self.adaptation_cache = {
                'final_loss': losses[-1],
                'losses': losses,
                'n_steps': n_steps
            }

            logger.info(
                "adaptation_completed",
                n_steps=n_steps,
                final_loss=losses[-1]
            )

        except Exception as e:
            logger.error("adaptation_failed", error=str(e))
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions.

        Args:
            features: Input features

        Returns:
            Predictions
        """
        try:
            self.model.eval()

            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).to(self.device)
                predictions = self.model(features_tensor)

            return predictions.cpu().numpy()

        except Exception as e:
            logger.error("prediction_failed", error=str(e))
            return np.zeros((len(features), self.output_dim))

    def adapt_and_predict(
        self,
        support_x: np.ndarray,
        support_y: np.ndarray,
        query_x: np.ndarray
    ) -> np.ndarray:
        """Adapt to new task and make predictions.

        Args:
            support_x: Support set features
            support_y: Support set labels
            query_x: Query features for prediction

        Returns:
            Predictions on query set
        """
        try:
            # Save original model state
            original_state = deepcopy(self.model.state_dict())

            # Adapt
            self.adapt(support_x, support_y)

            # Predict
            predictions = self.predict(query_x)

            # Restore original model
            self.model.load_state_dict(original_state)

            return predictions

        except Exception as e:
            logger.error("adapt_and_predict_failed", error=str(e))
            return np.zeros((len(query_x), self.output_dim))

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            features: Test features
            labels: Test labels

        Returns:
            Evaluation metrics
        """
        try:
            predictions = self.predict(features)

            mse = float(np.mean((predictions - labels) ** 2))
            mae = float(np.mean(np.abs(predictions - labels)))
            rmse = float(np.sqrt(mse))

            # Calculate R²
            ss_res = np.sum((labels - predictions) ** 2)
            ss_tot = np.sum((labels - np.mean(labels)) ** 2)
            r2 = float(1 - ss_res / (ss_tot + 1e-10))

            return {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }

        except Exception as e:
            logger.error("evaluation_failed", error=str(e))
            return {}

    def create_task_from_dataframe(
        self,
        data: pl.DataFrame,
        feature_cols: List[str],
        target_col: str
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Create task from Polars DataFrame.

        Args:
            data: Market data
            feature_cols: Feature column names
            target_col: Target column name

        Returns:
            Task dictionary with support and query sets
        """
        try:
            # Extract features and targets
            features = data.select(feature_cols).to_numpy()
            targets = data.select(target_col).to_numpy()

            # Split into support and query
            n_samples = len(features)
            support_size = min(self.config.support_size, n_samples // 2)
            query_size = min(self.config.query_size, n_samples - support_size)

            indices = np.random.permutation(n_samples)
            support_indices = indices[:support_size]
            query_indices = indices[support_size:support_size + query_size]

            support_x = torch.FloatTensor(features[support_indices])
            support_y = torch.FloatTensor(targets[support_indices])
            query_x = torch.FloatTensor(features[query_indices])
            query_y = torch.FloatTensor(targets[query_indices])

            return {
                'support': (support_x, support_y),
                'query': (query_x, query_y)
            }

        except Exception as e:
            logger.error("task_creation_failed", error=str(e))
            raise

    async def save_checkpoint(self, episode: int) -> None:
        """Save model checkpoint asynchronously.

        Args:
            episode: Current episode/iteration number
        """
        try:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / f"maml_ep{episode}.pt"

            checkpoint = {
                'episode': episode,
                'meta_iteration': self.meta_iteration,
                'model_state_dict': self.model.state_dict(),
                'task_encoder_state_dict': self.task_encoder.state_dict(),
                'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
                'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config,
                'meta_train_losses': self.meta_train_losses,
                'meta_val_losses': self.meta_val_losses
            }

            await asyncio.to_thread(
                torch.save,
                checkpoint,
                checkpoint_path
            )

            logger.info(
                "checkpoint_saved",
                episode=episode,
                path=str(checkpoint_path)
            )

        except Exception as e:
            logger.error("checkpoint_save_failed", error=str(e))

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Save path
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                'meta_iteration': self.meta_iteration,
                'model_state_dict': self.model.state_dict(),
                'task_encoder_state_dict': self.task_encoder.state_dict(),
                'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
                'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config,
                'meta_train_losses': self.meta_train_losses,
                'meta_val_losses': self.meta_val_losses,
                'input_dim': self.input_dim,
                'output_dim': self.output_dim
            }

            torch.save(checkpoint, save_path)

            logger.info("model_saved", path=path)

        except Exception as e:
            logger.error("save_failed", error=str(e))
            raise

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Load path
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)

            self.meta_iteration = checkpoint['meta_iteration']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.task_encoder.load_state_dict(checkpoint['task_encoder_state_dict'])
            self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
            self.meta_train_losses = checkpoint['meta_train_losses']
            self.meta_val_losses = checkpoint['meta_val_losses']

            logger.info("model_loaded", path=path)

        except Exception as e:
            logger.error("load_failed", error=str(e))
            raise

    def get_metrics_dataframe(self) -> pl.DataFrame:
        """Get training metrics as Polars DataFrame.

        Returns:
            DataFrame with training metrics
        """
        try:
            data = {
                'iteration': list(range(len(self.meta_train_losses))),
                'train_loss': self.meta_train_losses,
                'val_loss': self.meta_val_losses[:len(self.meta_train_losses)]
                if len(self.meta_val_losses) > 0
                else [0.0] * len(self.meta_train_losses)
            }

            return pl.DataFrame(data)

        except Exception as e:
            logger.error("metrics_dataframe_creation_failed", error=str(e))
            return pl.DataFrame()

    def compute_task_similarity(
        self,
        task1: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        task2: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Decimal:
        """Compute similarity between two tasks.

        Args:
            task1: First task
            task2: Second task

        Returns:
            Similarity score (0 to 1)
        """
        try:
            # Encode tasks
            support1_x, _ = task1['support']
            support2_x, _ = task2['support']

            support1_x = support1_x.to(self.device)
            support2_x = support2_x.to(self.device)

            with torch.no_grad():
                embedding1 = self.task_encoder(support1_x).mean(dim=0)
                embedding2 = self.task_encoder(support2_x).mean(dim=0)

                # Cosine similarity
                similarity = F.cosine_similarity(
                    embedding1.unsqueeze(0),
                    embedding2.unsqueeze(0)
                )

            return Decimal(str(float(similarity.item())))

        except Exception as e:
            logger.error("task_similarity_failed", error=str(e))
            return Decimal("0")
