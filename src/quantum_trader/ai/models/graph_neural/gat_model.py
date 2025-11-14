"""Graph Attention Network for market relationship modeling.

This module implements GAT for learning attention-based relationships between
assets, exchanges, and market participants in the trading ecosystem.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel

logger = get_logger(__name__)


class GraphAttentionLayer(nn.Module):
    """Single graph attention layer.

    Implements multi-head attention mechanism for graph-structured data.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 8,
        dropout: float = 0.3,
        alpha: float = 0.2
    ) -> None:
        """Initialize GAT layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            alpha: LeakyReLU negative slope
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha

        # Learnable parameters
        self.W = nn.Parameter(
            torch.zeros(size=(num_heads, in_features, out_features))
        )
        self.a = nn.Parameter(
            torch.zeros(size=(num_heads, 2 * out_features, 1))
        )

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features [batch_size, num_nodes, in_features]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]

        Returns:
            Updated node features [batch_size, num_nodes, num_heads * out_features]
        """
        batch_size, num_nodes, _ = x.size()

        # Apply linear transformation for each head
        # [batch_size, num_heads, num_nodes, out_features]
        h = torch.matmul(x.unsqueeze(1), self.W)

        # Compute attention scores
        # [batch_size, num_heads, num_nodes, num_nodes]
        attention = self._compute_attention(h, adj)

        # Apply attention to features
        # [batch_size, num_heads, num_nodes, out_features]
        h_prime = torch.matmul(attention, h)

        # Concatenate heads
        # [batch_size, num_nodes, num_heads * out_features]
        h_prime = h_prime.permute(0, 2, 1, 3).reshape(
            batch_size, num_nodes, self.num_heads * self.out_features
        )

        return h_prime

    def _compute_attention(
        self,
        h: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention coefficients.

        Args:
            h: Transformed features
            adj: Adjacency matrix

        Returns:
            Attention matrix
        """
        batch_size, num_heads, num_nodes, out_features = h.size()

        # Prepare for attention computation
        # [batch_size, num_heads, num_nodes, 1, out_features]
        h_i = h.unsqueeze(3)
        # [batch_size, num_heads, 1, num_nodes, out_features]
        h_j = h.unsqueeze(2)

        # Concatenate
        # [batch_size, num_heads, num_nodes, num_nodes, 2 * out_features]
        h_cat = torch.cat([
            h_i.expand(-1, -1, -1, num_nodes, -1),
            h_j.expand(-1, -1, num_nodes, -1, -1)
        ], dim=-1)

        # Compute attention logits
        # [batch_size, num_heads, num_nodes, num_nodes]
        e = self.leakyrelu(torch.matmul(h_cat, self.a).squeeze(-1))

        # Mask attention for non-connected nodes
        mask = (adj.unsqueeze(1) == 0).expand(-1, num_heads, -1, -1)
        e = e.masked_fill(mask, float('-inf'))

        # Apply softmax
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        return attention


class GATModel(BaseMLModel):
    """Graph Attention Network for market modeling.

    Models relationships between assets, detecting correlation patterns,
    lead-lag relationships, and market contagion effects.

    Attributes:
        config: Model configuration
        layers: List of GAT layers
        device: Torch device

    Example:
        >>> config = {
        ...     "input_dim": 64,
        ...     "hidden_dims": [128, 128, 64],
        ...     "output_dim": 32,
        ...     "num_heads": 8,
        ...     "dropout": "0.3",
        ...     "learning_rate": "0.001"
        ... }
        >>> model = GATModel(config)
        >>> model.train(node_features, adjacency_matrix)
        >>> embeddings = model.predict(test_features, test_adj)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize GAT model.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If config is invalid
        """
        super().__init__(config)

        self._validate_config()

        # Model parameters
        self.input_dim = config["input_dim"]
        self.hidden_dims = config["hidden_dims"]
        self.output_dim = config["output_dim"]
        self.num_heads = config.get("num_heads", 8)
        self.dropout = float(config.get("dropout", "0.3"))
        self.alpha = float(config.get("alpha", "0.2"))

        # Training parameters
        self.learning_rate = Decimal(str(config.get("learning_rate", "0.001")))
        self.weight_decay = Decimal(str(config.get("weight_decay", "0.0001")))
        self.num_epochs = config.get("num_epochs", 100)
        self.batch_size = config.get("batch_size", 32)

        # Device setup
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Build model
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(
            GraphAttentionLayer(
                self.input_dim,
                self.hidden_dims[0],
                self.num_heads,
                self.dropout,
                self.alpha
            )
        )

        # Hidden layers
        for i in range(len(self.hidden_dims) - 1):
            self.layers.append(
                GraphAttentionLayer(
                    self.hidden_dims[i] * self.num_heads,
                    self.hidden_dims[i + 1],
                    self.num_heads,
                    self.dropout,
                    self.alpha
                )
            )

        # Output layer (single head)
        self.output_layer = GraphAttentionLayer(
            self.hidden_dims[-1] * self.num_heads,
            self.output_dim,
            num_heads=1,
            dropout=0.0,
            alpha=self.alpha
        )

        self.model = nn.ModuleList([*self.layers, self.output_layer])
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay)
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Training state
        self.training_history: List[Dict[str, float]] = []

        logger.info(
            "GAT model initialized",
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            num_heads=self.num_heads,
            device=str(self.device)
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If required config missing
        """
        required_keys = [
            "input_dim",
            "hidden_dims",
            "output_dim"
        ]

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if not isinstance(self.config["hidden_dims"], list):
            raise ValueError("hidden_dims must be a list")

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through GAT.

        Args:
            x: Node features [batch_size, num_nodes, input_dim]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]

        Returns:
            Node embeddings [batch_size, num_nodes, output_dim]
        """
        # Apply GAT layers
        h = x
        for layer in self.layers:
            h = layer(h, adj)
            h = F.elu(h)
            h = F.dropout(h, self.dropout, training=self.training)

        # Output layer
        h = self.output_layer(h, adj)

        return h

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> None:
        """Train GAT model.

        Args:
            features: Combined [node_features, adjacency_matrix]
                Shape: [num_samples, num_nodes, input_dim + num_nodes]
            labels: Target node embeddings [num_samples, num_nodes, output_dim]

        Raises:
            ValueError: If input shape is invalid
        """
        try:
            # Split features into node features and adjacency
            num_nodes = labels.shape[1]
            node_features = features[:, :, :self.input_dim]
            adj_matrices = features[:, :, self.input_dim:self.input_dim + num_nodes]

            if adj_matrices.shape[2] != num_nodes:
                raise ValueError(
                    f"Expected {num_nodes} adjacency columns, "
                    f"got {adj_matrices.shape[2]}"
                )

            logger.info(
                "Starting GAT training",
                num_samples=len(features),
                num_nodes=num_nodes,
                epochs=self.num_epochs
            )

            # Convert to tensors
            X = torch.FloatTensor(node_features).to(self.device)
            A = torch.FloatTensor(adj_matrices).to(self.device)
            y = torch.FloatTensor(labels).to(self.device)

            # Training loop
            for epoch in range(self.num_epochs):
                self.model.train()
                epoch_loss = Decimal("0")
                num_batches = 0

                for i in range(0, len(X), self.batch_size):
                    batch_x = X[i:i + self.batch_size]
                    batch_a = A[i:i + self.batch_size]
                    batch_y = y[i:i + self.batch_size]

                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.forward(batch_x, batch_a)

                    # Compute loss
                    loss = self.criterion(outputs, batch_y)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += Decimal(str(loss.item()))
                    num_batches += 1

                avg_loss = float(epoch_loss / num_batches)
                self.training_history.append({
                    "epoch": epoch,
                    "loss": avg_loss
                })

                if epoch % 10 == 0:
                    logger.info(
                        "Training progress",
                        epoch=epoch,
                        loss=avg_loss
                    )

            logger.info("GAT training completed")

        except Exception as e:
            logger.error("Failed to train GAT", error=str(e))
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate node embeddings.

        Args:
            features: Combined [node_features, adjacency_matrix]

        Returns:
            Node embeddings
        """
        try:
            self.model.eval()

            # Split features
            node_features = features[:, :, :self.input_dim]
            adj_matrices = features[:, :, self.input_dim:]

            X = torch.FloatTensor(node_features).to(self.device)
            A = torch.FloatTensor(adj_matrices).to(self.device)

            with torch.no_grad():
                embeddings = self.forward(X, A)
                predictions = embeddings.cpu().numpy()

            logger.info("Generated predictions", shape=predictions.shape)

            return predictions

        except Exception as e:
            logger.error("Failed to generate predictions", error=str(e))
            raise

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            features: Input features
            labels: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            predictions = self.predict(features)

            # Calculate metrics
            mse = float(np.mean((predictions - labels) ** 2))
            mae = float(np.mean(np.abs(predictions - labels)))
            rmse = float(np.sqrt(mse))

            # Per-node metrics
            node_mse = np.mean((predictions - labels) ** 2, axis=(0, 2))

            metrics = {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "max_node_mse": float(node_mse.max()),
                "min_node_mse": float(node_mse.min())
            }

            logger.info("GAT evaluation completed", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("Failed to evaluate GAT", error=str(e))
            raise

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path to save model
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                "model_state": [layer.state_dict() for layer in self.model],
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config,
                "training_history": self.training_history
            }, save_path)

            logger.info("GAT model saved", path=path)

        except Exception as e:
            logger.error("Failed to save GAT model", error=str(e))
            raise

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: File path to load model from
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)

            for layer, state in zip(self.model, checkpoint["model_state"]):
                layer.load_state_dict(state)

            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.training_history = checkpoint["training_history"]

            logger.info("GAT model loaded", path=path)

        except Exception as e:
            logger.error("Failed to load GAT model", error=str(e))
            raise
