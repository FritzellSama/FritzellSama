"""Graph Convolutional Network for market relationship modeling.

This module implements GCN for learning market structure and asset relationships
through graph-structured convolutions.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel

logger = get_logger(__name__)


class GraphConvolutionLayer(nn.Module):
    """Single graph convolution layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """Initialize GCN layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to use bias
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features [batch_size, num_nodes, in_features]
            adj: Normalized adjacency [batch_size, num_nodes, num_nodes]

        Returns:
            Updated features [batch_size, num_nodes, out_features]
        """
        # Linear transformation
        support = torch.matmul(x, self.weight)

        # Graph convolution
        output = torch.matmul(adj, support)

        if self.bias is not None:
            output = output + self.bias

        return output


class GCNModel(BaseMLModel):
    """Graph Convolutional Network for market modeling.

    Models market structure through graph convolutions, capturing
    correlations and information flow between assets.

    Attributes:
        config: Model configuration
        gc_layers: List of graph convolution layers
        device: Torch device

    Example:
        >>> config = {
        ...     "input_dim": 64,
        ...     "hidden_dims": [128, 64],
        ...     "output_dim": 32,
        ...     "dropout": "0.5",
        ...     "learning_rate": "0.01"
        ... }
        >>> model = GCNModel(config)
        >>> model.train(features, labels)
        >>> predictions = model.predict(test_features)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize GCN model.

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
        self.dropout = float(config.get("dropout", "0.5"))

        # Training parameters
        self.learning_rate = Decimal(str(config.get("learning_rate", "0.01")))
        self.weight_decay = Decimal(str(config.get("weight_decay", "0.0005")))
        self.num_epochs = config.get("num_epochs", 200)
        self.batch_size = config.get("batch_size", 32)

        # Device setup
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Build model
        self.gc_layers = nn.ModuleList()

        # First layer
        self.gc_layers.append(
            GraphConvolutionLayer(self.input_dim, self.hidden_dims[0])
        )

        # Hidden layers
        for i in range(len(self.hidden_dims) - 1):
            self.gc_layers.append(
                GraphConvolutionLayer(self.hidden_dims[i], self.hidden_dims[i + 1])
            )

        # Output layer
        self.gc_layers.append(
            GraphConvolutionLayer(self.hidden_dims[-1], self.output_dim)
        )

        self.gc_layers.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.gc_layers.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay)
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Training state
        self.training_history: List[Dict[str, float]] = []

        logger.info(
            "GCN model initialized",
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            device=str(self.device)
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If required config missing
        """
        required_keys = ["input_dim", "hidden_dims", "output_dim"]

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if not isinstance(self.config["hidden_dims"], list):
            raise ValueError("hidden_dims must be a list")

    def _normalize_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        """Normalize adjacency matrix.

        Args:
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]

        Returns:
            Normalized adjacency with self-loops
        """
        # Add self-loops
        batch_size, num_nodes, _ = adj.size()
        identity = torch.eye(num_nodes).unsqueeze(0).expand(batch_size, -1, -1).to(adj.device)
        adj_with_loops = adj + identity

        # Compute degree matrix
        degree = adj_with_loops.sum(dim=2, keepdim=True)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0

        # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        norm_adj = degree_inv_sqrt * adj_with_loops * degree_inv_sqrt.transpose(1, 2)

        return norm_adj

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass through GCN.

        Args:
            x: Node features [batch_size, num_nodes, input_dim]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]

        Returns:
            Node embeddings [batch_size, num_nodes, output_dim]
        """
        # Normalize adjacency
        adj_norm = self._normalize_adjacency(adj)

        # Apply GCN layers
        h = x
        for i, gc_layer in enumerate(self.gc_layers[:-1]):
            h = gc_layer(h, adj_norm)
            h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)

        # Output layer (no activation)
        h = self.gc_layers[-1](h, adj_norm)

        return h

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train GCN model.

        Args:
            features: Combined [node_features, adjacency_matrix]
                Shape: [num_samples, num_nodes, input_dim + num_nodes]
            labels: Target embeddings [num_samples, num_nodes, output_dim]

        Raises:
            ValueError: If input shape is invalid
        """
        try:
            num_nodes = labels.shape[1]
            node_features = features[:, :, :self.input_dim]
            adj_matrices = features[:, :, self.input_dim:self.input_dim + num_nodes]

            logger.info(
                "Starting GCN training",
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
                self.gc_layers.train()
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
                self.training_history.append({"epoch": epoch, "loss": avg_loss})

                if epoch % 20 == 0:
                    logger.info("Training progress", epoch=epoch, loss=avg_loss)

            logger.info("GCN training completed")

        except Exception as e:
            logger.error("Failed to train GCN", error=str(e))
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions.

        Args:
            features: Input features

        Returns:
            Predicted embeddings
        """
        try:
            self.gc_layers.eval()

            node_features = features[:, :, :self.input_dim]
            adj_matrices = features[:, :, self.input_dim:]

            X = torch.FloatTensor(node_features).to(self.device)
            A = torch.FloatTensor(adj_matrices).to(self.device)

            with torch.no_grad():
                predictions = self.forward(X, A).cpu().numpy()

            logger.info("Generated predictions", shape=predictions.shape)

            return predictions

        except Exception as e:
            logger.error("Failed to generate predictions", error=str(e))
            raise

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            features: Input features
            labels: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            predictions = self.predict(features)

            mse = float(np.mean((predictions - labels) ** 2))
            mae = float(np.mean(np.abs(predictions - labels)))
            rmse = float(np.sqrt(mse))

            # Correlation
            pred_flat = predictions.reshape(-1)
            label_flat = labels.reshape(-1)
            correlation = float(np.corrcoef(pred_flat, label_flat)[0, 1])

            metrics = {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "correlation": correlation
            }

            logger.info("GCN evaluation completed", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("Failed to evaluate GCN", error=str(e))
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
                "model_state": self.gc_layers.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config,
                "training_history": self.training_history
            }, save_path)

            logger.info("GCN model saved", path=path)

        except Exception as e:
            logger.error("Failed to save GCN model", error=str(e))
            raise

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: File path to load model from
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)

            self.gc_layers.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.training_history = checkpoint["training_history"]

            logger.info("GCN model loaded", path=path)

        except Exception as e:
            logger.error("Failed to load GCN model", error=str(e))
            raise
