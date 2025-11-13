"""GraphSAGE model for inductive learning on market graphs.

This module implements GraphSAGE for scalable graph learning, enabling
generalization to unseen assets and market structures through neighborhood sampling.
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


class SAGEConvolution(nn.Module):
    """GraphSAGE convolution layer with neighborhood sampling."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggregator: str = "mean",
        normalize: bool = True
    ) -> None:
        """Initialize SAGE convolution layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            aggregator: Aggregation function ('mean', 'max', 'lstm')
            normalize: Whether to normalize output
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        self.normalize = normalize

        # Weight matrices
        self.weight_self = nn.Linear(in_features, out_features, bias=False)
        self.weight_neigh = nn.Linear(in_features, out_features, bias=False)

        # LSTM aggregator
        if aggregator == "lstm":
            self.lstm = nn.LSTM(in_features, in_features, batch_first=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight_self.weight)
        nn.init.xavier_uniform_(self.weight_neigh.weight)

        if self.aggregator == "lstm":
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features [batch_size, num_nodes, in_features]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]

        Returns:
            Updated features [batch_size, num_nodes, out_features]
        """
        batch_size, num_nodes, _ = x.size()

        # Self features
        h_self = self.weight_self(x)

        # Aggregate neighbor features
        if self.aggregator == "mean":
            h_neigh = self._aggregate_mean(x, adj)
        elif self.aggregator == "max":
            h_neigh = self._aggregate_max(x, adj)
        elif self.aggregator == "lstm":
            h_neigh = self._aggregate_lstm(x, adj)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

        h_neigh = self.weight_neigh(h_neigh)

        # Combine self and neighbor features
        h = h_self + h_neigh

        # Normalize
        if self.normalize:
            h = F.normalize(h, p=2, dim=-1)

        return h

    def _aggregate_mean(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Mean aggregation.

        Args:
            x: Node features
            adj: Adjacency matrix

        Returns:
            Aggregated features
        """
        # Normalize adjacency by degree
        degree = adj.sum(dim=2, keepdim=True)
        degree[degree == 0] = 1  # Avoid division by zero
        adj_norm = adj / degree

        # Aggregate
        h_neigh = torch.matmul(adj_norm, x)

        return h_neigh

    def _aggregate_max(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Max pooling aggregation.

        Args:
            x: Node features
            adj: Adjacency matrix

        Returns:
            Aggregated features
        """
        batch_size, num_nodes, feature_dim = x.size()

        # Expand for broadcasting
        x_expanded = x.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        adj_expanded = adj.unsqueeze(-1).expand(-1, -1, -1, feature_dim)

        # Mask non-neighbors
        x_masked = x_expanded * adj_expanded
        x_masked[adj_expanded == 0] = float('-inf')

        # Max pooling
        h_neigh, _ = torch.max(x_masked, dim=2)

        # Handle nodes with no neighbors
        h_neigh[h_neigh == float('-inf')] = 0

        return h_neigh

    def _aggregate_lstm(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """LSTM aggregation.

        Args:
            x: Node features
            adj: Adjacency matrix

        Returns:
            Aggregated features
        """
        batch_size, num_nodes, feature_dim = x.size()

        h_neigh_list = []

        for i in range(batch_size):
            batch_output = []

            for j in range(num_nodes):
                # Get neighbors for node j
                neighbors = adj[i, j].nonzero(as_tuple=True)[0]

                if len(neighbors) == 0:
                    # No neighbors - use zeros
                    batch_output.append(torch.zeros(feature_dim).to(x.device))
                else:
                    # Get neighbor features
                    neighbor_features = x[i, neighbors, :]

                    # Randomly permute (LSTM is permutation-sensitive)
                    perm = torch.randperm(len(neighbors))
                    neighbor_features = neighbor_features[perm]

                    # Apply LSTM
                    _, (h_n, _) = self.lstm(neighbor_features.unsqueeze(0))
                    batch_output.append(h_n.squeeze(0).squeeze(0))

            h_neigh_list.append(torch.stack(batch_output))

        h_neigh = torch.stack(h_neigh_list)

        return h_neigh


class GraphSAGEModel(BaseMLModel):
    """GraphSAGE model for inductive graph learning.

    Enables learning on large-scale dynamic market graphs through
    neighborhood sampling and inductive representation learning.

    Attributes:
        config: Model configuration
        sage_layers: List of SAGE convolution layers
        device: Torch device

    Example:
        >>> config = {
        ...     "input_dim": 64,
        ...     "hidden_dims": [128, 64],
        ...     "output_dim": 32,
        ...     "aggregator": "mean",
        ...     "dropout": "0.5",
        ...     "learning_rate": "0.01"
        ... }
        >>> model = GraphSAGEModel(config)
        >>> model.train(features, labels)
        >>> predictions = model.predict(test_features)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize GraphSAGE model.

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
        self.aggregator = config.get("aggregator", "mean")
        self.dropout = float(config.get("dropout", "0.5"))

        # Training parameters
        self.learning_rate = Decimal(str(config.get("learning_rate", "0.01")))
        self.weight_decay = Decimal(str(config.get("weight_decay", "0.0005")))
        self.num_epochs = config.get("num_epochs", 100)
        self.batch_size = config.get("batch_size", 32)

        # Device setup
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Build model
        self.sage_layers = nn.ModuleList()

        # First layer
        self.sage_layers.append(
            SAGEConvolution(
                self.input_dim,
                self.hidden_dims[0],
                aggregator=self.aggregator
            )
        )

        # Hidden layers
        for i in range(len(self.hidden_dims) - 1):
            self.sage_layers.append(
                SAGEConvolution(
                    self.hidden_dims[i],
                    self.hidden_dims[i + 1],
                    aggregator=self.aggregator
                )
            )

        # Output layer
        self.sage_layers.append(
            SAGEConvolution(
                self.hidden_dims[-1],
                self.output_dim,
                aggregator=self.aggregator,
                normalize=False
            )
        )

        self.sage_layers.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.sage_layers.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay)
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Training state
        self.training_history: List[Dict[str, float]] = []

        logger.info(
            "GraphSAGE model initialized",
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            aggregator=self.aggregator,
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

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass through GraphSAGE.

        Args:
            x: Node features [batch_size, num_nodes, input_dim]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]

        Returns:
            Node embeddings [batch_size, num_nodes, output_dim]
        """
        h = x

        for i, sage_layer in enumerate(self.sage_layers[:-1]):
            h = sage_layer(h, adj)
            h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)

        # Output layer
        h = self.sage_layers[-1](h, adj)

        return h

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train GraphSAGE model.

        Args:
            features: Combined [node_features, adjacency_matrix]
            labels: Target embeddings

        Raises:
            ValueError: If input shape is invalid
        """
        try:
            num_nodes = labels.shape[1]
            node_features = features[:, :, :self.input_dim]
            adj_matrices = features[:, :, self.input_dim:self.input_dim + num_nodes]

            logger.info(
                "Starting GraphSAGE training",
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
                self.sage_layers.train()
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

                if epoch % 10 == 0:
                    logger.info("Training progress", epoch=epoch, loss=avg_loss)

            logger.info("GraphSAGE training completed")

        except Exception as e:
            logger.error("Failed to train GraphSAGE", error=str(e))
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions.

        Args:
            features: Input features

        Returns:
            Predicted embeddings
        """
        try:
            self.sage_layers.eval()

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

            metrics = {
                "mse": mse,
                "mae": mae,
                "rmse": rmse
            }

            logger.info("GraphSAGE evaluation completed", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("Failed to evaluate GraphSAGE", error=str(e))
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
                "model_state": self.sage_layers.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config,
                "training_history": self.training_history
            }, save_path)

            logger.info("GraphSAGE model saved", path=path)

        except Exception as e:
            logger.error("Failed to save GraphSAGE model", error=str(e))
            raise

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: File path to load model from
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)

            self.sage_layers.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.training_history = checkpoint["training_history"]

            logger.info("GraphSAGE model loaded", path=path)

        except Exception as e:
            logger.error("Failed to load GraphSAGE model", error=str(e))
            raise
