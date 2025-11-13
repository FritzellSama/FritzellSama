"""Temporal Fusion Transformer for multi-horizon time series forecasting.

This module implements a simplified Temporal Fusion Transformer (TFT) architecture
for probabilistic time series forecasting with interpretable attention mechanisms.
"""

from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel

logger = get_logger(__name__)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for feature processing.

    Implements a gated residual network with skip connections
    for flexible feature transformation.

    Attributes:
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        """Initialize GRN.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            dropout: Dropout probability
        """
        super(GatedResidualNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.layernorm = nn.LayerNorm(output_dim)

        # Skip connection
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through GRN.

        Args:
            x: Input tensor

        Returns:
            Output tensor with gated residual connection
        """
        # Feed-forward
        h = self.fc1(x)
        h = torch.relu(h)
        h = self.fc2(h)
        h = self.dropout(h)

        # Gating
        gate = torch.sigmoid(self.gate(h))
        h = self.fc_out(h)
        h = gate * h

        # Skip connection
        if self.skip is not None:
            x = self.skip(x)

        return self.layernorm(x + h)


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer architecture.

    Simplified TFT implementation with variable selection networks,
    LSTM encoder-decoder, and multi-head attention.

    Attributes:
        hidden_dim: Hidden state dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_heads: int,
        num_outputs: int,
        dropout: float = 0.1
    ):
        """Initialize Temporal Fusion Transformer.

        Args:
            num_features: Number of input features
            hidden_dim: Hidden state dimension
            num_heads: Number of attention heads
            num_outputs: Number of output predictions
            dropout: Dropout probability
        """
        super(TemporalFusionTransformer, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Variable selection network
        self.variable_selection = GatedResidualNetwork(
            num_features, hidden_dim, hidden_dim, dropout
        )

        # LSTM encoder
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Temporal fusion
        self.fusion = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_outputs)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through TFT.

        Args:
            x: Input tensor (batch, sequence, features)
            return_attention: Whether to return attention weights

        Returns:
            Output predictions or (predictions, attention_weights)
        """
        # Variable selection
        h = self.variable_selection(x)

        # LSTM encoding
        h, _ = self.lstm(h)

        # Self-attention
        h_attended, attention_weights = self.attention(h, h, h)

        # Temporal fusion
        h = self.fusion(h_attended)

        # Output (use last time step)
        output = self.output_layer(h[:, -1, :])

        if return_attention:
            return output, attention_weights

        return output


class TemporalFusionModel(BaseMLModel):
    """Temporal Fusion Transformer model for trading predictions.

    Implements TFT for multi-horizon probabilistic forecasting with
    interpretable attention mechanisms and uncertainty quantification.

    Attributes:
        config: Model configuration
        model: TFT neural network
        optimizer: Training optimizer
        criterion: Loss function
        device: Computation device
        is_trained: Training status

    Examples:
        >>> config = {
        ...     "num_features": 20,
        ...     "hidden_dim": 128,
        ...     "num_heads": 4,
        ...     "num_outputs": 1,
        ...     "learning_rate": 0.001
        ... }
        >>> tft = TemporalFusionModel(config)
        >>> tft.train(features, labels)
        >>> predictions = tft.predict(test_features)
    """

    def __init__(self, config: Dict) -> None:
        """Initialize Temporal Fusion model.

        Args:
            config: Configuration with architecture and training params

        Raises:
            ValueError: If required config parameters missing
        """
        super().__init__(config)
        self._validate_config()

        # Architecture parameters
        self.num_features: int = int(config.get("num_features", os.getenv("TFT_NUM_FEATURES", "20")))
        self.hidden_dim: int = int(config.get("hidden_dim", os.getenv("TFT_HIDDEN_DIM", "128")))
        self.num_heads: int = int(config.get("num_heads", os.getenv("TFT_NUM_HEADS", "4")))
        self.num_outputs: int = int(config.get("num_outputs", os.getenv("TFT_NUM_OUTPUTS", "1")))
        self.dropout: float = float(config.get("dropout", os.getenv("TFT_DROPOUT", "0.1")))

        # Training parameters
        self.learning_rate: float = float(config.get("learning_rate", os.getenv("TFT_LEARNING_RATE", "0.001")))
        self.batch_size: int = int(config.get("batch_size", os.getenv("TFT_BATCH_SIZE", "64")))
        self.epochs: int = int(config.get("epochs", os.getenv("TFT_EPOCHS", "100")))
        self.patience: int = int(config.get("patience", os.getenv("TFT_PATIENCE", "15")))

        # Device configuration
        self.device = torch.device(
            config.get("device", os.getenv("TFT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        )

        # Initialize model components
        self.model: Optional[TemporalFusionTransformer] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion: nn.Module = nn.MSELoss()
        self.is_trained: bool = False

        self._initialize_model()

        logger.info(
            "Temporal Fusion Transformer initialized",
            num_features=self.num_features,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            device=str(self.device)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config validation fails
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        if "num_heads" in self.config:
            if self.config["num_heads"] < 1:
                raise ValueError("num_heads must be at least 1")

        if "hidden_dim" in self.config and "num_heads" in self.config:
            if self.config["hidden_dim"] % self.config["num_heads"] != 0:
                raise ValueError("hidden_dim must be divisible by num_heads")

    def _initialize_model(self) -> None:
        """Initialize the TFT architecture and optimizer."""
        self.model = TemporalFusionTransformer(
            num_features=self.num_features,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_outputs=self.num_outputs,
            dropout=self.dropout
        )

        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=float(os.getenv("TFT_WEIGHT_DECAY", "1e-5"))
        )

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the TFT model.

        Args:
            features: Training features (n_samples, seq_len, n_features)
            labels: Training labels (n_samples, n_outputs)

        Raises:
            ValueError: If input arrays invalid
        """
        try:
            if features.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"Features and labels must have same number of samples: "
                    f"{features.shape[0]} vs {labels.shape[0]}"
                )

            logger.info(
                "Starting TFT training",
                n_samples=features.shape[0],
                seq_len=features.shape[1] if len(features.shape) > 2 else 1,
                n_features=features.shape[2] if len(features.shape) > 2 else features.shape[1],
                epochs=self.epochs
            )

            # Prepare data
            X_tensor = torch.FloatTensor(features).to(self.device)
            y_tensor = torch.FloatTensor(labels).to(self.device)

            # Ensure 3D shape (batch, sequence, features)
            if len(X_tensor.shape) == 2:
                X_tensor = X_tensor.unsqueeze(1)

            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Training loop
            best_loss = float("inf")
            patience_counter = 0

            for epoch in range(self.epochs):
                epoch_loss = 0.0
                self.model.train()

                for batch_X, batch_y in dataloader:
                    self.optimizer.zero_grad()

                    # Forward pass
                    outputs = self.model(batch_X)

                    # Calculate loss
                    loss = self.criterion(outputs, batch_y)

                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        float(os.getenv("TFT_GRAD_CLIP", "1.0"))
                    )
                    self.optimizer.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(dataloader)

                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    logger.debug(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            self.is_trained = True

            logger.info(
                "TFT training completed",
                final_loss=float(best_loss),
                epochs_trained=epoch + 1
            )

        except Exception as e:
            logger.error(
                "TFT training failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions on provided features.

        Args:
            features: Features array (n_samples, seq_len, n_features)

        Returns:
            Predictions array (n_samples, n_outputs)

        Raises:
            RuntimeError: If model not trained
        """
        try:
            if not self.is_trained:
                raise RuntimeError("Model must be trained before prediction")

            self.model.eval()

            with torch.no_grad():
                X_tensor = torch.FloatTensor(features).to(self.device)

                # Ensure 3D shape
                if len(X_tensor.shape) == 2:
                    X_tensor = X_tensor.unsqueeze(1)

                predictions = self.model(X_tensor)

            return predictions.cpu().numpy()

        except Exception as e:
            logger.error(
                "Prediction failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def predict_with_attention(
        self,
        features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions and return attention weights.

        Args:
            features: Features array

        Returns:
            Tuple of (predictions, attention_weights)

        Raises:
            RuntimeError: If model not trained
        """
        try:
            if not self.is_trained:
                raise RuntimeError("Model must be trained before prediction")

            self.model.eval()

            with torch.no_grad():
                X_tensor = torch.FloatTensor(features).to(self.device)

                if len(X_tensor.shape) == 2:
                    X_tensor = X_tensor.unsqueeze(1)

                predictions, attention = self.model(X_tensor, return_attention=True)

            return predictions.cpu().numpy(), attention.cpu().numpy()

        except Exception as e:
            logger.error(
                "Attention prediction failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            features: Test features array
            labels: True labels array

        Returns:
            Dictionary of evaluation metrics

        Raises:
            RuntimeError: If model not trained
        """
        try:
            if not self.is_trained:
                raise RuntimeError("Model must be trained before evaluation")

            predictions = self.predict(features)

            # Calculate metrics
            mse = float(np.mean((predictions - labels) ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(predictions - labels)))

            # R² score
            ss_res = np.sum((labels - predictions) ** 2)
            ss_tot = np.sum((labels - np.mean(labels)) ** 2)
            r2 = float(1 - (ss_res / (ss_tot + 1e-10)))

            metrics = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }

            logger.info("TFT evaluation completed", **metrics)

            return metrics

        except Exception as e:
            logger.error(
                "Evaluation failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path to save model

        Raises:
            RuntimeError: If model not trained
        """
        try:
            if not self.is_trained:
                raise RuntimeError("Cannot save untrained model")

            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "is_trained": self.is_trained,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, path)

            logger.info("TFT model saved", path=path)

        except Exception as e:
            logger.error("Model save failed", error=str(e), path=path)
            raise

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: File path to load model from

        Raises:
            FileNotFoundError: If model file not found
        """
        try:
            if not Path(path).exists():
                raise FileNotFoundError(f"Model file not found: {path}")

            checkpoint = torch.load(path, map_location=self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.config = checkpoint.get("config", self.config)
            self.is_trained = checkpoint.get("is_trained", True)

            logger.info(
                "TFT model loaded",
                path=path,
                timestamp=checkpoint.get("timestamp")
            )

        except Exception as e:
            logger.error("Model load failed", error=str(e), path=path)
            raise
