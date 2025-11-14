"""Gated Recurrent Unit model for time series prediction.

This module implements GRU networks for modeling temporal dependencies
in market data and generating price/volatility predictions.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel

logger = get_logger(__name__)


class GRUNetwork(nn.Module):
    """GRU neural network architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.3,
        bidirectional: bool = False
    ) -> None:
        """Initialize GRU network.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of GRU layers
            output_dim: Output dimension
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional GRU
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Fully connected output layer
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, sequence_length, input_dim]
            hidden: Optional initial hidden state

        Returns:
            Output tensor [batch_size, output_dim]
        """
        # GRU forward
        gru_out, hidden = self.gru(x, hidden)

        # Take last timestep
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            last_hidden = torch.cat([
                hidden[-2, :, :],
                hidden[-1, :, :]
            ], dim=1)
        else:
            last_hidden = hidden[-1, :, :]

        # Apply dropout
        last_hidden = self.dropout(last_hidden)

        # Output layer
        output = self.fc(last_hidden)

        return output


class GRUModel(BaseMLModel):
    """GRU model for time series prediction.

    Implements GRU networks for learning temporal patterns in market data,
    predicting prices, volatility, and trading signals.

    Attributes:
        config: Model configuration
        network: GRU network
        device: Torch device

    Example:
        >>> config = {
        ...     "input_dim": 20,
        ...     "hidden_dim": 128,
        ...     "num_layers": 2,
        ...     "output_dim": 1,
        ...     "dropout": "0.3",
        ...     "learning_rate": "0.001",
        ...     "bidirectional": False
        ... }
        >>> model = GRUModel(config)
        >>> model.train(features, labels)
        >>> predictions = model.predict(test_features)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize GRU model.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If config is invalid
        """
        super().__init__(config)

        self._validate_config()

        # Model parameters
        self.input_dim = config["input_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config["num_layers"]
        self.output_dim = config["output_dim"]
        self.dropout = float(config.get("dropout", "0.3"))
        self.bidirectional = config.get("bidirectional", False)

        # Training parameters
        self.learning_rate = Decimal(str(config.get("learning_rate", "0.001")))
        self.weight_decay = Decimal(str(config.get("weight_decay", "0.00001")))
        self.batch_size = config.get("batch_size", 64)
        self.num_epochs = config.get("num_epochs", 100)
        self.gradient_clip = float(config.get("gradient_clip", "1.0"))

        # Device setup
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Build network
        self.network = GRUNetwork(
            self.input_dim,
            self.hidden_dim,
            self.num_layers,
            self.output_dim,
            self.dropout,
            self.bidirectional
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay)
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Training state
        self.training_history: List[Dict[str, float]] = []

        logger.info(
            "GRU model initialized",
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.output_dim,
            bidirectional=self.bidirectional,
            device=str(self.device)
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If required config missing
        """
        required_keys = [
            "input_dim",
            "hidden_dim",
            "num_layers",
            "output_dim"
        ]

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train GRU model.

        Args:
            features: Input sequences [num_samples, sequence_length, input_dim]
            labels: Target values [num_samples, output_dim]

        Raises:
            ValueError: If input shape is invalid
        """
        try:
            if features.shape[2] != self.input_dim:
                raise ValueError(
                    f"Expected input_dim {self.input_dim}, got {features.shape[2]}"
                )

            if labels.shape[1] != self.output_dim:
                raise ValueError(
                    f"Expected output_dim {self.output_dim}, got {labels.shape[1]}"
                )

            logger.info(
                "Starting GRU training",
                num_samples=len(features),
                sequence_length=features.shape[1],
                epochs=self.num_epochs
            )

            # Convert to tensors
            X = torch.FloatTensor(features).to(self.device)
            y = torch.FloatTensor(labels).to(self.device)

            # Training loop
            for epoch in range(self.num_epochs):
                self.network.train()
                epoch_loss = Decimal("0")
                num_batches = 0

                # Batch training
                for i in range(0, len(X), self.batch_size):
                    batch_X = X[i:i + self.batch_size]
                    batch_y = y[i:i + self.batch_size]

                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.network(batch_X)

                    # Compute loss
                    loss = self.criterion(outputs, batch_y)

                    # Backward pass
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        self.gradient_clip
                    )

                    self.optimizer.step()

                    epoch_loss += Decimal(str(loss.item()))
                    num_batches += 1

                # Log epoch metrics
                avg_loss = float(epoch_loss / num_batches)
                self.training_history.append({
                    "epoch": epoch,
                    "loss": avg_loss
                })

                if epoch % 10 == 0:
                    logger.info("Training progress", epoch=epoch, loss=avg_loss)

            logger.info("GRU training completed")

        except Exception as e:
            logger.error("Failed to train GRU model", error=str(e))
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions.

        Args:
            features: Input sequences

        Returns:
            Predictions [num_samples, output_dim]
        """
        try:
            self.network.eval()

            X = torch.FloatTensor(features).to(self.device)

            with torch.no_grad():
                predictions = self.network(X).cpu().numpy()

            logger.info("Generated predictions", shape=predictions.shape)

            return predictions

        except Exception as e:
            logger.error("Failed to generate predictions", error=str(e))
            raise

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            features: Input sequences
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

            # Directional accuracy (for price prediction)
            if self.output_dim == 1:
                pred_direction = (predictions[:, 0] > 0).astype(int)
                true_direction = (labels[:, 0] > 0).astype(int)
                directional_accuracy = float(
                    np.mean(pred_direction == true_direction)
                )
            else:
                directional_accuracy = 0.0

            # R-squared
            ss_res = np.sum((labels - predictions) ** 2)
            ss_tot = np.sum((labels - np.mean(labels)) ** 2)
            r2 = float(1 - (ss_res / (ss_tot + 1e-10)))

            metrics = {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "directional_accuracy": directional_accuracy
            }

            logger.info("GRU evaluation completed", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("Failed to evaluate GRU model", error=str(e))
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
                "model_state": self.network.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config,
                "training_history": self.training_history
            }, save_path)

            logger.info("GRU model saved", path=path)

        except Exception as e:
            logger.error("Failed to save GRU model", error=str(e))
            raise

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: File path to load model from
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)

            self.network.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.training_history = checkpoint["training_history"]

            logger.info("GRU model loaded", path=path)

        except Exception as e:
            logger.error("Failed to load GRU model", error=str(e))
            raise

    def predict_sequence(
        self,
        features: np.ndarray,
        num_steps: int
    ) -> np.ndarray:
        """Generate multi-step ahead predictions.

        Args:
            features: Initial input sequence [1, sequence_length, input_dim]
            num_steps: Number of steps to predict

        Returns:
            Predictions for each step [num_steps, output_dim]
        """
        try:
            self.network.eval()

            predictions = []
            current_input = torch.FloatTensor(features).to(self.device)

            with torch.no_grad():
                for _ in range(num_steps):
                    # Predict next step
                    pred = self.network(current_input)
                    predictions.append(pred.cpu().numpy()[0])

                    # Update input (shift and append prediction)
                    # Note: This assumes output matches input features
                    if self.output_dim == self.input_dim:
                        new_step = pred.unsqueeze(1)
                        current_input = torch.cat([
                            current_input[:, 1:, :],
                            new_step
                        ], dim=1)

            predictions = np.array(predictions)

            logger.info(
                "Generated sequence predictions",
                num_steps=num_steps,
                shape=predictions.shape
            )

            return predictions

        except Exception as e:
            logger.error("Failed to generate sequence predictions", error=str(e))
            raise
