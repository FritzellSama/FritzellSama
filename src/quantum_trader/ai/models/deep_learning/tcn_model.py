"""Temporal Convolutional Network for time series prediction.

This module implements a TCN architecture optimized for financial time series
forecasting with causal convolutions and dilated residual blocks.
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


class TemporalBlock(nn.Module):
    """Temporal block with dilated causal convolutions.

    Implements a residual block with dilated causal convolutions,
    weight normalization, and dropout for temporal modeling.

    Attributes:
        conv1: First dilated causal convolution
        conv2: Second dilated causal convolution
        downsample: Downsampling layer for residual connection
        relu: ReLU activation
        dropout: Dropout layer
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        """Initialize temporal block.

        Args:
            n_inputs: Number of input channels
            n_outputs: Number of output channels
            kernel_size: Convolutional kernel size
            stride: Convolution stride
            dilation: Dilation factor
            padding: Padding size
            dropout: Dropout probability
        """
        super(TemporalBlock, self).__init__()

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(
                n_inputs, n_outputs, kernel_size,
                stride=stride, padding=padding, dilation=dilation
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(
                n_outputs, n_outputs, kernel_size,
                stride=stride, padding=padding, dilation=dilation
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through temporal block.

        Args:
            x: Input tensor (batch, channels, sequence)

        Returns:
            Output tensor (batch, channels, sequence)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Chomping layer to ensure causality in temporal convolutions."""

    def __init__(self, chomp_size: int):
        """Initialize chomp layer.

        Args:
            chomp_size: Number of elements to chomp
        """
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Remove trailing elements to ensure causality.

        Args:
            x: Input tensor

        Returns:
            Chomped tensor
        """
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network architecture.

    Stacks multiple temporal blocks with increasing dilation
    to capture long-range temporal dependencies.

    Attributes:
        network: Sequential network of temporal blocks
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2
    ):
        """Initialize TCN.

        Args:
            num_inputs: Number of input features
            num_channels: List of channel sizes for each layer
            kernel_size: Convolutional kernel size
            dropout: Dropout probability
        """
        super(TemporalConvNet, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TCN.

        Args:
            x: Input tensor (batch, features, sequence)

        Returns:
            Output tensor (batch, channels, sequence)
        """
        return self.network(x)


class TCNModel(BaseMLModel):
    """Temporal Convolutional Network for trading predictions.

    Implements TCN architecture for time series forecasting with
    support for multi-step ahead predictions and uncertainty estimation.

    Attributes:
        config: Model configuration
        model: TCN neural network
        optimizer: Training optimizer
        criterion: Loss function
        device: Computation device (CPU/GPU)
        is_trained: Training status

    Examples:
        >>> config = {
        ...     "num_channels": [64, 128, 64],
        ...     "kernel_size": 3,
        ...     "dropout": 0.2,
        ...     "learning_rate": 0.001
        ... }
        >>> tcn = TCNModel(config)
        >>> tcn.train(features, labels)
        >>> predictions = tcn.predict(test_features)
    """

    def __init__(self, config: Dict) -> None:
        """Initialize TCN model.

        Args:
            config: Configuration with architecture, training params, etc.

        Raises:
            ValueError: If required config parameters missing
        """
        super().__init__(config)
        self._validate_config()

        # Architecture parameters
        self.num_inputs: int = int(config.get("num_inputs", os.getenv("TCN_NUM_INPUTS", "10")))
        self.num_outputs: int = int(config.get("num_outputs", os.getenv("TCN_NUM_OUTPUTS", "1")))
        self.num_channels: List[int] = config.get("num_channels", [int(x) for x in os.getenv("TCN_NUM_CHANNELS", "64,128,64").split(",")])
        self.kernel_size: int = int(config.get("kernel_size", os.getenv("TCN_KERNEL_SIZE", "3")))
        self.dropout: float = float(config.get("dropout", os.getenv("TCN_DROPOUT", "0.2")))

        # Training parameters
        self.learning_rate: float = float(config.get("learning_rate", os.getenv("TCN_LEARNING_RATE", "0.001")))
        self.batch_size: int = int(config.get("batch_size", os.getenv("TCN_BATCH_SIZE", "32")))
        self.epochs: int = int(config.get("epochs", os.getenv("TCN_EPOCHS", "100")))
        self.patience: int = int(config.get("patience", os.getenv("TCN_PATIENCE", "10")))

        # Device configuration
        self.device = torch.device(
            config.get("device", os.getenv("TCN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        )

        # Initialize model components
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion: nn.Module = nn.MSELoss()
        self.is_trained: bool = False

        self._initialize_model()

        logger.info(
            "TCN model initialized",
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            num_channels=self.num_channels,
            device=str(self.device)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config validation fails
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        if "num_channels" in self.config:
            if not self.config["num_channels"]:
                raise ValueError("num_channels must not be empty")

    def _initialize_model(self) -> None:
        """Initialize the TCN architecture and optimizer."""
        # Build TCN
        self.tcn = TemporalConvNet(
            num_inputs=self.num_inputs,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        )

        # Output layer
        self.linear = nn.Linear(self.num_channels[-1], self.num_outputs)

        # Full model
        self.model = nn.Sequential(self.tcn, nn.AdaptiveAvgPool1d(1), nn.Flatten(), self.linear)
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=float(os.getenv("TCN_WEIGHT_DECAY", "1e-5"))
        )

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the TCN model.

        Args:
            features: Training features (n_samples, n_features, seq_len)
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
                "Starting TCN training",
                n_samples=features.shape[0],
                n_features=features.shape[1] if len(features.shape) > 1 else 1,
                epochs=self.epochs
            )

            # Prepare data
            X_tensor = torch.FloatTensor(features).to(self.device)
            y_tensor = torch.FloatTensor(labels).to(self.device)

            # Ensure correct shape (batch, features, sequence)
            if len(X_tensor.shape) == 2:
                X_tensor = X_tensor.unsqueeze(-1)

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
                    self.optimizer.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(dataloader)

                # Early stopping check
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
                "TCN training completed",
                final_loss=float(best_loss),
                epochs_trained=epoch + 1
            )

        except Exception as e:
            logger.error(
                "TCN training failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions on provided features.

        Args:
            features: Features array for prediction (n_samples, n_features, seq_len)

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

                # Ensure correct shape
                if len(X_tensor.shape) == 2:
                    X_tensor = X_tensor.unsqueeze(-1)

                predictions = self.model(X_tensor)

            return predictions.cpu().numpy()

        except Exception as e:
            logger.error(
                "Prediction failed",
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

            logger.info("Model evaluation completed", **metrics)

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

            logger.info("Model saved", path=path)

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
                "Model loaded",
                path=path,
                timestamp=checkpoint.get("timestamp")
            )

        except Exception as e:
            logger.error("Model load failed", error=str(e), path=path)
            raise
