"""WaveNet model for time series forecasting.

This module implements a WaveNet-inspired architecture with dilated causal
convolutions for capturing long-range dependencies in financial time series.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel

logger = get_logger(__name__)


class CausalConv1d(nn.Module):
    """Causal 1D convolution for time series.

    Ensures that predictions at time t only depend on inputs up to time t.

    Attributes:
        conv: 1D convolution layer
        padding: Padding size for causality
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ):
        """Initialize causal convolution.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            dilation: Dilation factor
        """
        super(CausalConv1d, self).__init__()

        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal padding.

        Args:
            x: Input tensor (batch, channels, sequence)

        Returns:
            Output tensor maintaining causality
        """
        output = self.conv(x)
        # Remove future information
        if self.padding > 0:
            output = output[:, :, :-self.padding]
        return output


class ResidualBlock(nn.Module):
    """Residual block with dilated causal convolutions.

    Implements a gated activation unit with skip and residual connections.

    Attributes:
        dilated_conv: Dilated causal convolution
        gate_conv: Gating convolution
        residual_conv: Residual projection
        skip_conv: Skip connection projection
    """

    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        """Initialize residual block.

        Args:
            residual_channels: Number of residual channels
            skip_channels: Number of skip channels
            kernel_size: Convolution kernel size
            dilation: Dilation factor
            dropout: Dropout probability
        """
        super(ResidualBlock, self).__init__()

        self.dilated_conv = CausalConv1d(
            residual_channels,
            residual_channels * 2,
            kernel_size,
            dilation
        )

        self.dropout = nn.Dropout(dropout)

        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through residual block.

        Args:
            x: Input tensor

        Returns:
            Tuple of (residual_output, skip_output)
        """
        # Dilated causal convolution
        h = self.dilated_conv(x)

        # Gated activation
        tanh_out, sigmoid_out = h.chunk(2, dim=1)
        h = torch.tanh(tanh_out) * torch.sigmoid(sigmoid_out)

        h = self.dropout(h)

        # Residual and skip connections
        residual = self.residual_conv(h)
        skip = self.skip_conv(h)

        return (x + residual), skip


class WaveNet(nn.Module):
    """WaveNet architecture for time series.

    Stacks residual blocks with exponentially increasing dilation
    to capture multi-scale temporal patterns.

    Attributes:
        input_conv: Initial causal convolution
        residual_blocks: Stack of residual blocks
        output_conv: Output projection layers
    """

    def __init__(
        self,
        input_channels: int,
        residual_channels: int,
        skip_channels: int,
        output_channels: int,
        num_layers: int,
        kernel_size: int = 2,
        dropout: float = 0.2
    ):
        """Initialize WaveNet.

        Args:
            input_channels: Number of input features
            residual_channels: Channels in residual blocks
            skip_channels: Channels in skip connections
            output_channels: Number of output predictions
            num_layers: Number of residual layers
            kernel_size: Convolution kernel size
            dropout: Dropout probability
        """
        super(WaveNet, self).__init__()

        # Initial causal convolution
        self.input_conv = CausalConv1d(
            input_channels,
            residual_channels,
            kernel_size=1
        )

        # Residual blocks with exponentially increasing dilation
        self.residual_blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.residual_blocks.append(
                ResidualBlock(
                    residual_channels,
                    skip_channels,
                    kernel_size,
                    dilation,
                    dropout
                )
            )

        # Output layers
        self.output_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, 1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, output_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through WaveNet.

        Args:
            x: Input tensor (batch, channels, sequence)

        Returns:
            Output predictions
        """
        # Initial convolution
        h = self.input_conv(x)

        # Residual blocks with skip connections
        skip_connections = []
        for block in self.residual_blocks:
            h, skip = block(h)
            skip_connections.append(skip)

        # Sum skip connections
        skip_sum = torch.sum(torch.stack(skip_connections), dim=0)

        # Output
        output = self.output_conv(skip_sum)

        return output


class WaveNetModel(BaseMLModel):
    """WaveNet model for trading predictions.

    Implements WaveNet architecture for multi-horizon time series forecasting
    with dilated causal convolutions.

    Attributes:
        config: Model configuration
        model: WaveNet neural network
        optimizer: Training optimizer
        criterion: Loss function
        device: Computation device
        is_trained: Training status

    Examples:
        >>> config = {
        ...     "input_channels": 10,
        ...     "residual_channels": 32,
        ...     "skip_channels": 64,
        ...     "num_layers": 10,
        ...     "learning_rate": 0.001
        ... }
        >>> wavenet = WaveNetModel(config)
        >>> wavenet.train(features, labels)
        >>> predictions = wavenet.predict(test_features)
    """

    def __init__(self, config: Dict) -> None:
        """Initialize WaveNet model.

        Args:
            config: Configuration with architecture and training params

        Raises:
            ValueError: If required config parameters missing
        """
        super().__init__(config)
        self._validate_config()

        # Architecture parameters
        self.input_channels: int = int(config.get("input_channels", os.getenv("WAVENET_INPUT_CHANNELS", "10")))
        self.residual_channels: int = int(config.get("residual_channels", os.getenv("WAVENET_RESIDUAL_CHANNELS", "32")))
        self.skip_channels: int = int(config.get("skip_channels", os.getenv("WAVENET_SKIP_CHANNELS", "64")))
        self.output_channels: int = int(config.get("output_channels", os.getenv("WAVENET_OUTPUT_CHANNELS", "1")))
        self.num_layers: int = int(config.get("num_layers", os.getenv("WAVENET_NUM_LAYERS", "10")))
        self.kernel_size: int = int(config.get("kernel_size", os.getenv("WAVENET_KERNEL_SIZE", "2")))
        self.dropout: float = float(config.get("dropout", os.getenv("WAVENET_DROPOUT", "0.2")))

        # Training parameters
        self.learning_rate: float = float(config.get("learning_rate", os.getenv("WAVENET_LEARNING_RATE", "0.001")))
        self.batch_size: int = int(config.get("batch_size", os.getenv("WAVENET_BATCH_SIZE", "32")))
        self.epochs: int = int(config.get("epochs", os.getenv("WAVENET_EPOCHS", "100")))
        self.patience: int = int(config.get("patience", os.getenv("WAVENET_PATIENCE", "10")))

        # Device configuration
        self.device = torch.device(
            config.get("device", os.getenv("WAVENET_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        )

        # Initialize model components
        self.model: Optional[WaveNet] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion: nn.Module = nn.MSELoss()
        self.is_trained: bool = False

        self._initialize_model()

        logger.info(
            "WaveNet model initialized",
            input_channels=self.input_channels,
            residual_channels=self.residual_channels,
            num_layers=self.num_layers,
            device=str(self.device)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config validation fails
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        if "num_layers" in self.config:
            if self.config["num_layers"] < 1:
                raise ValueError("num_layers must be at least 1")

    def _initialize_model(self) -> None:
        """Initialize the WaveNet architecture and optimizer."""
        self.model = WaveNet(
            input_channels=self.input_channels,
            residual_channels=self.residual_channels,
            skip_channels=self.skip_channels,
            output_channels=self.output_channels,
            num_layers=self.num_layers,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        )

        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=float(os.getenv("WAVENET_WEIGHT_DECAY", "1e-5"))
        )

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the WaveNet model.

        Args:
            features: Training features (n_samples, n_channels, seq_len)
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
                "Starting WaveNet training",
                n_samples=features.shape[0],
                n_channels=features.shape[1] if len(features.shape) > 1 else 1,
                epochs=self.epochs
            )

            # Prepare data
            X_tensor = torch.FloatTensor(features).to(self.device)
            y_tensor = torch.FloatTensor(labels).to(self.device)

            # Ensure correct shape (batch, channels, sequence)
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

                    # Take last time step for prediction
                    outputs = outputs[:, :, -1]

                    # Calculate loss
                    loss = self.criterion(outputs, batch_y)

                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        float(os.getenv("WAVENET_GRAD_CLIP", "1.0"))
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
                "WaveNet training completed",
                final_loss=float(best_loss),
                epochs_trained=epoch + 1
            )

        except Exception as e:
            logger.error(
                "WaveNet training failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions on provided features.

        Args:
            features: Features array (n_samples, n_channels, seq_len)

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

                outputs = self.model(X_tensor)

                # Take last time step
                predictions = outputs[:, :, -1]

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

            logger.info("WaveNet evaluation completed", **metrics)

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

            logger.info("WaveNet model saved", path=path)

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
                "WaveNet model loaded",
                path=path,
                timestamp=checkpoint.get("timestamp")
            )

        except Exception as e:
            logger.error("Model load failed", error=str(e), path=path)
            raise
