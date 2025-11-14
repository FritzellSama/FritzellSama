"""
WaveNet Model for Trading

Production-ready WaveNet implementation for time series forecasting.
Uses dilated causal convolutions with residual and skip connections.
"""

import asyncio
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from abc import ABC, abstractmethod
import os
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from structlog import get_logger

logger = get_logger(__name__)

# Set high precision
getcontext().prec = 28


class BaseMLModel(ABC):
    """Abstract base for all ML models"""

    def __init__(self, config: Dict) -> None:
        ...

    @abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass


class CausalConv1d(nn.Module):
    """Causal convolution layer."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class ResidualBlock(nn.Module):
    """WaveNet residual block with gated activation."""

    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super(ResidualBlock, self).__init__()

        # Dilated causal convolutions
        self.conv_filter = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation)
        self.conv_gate = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation)

        # 1x1 convolutions
        self.conv_residual = nn.Conv1d(residual_channels, residual_channels, 1)
        self.conv_skip = nn.Conv1d(residual_channels, skip_channels, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Gated activation
        filter_out = self.conv_filter(x)
        gate_out = self.conv_gate(x)
        gated = torch.tanh(filter_out) * torch.sigmoid(gate_out)

        gated = self.dropout(gated)

        # Residual and skip connections
        residual = self.conv_residual(gated)
        skip = self.conv_skip(gated)

        return (x + residual), skip


class WaveNetArchitecture(nn.Module):
    """WaveNet architecture."""

    def __init__(
        self,
        input_channels: int,
        residual_channels: int,
        skip_channels: int,
        num_layers: int,
        kernel_size: int,
        output_size: int,
        dropout: float = 0.2
    ):
        super(WaveNetArchitecture, self).__init__()

        self.input_channels = input_channels

        # Input projection
        self.input_conv = nn.Conv1d(input_channels, residual_channels, 1)

        # Residual blocks with increasing dilation
        self.residual_blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.residual_blocks.append(
                ResidualBlock(residual_channels, skip_channels, kernel_size, dilation, dropout)
            )

        # Output layers
        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.output_conv2 = nn.Conv1d(skip_channels, output_size, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Input projection
        x = self.input_conv(x)

        # Accumulate skip connections
        skip_connections = []

        # Pass through residual blocks
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        # Sum skip connections
        skip_sum = torch.stack(skip_connections).sum(dim=0)

        # Output processing
        out = self.relu(skip_sum)
        out = self.output_conv1(out)
        out = self.relu(out)
        out = self.output_conv2(out)

        # Take last timestep
        out = out[:, :, -1]

        return out


class WaveNetModel(BaseMLModel):
    """
    WaveNet model for trading time series.

    Uses dilated causal convolutions with residual and skip connections
    for long-range temporal dependency modeling.

    Attributes:
        config: Configuration dictionary
        model: PyTorch WaveNet model
        device: Computation device
        is_trained: Training status

    Example:
        >>> config = {"model": {"wavenet": {"residual_channels": 64, "num_layers": 10}}}
        >>> model = WaveNetModel(config)
        >>> model.train(features, labels)
        >>> predictions = model.predict(new_features)
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialize WaveNet model.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__(config)
        self.config = config
        self._validate_config()

        wavenet_config = self.config.get("model", {}).get("wavenet", {})

        # Model architecture
        self.residual_channels: int = wavenet_config.get(
            "residual_channels",
            int(os.getenv("WAVENET_RESIDUAL_CHANNELS", "64"))
        )
        self.skip_channels: int = wavenet_config.get(
            "skip_channels",
            int(os.getenv("WAVENET_SKIP_CHANNELS", "64"))
        )
        self.num_layers: int = wavenet_config.get("num_layers", int(os.getenv("WAVENET_NUM_LAYERS", "10")))
        self.kernel_size: int = wavenet_config.get("kernel_size", int(os.getenv("WAVENET_KERNEL_SIZE", "2")))
        self.dropout: float = float(wavenet_config.get("dropout", os.getenv("WAVENET_DROPOUT", "0.2")))
        self.output_size: int = wavenet_config.get("output_size", int(os.getenv("WAVENET_OUTPUT_SIZE", "1")))

        # Training parameters
        self.learning_rate: float = float(
            wavenet_config.get("learning_rate", os.getenv("WAVENET_LEARNING_RATE", "0.001"))
        )
        self.batch_size: int = wavenet_config.get("batch_size", int(os.getenv("WAVENET_BATCH_SIZE", "32")))
        self.num_epochs: int = wavenet_config.get("num_epochs", int(os.getenv("WAVENET_NUM_EPOCHS", "100")))
        self.patience: int = wavenet_config.get("patience", int(os.getenv("WAVENET_PATIENCE", "15")))

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.model: Optional[WaveNetArchitecture] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion: nn.Module = nn.MSELoss()
        self.is_trained: bool = False

        logger.info(
            "wavenet_model_initialized",
            residual_channels=self.residual_channels,
            num_layers=self.num_layers,
            device=str(self.device)
        )

    def _validate_config(self) -> None:
        """Validate configuration."""
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        wavenet_config = self.config.get("model", {}).get("wavenet", {})

        if wavenet_config:
            num_layers = wavenet_config.get("num_layers", 10)
            if num_layers <= 0:
                raise ValueError("num_layers must be positive")

            residual_channels = wavenet_config.get("residual_channels", 64)
            if residual_channels <= 0:
                raise ValueError("residual_channels must be positive")

    def _initialize_model(self, input_channels: int) -> None:
        """Initialize model architecture."""
        self.model = WaveNetArchitecture(
            input_channels=input_channels,
            residual_channels=self.residual_channels,
            skip_channels=self.skip_channels,
            num_layers=self.num_layers,
            kernel_size=self.kernel_size,
            output_size=self.output_size,
            dropout=self.dropout
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        logger.info("wavenet_network_created", input_channels=input_channels)

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Train WaveNet model.

        Args:
            features: Training features (n_samples, sequence_length, n_features)
            labels: Training labels (n_samples, output_size)

        Raises:
            ValueError: If inputs are invalid
        """
        try:
            if not isinstance(features, np.ndarray):
                raise ValueError("features must be numpy array")
            if not isinstance(labels, np.ndarray):
                raise ValueError("labels must be numpy array")

            if len(features) == 0 or len(labels) == 0:
                raise ValueError("Cannot train on empty data")

            # Ensure 3D features
            if len(features.shape) == 2:
                features = features[:, :, np.newaxis]

            n_samples, seq_length, n_features = features.shape

            logger.info("wavenet_training_started", n_samples=n_samples, n_features=n_features)

            # Initialize model
            if self.model is None:
                self._initialize_model(n_features)

            # Prepare data (transpose to channels-first)
            features_transposed = np.transpose(features, (0, 2, 1))
            X_tensor = torch.FloatTensor(features_transposed).to(self.device)
            y_tensor = torch.FloatTensor(labels).to(self.device)

            if len(y_tensor.shape) == 1:
                y_tensor = y_tensor.unsqueeze(1)

            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Training loop
            best_loss = float('inf')
            patience_counter = 0

            for epoch in range(self.num_epochs):
                self.model.train()
                epoch_loss = 0.0

                for batch_X, batch_y in dataloader:
                    self.optimizer.zero_grad()

                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)

                    loss.backward()
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
                    logger.debug("wavenet_training_epoch", epoch=epoch + 1, loss=avg_loss)

                if patience_counter >= self.patience:
                    logger.info("wavenet_early_stopping", epoch=epoch + 1, best_loss=best_loss)
                    break

            self.is_trained = True

            logger.info("wavenet_training_completed", final_loss=best_loss, epochs=epoch + 1)

        except Exception as e:
            logger.error("wavenet_training_failed", error=str(e))
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            features: Input features (n_samples, sequence_length, n_features)

        Returns:
            Predictions array
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")

            if len(features.shape) == 2:
                features = features[:, :, np.newaxis]

            # Transpose to channels-first
            features_transposed = np.transpose(features, (0, 2, 1))
            X_tensor = torch.FloatTensor(features_transposed).to(self.device)

            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor)

            predictions_np = predictions.cpu().numpy().squeeze()

            logger.debug("wavenet_predictions_generated", n_samples=len(predictions_np))

            return predictions_np

        except Exception as e:
            logger.error("wavenet_prediction_failed", error=str(e))
            raise

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")

            predictions = self.predict(features)

            # Ensure same shape
            if len(predictions.shape) != len(labels.shape):
                if len(labels.shape) == 1:
                    labels = labels.reshape(-1, 1)
                if len(predictions.shape) == 1:
                    predictions = predictions.reshape(-1, 1)

            mse = np.mean((labels - predictions) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(labels - predictions))

            ss_res = np.sum((labels - predictions) ** 2)
            ss_tot = np.sum((labels - np.mean(labels)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))

            metrics = {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "r2_score": float(r2)
            }

            logger.info("wavenet_evaluation_completed", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("wavenet_evaluation_failed", error=str(e))
            raise

    def save(self, path: str) -> None:
        """Save model to disk."""
        try:
            if not self.is_trained:
                raise ValueError("Cannot save untrained model")

            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            save_dict = {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config,
                "residual_channels": self.residual_channels,
                "skip_channels": self.skip_channels,
                "num_layers": self.num_layers,
                "kernel_size": self.kernel_size,
                "output_size": self.output_size,
                "input_channels": self.model.input_channels,
                "is_trained": self.is_trained
            }

            torch.save(save_dict, path)
            logger.info("wavenet_model_saved", path=path)

        except Exception as e:
            logger.error("wavenet_save_failed", path=path, error=str(e))
            raise

    def load(self, path: str) -> None:
        """Load model from disk."""
        try:
            if not Path(path).exists():
                raise FileNotFoundError(f"Model file not found: {path}")

            checkpoint = torch.load(path, map_location=self.device)

            # Restore configuration
            self.residual_channels = checkpoint["residual_channels"]
            self.skip_channels = checkpoint["skip_channels"]
            self.num_layers = checkpoint["num_layers"]
            self.kernel_size = checkpoint["kernel_size"]
            self.output_size = checkpoint["output_size"]

            # Recreate model
            self._initialize_model(checkpoint["input_channels"])

            # Load states
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.is_trained = checkpoint["is_trained"]

            logger.info("wavenet_model_loaded", path=path)

        except Exception as e:
            logger.error("wavenet_load_failed", path=path, error=str(e))
            raise
