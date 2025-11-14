"""
Temporal Convolutional Network for Trading

Production-ready TCN implementation for time series forecasting and trading signals.
Uses dilated causal convolutions for long-range temporal dependencies.
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


class TemporalBlock(nn.Module):
    """Temporal convolutional block with residual connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super(TemporalBlock, self).__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.chomp1 = nn.ConstantPad1d((0, -padding), 0)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.chomp2 = nn.ConstantPad1d((0, -padding), 0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNNetwork(nn.Module):
    """Temporal Convolutional Network architecture."""

    def __init__(
        self,
        input_size: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        super(TCNNetwork, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation_size,
                    dropout
                )
            )

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x shape: (batch, features, sequence_length)
        y = self.network(x)
        # Take last timestep
        y = y[:, :, -1]
        return self.fc(y)


class TCNModel(BaseMLModel):
    """
    Temporal Convolutional Network for trading.

    Implements TCN with dilated causal convolutions for time series
    prediction with long-range dependencies.

    Attributes:
        config: Configuration dictionary
        model: PyTorch TCN model
        device: Computation device (CPU/GPU)
        is_trained: Training status

    Example:
        >>> config = {"model": {"tcn": {"num_channels": [64, 128, 256]}}}
        >>> model = TCNModel(config)
        >>> model.train(features, labels)
        >>> predictions = model.predict(new_features)
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialize TCN model.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__(config)
        self.config = config
        self._validate_config()

        tcn_config = self.config.get("model", {}).get("tcn", {})

        # Model architecture parameters
        self.num_channels: List[int] = tcn_config.get(
            "num_channels",
            [int(x) for x in os.getenv("TCN_NUM_CHANNELS", "64,128,256").split(",")]
        )
        self.kernel_size: int = tcn_config.get("kernel_size", int(os.getenv("TCN_KERNEL_SIZE", "3")))
        self.dropout: float = float(tcn_config.get("dropout", os.getenv("TCN_DROPOUT", "0.2")))
        self.input_size: Optional[int] = tcn_config.get("input_size")
        self.output_size: int = tcn_config.get("output_size", int(os.getenv("TCN_OUTPUT_SIZE", "1")))

        # Training parameters
        self.learning_rate: float = float(tcn_config.get("learning_rate", os.getenv("TCN_LEARNING_RATE", "0.001")))
        self.batch_size: int = tcn_config.get("batch_size", int(os.getenv("TCN_BATCH_SIZE", "32")))
        self.num_epochs: int = tcn_config.get("num_epochs", int(os.getenv("TCN_NUM_EPOCHS", "100")))
        self.patience: int = tcn_config.get("patience", int(os.getenv("TCN_PATIENCE", "10")))

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model and optimizer
        self.model: Optional[TCNNetwork] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion: nn.Module = nn.MSELoss()
        self.is_trained: bool = False

        logger.info(
            "tcn_model_initialized",
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            device=str(self.device)
        )

    def _validate_config(self) -> None:
        """
        Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        tcn_config = self.config.get("model", {}).get("tcn", {})

        if tcn_config:
            num_channels = tcn_config.get("num_channels", [64, 128, 256])
            if not isinstance(num_channels, list) or len(num_channels) == 0:
                raise ValueError("num_channels must be non-empty list")

            kernel_size = tcn_config.get("kernel_size", 3)
            if kernel_size < 2:
                raise ValueError("kernel_size must be at least 2")

    def _initialize_model(self, input_size: int) -> None:
        """Initialize model architecture."""
        self.model = TCNNetwork(
            input_size=input_size,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            output_size=self.output_size
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        logger.info("tcn_network_created", input_size=input_size, output_size=self.output_size)

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Train TCN model.

        Args:
            features: Training features (n_samples, sequence_length, n_features)
            labels: Training labels (n_samples, output_size)

        Raises:
            ValueError: If inputs are invalid
        """
        try:
            # Validate inputs
            if not isinstance(features, np.ndarray):
                raise ValueError("features must be numpy array")
            if not isinstance(labels, np.ndarray):
                raise ValueError("labels must be numpy array")

            if len(features) == 0 or len(labels) == 0:
                raise ValueError("Cannot train on empty data")

            if len(features) != len(labels):
                raise ValueError("features and labels must have same length")

            # Ensure 3D features (samples, sequence, features)
            if len(features.shape) == 2:
                features = features[:, :, np.newaxis]

            n_samples, seq_length, n_features = features.shape

            logger.info(
                "tcn_training_started",
                n_samples=n_samples,
                seq_length=seq_length,
                n_features=n_features
            )

            # Initialize model if not done
            if self.model is None:
                self._initialize_model(n_features)

            # Prepare data
            # TCN expects (batch, features, sequence) format
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

                # Early stopping check
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    logger.debug(f"tcn_training_epoch", epoch=epoch + 1, loss=avg_loss)

                if patience_counter >= self.patience:
                    logger.info("tcn_early_stopping", epoch=epoch + 1, best_loss=best_loss)
                    break

            self.is_trained = True

            logger.info(
                "tcn_training_completed",
                final_loss=best_loss,
                epochs_trained=epoch + 1
            )

        except Exception as e:
            logger.error("tcn_training_failed", error=str(e))
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            features: Input features (n_samples, sequence_length, n_features)

        Returns:
            Predictions array

        Raises:
            ValueError: If model not trained or inputs invalid
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")

            if not isinstance(features, np.ndarray):
                raise ValueError("features must be numpy array")

            # Ensure 3D features
            if len(features.shape) == 2:
                features = features[:, :, np.newaxis]

            # Transpose to (batch, features, sequence)
            features_transposed = np.transpose(features, (0, 2, 1))
            X_tensor = torch.FloatTensor(features_transposed).to(self.device)

            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor)

            predictions_np = predictions.cpu().numpy().squeeze()

            logger.debug("tcn_predictions_generated", n_samples=len(predictions_np))

            return predictions_np

        except Exception as e:
            logger.error("tcn_prediction_failed", error=str(e))
            raise

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            features: Test features
            labels: True labels

        Returns:
            Dictionary of evaluation metrics

        Raises:
            ValueError: If model not trained or inputs invalid
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")

            predictions = self.predict(features)

            # Ensure same shape
            if len(predictions.shape) == 1 and len(labels.shape) == 1:
                pass
            elif len(predictions.shape) != len(labels.shape):
                if len(labels.shape) == 1:
                    labels = labels.reshape(-1, 1)
                if len(predictions.shape) == 1:
                    predictions = predictions.reshape(-1, 1)

            # Calculate metrics
            mse = np.mean((labels - predictions) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(labels - predictions))

            # R² score
            ss_res = np.sum((labels - predictions) ** 2)
            ss_tot = np.sum((labels - np.mean(labels)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))

            # Direction accuracy (for trading)
            if len(labels) > 1:
                direction_correct = np.sum(np.sign(predictions[1:] - predictions[:-1]) == np.sign(labels[1:] - labels[:-1]))
                direction_accuracy = direction_correct / (len(labels) - 1)
            else:
                direction_accuracy = 0.0

            metrics = {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "r2_score": float(r2),
                "direction_accuracy": float(direction_accuracy)
            }

            logger.info("tcn_evaluation_completed", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("tcn_evaluation_failed", error=str(e))
            raise

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: File path to save model

        Raises:
            ValueError: If model not trained
        """
        try:
            if not self.is_trained:
                raise ValueError("Cannot save untrained model")

            # Create directory if needed
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Save model state and config
            save_dict = {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config,
                "num_channels": self.num_channels,
                "kernel_size": self.kernel_size,
                "dropout": self.dropout,
                "input_size": self.model.network[0].conv1.in_channels,
                "output_size": self.output_size,
                "is_trained": self.is_trained
            }

            torch.save(save_dict, path)

            logger.info("tcn_model_saved", path=path)

        except Exception as e:
            logger.error("tcn_save_failed", path=path, error=str(e))
            raise

    def load(self, path: str) -> None:
        """
        Load model from disk.

        Args:
            path: File path to load model from

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        try:
            if not Path(path).exists():
                raise FileNotFoundError(f"Model file not found: {path}")

            checkpoint = torch.load(path, map_location=self.device)

            # Restore configuration
            self.num_channels = checkpoint["num_channels"]
            self.kernel_size = checkpoint["kernel_size"]
            self.dropout = checkpoint["dropout"]
            self.output_size = checkpoint["output_size"]

            # Recreate model
            self._initialize_model(checkpoint["input_size"])

            # Load states
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.is_trained = checkpoint["is_trained"]

            logger.info("tcn_model_loaded", path=path)

        except Exception as e:
            logger.error("tcn_load_failed", path=path, error=str(e))
            raise
