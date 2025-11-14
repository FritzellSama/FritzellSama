"""
PatchTST (Patch Time Series Transformer) Model.

This module implements the PatchTST architecture for time series forecasting,
which uses patch-based attention mechanism for efficient processing.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from structlog import get_logger

logger = get_logger(__name__)


class PatchEmbedding(nn.Module):
    """Patch embedding layer for time series.

    Converts time series into patches and embeds them.

    Attributes:
        patch_len: Length of each patch
        stride: Stride between patches
        d_model: Embedding dimension
    """

    def __init__(
        self,
        patch_len: int,
        stride: int,
        d_model: int,
        input_channels: int
    ) -> None:
        """Initialize patch embedding.

        Args:
            patch_len: Length of each patch
            stride: Stride between patches
            d_model: Embedding dimension
            input_channels: Number of input channels
        """
        super(PatchEmbedding, self).__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        # Linear projection from patch to embedding
        self.projection = nn.Linear(patch_len * input_channels, d_model)

    def forward(self, x: Tensor) -> Tuple[Tensor, int]:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, input_channels)

        Returns:
            Tuple of (embedded_patches, num_patches)
        """
        batch_size, seq_len, input_channels = x.shape

        # Calculate number of patches
        num_patches = (seq_len - self.patch_len) // self.stride + 1

        # Extract patches using unfold
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # Shape: (batch_size, num_patches, input_channels, patch_len)

        # Reshape for projection
        patches = patches.reshape(batch_size, num_patches, -1)
        # Shape: (batch_size, num_patches, patch_len * input_channels)

        # Project to embedding dimension
        embedded = self.projection(patches)
        # Shape: (batch_size, num_patches, d_model)

        return embedded, num_patches


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer.

    Attributes:
        d_model: Model dimension
        max_len: Maximum sequence length
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class PatchTST(nn.Module):
    """PatchTST model for time series forecasting.

    Implements patch-based transformer for efficient time series processing.

    Attributes:
        config: Configuration dictionary
        patch_len: Patch length
        stride: Patch stride
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        forecast_horizon: Forecasting horizon
    """

    def __init__(
        self,
        config: Dict[str, Any],
        input_channels: int,
        forecast_horizon: int
    ) -> None:
        """Initialize PatchTST model.

        Args:
            config: Configuration dictionary containing:
                - models.patchtst.patch_len
                - models.patchtst.stride
                - models.patchtst.d_model
                - models.patchtst.nhead
                - models.patchtst.num_layers
                - models.patchtst.dim_feedforward
                - models.patchtst.dropout
            input_channels: Number of input features
            forecast_horizon: Number of steps to forecast
        """
        super(PatchTST, self).__init__()

        self.config = config
        self.input_channels = input_channels
        self.forecast_horizon = forecast_horizon
        self._validate_config()

        model_config = self.config["models"]["patchtst"]
        self.patch_len = model_config["patch_len"]
        self.stride = model_config["stride"]
        self.d_model = model_config["d_model"]
        self.nhead = model_config["nhead"]
        self.num_layers = model_config["num_layers"]
        self.dim_feedforward = model_config["dim_feedforward"]
        self.dropout = model_config["dropout"]

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            self.patch_len,
            self.stride,
            self.d_model,
            input_channels
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim_feedforward, forecast_horizon * input_channels)
        )

        logger.info("patchtst_initialized",
                   patch_len=self.patch_len,
                   d_model=self.d_model,
                   nhead=self.nhead,
                   num_layers=self.num_layers)

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config parameters are missing
        """
        required_keys = [
            "models.patchtst.patch_len",
            "models.patchtst.stride",
            "models.patchtst.d_model",
            "models.patchtst.nhead",
            "models.patchtst.num_layers",
            "models.patchtst.dim_feedforward",
            "models.patchtst.dropout"
        ]

        for key in required_keys:
            parts = key.split(".")
            current = self.config
            for part in parts:
                if part not in current:
                    raise ValueError(f"Missing required config key: {key}")
                current = current[part]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, input_channels)

        Returns:
            Predictions (batch_size, forecast_horizon, input_channels)
        """
        batch_size = x.size(0)

        # Embed patches
        embedded, num_patches = self.patch_embedding(x)
        # Shape: (batch_size, num_patches, d_model)

        # Add positional encoding
        embedded = self.pos_encoder(embedded)

        # Transformer encoding
        encoded = self.transformer_encoder(embedded)
        # Shape: (batch_size, num_patches, d_model)

        # Global average pooling over patches
        pooled = encoded.mean(dim=1)
        # Shape: (batch_size, d_model)

        # Generate predictions
        predictions = self.prediction_head(pooled)
        # Shape: (batch_size, forecast_horizon * input_channels)

        # Reshape to (batch_size, forecast_horizon, input_channels)
        predictions = predictions.view(batch_size, self.forecast_horizon, self.input_channels)

        return predictions


class PatchTSTTrader:
    """Production-ready PatchTST trading model wrapper.

    Wrapper class for training and deploying PatchTST models
    for trading applications.

    Attributes:
        config: Configuration dictionary
        model: PatchTST model
        device: PyTorch device
        optimizer: Optimizer
        criterion: Loss function
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize PatchTST trader.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._validate_config()

        self.device = torch.device(
            self.config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model: Optional[PatchTST] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.scaler: Optional[Dict[str, Any]] = None

        logger.info("patchtst_trader_initialized", device=str(self.device))

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If required config is missing
        """
        required_keys = [
            "models.patchtst",
            "training"
        ]

        for key in required_keys:
            parts = key.split(".")
            current = self.config
            for part in parts:
                if part not in current:
                    raise ValueError(f"Missing required config key: {key}")
                current = current[part]

    def initialize_model(
        self,
        input_channels: int,
        forecast_horizon: int
    ) -> None:
        """Initialize PatchTST model.

        Args:
            input_channels: Number of input features
            forecast_horizon: Forecast horizon
        """
        self.model = PatchTST(
            self.config,
            input_channels,
            forecast_horizon
        ).to(self.device)

        # Initialize optimizer
        lr = float(self.config["training"]["learning_rate"])
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=float(self.config["training"].get("weight_decay", 0.01))
        )

        # Initialize loss function
        self.criterion = nn.MSELoss()

        # Initialize scaler for normalization
        self.scaler = {
            "mean": None,
            "std": None
        }

        logger.info("model_initialized",
                   input_channels=input_channels,
                   forecast_horizon=forecast_horizon,
                   parameters=sum(p.numel() for p in self.model.parameters()))

    def _normalize_data(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize data using z-score normalization.

        Args:
            data: Data to normalize
            fit: Whether to fit scaler

        Returns:
            Normalized data
        """
        if fit or self.scaler["mean"] is None:
            self.scaler["mean"] = np.mean(data, axis=(0, 1), keepdims=True)
            self.scaler["std"] = np.std(data, axis=(0, 1), keepdims=True) + 1e-8

        return (data - self.scaler["mean"]) / self.scaler["std"]

    def _denormalize_data(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data.

        Args:
            data: Normalized data

        Returns:
            Original scale data
        """
        if self.scaler["mean"] is None:
            return data

        return data * self.scaler["std"] + self.scaler["mean"]

    async def train_step(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        fit_scaler: bool = False
    ) -> Decimal:
        """Perform single training step.

        Args:
            x_batch: Input batch (batch_size, seq_len, features)
            y_batch: Target batch (batch_size, horizon, features)
            fit_scaler: Whether to fit the data scaler

        Returns:
            Loss value

        Raises:
            RuntimeError: If model not initialized
        """
        if self.model is None or self.optimizer is None or self.criterion is None:
            raise RuntimeError("Model not initialized. Call initialize_model first.")

        self.model.train()

        try:
            # Normalize data
            x_normalized = self._normalize_data(x_batch, fit=fit_scaler)
            y_normalized = self._normalize_data(y_batch, fit=False)

            # Convert to tensors
            x_tensor = torch.FloatTensor(x_normalized).to(self.device)
            y_tensor = torch.FloatTensor(y_normalized).to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(x_tensor)

            # Calculate loss
            loss = self.criterion(predictions, y_tensor)

            # Backward pass
            loss.backward()

            # Gradient clipping
            max_grad_norm = self.config["training"].get("max_grad_norm", 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            self.optimizer.step()

            return Decimal(str(loss.item()))

        except Exception as e:
            logger.error("training_step_failed", error=str(e))
            raise

    async def predict(
        self,
        x: np.ndarray,
        return_normalized: bool = False
    ) -> np.ndarray:
        """Make predictions.

        Args:
            x: Input data (batch_size, seq_len, features)
            return_normalized: Whether to return normalized predictions

        Returns:
            Predictions (batch_size, horizon, features)

        Raises:
            RuntimeError: If model not initialized
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        self.model.eval()

        try:
            # Normalize input
            x_normalized = self._normalize_data(x, fit=False)

            with torch.no_grad():
                x_tensor = torch.FloatTensor(x_normalized).to(self.device)
                predictions = self.model(x_tensor)
                predictions_np = predictions.cpu().numpy()

            # Denormalize if requested
            if not return_normalized and self.scaler["mean"] is not None:
                predictions_np = self._denormalize_data(predictions_np)

            return predictions_np

        except Exception as e:
            logger.error("prediction_failed", error=str(e))
            raise

    async def evaluate(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Decimal]:
        """Evaluate model performance.

        Args:
            x_test: Test input data
            y_test: Test target data

        Returns:
            Dictionary of evaluation metrics

        Raises:
            RuntimeError: If model not initialized
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        try:
            predictions = await self.predict(x_test, return_normalized=False)

            # Calculate metrics
            mse = np.mean((predictions - y_test) ** 2)
            mae = np.mean(np.abs(predictions - y_test))
            rmse = np.sqrt(mse)

            # Calculate MAPE (avoid division by zero)
            mask = y_test != 0
            mape = np.mean(np.abs((y_test[mask] - predictions[mask]) / y_test[mask])) * 100

            metrics = {
                "mse": Decimal(str(mse)),
                "mae": Decimal(str(mae)),
                "rmse": Decimal(str(rmse)),
                "mape": Decimal(str(mape))
            }

            logger.info("model_evaluated",
                       metrics={k: str(v) for k, v in metrics.items()})

            return metrics

        except Exception as e:
            logger.error("evaluation_failed", error=str(e))
            raise

    def save_model(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Save path

        Raises:
            RuntimeError: If model not initialized
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'config': self.config
        }, path)

        logger.info("model_saved", path=path)

    def load_model(
        self,
        path: str,
        input_channels: int,
        forecast_horizon: int
    ) -> None:
        """Load model from disk.

        Args:
            path: Model path
            input_channels: Input channels
            forecast_horizon: Forecast horizon

        Raises:
            FileNotFoundError: If model file not found
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.initialize_model(input_channels, forecast_horizon)

        if self.model is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint.get('scaler', {"mean": None, "std": None})

        logger.info("model_loaded", path=path)
