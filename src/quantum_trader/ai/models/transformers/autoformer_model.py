"""Autoformer transformer model for time series forecasting.

This module implements the Autoformer architecture, a state-of-the-art transformer
model designed specifically for long-term time series forecasting. It features
series decomposition and auto-correlation mechanisms for improved performance.

The Autoformer is particularly well-suited for:
- Long-term price prediction
- Multi-horizon forecasting
- Capturing seasonal patterns
- Handling complex temporal dependencies

Reference:
    Wu, H., Xu, J., Wang, J., & Long, M. (2021).
    Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting.
    NeurIPS 2021.

Example:
    ```python
    from quantum_trader.ai.models.transformers.autoformer_model import AutoformerModel
    import numpy as np

    config = {
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 2048,
        "dropout": 0.1,
        "seq_len": 96,
        "label_len": 48,
        "pred_len": 24,
        "learning_rate": 0.0001,
        "batch_size": 32,
        "epochs": 100,
    }

    model = AutoformerModel(config)
    model.train(train_features, train_labels)
    predictions = model.predict(test_features)
    ```
"""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel

logger = get_logger(__name__)


class MovingAvgDecomposition(nn.Module):
    """Moving average decomposition block.

    Decomposes time series into trend and seasonal components using
    moving average filters.
    """

    def __init__(self, kernel_size: int) -> None:
        """Initialize decomposition block.

        Args:
            kernel_size: Size of moving average kernel
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose input into trend and seasonal components.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tuple of (seasonal, trend) tensors
        """
        # Padding on both ends
        num_pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, num_pad, 1)
        end = x[:, -1:, :].repeat(1, num_pad, 1)
        x_padded = torch.cat([front, x, end], dim=1)

        # Apply moving average
        x_trend = self.avg(x_padded.permute(0, 2, 1))
        x_trend = x_trend.permute(0, 2, 1)

        # Extract seasonal component
        x_seasonal = x - x_trend

        return x_seasonal, x_trend


class AutoCorrelation(nn.Module):
    """Auto-correlation mechanism for time series.

    Replaces standard attention with auto-correlation to better capture
    temporal dependencies in time series data.
    """

    def __init__(
        self,
        attention_dropout: float = 0.1,
        output_attention: bool = False
    ) -> None:
        """Initialize auto-correlation mechanism.

        Args:
            attention_dropout: Dropout rate for attention weights
            output_attention: Whether to output attention weights
        """
        super().__init__()
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg(
        self,
        values: torch.Tensor,
        corr: torch.Tensor
    ) -> torch.Tensor:
        """Time delay aggregation.

        Args:
            values: Value tensor
            corr: Correlation tensor

        Returns:
            Aggregated tensor
        """
        batch_size, head, seq_len, d_k = values.shape
        top_k = int(seq_len * 0.25)  # Use top 25% correlations

        # Find top-k delays
        mean_value = torch.mean(corr, dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)

        # Normalize weights
        tmp_corr = torch.softmax(weights, dim=-1)

        # Aggregate with time delays
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(values, shifts=-int(index[i]), dims=2)
            delays_agg = delays_agg + pattern * \
                (tmp_corr[:, i].unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(
                    1, head, seq_len, d_k))

        return delays_agg

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of auto-correlation.

        Args:
            queries: Query tensor
            keys: Key tensor
            values: Value tensor

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = queries.shape
        _, S, _ = keys.shape

        if seq_len > S:
            zeros = torch.zeros_like(queries[:, :(seq_len - S), :]).float()
            keys = torch.cat([keys, zeros], dim=1)
            values = torch.cat([values, zeros], dim=1)
        else:
            keys = keys[:, :seq_len, :]
            values = values[:, :seq_len, :]

        # Compute auto-correlation using FFT
        q_fft = torch.fft.rfft(queries.contiguous(), dim=1)
        k_fft = torch.fft.rfft(keys.contiguous(), dim=1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=1)

        # Time delay aggregation
        V = self.time_delay_agg(values.unsqueeze(1), corr.unsqueeze(1))
        V = V.squeeze(1)

        if self.output_attention:
            return V, corr
        else:
            return V, None


class AutoformerLayer(nn.Module):
    """Single Autoformer encoder/decoder layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        moving_avg: int
    ) -> None:
        """Initialize Autoformer layer.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            moving_avg: Moving average window size
        """
        super().__init__()
        self.attention = AutoCorrelation(attention_dropout=dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.decomp1 = MovingAvgDecomposition(moving_avg)
        self.decomp2 = MovingAvgDecomposition(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through layer.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Auto-correlation
        new_x, _ = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)

        # Feed-forward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)

        return res


class AutoformerModel(BaseMLModel):
    """Autoformer model for time series forecasting.

    Production-ready implementation of the Autoformer architecture with:
    - Series decomposition
    - Auto-correlation mechanism
    - Multi-horizon forecasting
    - Configurable architecture
    - GPU acceleration support

    Attributes:
        model: PyTorch model
        device: Computation device (CPU/CUDA)
        optimizer: Training optimizer
        criterion: Loss function
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Autoformer model.

        Args:
            config: Configuration dictionary with required keys:
                - d_model: Model dimension (default: 512)
                - n_heads: Number of attention heads (default: 8)
                - e_layers: Number of encoder layers (default: 2)
                - d_layers: Number of decoder layers (default: 1)
                - d_ff: Feed-forward dimension (default: 2048)
                - dropout: Dropout rate (default: 0.1)
                - moving_avg: Moving average window (default: 25)
                - seq_len: Input sequence length (default: 96)
                - label_len: Decoder start sequence length (default: 48)
                - pred_len: Prediction length (default: 24)
                - learning_rate: Learning rate (default: 0.0001)
                - batch_size: Batch size (default: 32)
                - epochs: Training epochs (default: 100)
                - device: Device ('cuda' or 'cpu', default: auto)
        """
        super().__init__(config)

        # Model architecture parameters
        self.d_model = int(config.get("d_model", 512))
        self.n_heads = int(config.get("n_heads", 8))
        self.e_layers = int(config.get("e_layers", 2))
        self.d_layers = int(config.get("d_layers", 1))
        self.d_ff = int(config.get("d_ff", 2048))
        self.dropout = float(config.get("dropout", 0.1))
        self.moving_avg = int(config.get("moving_avg", 25))

        # Sequence parameters
        self.seq_len = int(config.get("seq_len", 96))
        self.label_len = int(config.get("label_len", 48))
        self.pred_len = int(config.get("pred_len", 24))

        # Training parameters
        self.learning_rate = float(config.get("learning_rate", 0.0001))
        self.batch_size = int(config.get("batch_size", 32))
        self.epochs = int(config.get("epochs", 100))

        # Device configuration
        device_config = config.get("device", "auto")
        if device_config == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_config)

        # Initialize model components
        self._build_model()

        logger.info(
            "Autoformer model initialized",
            d_model=self.d_model,
            n_heads=self.n_heads,
            device=str(self.device),
            parameters=sum(p.numel() for p in self.model.parameters())
        )

    def _build_model(self) -> None:
        """Build the Autoformer neural network architecture."""
        # Input embedding
        self.enc_embedding = nn.Linear(1, self.d_model)
        self.dec_embedding = nn.Linear(1, self.d_model)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            AutoformerLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                moving_avg=self.moving_avg
            )
            for _ in range(self.e_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            AutoformerLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                moving_avg=self.moving_avg
            )
            for _ in range(self.d_layers)
        ])

        # Output projection
        self.projection = nn.Linear(self.d_model, 1)

        # Decomposition for trend initialization
        self.decomp = MovingAvgDecomposition(self.moving_avg)

        # Create complete model
        self.model = nn.ModuleDict({
            'enc_embedding': self.enc_embedding,
            'dec_embedding': self.dec_embedding,
            'encoder_layers': self.encoder_layers,
            'decoder_layers': self.decoder_layers,
            'projection': self.projection,
            'decomp': self.decomp,
        })

        self.model.to(self.device)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        self.criterion = nn.MSELoss()

    def _prepare_data(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> DataLoader:
        """Prepare data for training/prediction.

        Args:
            features: Input features
            labels: Optional labels

        Returns:
            DataLoader instance
        """
        # Reshape if needed
        if len(features.shape) == 2:
            features = features.reshape(features.shape[0], features.shape[1], 1)

        features_tensor = torch.FloatTensor(features).to(self.device)

        if labels is not None:
            if len(labels.shape) == 1:
                labels = labels.reshape(-1, 1, 1)
            elif len(labels.shape) == 2:
                labels = labels.reshape(labels.shape[0], labels.shape[1], 1)

            labels_tensor = torch.FloatTensor(labels).to(self.device)
            dataset = TensorDataset(features_tensor, labels_tensor)
        else:
            dataset = TensorDataset(features_tensor)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(labels is not None),
            drop_last=False
        )

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the Autoformer model.

        Args:
            features: Input time series data of shape (n_samples, seq_len, n_features)
            labels: Target values of shape (n_samples, pred_len)

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If training fails
        """
        try:
            logger.info("Starting Autoformer training", epochs=self.epochs)

            # Validate inputs
            self.validate_input(features, labels)

            # Prepare data
            train_loader = self._prepare_data(features, labels)

            # Training loop
            self.model.train()
            training_losses = []

            for epoch in range(self.epochs):
                epoch_loss = 0.0
                batch_count = 0

                for batch in train_loader:
                    if len(batch) == 2:
                        batch_x, batch_y = batch
                    else:
                        batch_x = batch[0]
                        continue

                    self.optimizer.zero_grad()

                    # Forward pass
                    # Encoder
                    enc_out = self.enc_embedding(batch_x)
                    for layer in self.encoder_layers:
                        enc_out = layer(enc_out)

                    # Decoder (using label_len for teacher forcing)
                    dec_inp = torch.zeros(
                        batch_x.shape[0],
                        self.pred_len,
                        batch_x.shape[-1]
                    ).to(self.device)

                    dec_inp = torch.cat([
                        batch_x[:, -self.label_len:, :],
                        dec_inp
                    ], dim=1)

                    dec_out = self.dec_embedding(dec_inp)
                    for layer in self.decoder_layers:
                        dec_out = layer(dec_out)

                    # Output projection
                    outputs = self.projection(dec_out[:, -self.pred_len:, :])

                    # Compute loss
                    loss = self.criterion(outputs, batch_y)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    batch_count += 1

                avg_loss = epoch_loss / max(batch_count, 1)
                training_losses.append(avg_loss)

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        "Training progress",
                        epoch=epoch + 1,
                        loss=f"{avg_loss:.6f}"
                    )

            # Update training status
            self.is_trained = True
            self.last_trained = datetime.utcnow()
            self.training_history.append({
                "timestamp": self.last_trained.isoformat(),
                "epochs": self.epochs,
                "final_loss": training_losses[-1],
                "losses": training_losses,
            })

            logger.info(
                "Training completed",
                final_loss=f"{training_losses[-1]:.6f}",
                epochs=self.epochs
            )

        except Exception as e:
            logger.error("Training failed", error=str(e))
            raise RuntimeError(f"Training failed: {e}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions using the trained model.

        Args:
            features: Input time series data of shape (n_samples, seq_len, n_features)

        Returns:
            Predictions of shape (n_samples, pred_len)

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If model is not trained or prediction fails
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        try:
            logger.debug("Generating predictions", samples=features.shape[0])

            # Validate inputs
            self.validate_input(features)

            # Prepare data
            test_loader = self._prepare_data(features)

            # Prediction
            self.model.eval()
            predictions = []

            with torch.no_grad():
                for batch in test_loader:
                    batch_x = batch[0]

                    # Encoder
                    enc_out = self.enc_embedding(batch_x)
                    for layer in self.encoder_layers:
                        enc_out = layer(enc_out)

                    # Decoder
                    dec_inp = torch.zeros(
                        batch_x.shape[0],
                        self.pred_len,
                        batch_x.shape[-1]
                    ).to(self.device)

                    dec_inp = torch.cat([
                        batch_x[:, -self.label_len:, :],
                        dec_inp
                    ], dim=1)

                    dec_out = self.dec_embedding(dec_inp)
                    for layer in self.decoder_layers:
                        dec_out = layer(dec_out)

                    # Output projection
                    outputs = self.projection(dec_out[:, -self.pred_len:, :])

                    predictions.append(outputs.cpu().numpy())

            predictions = np.concatenate(predictions, axis=0)
            predictions = predictions.reshape(predictions.shape[0], -1)

            logger.debug("Predictions generated", shape=predictions.shape)

            return predictions

        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            raise RuntimeError(f"Prediction failed: {e}")

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            features: Input features
            labels: True labels

        Returns:
            Dictionary of evaluation metrics

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If evaluation fails
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        try:
            logger.debug("Evaluating model")

            # Generate predictions
            predictions = self.predict(features)

            # Flatten if needed
            if len(labels.shape) > 1:
                labels_flat = labels.reshape(-1)
            else:
                labels_flat = labels

            predictions_flat = predictions.reshape(-1)

            # Calculate metrics
            mse = float(np.mean((predictions_flat - labels_flat) ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(predictions_flat - labels_flat)))

            # Avoid division by zero
            labels_std = np.std(labels_flat)
            if labels_std > 0:
                mape = float(np.mean(np.abs(
                    (labels_flat - predictions_flat) / (labels_flat + 1e-8)
                )) * 100)
            else:
                mape = float('inf')

            metrics = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
            }

            logger.info("Evaluation completed", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("Evaluation failed", error=str(e))
            raise RuntimeError(f"Evaluation failed: {e}")

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path for saving

        Raises:
            IOError: If save fails
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save model state
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
            }, save_path)

            # Save metadata
            self.save_metadata(str(save_path))

            logger.info("Model saved successfully", path=str(save_path))

        except Exception as e:
            logger.error("Failed to save model", error=str(e))
            raise IOError(f"Failed to save model: {e}")

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: File path for loading

        Raises:
            IOError: If load fails
        """
        try:
            load_path = Path(path)

            if not load_path.exists():
                raise FileNotFoundError(f"Model file not found: {load_path}")

            # Load checkpoint
            checkpoint = torch.load(load_path, map_location=self.device)

            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load metadata
            self.load_metadata(str(load_path))

            logger.info("Model loaded successfully", path=str(load_path))

        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise IOError(f"Failed to load model: {e}")
