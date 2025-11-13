"""
Informer Transformer Model for Time Series Forecasting.

This module implements the Informer architecture, designed for efficient
long-sequence time-series forecasting with ProbSparse self-attention.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel
from quantum_trader.exceptions import ModelError, ValidationError

logger = get_logger(__name__)


class ProbAttention(nn.Module):
    """ProbSparse Self-Attention mechanism.

    Implements the efficient attention mechanism from the Informer paper
    that reduces complexity from O(L^2) to O(L log L).
    """

    def __init__(
        self,
        mask_flag: bool,
        factor: int,
        attention_dropout: float,
        output_attention: bool
    ) -> None:
        """Initialize ProbAttention.

        Args:
            mask_flag: Whether to use masking
            factor: Sampling factor for ProbSparse
            attention_dropout: Dropout rate for attention
            output_attention: Whether to output attention weights
        """
        super(ProbAttention, self).__init__()
        self.mask_flag = mask_flag
        self.factor = factor
        self.attention_dropout = attention_dropout
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_qk(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        sample_k: int,
        n_top: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ProbSparse Q-K product.

        Args:
            Q: Query tensor
            K: Key tensor
            sample_k: Number of samples
            n_top: Top-k to select

        Returns:
            Tuple of (Q_reduced, M_top)
        """
        B, H, L_Q, E = Q.shape
        _, _, L_K, _ = K.shape

        # Sample K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]

        # Compute Q * K_sample
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)
        ).squeeze(-2)

        # Find top-k queries
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # Reduce Q
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]

        return Q_reduce, M_top

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            queries: Query tensor [B, L, H, E]
            keys: Key tensor [B, S, H, E]
            values: Value tensor [B, S, H, D]
            attn_mask: Attention mask

        Returns:
            Tuple of (output, attention)
        """
        B, L_Q, H, E = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_qk(queries, keys, sample_k=U_part, n_top=u)

        # Compute attention
        scale = 1. / np.sqrt(E)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)

        # Add scale
        scores_top = scores_top * scale

        # Compute context
        context = torch.matmul(scores_top, values)
        context = context.transpose(2, 1).contiguous()

        return context, None if not self.output_attention else scores_top


class AttentionLayer(nn.Module):
    """Attention layer wrapper."""

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None
    ) -> None:
        """Initialize attention layer.

        Args:
            attention: Attention mechanism
            d_model: Model dimension
            n_heads: Number of attention heads
            d_keys: Key dimension (default: d_model // n_heads)
            d_values: Value dimension (default: d_model // n_heads)
        """
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            queries: Query tensor
            keys: Key tensor
            values: Value tensor
            attn_mask: Attention mask

        Returns:
            Tuple of (output, attention)
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        out = out.view(B, L, -1)
        out = self.out_projection(out)

        return out, attn


class InformerModel(BaseMLModel):
    """Informer Transformer for time series forecasting.

    Implements the Informer architecture with ProbSparse attention
    for efficient long-sequence time-series forecasting.

    Attributes:
        config: Model configuration
        model: PyTorch model
        device: Computation device
        input_size: Input feature dimension
        output_size: Output dimension
        seq_len: Input sequence length
        label_len: Label sequence length
        pred_len: Prediction length

    Example:
        >>> config = {
        ...     "input_size": 10,
        ...     "output_size": 1,
        ...     "d_model": 512,
        ...     "n_heads": 8,
        ...     "e_layers": 2,
        ...     "d_layers": 1,
        ...     "d_ff": 2048,
        ...     "dropout": 0.1,
        ...     "seq_len": 96,
        ...     "label_len": 48,
        ...     "pred_len": 24,
        ...     "learning_rate": 0.0001,
        ...     "batch_size": 32,
        ...     "epochs": 100
        ... }
        >>> model = InformerModel(config)
        >>> model.train(features, labels)
        >>> predictions = model.predict(test_features)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Informer model.

        Args:
            config: Model configuration

        Raises:
            ValidationError: If configuration is invalid
        """
        super().__init__(config)

        self._validate_config()

        self.input_size = config["input_size"]
        self.output_size = config["output_size"]
        self.d_model = config.get("d_model", 512)
        self.n_heads = config.get("n_heads", 8)
        self.e_layers = config.get("e_layers", 2)
        self.d_layers = config.get("d_layers", 1)
        self.d_ff = config.get("d_ff", 2048)
        self.dropout = config.get("dropout", 0.1)
        self.seq_len = config.get("seq_len", 96)
        self.label_len = config.get("label_len", 48)
        self.pred_len = config.get("pred_len", 24)
        self.learning_rate = Decimal(str(config.get("learning_rate", 0.0001)))
        self.batch_size = config.get("batch_size", 32)
        self.epochs = config.get("epochs", 100)
        self.factor = config.get("factor", 5)
        self.output_attention = config.get("output_attention", False)

        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = self._build_model()
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.learning_rate)
        )
        self.criterion = nn.MSELoss()

        self._trained = False

        logger.info(
            "Informer model initialized",
            input_size=self.input_size,
            output_size=self.output_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            device=str(self.device)
        )

    def _validate_config(self) -> None:
        """Validate model configuration.

        Raises:
            ValidationError: If configuration is invalid
        """
        required_fields = ["input_size", "output_size"]
        for field in required_fields:
            if field not in self.config:
                raise ValidationError(f"Missing required config field: {field}")

        if self.config["input_size"] < 1:
            raise ValidationError("input_size must be >= 1")

        if self.config["output_size"] < 1:
            raise ValidationError("output_size must be >= 1")

    def _build_model(self) -> nn.Module:
        """Build Informer model architecture.

        Returns:
            PyTorch model
        """
        # Input embedding
        enc_embedding = nn.Linear(self.input_size, self.d_model)

        # Encoder layers
        encoder_layers = []
        for _ in range(self.e_layers):
            attention = ProbAttention(
                mask_flag=False,
                factor=self.factor,
                attention_dropout=self.dropout,
                output_attention=self.output_attention
            )
            attention_layer = AttentionLayer(
                attention,
                self.d_model,
                self.n_heads
            )
            encoder_layers.append(attention_layer)

        # Decoder layers
        decoder_layers = []
        for _ in range(self.d_layers):
            attention = ProbAttention(
                mask_flag=True,
                factor=self.factor,
                attention_dropout=self.dropout,
                output_attention=self.output_attention
            )
            attention_layer = AttentionLayer(
                attention,
                self.d_model,
                self.n_heads
            )
            decoder_layers.append(attention_layer)

        # Output projection
        projection = nn.Linear(self.d_model, self.output_size)

        # Create model
        model = nn.ModuleDict({
            'enc_embedding': enc_embedding,
            'encoder_layers': nn.ModuleList(encoder_layers),
            'decoder_layers': nn.ModuleList(decoder_layers),
            'projection': projection
        })

        return model

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None
    ) -> None:
        """Train the Informer model.

        Args:
            features: Training features [N, seq_len, input_size]
            labels: Training labels [N, pred_len, output_size]
            val_features: Validation features (optional)
            val_labels: Validation labels (optional)

        Raises:
            ModelError: If training fails
        """
        try:
            logger.info("Starting Informer training", epochs=self.epochs)

            # Convert to tensors
            X_train = torch.FloatTensor(features).to(self.device)
            y_train = torch.FloatTensor(labels).to(self.device)

            if val_features is not None and val_labels is not None:
                X_val = torch.FloatTensor(val_features).to(self.device)
                y_val = torch.FloatTensor(val_labels).to(self.device)
                use_validation = True
            else:
                use_validation = False

            # Training loop
            for epoch in range(self.epochs):
                self.model.train()
                epoch_loss = Decimal("0")

                # Mini-batch training
                num_batches = (len(X_train) + self.batch_size - 1) // self.batch_size

                for i in range(0, len(X_train), self.batch_size):
                    batch_X = X_train[i:i + self.batch_size]
                    batch_y = y_train[i:i + self.batch_size]

                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self._forward(batch_X)
                    loss = self.criterion(outputs, batch_y)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += Decimal(str(loss.item()))

                avg_loss = epoch_loss / Decimal(str(num_batches))

                # Validation
                if use_validation and (epoch + 1) % 10 == 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_outputs = self._forward(X_val)
                        val_loss = self.criterion(val_outputs, y_val)

                    logger.info(
                        "Training progress",
                        epoch=epoch + 1,
                        train_loss=float(avg_loss),
                        val_loss=float(val_loss.item())
                    )
                elif (epoch + 1) % 10 == 0:
                    logger.info(
                        "Training progress",
                        epoch=epoch + 1,
                        train_loss=float(avg_loss)
                    )

            self._trained = True
            logger.info("Training completed successfully")

        except Exception as e:
            logger.error("Training failed", error=str(e))
            raise ModelError(f"Training failed: {e}") from e

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions.

        Args:
            features: Input features [N, seq_len, input_size]

        Returns:
            Predictions [N, pred_len, output_size]

        Raises:
            ModelError: If prediction fails
        """
        try:
            if not self._trained:
                logger.warning("Model not trained, predictions may be unreliable")

            self.model.eval()

            with torch.no_grad():
                X = torch.FloatTensor(features).to(self.device)
                outputs = self._forward(X)
                predictions = outputs.cpu().numpy()

            return predictions

        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            raise ModelError(f"Prediction failed: {e}") from e

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor [B, seq_len, input_size]

        Returns:
            Output tensor [B, pred_len, output_size]
        """
        # Embedding
        enc_out = self.model['enc_embedding'](x)

        # Encoder
        for layer in self.model['encoder_layers']:
            enc_out, _ = layer(enc_out, enc_out, enc_out)

        # Decoder (simplified - using encoder output)
        dec_out = enc_out

        for layer in self.model['decoder_layers']:
            dec_out, _ = layer(dec_out, enc_out, enc_out)

        # Projection
        output = self.model['projection'](dec_out[:, -self.pred_len:, :])

        return output

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            features: Test features
            labels: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            self.model.eval()

            with torch.no_grad():
                X = torch.FloatTensor(features).to(self.device)
                y = torch.FloatTensor(labels).to(self.device)

                outputs = self._forward(X)
                mse = self.criterion(outputs, y).item()

                # Calculate additional metrics
                mae = F.l1_loss(outputs, y).item()
                rmse = np.sqrt(mse)

            metrics = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae
            }

            logger.info("Model evaluation completed", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("Evaluation failed", error=str(e))
            raise ModelError(f"Evaluation failed: {e}") from e

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save model

        Raises:
            ModelError: If save fails
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'trained': self._trained
            }, path)

            logger.info("Model saved", path=path)

        except Exception as e:
            logger.error("Failed to save model", error=str(e))
            raise ModelError(f"Failed to save model: {e}") from e

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to load model from

        Raises:
            ModelError: If load fails
        """
        try:
            if not os.path.exists(path):
                raise ModelError(f"Model file not found: {path}")

            checkpoint = torch.load(path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self._trained = checkpoint.get('trained', False)

            logger.info("Model loaded", path=path)

        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise ModelError(f"Failed to load model: {e}") from e
