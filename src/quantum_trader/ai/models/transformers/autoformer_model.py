"""
Autoformer Model for Long-term Time Series Forecasting.

Implements Autoformer with decomposition architecture for accurate
long-term financial time series prediction.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class AutoformerConfig:
    """Configuration for Autoformer model."""

    input_dim: int
    output_dim: int = 1
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    moving_avg: int = 25
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 24
    dropout: Decimal = Decimal("0.05")
    activation: str = "gelu"


class SeriesDecomposition(nn.Module):
    """Series decomposition block for trend and seasonal components."""

    def __init__(self, kernel_size: int):
        """
        Initialize series decomposition.

        Args:
            kernel_size: Moving average kernel size
        """
        super(SeriesDecomposition, self).__init__()

        self.moving_avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose series into trend and seasonal components.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Tuple of (trend, seasonal)
        """
        # Permute for conv operation
        x_permuted = x.permute(0, 2, 1)

        # Extract trend using moving average
        trend = self.moving_avg(x_permuted)
        trend = trend.permute(0, 2, 1)

        # Seasonal component is residual
        seasonal = x - trend

        return trend, seasonal


class AutoCorrelation(nn.Module):
    """Auto-correlation mechanism for time series."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        factor: int = 1
    ):
        """
        Initialize auto-correlation.

        Args:
            d_model: Model dimension
            n_heads: Number of heads
            factor: Factor for top-k selection
        """
        super(AutoCorrelation, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor

        self.d_keys = d_model // n_heads
        self.d_values = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def time_delay_agg_training(
        self,
        values: torch.Tensor,
        corr: torch.Tensor
    ) -> torch.Tensor:
        """
        Time delay aggregation for training.

        Args:
            values: Value tensor
            corr: Correlation tensor

        Returns:
            Aggregated tensor
        """
        batch, head, channel, length = values.shape

        # Find top-k correlations
        top_k = int(self.factor * torch.log(torch.tensor(length, dtype=torch.float32)).item())
        top_k = max(1, min(top_k, length))

        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]

        weights = torch.stack([mean_value[:, idx] for idx in index], dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)

        # Time delay aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()

        for i, idx in enumerate(index):
            pattern = torch.roll(tmp_values, -int(idx), dims=-1)
            delays_agg += pattern[..., :length] * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1)

        return delays_agg

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through auto-correlation.

        Args:
            queries: Query tensor
            keys: Key tensor
            values: Value tensor

        Returns:
            Tuple of (output, attention)
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Compute auto-correlation using FFT
        queries = queries.permute(0, 2, 3, 1)
        keys = keys.permute(0, 2, 3, 1)
        values = values.permute(0, 2, 3, 1)

        # FFT-based auto-correlation
        q_fft = torch.fft.rfft(queries, dim=-1)
        k_fft = torch.fft.rfft(keys, dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # Time delay aggregation
        V = self.time_delay_agg_training(values, corr)

        # Reshape and project
        V = V.permute(0, 3, 1, 2).contiguous()
        V = V.view(B, L, -1)
        out = self.out_projection(V)

        return out, corr


class EncoderLayer(nn.Module):
    """Autoformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        moving_avg: int,
        dropout: float,
        activation: str = "gelu"
    ):
        """
        Initialize encoder layer.

        Args:
            d_model: Model dimension
            n_heads: Number of heads
            d_ff: Feed-forward dimension
            moving_avg: Moving average window
            dropout: Dropout rate
            activation: Activation function
        """
        super(EncoderLayer, self).__init__()

        self.attention = AutoCorrelation(d_model, n_heads)
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)

        self.dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)

        if activation == "gelu":
            self.activation = F.gelu
        else:
            self.activation = F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder layer.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Auto-correlation
        new_x, attn = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)

        # Feed-forward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        res, _ = self.decomp2(x + y)

        return res


class DecoderLayer(nn.Module):
    """Autoformer decoder layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        moving_avg: int,
        dropout: float,
        activation: str = "gelu"
    ):
        """
        Initialize decoder layer.

        Args:
            d_model: Model dimension
            n_heads: Number of heads
            d_ff: Feed-forward dimension
            moving_avg: Moving average window
            dropout: Dropout rate
            activation: Activation function
        """
        super(DecoderLayer, self).__init__()

        self.self_attention = AutoCorrelation(d_model, n_heads)
        self.cross_attention = AutoCorrelation(d_model, n_heads)

        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.decomp3 = SeriesDecomposition(moving_avg)

        self.dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)

        if activation == "gelu":
            self.activation = F.gelu
        else:
            self.activation = F.relu

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through decoder layer.

        Args:
            x: Input tensor
            cross: Cross-attention tensor from encoder

        Returns:
            Tuple of (seasonal output, trend output)
        """
        # Self-attention
        x_sa, _ = self.self_attention(x, x, x)
        x = x + self.dropout(x_sa)
        x, trend1 = self.decomp1(x)

        # Cross-attention
        x_ca, _ = self.cross_attention(x, cross, cross)
        x = x + self.dropout(x_ca)
        x, trend2 = self.decomp2(x)

        # Feed-forward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3

        return x, residual_trend


class AutoformerModel(nn.Module):
    """Complete Autoformer model."""

    def __init__(self, config: AutoformerConfig):
        """
        Initialize Autoformer.

        Args:
            config: Model configuration
        """
        super(AutoformerModel, self).__init__()

        self.config = config

        # Input embedding
        self.enc_embedding = nn.Linear(config.input_dim, config.d_model)

        # Decomposition
        self.decomp = SeriesDecomposition(config.moving_avg)

        # Encoder
        dropout = float(config.dropout)

        self.encoder = nn.ModuleList([
            EncoderLayer(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.moving_avg,
                dropout,
                config.activation
            )
            for _ in range(config.e_layers)
        ])

        # Decoder
        self.dec_embedding = nn.Linear(config.input_dim, config.d_model)

        self.decoder = nn.ModuleList([
            DecoderLayer(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.moving_avg,
                dropout,
                config.activation
            )
            for _ in range(config.d_layers)
        ])

        # Projection
        self.projection = nn.Linear(config.d_model, config.output_dim)

        logger.info(
            "Initialized Autoformer model",
            extra={
                "input_dim": config.input_dim,
                "output_dim": config.output_dim,
                "d_model": config.d_model,
                "e_layers": config.e_layers,
                "d_layers": config.d_layers,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_dec: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through Autoformer.

        Args:
            x_enc: Encoder input
            x_dec: Decoder input

        Returns:
            Predictions
        """
        # Encoder
        enc_out = self.enc_embedding(x_enc)

        for layer in self.encoder:
            enc_out = layer(enc_out)

        # Decoder
        seasonal_init, trend_init = self.decomp(x_dec)
        trend_init = torch.mean(trend_init, dim=1, keepdim=True).repeat(
            1, self.config.pred_len, 1
        )

        dec_out = self.dec_embedding(seasonal_init)

        seasonal_part = dec_out
        trend_part = trend_init

        for layer in self.decoder:
            seasonal_part, trend_layer = layer(seasonal_part, enc_out)
            trend_part = trend_part + trend_layer

        # Final prediction (seasonal + trend)
        dec_out = seasonal_part + trend_part

        # Project to output dimension
        output = self.projection(dec_out)

        return output[:, -self.config.pred_len :, :]

    def predict(self, x_enc: torch.Tensor, x_dec: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.

        Args:
            x_enc: Encoder input
            x_dec: Decoder input

        Returns:
            Predictions
        """
        self.eval()

        with torch.no_grad():
            output = self.forward(x_enc, x_dec)

        return output


class AutoformerTrainer:
    """Trainer for Autoformer model."""

    def __init__(
        self,
        model: AutoformerModel,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Autoformer model
            device: Computing device
        """
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        logger.info(
            "Initialized Autoformer trainer",
            extra={
                "device": str(self.device),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    async def train_epoch(
        self,
        train_df: pl.DataFrame,
        feature_cols: List[str],
        label_col: str,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        batch_size: int = 32
    ) -> Dict[str, Decimal]:
        """
        Train for one epoch.

        Args:
            train_df: Training data
            feature_cols: Feature column names
            label_col: Label column name
            optimizer: Optimizer
            criterion: Loss criterion
            batch_size: Batch size

        Returns:
            Training metrics
        """
        self.model.train()

        features = train_df.select(feature_cols).to_numpy()
        labels = train_df.select(label_col).to_numpy()

        config = self.model.config
        total_loss = Decimal("0")
        num_batches = 0

        # Create sequences
        for i in range(0, len(features) - config.seq_len - config.pred_len, batch_size):
            batch_x_enc = []
            batch_x_dec = []
            batch_y = []

            for j in range(i, min(i + batch_size, len(features) - config.seq_len - config.pred_len)):
                # Encoder input
                x_enc = features[j : j + config.seq_len]

                # Decoder input (label_len from encoder + zeros for prediction)
                x_dec_label = features[j + config.seq_len - config.label_len : j + config.seq_len]
                x_dec_pred = torch.zeros((config.pred_len, len(feature_cols)))
                x_dec = torch.cat([
                    torch.tensor(x_dec_label, dtype=torch.float32),
                    x_dec_pred
                ], dim=0)

                # Target
                y = labels[j + config.seq_len : j + config.seq_len + config.pred_len]

                batch_x_enc.append(torch.tensor(x_enc, dtype=torch.float32))
                batch_x_dec.append(x_dec)
                batch_y.append(torch.tensor(y, dtype=torch.float32))

            if not batch_x_enc:
                continue

            X_enc = torch.stack(batch_x_enc).to(self.device)
            X_dec = torch.stack(batch_x_dec).to(self.device)
            Y = torch.stack(batch_y).to(self.device)

            optimizer.zero_grad()

            # Forward pass
            predictions = self.model(X_enc, X_dec)

            # Compute loss
            loss = criterion(predictions, Y)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += Decimal(str(loss.item()))
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else Decimal("0")

        return {
            "avg_loss": avg_loss,
            "num_batches": Decimal(str(num_batches))
        }

    async def evaluate(
        self,
        eval_df: pl.DataFrame,
        feature_cols: List[str],
        label_col: str,
        criterion: nn.Module
    ) -> Dict[str, Decimal]:
        """
        Evaluate model.

        Args:
            eval_df: Evaluation data
            feature_cols: Feature column names
            label_col: Label column name
            criterion: Loss criterion

        Returns:
            Evaluation metrics
        """
        self.model.eval()

        features = eval_df.select(feature_cols).to_numpy()
        labels = eval_df.select(label_col).to_numpy()

        config = self.model.config
        total_loss = Decimal("0")
        num_samples = 0

        with torch.no_grad():
            for i in range(0, len(features) - config.seq_len - config.pred_len):
                # Prepare inputs
                x_enc = torch.tensor(
                    features[i : i + config.seq_len],
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0)

                x_dec_label = features[i + config.seq_len - config.label_len : i + config.seq_len]
                x_dec_pred = torch.zeros((config.pred_len, len(feature_cols)))
                x_dec = torch.cat([
                    torch.tensor(x_dec_label, dtype=torch.float32),
                    x_dec_pred
                ], dim=0).unsqueeze(0).to(self.device)

                y = torch.tensor(
                    labels[i + config.seq_len : i + config.seq_len + config.pred_len],
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0)

                # Predict
                predictions = self.model(x_enc, x_dec)

                # Compute loss
                loss = criterion(predictions, y)

                total_loss += Decimal(str(loss.item()))
                num_samples += 1

        avg_loss = total_loss / num_samples if num_samples > 0 else Decimal("0")

        return {
            "eval_loss": avg_loss,
            "num_samples": Decimal(str(num_samples))
        }
