"""
Attention-based Neural Network for Time Series Prediction.

Implements multi-head self-attention mechanism for capturing temporal
dependencies in financial time series data.
"""

import logging
import math
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
class AttentionConfig:
    """Configuration for attention model."""

    input_dim: int
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    output_dim: int = 1
    dropout: Decimal = Decimal("0.1")
    max_seq_length: int = 512
    use_positional_encoding: bool = True
    attention_dropout: Decimal = Decimal("0.1")


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence data."""

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, : x.size(1), :]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.

        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            mask: Optional attention mask

        Returns:
            Tuple of (attention output, attention weights)
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Compute attention output
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-head attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Tuple of (output tensor, attention weights)
        """
        batch_size = x.size(0)

        # Linear projections and reshape for multi-head
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        # Final linear projection
        output = self.W_o(attn_output)

        return output, attn_weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize feed-forward network.

        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class AttentionLayer(nn.Module):
    """Single attention layer with residual connections and layer norm."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Initialize attention layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super(AttentionLayer, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through attention layer.

        Args:
            x: Input tensor
            mask: Optional attention mask

        Returns:
            Tuple of (output tensor, attention weights)
        """
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x, attn_weights


class AttentionModel(nn.Module):
    """Complete attention-based model for time series prediction."""

    def __init__(self, config: AttentionConfig):
        """
        Initialize attention model.

        Args:
            config: Model configuration
        """
        super(AttentionModel, self).__init__()

        self.config = config

        # Input embedding
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)

        # Positional encoding
        if config.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(
                config.hidden_dim,
                config.max_seq_length
            )

        # Attention layers
        d_ff = config.hidden_dim * 4
        dropout = float(config.dropout)

        self.layers = nn.ModuleList([
            AttentionLayer(
                config.hidden_dim,
                config.num_heads,
                d_ff,
                dropout
            )
            for _ in range(config.num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.output_dim)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._initialize_weights()

        logger.info(
            "Initialized attention model",
            extra={
                "input_dim": config.input_dim,
                "hidden_dim": config.hidden_dim,
                "num_heads": config.num_heads,
                "num_layers": config.num_layers,
                "output_dim": config.output_dim,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Optional attention mask

        Returns:
            Tuple of (predictions, attention weights from each layer)
        """
        # Project input to hidden dimension
        x = self.input_projection(x)
        x = self.dropout(x)

        # Add positional encoding
        if self.config.use_positional_encoding:
            x = self.pos_encoder(x)

        # Apply attention layers
        attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)

        # Use last timestep for prediction (or mean pooling)
        # Here we use the last timestep
        x = x[:, -1, :]

        # Project to output dimension
        output = self.output_projection(x)

        return output, attention_weights

    def predict(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Make predictions.

        Args:
            x: Input tensor
            return_attention: Whether to return attention weights

        Returns:
            Predictions (and attention weights if requested)
        """
        self.eval()

        with torch.no_grad():
            output, attention_weights = self.forward(x)

            if return_attention:
                return output, attention_weights
            return output

    def get_attention_weights(
        self,
        x: torch.Tensor,
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Get attention weights for visualization.

        Args:
            x: Input tensor
            layer_idx: Specific layer index (None for all layers)

        Returns:
            Attention weights tensor
        """
        self.eval()

        with torch.no_grad():
            _, attention_weights = self.forward(x)

            if layer_idx is not None:
                return attention_weights[layer_idx]
            return torch.stack(attention_weights)


class AttentionTrainer:
    """Trainer for attention model."""

    def __init__(
        self,
        model: AttentionModel,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Attention model to train
            device: Computing device
        """
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        logger.info(
            "Initialized attention trainer",
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

        # Prepare data
        features = train_df.select(feature_cols).to_numpy()
        labels = train_df.select(label_col).to_numpy()

        num_samples = len(features)
        total_loss = Decimal("0")
        num_batches = 0

        # Training loop
        for i in range(0, num_samples, batch_size):
            batch_features = features[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            # Convert to tensors
            X = torch.tensor(
                batch_features,
                dtype=torch.float32,
                device=self.device
            )
            y = torch.tensor(
                batch_labels,
                dtype=torch.float32,
                device=self.device
            )

            # Forward pass
            optimizer.zero_grad()
            predictions, _ = self.model(X)

            # Compute loss
            loss = criterion(predictions.squeeze(), y.squeeze())

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

        X = torch.tensor(features, dtype=torch.float32, device=self.device)
        y = torch.tensor(labels, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            predictions, _ = self.model(X)
            loss = criterion(predictions.squeeze(), y.squeeze())

        return {
            "eval_loss": Decimal(str(loss.item()))
        }
