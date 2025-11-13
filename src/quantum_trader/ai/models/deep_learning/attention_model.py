"""
Attention-based Deep Learning Model for Time Series Prediction.

This module implements multi-head self-attention mechanisms for
financial time series forecasting and trading signal generation.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class AttentionConfig:
    """Configuration for attention model."""

    input_dim: int  # Input feature dimension
    hidden_dim: int  # Hidden layer dimension
    num_heads: int  # Number of attention heads
    num_layers: int  # Number of transformer layers
    dropout: Decimal  # Dropout rate
    output_dim: int  # Output dimension
    max_seq_length: int  # Maximum sequence length
    learning_rate: Decimal  # Learning rate
    device: str  # 'cpu' or 'cuda'
    model_path: str  # Path to save/load models
    use_positional_encoding: bool  # Use positional encoding
    attention_type: str  # 'scaled_dot_product' or 'additive'


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

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
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ) -> None:
        """Initialize multi-head attention.

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
        self.attention_weights: Optional[torch.Tensor] = None

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split tensor into multiple heads.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Reshaped tensor [batch, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention.

        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Compute output
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through multi-head attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = x.size(0)

        # Linear projections and split into heads
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask
        )

        # Store attention weights for visualization
        self.attention_weights = attention_weights.detach()

        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(
            batch_size, -1, self.d_model
        )

        # Final linear projection
        output = self.W_o(attention_output)

        return output, attention_weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int = 2048,
        dropout: float = 0.1
    ) -> None:
        """Initialize feed-forward network.

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
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.1
    ) -> None:
        """Initialize transformer encoder layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super(TransformerEncoderLayer, self).__init__()

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
        """Forward pass through encoder layer.

        Args:
            x: Input tensor
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        # Multi-head attention with residual connection
        attention_output, attention_weights = self.attention(x, mask)
        x = x + self.dropout1(attention_output)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x, attention_weights


class AttentionModel(nn.Module):
    """Attention-based model for time series prediction."""

    def __init__(self, config: AttentionConfig) -> None:
        """Initialize attention model.

        Args:
            config: Model configuration
        """
        super(AttentionModel, self).__init__()

        self.config = config

        # Input projection
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)

        # Positional encoding
        if config.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(
                config.hidden_dim,
                config.max_seq_length
            )
        else:
            self.pos_encoder = None

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                config.hidden_dim,
                config.num_heads,
                config.hidden_dim * 4,
                float(config.dropout)
            )
            for _ in range(config.num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.output_dim)

        # Dropout
        self.dropout = nn.Dropout(float(config.dropout))

        # Store attention weights for each layer
        self.layer_attention_weights: List[torch.Tensor] = []

        self._initialize_weights()

        logger.info(
            "Attention model initialized",
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers
        )

    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through model.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights_list)
        """
        # Input projection
        x = self.input_projection(x)
        x = self.dropout(x)

        # Add positional encoding
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)

        # Pass through encoder layers
        self.layer_attention_weights = []
        for encoder_layer in self.encoder_layers:
            x, attention_weights = encoder_layer(x, mask)
            self.layer_attention_weights.append(attention_weights)

        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)

        # Output projection
        output = self.output_projection(x)

        return output, self.layer_attention_weights

    def get_attention_weights(self) -> List[torch.Tensor]:
        """Get attention weights from all layers.

        Returns:
            List of attention weight tensors
        """
        return self.layer_attention_weights


class AttentionTrader:
    """Trading system using attention-based predictions."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize attention trader.

        Args:
            config: Configuration dictionary

        Example:
            >>> config = {
            ...     "input_dim": 50,
            ...     "hidden_dim": 256,
            ...     "num_heads": 8,
            ...     "num_layers": 4,
            ...     "output_dim": 3
            ... }
            >>> trader = AttentionTrader(config)
        """
        self.config = self._build_config(config)

        # Initialize model
        self.model = AttentionModel(self.config).to(self.config.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config.learning_rate)
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Training statistics
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "accuracy": []
        }

        logger.info(
            "Attention trader initialized",
            device=self.config.device,
            num_params=sum(p.numel() for p in self.model.parameters())
        )

    def _build_config(self, config: Dict[str, Any]) -> AttentionConfig:
        """Build AttentionConfig from dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            AttentionConfig instance
        """
        return AttentionConfig(
            input_dim=config["input_dim"],
            hidden_dim=config.get("hidden_dim", 256),
            num_heads=config.get("num_heads", 8),
            num_layers=config.get("num_layers", 4),
            dropout=Decimal(str(config.get("dropout", "0.1"))),
            output_dim=config["output_dim"],
            max_seq_length=config.get("max_seq_length", 1000),
            learning_rate=Decimal(str(config.get("learning_rate", "0.001"))),
            device=config.get("device", "cpu"),
            model_path=config.get("model_path", "/tmp/attention_model.pt"),
            use_positional_encoding=config.get("use_positional_encoding", True),
            attention_type=config.get("attention_type", "scaled_dot_product")
        )

    async def train(
        self,
        train_data: pl.DataFrame,
        val_data: pl.DataFrame,
        epochs: int
    ) -> Dict[str, List[float]]:
        """Train the attention model.

        Args:
            train_data: Training data DataFrame
            val_data: Validation data DataFrame
            epochs: Number of training epochs

        Returns:
            Training history dictionary

        Example:
            >>> history = await trader.train(train_df, val_df, epochs=100)
        """
        try:
            logger.info("Starting training", epochs=epochs)

            for epoch in range(epochs):
                # Training phase
                train_loss = await self._train_epoch(train_data)
                self.training_history["train_loss"].append(train_loss)

                # Validation phase
                val_loss, accuracy = await self._validate(val_data)
                self.training_history["val_loss"].append(val_loss)
                self.training_history["accuracy"].append(accuracy)

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        "Training progress",
                        epoch=epoch + 1,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        accuracy=accuracy
                    )

            logger.info("Training completed")
            return self.training_history

        except Exception as e:
            logger.error("Training failed", error=str(e))
            raise

    async def _train_epoch(self, data: pl.DataFrame) -> float:
        """Train for one epoch.

        Args:
            data: Training data

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Convert DataFrame to tensors
        features = torch.FloatTensor(
            data.select([c for c in data.columns if c != "label"]).to_numpy()
        ).to(self.config.device)

        labels = torch.LongTensor(
            data.select("label").to_numpy().flatten()
        ).to(self.config.device)

        # Add sequence dimension
        features = features.unsqueeze(1)

        # Forward pass
        outputs, _ = self.model(features)
        loss = self.criterion(outputs, labels)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        return total_loss / num_batches

    async def _validate(self, data: pl.DataFrame) -> Tuple[float, float]:
        """Validate model.

        Args:
            data: Validation data

        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()

        features = torch.FloatTensor(
            data.select([c for c in data.columns if c != "label"]).to_numpy()
        ).to(self.config.device)

        labels = torch.LongTensor(
            data.select("label").to_numpy().flatten()
        ).to(self.config.device)

        features = features.unsqueeze(1)

        with torch.no_grad():
            outputs, _ = self.model(features)
            loss = self.criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).float().mean().item()

        return loss.item(), accuracy

    async def predict(
        self,
        features: np.ndarray
    ) -> Tuple[int, Decimal]:
        """Make prediction for given features.

        Args:
            features: Input features

        Returns:
            Tuple of (predicted_class, confidence)

        Example:
            >>> features = np.random.randn(50)
            >>> action, confidence = await trader.predict(features)
        """
        try:
            self.model.eval()

            # Prepare input
            features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
            features_tensor = features_tensor.to(self.config.device)

            with torch.no_grad():
                outputs, attention_weights = self.model(features_tensor)
                probabilities = F.softmax(outputs, dim=-1)

                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = Decimal(str(probabilities[0, predicted_class].item()))

            return predicted_class, confidence

        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            raise

    def get_attention_weights(self) -> List[np.ndarray]:
        """Get attention weights from last forward pass.

        Returns:
            List of attention weight arrays
        """
        return [w.cpu().numpy() for w in self.model.get_attention_weights()]

    def save(self, path: Optional[str] = None) -> None:
        """Save model to disk.

        Args:
            path: Path to save model (uses config path if None)
        """
        try:
            save_path = path or self.config.model_path
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "training_history": self.training_history
            }, save_path)

            logger.info("Model saved", path=save_path)

        except Exception as e:
            logger.error("Model save failed", error=str(e))
            raise

    def load(self, path: Optional[str] = None) -> None:
        """Load model from disk.

        Args:
            path: Path to load model from (uses config path if None)
        """
        try:
            load_path = path or self.config.model_path
            checkpoint = torch.load(load_path, map_location=self.config.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.training_history = checkpoint.get("training_history", {})

            logger.info("Model loaded", path=load_path)

        except Exception as e:
            logger.error("Model load failed", error=str(e))
            raise

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary statistics.

        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_layers": self.config.num_layers,
            "num_heads": self.config.num_heads,
            "hidden_dim": self.config.hidden_dim,
            "training_epochs": len(self.training_history.get("train_loss", []))
        }
