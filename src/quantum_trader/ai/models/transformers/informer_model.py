"""Informer model for long-sequence time-series forecasting.

This module implements the Informer architecture for efficient
long-range time series prediction with reduced computational complexity.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel

logger = get_logger(__name__)


class ProbAttention(nn.Module):
    """ProbSparse self-attention mechanism."""

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale: Optional[float] = None,
        attention_dropout: float = 0.1
    ) -> None:
        """Initialize ProbSparse attention.

        Args:
            mask_flag: Whether to use masking
            factor: Sampling factor for ProbSparse
            scale: Attention scale
            attention_dropout: Dropout rate
        """
        super().__init__()

        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            queries: Query tensor
            keys: Key tensor
            values: Value tensor
            attn_mask: Optional attention mask

        Returns:
            Tuple of (output, attention)
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        # Sample queries
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = min(U_part, L_Q)

        # Calculate scores
        scores_top = torch.matmul(
            queries[:, :u, :, :],
            keys.transpose(-2, -1)
        ) / np.sqrt(D)

        # ProbSparse sampling
        M = scores_top.max(dim=-1)[0] - torch.div(
            scores_top.sum(dim=-1),
            L_K
        )

        # Select top-k
        M_top = M.topk(u, sorted=False)[1]

        # Expand to original size
        queries_reduce = queries[
            torch.arange(B)[:, None, None],
            M_top,
            torch.arange(H)[None, :, None]
        ]

        # Calculate attention
        scores = torch.matmul(
            queries_reduce,
            keys.transpose(-2, -1)
        ) / np.sqrt(D)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        attn = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(attn, values)

        return output, attn


class InformerModel(BaseMLModel):
    """Informer model for time series forecasting.

    Efficient transformer for long-sequence forecasting with
    ProbSparse self-attention and distilling architecture.

    Attributes:
        config: Model configuration
        encoder: Encoder network
        decoder: Decoder network
        device: Torch device

    Example:
        >>> config = {
        ...     "input_dim": 20,
        ...     "d_model": 512,
        ...     "n_heads": 8,
        ...     "e_layers": 2,
        ...     "d_layers": 1,
        ...     "d_ff": 2048,
        ...     "dropout": "0.1",
        ...     "seq_len": 96,
        ...     "label_len": 48,
        ...     "pred_len": 24,
        ...     "learning_rate": "0.0001"
        ... }
        >>> model = InformerModel(config)
        >>> model.train(features, labels)
        >>> predictions = model.predict(test_features)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Informer model.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If config is invalid
        """
        super().__init__(config)

        self._validate_config()

        # Model parameters
        self.input_dim = config["input_dim"]
        self.d_model = config.get("d_model", 512)
        self.n_heads = config.get("n_heads", 8)
        self.e_layers = config.get("e_layers", 2)
        self.d_layers = config.get("d_layers", 1)
        self.d_ff = config.get("d_ff", 2048)
        self.dropout = float(config.get("dropout", "0.1"))
        self.activation = config.get("activation", "gelu")
        self.factor = config.get("factor", 5)

        # Sequence parameters
        self.seq_len = config["seq_len"]
        self.label_len = config["label_len"]
        self.pred_len = config["pred_len"]
        self.output_dim = config.get("output_dim", self.input_dim)

        # Training parameters
        self.learning_rate = Decimal(str(config.get("learning_rate", "0.0001")))
        self.weight_decay = Decimal(str(config.get("weight_decay", "0.00001")))
        self.batch_size = config.get("batch_size", 32)
        self.num_epochs = config.get("num_epochs", 10)

        # Device setup
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Build encoder-decoder
        self._build_model()

        # Optimizer
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay)
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Training state
        self.training_history: List[Dict[str, float]] = []

        logger.info(
            "Informer model initialized",
            d_model=self.d_model,
            n_heads=self.n_heads,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            device=str(self.device)
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If required config missing
        """
        required_keys = [
            "input_dim",
            "seq_len",
            "label_len",
            "pred_len"
        ]

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def _build_model(self) -> None:
        """Build Informer encoder-decoder architecture."""
        # Input embedding
        self.enc_embedding = nn.Linear(self.input_dim, self.d_model)
        self.dec_embedding = nn.Linear(self.input_dim, self.d_model)

        # Positional encoding
        self.position_enc = nn.Parameter(
            torch.zeros(1, self.seq_len, self.d_model)
        )
        self.position_dec = nn.Parameter(
            torch.zeros(1, self.label_len + self.pred_len, self.d_model)
        )

        # Encoder layers (simplified - full implementation would use custom encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.e_layers
        )

        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.d_layers
        )

        # Output projection
        self.projection = nn.Linear(self.d_model, self.output_dim)

    def parameters(self) -> List:
        """Get model parameters.

        Returns:
            List of parameters
        """
        params = []
        for component in [
            self.enc_embedding,
            self.dec_embedding,
            self.encoder,
            self.decoder,
            self.projection
        ]:
            params.extend(component.parameters())

        params.append(self.position_enc)
        params.append(self.position_dec)

        return params

    def forward(
        self,
        x_enc: torch.Tensor,
        x_dec: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_enc: Encoder input [batch_size, seq_len, input_dim]
            x_dec: Decoder input [batch_size, label_len + pred_len, input_dim]

        Returns:
            Predictions [batch_size, pred_len, output_dim]
        """
        # Encoder
        enc_out = self.enc_embedding(x_enc)
        enc_out = enc_out + self.position_enc[:, :x_enc.size(1), :]
        enc_out = self.encoder(enc_out)

        # Decoder
        dec_out = self.dec_embedding(x_dec)
        dec_out = dec_out + self.position_dec[:, :x_dec.size(1), :]
        dec_out = self.decoder(dec_out, enc_out)

        # Projection
        output = self.projection(dec_out)

        # Return only prediction part
        return output[:, -self.pred_len:, :]

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train Informer model.

        Args:
            features: Input sequences [num_samples, seq_len, input_dim]
            labels: Target sequences [num_samples, pred_len, output_dim]

        Raises:
            ValueError: If input shape is invalid
        """
        try:
            logger.info(
                "Starting Informer training",
                num_samples=len(features),
                epochs=self.num_epochs
            )

            # Convert to tensors
            X_enc = torch.FloatTensor(features).to(self.device)

            # Prepare decoder input (last label_len from encoder + zeros for pred_len)
            X_dec = torch.cat([
                X_enc[:, -self.label_len:, :],
                torch.zeros(len(features), self.pred_len, self.input_dim).to(self.device)
            ], dim=1)

            y = torch.FloatTensor(labels).to(self.device)

            # Training loop
            for epoch in range(self.num_epochs):
                epoch_loss = Decimal("0")
                num_batches = 0

                for i in range(0, len(X_enc), self.batch_size):
                    batch_enc = X_enc[i:i + self.batch_size]
                    batch_dec = X_dec[i:i + self.batch_size]
                    batch_y = y[i:i + self.batch_size]

                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.forward(batch_enc, batch_dec)

                    # Compute loss
                    loss = self.criterion(outputs, batch_y)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += Decimal(str(loss.item()))
                    num_batches += 1

                avg_loss = float(epoch_loss / num_batches)
                self.training_history.append({"epoch": epoch, "loss": avg_loss})

                if epoch % 1 == 0:
                    logger.info("Training progress", epoch=epoch, loss=avg_loss)

            logger.info("Informer training completed")

        except Exception as e:
            logger.error("Failed to train Informer", error=str(e))
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions.

        Args:
            features: Input sequences

        Returns:
            Predictions [num_samples, pred_len, output_dim]
        """
        try:
            X_enc = torch.FloatTensor(features).to(self.device)

            # Prepare decoder input
            X_dec = torch.cat([
                X_enc[:, -self.label_len:, :],
                torch.zeros(len(features), self.pred_len, self.input_dim).to(self.device)
            ], dim=1)

            with torch.no_grad():
                predictions = self.forward(X_enc, X_dec).cpu().numpy()

            logger.info("Generated predictions", shape=predictions.shape)

            return predictions

        except Exception as e:
            logger.error("Failed to generate predictions", error=str(e))
            raise

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            features: Input sequences
            labels: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            predictions = self.predict(features)

            mse = float(np.mean((predictions - labels) ** 2))
            mae = float(np.mean(np.abs(predictions - labels)))
            rmse = float(np.sqrt(mse))

            metrics = {
                "mse": mse,
                "mae": mae,
                "rmse": rmse
            }

            logger.info("Informer evaluation completed", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("Failed to evaluate Informer", error=str(e))
            raise

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path to save model
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                "enc_embedding": self.enc_embedding.state_dict(),
                "dec_embedding": self.dec_embedding.state_dict(),
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "projection": self.projection.state_dict(),
                "position_enc": self.position_enc,
                "position_dec": self.position_dec,
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
                "training_history": self.training_history
            }, save_path)

            logger.info("Informer model saved", path=path)

        except Exception as e:
            logger.error("Failed to save Informer model", error=str(e))
            raise

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: File path to load model from
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)

            self.enc_embedding.load_state_dict(checkpoint["enc_embedding"])
            self.dec_embedding.load_state_dict(checkpoint["dec_embedding"])
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.decoder.load_state_dict(checkpoint["decoder"])
            self.projection.load_state_dict(checkpoint["projection"])
            self.position_enc = checkpoint["position_enc"]
            self.position_dec = checkpoint["position_dec"]
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.training_history = checkpoint["training_history"]

            logger.info("Informer model loaded", path=path)

        except Exception as e:
            logger.error("Failed to load Informer model", error=str(e))
            raise
