"""Variational Autoencoder for anomaly detection and feature learning.

This module implements a VAE for unsupervised learning of latent representations
and anomaly detection in market data.
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


class VAE(nn.Module):
    """Variational Autoencoder architecture.

    Implements a VAE with encoder-decoder structure for learning
    latent representations and generating synthetic data.

    Attributes:
        encoder: Encoder network
        decoder: Decoder network
        fc_mu: Mean layer for latent distribution
        fc_logvar: Log variance layer for latent distribution
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout: float = 0.2
    ):
        """Initialize VAE.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            latent_dim: Latent space dimension
            dropout: Dropout probability
        """
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor

        Returns:
            Tuple of (mean, log_variance)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling.

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction.

        Args:
            z: Latent vector

        Returns:
            Reconstructed output
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE.

        Args:
            x: Input tensor

        Returns:
            Tuple of (reconstruction, mean, log_variance)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


class VAEModel(BaseMLModel):
    """Variational Autoencoder model for trading applications.

    Implements VAE for unsupervised learning, anomaly detection,
    and generating synthetic market scenarios.

    Attributes:
        config: Model configuration
        model: VAE neural network
        optimizer: Training optimizer
        device: Computation device
        is_trained: Training status

    Examples:
        >>> config = {
        ...     "input_dim": 50,
        ...     "hidden_dims": [128, 64, 32],
        ...     "latent_dim": 16,
        ...     "learning_rate": 0.001
        ... }
        >>> vae = VAEModel(config)
        >>> vae.train(features, features)  # Unsupervised
        >>> latent = vae.encode(test_features)
        >>> anomalies = vae.detect_anomalies(test_features)
    """

    def __init__(self, config: Dict) -> None:
        """Initialize VAE model.

        Args:
            config: Configuration with architecture and training params

        Raises:
            ValueError: If required config parameters missing
        """
        super().__init__(config)
        self._validate_config()

        # Architecture parameters
        self.input_dim: int = int(config.get("input_dim", os.getenv("VAE_INPUT_DIM", "50")))
        self.hidden_dims: List[int] = config.get("hidden_dims", [int(x) for x in os.getenv("VAE_HIDDEN_DIMS", "128,64,32").split(",")])
        self.latent_dim: int = int(config.get("latent_dim", os.getenv("VAE_LATENT_DIM", "16")))
        self.dropout: float = float(config.get("dropout", os.getenv("VAE_DROPOUT", "0.2")))

        # Training parameters
        self.learning_rate: float = float(config.get("learning_rate", os.getenv("VAE_LEARNING_RATE", "0.001")))
        self.batch_size: int = int(config.get("batch_size", os.getenv("VAE_BATCH_SIZE", "128")))
        self.epochs: int = int(config.get("epochs", os.getenv("VAE_EPOCHS", "100")))
        self.beta: float = float(config.get("beta", os.getenv("VAE_BETA", "1.0")))  # KL divergence weight

        # Device configuration
        self.device = torch.device(
            config.get("device", os.getenv("VAE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        )

        # Initialize model components
        self.model: Optional[VAE] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.is_trained: bool = False

        # Anomaly detection threshold
        self.anomaly_threshold: Optional[float] = None

        self._initialize_model()

        logger.info(
            "VAE model initialized",
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            device=str(self.device)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config validation fails
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        if "latent_dim" in self.config:
            if self.config["latent_dim"] < 1:
                raise ValueError("latent_dim must be at least 1")

        if "hidden_dims" in self.config:
            if not self.config["hidden_dims"]:
                raise ValueError("hidden_dims must not be empty")

    def _initialize_model(self) -> None:
        """Initialize the VAE architecture and optimizer."""
        self.model = VAE(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            dropout=self.dropout
        )

        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=float(os.getenv("VAE_WEIGHT_DECAY", "1e-5"))
        )

    def _vae_loss(
        self,
        reconstruction: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate VAE loss (reconstruction + KL divergence).

        Args:
            reconstruction: Reconstructed output
            x: Original input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            Tuple of (total_loss, reconstruction_loss, kl_loss)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x, reduction="sum")

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the VAE model.

        Args:
            features: Training features (n_samples, n_features)
            labels: Not used (VAE is unsupervised), but kept for API compatibility

        Raises:
            ValueError: If input arrays invalid
        """
        try:
            logger.info(
                "Starting VAE training",
                n_samples=features.shape[0],
                n_features=features.shape[1],
                epochs=self.epochs
            )

            # Prepare data (use features as both input and target)
            X_tensor = torch.FloatTensor(features).to(self.device)
            dataset = TensorDataset(X_tensor, X_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Training loop
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                epoch_recon_loss = 0.0
                epoch_kl_loss = 0.0
                self.model.train()

                for batch_X, _ in dataloader:
                    self.optimizer.zero_grad()

                    # Forward pass
                    reconstruction, mu, logvar = self.model(batch_X)

                    # Calculate loss
                    loss, recon_loss, kl_loss = self._vae_loss(reconstruction, batch_X, mu, logvar)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    epoch_recon_loss += recon_loss.item()
                    epoch_kl_loss += kl_loss.item()

                avg_loss = epoch_loss / len(dataloader.dataset)
                avg_recon = epoch_recon_loss / len(dataloader.dataset)
                avg_kl = epoch_kl_loss / len(dataloader.dataset)

                if (epoch + 1) % 10 == 0:
                    logger.debug(
                        f"Epoch {epoch + 1}/{self.epochs}",
                        loss=f"{avg_loss:.6f}",
                        recon_loss=f"{avg_recon:.6f}",
                        kl_loss=f"{avg_kl:.6f}"
                    )

            # Calculate anomaly threshold (95th percentile of reconstruction errors)
            self.model.eval()
            with torch.no_grad():
                reconstruction, _, _ = self.model(X_tensor)
                errors = torch.mean((X_tensor - reconstruction) ** 2, dim=1)
                self.anomaly_threshold = float(torch.quantile(errors, 0.95))

            self.is_trained = True

            logger.info(
                "VAE training completed",
                final_loss=float(avg_loss),
                anomaly_threshold=self.anomaly_threshold
            )

        except Exception as e:
            logger.error(
                "VAE training failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Reconstruct input features (for API compatibility).

        Args:
            features: Features array for reconstruction

        Returns:
            Reconstructed features array

        Raises:
            RuntimeError: If model not trained
        """
        try:
            if not self.is_trained:
                raise RuntimeError("Model must be trained before prediction")

            self.model.eval()

            with torch.no_grad():
                X_tensor = torch.FloatTensor(features).to(self.device)
                reconstruction, _, _ = self.model(X_tensor)

            return reconstruction.cpu().numpy()

        except Exception as e:
            logger.error(
                "Reconstruction failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode features to latent space.

        Args:
            features: Features array to encode

        Returns:
            Latent representations

        Raises:
            RuntimeError: If model not trained
        """
        try:
            if not self.is_trained:
                raise RuntimeError("Model must be trained before encoding")

            self.model.eval()

            with torch.no_grad():
                X_tensor = torch.FloatTensor(features).to(self.device)
                mu, _ = self.model.encode(X_tensor)

            return mu.cpu().numpy()

        except Exception as e:
            logger.error(
                "Encoding failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode latent vectors to feature space.

        Args:
            latent: Latent vectors to decode

        Returns:
            Decoded features

        Raises:
            RuntimeError: If model not trained
        """
        try:
            if not self.is_trained:
                raise RuntimeError("Model must be trained before decoding")

            self.model.eval()

            with torch.no_grad():
                z_tensor = torch.FloatTensor(latent).to(self.device)
                reconstruction = self.model.decode(z_tensor)

            return reconstruction.cpu().numpy()

        except Exception as e:
            logger.error(
                "Decoding failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def detect_anomalies(self, features: np.ndarray) -> np.ndarray:
        """Detect anomalies based on reconstruction error.

        Args:
            features: Features to check for anomalies

        Returns:
            Boolean array indicating anomalies

        Raises:
            RuntimeError: If model not trained
        """
        try:
            if not self.is_trained:
                raise RuntimeError("Model must be trained before anomaly detection")

            if self.anomaly_threshold is None:
                raise RuntimeError("Anomaly threshold not set")

            self.model.eval()

            with torch.no_grad():
                X_tensor = torch.FloatTensor(features).to(self.device)
                reconstruction, _, _ = self.model(X_tensor)

                # Calculate reconstruction errors
                errors = torch.mean((X_tensor - reconstruction) ** 2, dim=1)
                anomalies = errors > self.anomaly_threshold

            return anomalies.cpu().numpy()

        except Exception as e:
            logger.error(
                "Anomaly detection failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            features: Test features array
            labels: Not used (VAE is unsupervised)

        Returns:
            Dictionary of evaluation metrics

        Raises:
            RuntimeError: If model not trained
        """
        try:
            if not self.is_trained:
                raise RuntimeError("Model must be trained before evaluation")

            self.model.eval()

            with torch.no_grad():
                X_tensor = torch.FloatTensor(features).to(self.device)
                reconstruction, mu, logvar = self.model(X_tensor)

                # Calculate losses
                total_loss, recon_loss, kl_loss = self._vae_loss(reconstruction, X_tensor, mu, logvar)

            metrics = {
                "total_loss": float(total_loss.item() / len(features)),
                "reconstruction_loss": float(recon_loss.item() / len(features)),
                "kl_divergence": float(kl_loss.item() / len(features)),
                "anomaly_threshold": self.anomaly_threshold if self.anomaly_threshold else 0.0
            }

            logger.info("VAE evaluation completed", **metrics)

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
                "anomaly_threshold": self.anomaly_threshold,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, path)

            logger.info("VAE model saved", path=path)

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
            self.anomaly_threshold = checkpoint.get("anomaly_threshold")

            logger.info(
                "VAE model loaded",
                path=path,
                timestamp=checkpoint.get("timestamp")
            )

        except Exception as e:
            logger.error("Model load failed", error=str(e), path=path)
            raise
