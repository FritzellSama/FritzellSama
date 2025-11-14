"""
Variational Autoencoder for Trading

Production-ready VAE for dimensionality reduction, anomaly detection,
and feature learning from market data.
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


class VAEEncoder(nn.Module):
    """Encoder network for VAE."""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int, dropout: float = 0.2):
        super(VAEEncoder, self).__init__()

        # Build encoder layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    """Decoder network for VAE."""

    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.2):
        super(VAEDecoder, self).__init__()

        # Build decoder layers (reverse of encoder)
        layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)


class VAENetwork(nn.Module):
    """Complete VAE architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout: float = 0.2
    ):
        super(VAENetwork, self).__init__()

        self.encoder = VAEEncoder(input_dim, hidden_dims, latent_dim, dropout)
        self.decoder = VAEDecoder(latent_dim, hidden_dims, input_dim, dropout)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

    def encode(self, x):
        """Encode input to latent space."""
        mu, logvar = self.encoder(x)
        return mu

    def decode(self, z):
        """Decode from latent space."""
        return self.decoder(z)


class VAEModel(BaseMLModel):
    """
    Variational Autoencoder for trading applications.

    Uses VAE for dimensionality reduction, anomaly detection,
    and learning latent representations of market states.

    Attributes:
        config: Configuration dictionary
        model: PyTorch VAE model
        device: Computation device
        is_trained: Training status

    Example:
        >>> config = {"model": {"vae": {"latent_dim": 32, "hidden_dims": [128, 64]}}}
        >>> model = VAEModel(config)
        >>> model.train(features, features)  # Unsupervised
        >>> latent_features = model.encode(new_features)
        >>> anomaly_scores = model.anomaly_score(test_features)
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialize VAE model.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__(config)
        self.config = config
        self._validate_config()

        vae_config = self.config.get("model", {}).get("vae", {})

        # Model architecture
        self.latent_dim: int = vae_config.get("latent_dim", int(os.getenv("VAE_LATENT_DIM", "32")))
        self.hidden_dims: List[int] = vae_config.get(
            "hidden_dims",
            [int(x) for x in os.getenv("VAE_HIDDEN_DIMS", "128,64").split(",")]
        )
        self.dropout: float = float(vae_config.get("dropout", os.getenv("VAE_DROPOUT", "0.2")))

        # Loss weights
        self.beta: float = float(vae_config.get("beta", os.getenv("VAE_BETA", "1.0")))  # KL divergence weight

        # Training parameters
        self.learning_rate: float = float(vae_config.get("learning_rate", os.getenv("VAE_LEARNING_RATE", "0.001")))
        self.batch_size: int = vae_config.get("batch_size", int(os.getenv("VAE_BATCH_SIZE", "32")))
        self.num_epochs: int = vae_config.get("num_epochs", int(os.getenv("VAE_NUM_EPOCHS", "100")))
        self.patience: int = vae_config.get("patience", int(os.getenv("VAE_PATIENCE", "15")))

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.model: Optional[VAENetwork] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.is_trained: bool = False

        logger.info(
            "vae_model_initialized",
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            beta=self.beta,
            device=str(self.device)
        )

    def _validate_config(self) -> None:
        """Validate configuration."""
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        vae_config = self.config.get("model", {}).get("vae", {})

        if vae_config:
            latent_dim = vae_config.get("latent_dim", 32)
            if latent_dim <= 0:
                raise ValueError("latent_dim must be positive")

    def _initialize_model(self, input_dim: int) -> None:
        """Initialize model architecture."""
        self.model = VAENetwork(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            dropout=self.dropout
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        logger.info("vae_network_created", input_dim=input_dim, latent_dim=self.latent_dim)

    def _vae_loss(self, reconstruction, x, mu, logvar):
        """Calculate VAE loss (reconstruction + KL divergence)."""
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(reconstruction, x, reduction='sum')

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Train VAE model (labels are ignored - unsupervised).

        Args:
            features: Training features
            labels: Ignored (VAE is unsupervised)

        Raises:
            ValueError: If inputs are invalid
        """
        try:
            if not isinstance(features, np.ndarray):
                raise ValueError("features must be numpy array")

            if len(features) == 0:
                raise ValueError("Cannot train on empty data")

            # Ensure 2D
            if len(features.shape) == 1:
                features = features.reshape(-1, 1)

            n_samples, n_features = features.shape

            logger.info("vae_training_started", n_samples=n_samples, n_features=n_features)

            # Initialize model
            if self.model is None:
                self._initialize_model(n_features)

            # Prepare data
            X_tensor = torch.FloatTensor(features).to(self.device)
            dataset = TensorDataset(X_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Training loop
            best_loss = float('inf')
            patience_counter = 0

            for epoch in range(self.num_epochs):
                self.model.train()
                epoch_loss = 0.0
                epoch_recon_loss = 0.0
                epoch_kl_loss = 0.0

                for (batch_x,) in dataloader:
                    self.optimizer.zero_grad()

                    reconstruction, mu, logvar = self.model(batch_x)
                    loss, recon_loss, kl_loss = self._vae_loss(reconstruction, batch_x, mu, logvar)

                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    epoch_recon_loss += recon_loss.item()
                    epoch_kl_loss += kl_loss.item()

                avg_loss = epoch_loss / len(dataloader)
                avg_recon = epoch_recon_loss / len(dataloader)
                avg_kl = epoch_kl_loss / len(dataloader)

                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    logger.debug(
                        "vae_training_epoch",
                        epoch=epoch + 1,
                        loss=avg_loss,
                        recon_loss=avg_recon,
                        kl_loss=avg_kl
                    )

                if patience_counter >= self.patience:
                    logger.info("vae_early_stopping", epoch=epoch + 1, best_loss=best_loss)
                    break

            self.is_trained = True

            logger.info("vae_training_completed", final_loss=best_loss, epochs=epoch + 1)

        except Exception as e:
            logger.error("vae_training_failed", error=str(e))
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Reconstruct features (for anomaly detection).

        Args:
            features: Input features

        Returns:
            Reconstructed features
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")

            if len(features.shape) == 1:
                features = features.reshape(-1, 1)

            X_tensor = torch.FloatTensor(features).to(self.device)

            self.model.eval()
            with torch.no_grad():
                reconstruction, _, _ = self.model(X_tensor)

            reconstruction_np = reconstruction.cpu().numpy()

            logger.debug("vae_reconstruction_generated", n_samples=len(reconstruction_np))

            return reconstruction_np

        except Exception as e:
            logger.error("vae_prediction_failed", error=str(e))
            raise

    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Encode features to latent space.

        Args:
            features: Input features

        Returns:
            Latent representations
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before encoding")

            if len(features.shape) == 1:
                features = features.reshape(-1, 1)

            X_tensor = torch.FloatTensor(features).to(self.device)

            self.model.eval()
            with torch.no_grad():
                latent = self.model.encode(X_tensor)

            latent_np = latent.cpu().numpy()

            logger.debug("vae_encoding_generated", n_samples=len(latent_np))

            return latent_np

        except Exception as e:
            logger.error("vae_encoding_failed", error=str(e))
            raise

    def anomaly_score(self, features: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores (reconstruction error).

        Args:
            features: Input features

        Returns:
            Anomaly scores (higher = more anomalous)
        """
        try:
            reconstruction = self.predict(features)

            # Calculate reconstruction error
            errors = np.mean((features - reconstruction) ** 2, axis=1)

            logger.debug("anomaly_scores_calculated", n_samples=len(errors))

            return errors

        except Exception as e:
            logger.error("anomaly_score_failed", error=str(e))
            raise

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model (reconstruction quality).

        Args:
            features: Test features
            labels: Ignored (unsupervised)

        Returns:
            Evaluation metrics
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")

            reconstruction = self.predict(features)

            # Calculate metrics
            mse = np.mean((features - reconstruction) ** 2)
            mae = np.mean(np.abs(features - reconstruction))

            # Calculate per-feature correlation
            correlations = []
            for i in range(features.shape[1]):
                corr = np.corrcoef(features[:, i], reconstruction[:, i])[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0.0)

            avg_correlation = np.mean(correlations)

            metrics = {
                "reconstruction_mse": float(mse),
                "reconstruction_mae": float(mae),
                "avg_correlation": float(avg_correlation)
            }

            logger.info("vae_evaluation_completed", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("vae_evaluation_failed", error=str(e))
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
                "latent_dim": self.latent_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "beta": self.beta,
                "is_trained": self.is_trained
            }

            torch.save(save_dict, path)
            logger.info("vae_model_saved", path=path)

        except Exception as e:
            logger.error("vae_save_failed", path=path, error=str(e))
            raise

    def load(self, path: str) -> None:
        """Load model from disk."""
        try:
            if not Path(path).exists():
                raise FileNotFoundError(f"Model file not found: {path}")

            checkpoint = torch.load(path, map_location=self.device)

            # Restore configuration
            self.latent_dim = checkpoint["latent_dim"]
            self.hidden_dims = checkpoint["hidden_dims"]
            self.dropout = checkpoint["dropout"]
            self.beta = checkpoint["beta"]

            # Determine input_dim from saved model
            input_dim = list(checkpoint["model_state"].values())[0].shape[1]

            # Recreate model
            self._initialize_model(input_dim)

            # Load states
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.is_trained = checkpoint["is_trained"]

            logger.info("vae_model_loaded", path=path)

        except Exception as e:
            logger.error("vae_load_failed", path=path, error=str(e))
            raise
