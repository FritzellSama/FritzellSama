"""
Autoencoder Model for Anomaly Detection and Feature Learning.

This module implements various autoencoder architectures for
detecting market anomalies, dimensionality reduction, and feature extraction.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class AutoencoderType(Enum):
    """Types of autoencoder architectures."""

    VANILLA = "vanilla"  # Standard autoencoder
    VARIATIONAL = "variational"  # VAE
    DENOISING = "denoising"  # Denoising autoencoder
    SPARSE = "sparse"  # Sparse autoencoder
    CONTRACTIVE = "contractive"  # Contractive autoencoder


@dataclass
class AutoencoderConfig:
    """Configuration for autoencoder model."""

    input_dim: int  # Input feature dimension
    hidden_dims: List[int]  # Hidden layer dimensions
    latent_dim: int  # Latent space dimension
    autoencoder_type: str  # Type of autoencoder
    dropout: Decimal  # Dropout rate
    learning_rate: Decimal  # Learning rate
    beta: Decimal  # KL divergence weight (VAE)
    sparsity_param: Decimal  # Sparsity parameter
    sparsity_weight: Decimal  # Sparsity regularization weight
    noise_factor: Decimal  # Noise factor for denoising
    device: str  # 'cpu' or 'cuda'
    model_path: str  # Path to save/load models
    reconstruction_loss: str  # 'mse' or 'bce'


@dataclass
class AnomalyScore:
    """Anomaly detection result."""

    reconstruction_error: Decimal
    is_anomaly: bool
    threshold: Decimal
    timestamp: datetime
    latent_representation: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Encoder(nn.Module):
    """Encoder network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout: float = 0.1
    ) -> None:
        """Initialize encoder.

        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            latent_dim: Latent dimension
            dropout: Dropout rate
        """
        super(Encoder, self).__init__()

        # Build encoder layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.latent = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder.

        Args:
            x: Input tensor

        Returns:
            Latent representation
        """
        encoded = self.encoder(x)
        latent = self.latent(encoded)
        return latent


class Decoder(nn.Module):
    """Decoder network."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1
    ) -> None:
        """Initialize decoder.

        Args:
            latent_dim: Latent dimension
            hidden_dims: Hidden layer dimensions (reversed)
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super(Decoder, self).__init__()

        # Build decoder layers
        layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.decoder = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder.

        Args:
            z: Latent representation

        Returns:
            Reconstructed output
        """
        decoded = self.decoder(z)
        output = self.output(decoded)
        return output


class VariationalEncoder(nn.Module):
    """Variational encoder for VAE."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout: float = 0.1
    ) -> None:
        """Initialize variational encoder.

        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            latent_dim: Latent dimension
            dropout: Dropout rate
        """
        super(VariationalEncoder, self).__init__()

        # Build encoder layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Mean and variance layers
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through variational encoder.

        Args:
            x: Input tensor

        Returns:
            Tuple of (latent, mu, logvar)
        """
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar


class VanillaAutoencoder(nn.Module):
    """Standard autoencoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        """Initialize vanilla autoencoder.

        Args:
            config: Autoencoder configuration
        """
        super(VanillaAutoencoder, self).__init__()

        self.encoder = Encoder(
            config.input_dim,
            config.hidden_dims,
            config.latent_dim,
            float(config.dropout)
        )

        self.decoder = Decoder(
            config.latent_dim,
            config.hidden_dims,
            config.input_dim,
            float(config.dropout)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Tuple of (reconstruction, latent)
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


class VariationalAutoencoder(nn.Module):
    """Variational autoencoder (VAE)."""

    def __init__(self, config: AutoencoderConfig) -> None:
        """Initialize VAE.

        Args:
            config: Autoencoder configuration
        """
        super(VariationalAutoencoder, self).__init__()

        self.encoder = VariationalEncoder(
            config.input_dim,
            config.hidden_dims,
            config.latent_dim,
            float(config.dropout)
        )

        self.decoder = Decoder(
            config.latent_dim,
            config.hidden_dims,
            config.input_dim,
            float(config.dropout)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Tuple of (reconstruction, latent, mu, logvar)
        """
        latent, mu, logvar = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent, mu, logvar


class AutoencoderModel:
    """Autoencoder model for anomaly detection and feature learning."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize autoencoder model.

        Args:
            config: Configuration dictionary

        Example:
            >>> config = {
            ...     "input_dim": 50,
            ...     "hidden_dims": [256, 128, 64],
            ...     "latent_dim": 32,
            ...     "autoencoder_type": "vanilla"
            ... }
            >>> model = AutoencoderModel(config)
        """
        self.config = self._build_config(config)

        # Initialize model
        self.model = self._build_model()
        self.model = self.model.to(self.config.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config.learning_rate)
        )

        # Anomaly detection threshold
        self.anomaly_threshold: Optional[Decimal] = None

        # Training statistics
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "reconstruction_error": []
        }

        logger.info(
            "Autoencoder model initialized",
            type=self.config.autoencoder_type,
            latent_dim=self.config.latent_dim,
            device=self.config.device
        )

    def _build_config(self, config: Dict[str, Any]) -> AutoencoderConfig:
        """Build AutoencoderConfig from dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            AutoencoderConfig instance
        """
        return AutoencoderConfig(
            input_dim=config["input_dim"],
            hidden_dims=config.get("hidden_dims", [256, 128, 64]),
            latent_dim=config.get("latent_dim", 32),
            autoencoder_type=config.get("autoencoder_type", "vanilla"),
            dropout=Decimal(str(config.get("dropout", "0.1"))),
            learning_rate=Decimal(str(config.get("learning_rate", "0.001"))),
            beta=Decimal(str(config.get("beta", "1.0"))),
            sparsity_param=Decimal(str(config.get("sparsity_param", "0.05"))),
            sparsity_weight=Decimal(str(config.get("sparsity_weight", "0.1"))),
            noise_factor=Decimal(str(config.get("noise_factor", "0.2"))),
            device=config.get("device", "cpu"),
            model_path=config.get("model_path", "/tmp/autoencoder_model.pt"),
            reconstruction_loss=config.get("reconstruction_loss", "mse")
        )

    def _build_model(self) -> nn.Module:
        """Build autoencoder model based on type.

        Returns:
            Autoencoder model
        """
        ae_type = AutoencoderType(self.config.autoencoder_type)

        if ae_type == AutoencoderType.VARIATIONAL:
            return VariationalAutoencoder(self.config)
        else:
            # Default to vanilla autoencoder
            return VanillaAutoencoder(self.config)

    async def train(
        self,
        train_data: pl.DataFrame,
        val_data: pl.DataFrame,
        epochs: int
    ) -> Dict[str, List[float]]:
        """Train the autoencoder.

        Args:
            train_data: Training data DataFrame
            val_data: Validation data DataFrame
            epochs: Number of training epochs

        Returns:
            Training history dictionary

        Example:
            >>> history = await model.train(train_df, val_df, epochs=100)
        """
        try:
            logger.info("Starting training", epochs=epochs)

            for epoch in range(epochs):
                # Training phase
                train_loss = await self._train_epoch(train_data)
                self.training_history["train_loss"].append(train_loss)

                # Validation phase
                val_loss = await self._validate(val_data)
                self.training_history["val_loss"].append(val_loss)

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        "Training progress",
                        epoch=epoch + 1,
                        train_loss=train_loss,
                        val_loss=val_loss
                    )

            # Compute anomaly threshold from validation data
            await self._compute_anomaly_threshold(val_data)

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

        # Convert to tensor
        features = torch.FloatTensor(data.to_numpy()).to(self.config.device)

        # Add noise for denoising autoencoder
        if self.config.autoencoder_type == "denoising":
            noise = torch.randn_like(features) * float(self.config.noise_factor)
            noisy_features = features + noise
        else:
            noisy_features = features

        # Forward pass
        if isinstance(self.model, VariationalAutoencoder):
            reconstruction, latent, mu, logvar = self.model(noisy_features)

            # VAE loss
            recon_loss = self._compute_reconstruction_loss(reconstruction, features)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + float(self.config.beta) * kl_loss
        else:
            reconstruction, latent = self.model(noisy_features)
            loss = self._compute_reconstruction_loss(reconstruction, features)

            # Add sparsity regularization for sparse autoencoder
            if self.config.autoencoder_type == "sparse":
                sparsity_loss = self._compute_sparsity_loss(latent)
                loss = loss + float(self.config.sparsity_weight) * sparsity_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    async def _validate(self, data: pl.DataFrame) -> float:
        """Validate model.

        Args:
            data: Validation data

        Returns:
            Validation loss
        """
        self.model.eval()

        features = torch.FloatTensor(data.to_numpy()).to(self.config.device)

        with torch.no_grad():
            if isinstance(self.model, VariationalAutoencoder):
                reconstruction, latent, mu, logvar = self.model(features)
                recon_loss = self._compute_reconstruction_loss(reconstruction, features)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + float(self.config.beta) * kl_loss
            else:
                reconstruction, latent = self.model(features)
                loss = self._compute_reconstruction_loss(reconstruction, features)

        return loss.item()

    def _compute_reconstruction_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss.

        Args:
            reconstruction: Reconstructed output
            target: Target input

        Returns:
            Loss value
        """
        if self.config.reconstruction_loss == "mse":
            return F.mse_loss(reconstruction, target)
        elif self.config.reconstruction_loss == "bce":
            return F.binary_cross_entropy(torch.sigmoid(reconstruction), target)
        else:
            return F.mse_loss(reconstruction, target)

    def _compute_sparsity_loss(self, latent: torch.Tensor) -> torch.Tensor:
        """Compute sparsity regularization loss.

        Args:
            latent: Latent representation

        Returns:
            Sparsity loss
        """
        rho = float(self.config.sparsity_param)
        rho_hat = torch.mean(torch.sigmoid(latent), dim=0)

        kl_div = rho * torch.log(rho / (rho_hat + 1e-10)) + \
                 (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-10))

        return torch.sum(kl_div)

    async def _compute_anomaly_threshold(self, data: pl.DataFrame) -> None:
        """Compute anomaly detection threshold from validation data.

        Args:
            data: Validation data
        """
        try:
            self.model.eval()

            features = torch.FloatTensor(data.to_numpy()).to(self.config.device)

            with torch.no_grad():
                if isinstance(self.model, VariationalAutoencoder):
                    reconstruction, _, _, _ = self.model(features)
                else:
                    reconstruction, _ = self.model(features)

                # Compute reconstruction errors
                errors = torch.mean((reconstruction - features) ** 2, dim=1)
                errors_np = errors.cpu().numpy()

                # Set threshold at 95th percentile
                self.anomaly_threshold = Decimal(str(np.percentile(errors_np, 95)))

            logger.info(
                "Anomaly threshold computed",
                threshold=float(self.anomaly_threshold)
            )

        except Exception as e:
            logger.error("Threshold computation failed", error=str(e))
            raise

    async def detect_anomaly(
        self,
        features: np.ndarray
    ) -> AnomalyScore:
        """Detect anomaly in input features.

        Args:
            features: Input features

        Returns:
            AnomalyScore with detection results

        Example:
            >>> features = np.random.randn(50)
            >>> result = await model.detect_anomaly(features)
            >>> if result.is_anomaly:
            ...     print(f"Anomaly detected! Score: {result.reconstruction_error}")
        """
        try:
            self.model.eval()

            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.config.device)

            with torch.no_grad():
                if isinstance(self.model, VariationalAutoencoder):
                    reconstruction, latent, _, _ = self.model(features_tensor)
                else:
                    reconstruction, latent = self.model(features_tensor)

                # Compute reconstruction error
                error = torch.mean((reconstruction - features_tensor) ** 2).item()
                reconstruction_error = Decimal(str(error))

                # Check if anomaly
                if self.anomaly_threshold is None:
                    # Use default threshold if not computed
                    self.anomaly_threshold = Decimal("1.0")

                is_anomaly = reconstruction_error > self.anomaly_threshold

                latent_np = latent.cpu().numpy().flatten()

            logger.debug(
                "Anomaly detection completed",
                reconstruction_error=float(reconstruction_error),
                is_anomaly=is_anomaly
            )

            return AnomalyScore(
                reconstruction_error=reconstruction_error,
                is_anomaly=is_anomaly,
                threshold=self.anomaly_threshold,
                timestamp=datetime.utcnow(),
                latent_representation=latent_np
            )

        except Exception as e:
            logger.error("Anomaly detection failed", error=str(e))
            raise

    async def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode features to latent representation.

        Args:
            features: Input features

        Returns:
            Latent representation

        Example:
            >>> latent = await model.encode(features)
        """
        try:
            self.model.eval()

            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.config.device)

            with torch.no_grad():
                if isinstance(self.model, VariationalAutoencoder):
                    latent, _, _ = self.model.encoder(features_tensor)
                else:
                    latent = self.model.encoder(features_tensor)

                return latent.cpu().numpy().flatten()

        except Exception as e:
            logger.error("Encoding failed", error=str(e))
            raise

    async def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode latent representation to features.

        Args:
            latent: Latent representation

        Returns:
            Reconstructed features

        Example:
            >>> reconstructed = await model.decode(latent)
        """
        try:
            self.model.eval()

            latent_tensor = torch.FloatTensor(latent).unsqueeze(0).to(self.config.device)

            with torch.no_grad():
                reconstruction = self.model.decoder(latent_tensor)
                return reconstruction.cpu().numpy().flatten()

        except Exception as e:
            logger.error("Decoding failed", error=str(e))
            raise

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
                "training_history": self.training_history,
                "anomaly_threshold": self.anomaly_threshold
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
            self.anomaly_threshold = checkpoint.get("anomaly_threshold")

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
            "latent_dim": self.config.latent_dim,
            "hidden_dims": self.config.hidden_dims,
            "autoencoder_type": self.config.autoencoder_type,
            "anomaly_threshold": float(self.anomaly_threshold) if self.anomaly_threshold else None,
            "training_epochs": len(self.training_history.get("train_loss", []))
        }
