"""
Autoencoder Model for Feature Learning and Anomaly Detection.

Implements variational and denoising autoencoders for unsupervised
feature learning and anomaly detection in trading data.
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
class AutoencoderConfig:
    """Configuration for autoencoder model."""

    input_dim: int
    latent_dim: int = 64
    hidden_dims: List[int] = None
    dropout: Decimal = Decimal("0.1")
    use_batch_norm: bool = True
    activation: str = "relu"
    variational: bool = False
    denoising: bool = False
    noise_factor: Decimal = Decimal("0.1")

    def __post_init__(self):
        """Set default hidden dimensions if not provided."""
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class Encoder(nn.Module):
    """Encoder network for autoencoder."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int],
        dropout: float,
        use_batch_norm: bool,
        variational: bool
    ):
        """
        Initialize encoder.

        Args:
            input_dim: Input dimension
            latent_dim: Latent dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            variational: Whether this is a variational autoencoder
        """
        super(Encoder, self).__init__()

        self.variational = variational

        # Build encoder layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Latent space projection
        if variational:
            # Separate projections for mean and log variance
            self.fc_mu = nn.Linear(prev_dim, latent_dim)
            self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        else:
            self.fc_latent = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor

        Returns:
            Latent representation (and mu, logvar for VAE)
        """
        h = self.encoder(x)

        if self.variational:
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar
        else:
            z = self.fc_latent(h)
            return (z,)


class Decoder(nn.Module):
    """Decoder network for autoencoder."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout: float,
        use_batch_norm: bool
    ):
        """
        Initialize decoder.

        Args:
            latent_dim: Latent dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions (reversed from encoder)
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(Decoder, self).__init__()

        # Build decoder layers (reverse of encoder)
        layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            z: Latent representation

        Returns:
            Reconstructed output
        """
        return self.decoder(z)


class AutoencoderModel(nn.Module):
    """Complete autoencoder model."""

    def __init__(self, config: AutoencoderConfig):
        """
        Initialize autoencoder.

        Args:
            config: Autoencoder configuration
        """
        super(AutoencoderModel, self).__init__()

        self.config = config

        dropout = float(config.dropout)

        # Encoder
        self.encoder = Encoder(
            config.input_dim,
            config.latent_dim,
            config.hidden_dims,
            dropout,
            config.use_batch_norm,
            config.variational
        )

        # Decoder
        self.decoder = Decoder(
            config.latent_dim,
            config.input_dim,
            config.hidden_dims,
            dropout,
            config.use_batch_norm
        )

        # Initialize weights
        self._initialize_weights()

        logger.info(
            "Initialized autoencoder model",
            extra={
                "input_dim": config.input_dim,
                "latent_dim": config.latent_dim,
                "hidden_dims": config.hidden_dims,
                "variational": config.variational,
                "denoising": config.denoising,
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

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add noise for denoising autoencoder.

        Args:
            x: Input tensor

        Returns:
            Noisy input
        """
        noise = torch.randn_like(x) * float(self.config.noise_factor)
        return x + noise

    def forward(
        self,
        x: torch.Tensor,
        add_noise: bool = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through autoencoder.

        Args:
            x: Input tensor
            add_noise: Whether to add noise (for denoising AE)

        Returns:
            Tuple of (reconstruction, latent, [mu, logvar])
        """
        # Add noise if denoising autoencoder
        if add_noise is None:
            add_noise = self.config.denoising and self.training

        if add_noise:
            x_noisy = self.add_noise(x)
        else:
            x_noisy = x

        # Encode
        encoder_output = self.encoder(x_noisy)

        if self.config.variational:
            mu, logvar = encoder_output
            z = self.reparameterize(mu, logvar)
        else:
            z = encoder_output[0]
            mu = logvar = None

        # Decode
        reconstruction = self.decoder(z)

        if self.config.variational:
            return reconstruction, z, mu, logvar
        else:
            return reconstruction, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.

        Args:
            x: Input tensor

        Returns:
            Latent representation
        """
        with torch.no_grad():
            encoder_output = self.encoder(x)

            if self.config.variational:
                mu, logvar = encoder_output
                z = self.reparameterize(mu, logvar)
            else:
                z = encoder_output[0]

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output.

        Args:
            z: Latent tensor

        Returns:
            Reconstructed output
        """
        with torch.no_grad():
            reconstruction = self.decoder(z)

        return reconstruction

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input through autoencoder.

        Args:
            x: Input tensor

        Returns:
            Reconstructed output
        """
        self.eval()

        with torch.no_grad():
            output = self.forward(x, add_noise=False)
            reconstruction = output[0]

        return reconstruction

    def compute_reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute reconstruction error.

        Args:
            x: Input tensor
            reduction: Reduction method (mean, sum, none)

        Returns:
            Reconstruction error
        """
        reconstruction = self.reconstruct(x)

        # Compute MSE
        error = F.mse_loss(reconstruction, x, reduction=reduction)

        return error


class AutoencoderLoss(nn.Module):
    """Loss function for autoencoder."""

    def __init__(
        self,
        variational: bool = False,
        beta: Decimal = Decimal("1.0")
    ):
        """
        Initialize loss function.

        Args:
            variational: Whether this is for VAE
            beta: Weight for KL divergence (for beta-VAE)
        """
        super(AutoencoderLoss, self).__init__()

        self.variational = variational
        self.beta = float(beta)

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss.

        Args:
            reconstruction: Reconstructed output
            target: Target input
            mu: Mean (for VAE)
            logvar: Log variance (for VAE)

        Returns:
            Tuple of (total loss, loss components dict)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, target)

        losses = {"reconstruction_loss": recon_loss}

        if self.variational and mu is not None and logvar is not None:
            # KL divergence loss
            kl_loss = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp()
            )
            losses["kl_loss"] = kl_loss

            # Total loss
            total_loss = recon_loss + self.beta * kl_loss
        else:
            total_loss = recon_loss

        losses["total_loss"] = total_loss

        return total_loss, losses


class AnomalyDetector:
    """Anomaly detector using autoencoder."""

    def __init__(
        self,
        model: AutoencoderModel,
        threshold: Optional[Decimal] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize anomaly detector.

        Args:
            model: Trained autoencoder model
            threshold: Anomaly threshold (computed if not provided)
            device: Computing device
        """
        self.model = model
        self.threshold = threshold
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "Initialized anomaly detector",
            extra={
                "threshold": str(threshold) if threshold else "auto",
                "device": str(self.device),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    def fit_threshold(
        self,
        normal_df: pl.DataFrame,
        feature_cols: List[str],
        percentile: Decimal = Decimal("95")
    ) -> Decimal:
        """
        Fit anomaly threshold using normal data.

        Args:
            normal_df: DataFrame with normal samples
            feature_cols: Feature column names
            percentile: Percentile for threshold

        Returns:
            Computed threshold
        """
        # Compute reconstruction errors on normal data
        features = normal_df.select(feature_cols).to_numpy()
        X = torch.tensor(features, dtype=torch.float32, device=self.device)

        errors = []
        batch_size = 256

        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            error = self.model.compute_reconstruction_error(batch, reduction="none")
            errors.append(error.cpu().numpy())

        import numpy as np

        all_errors = np.concatenate(errors)

        # Set threshold at specified percentile
        threshold_value = np.percentile(all_errors, float(percentile))
        self.threshold = Decimal(str(threshold_value))

        logger.info(
            "Fitted anomaly threshold",
            extra={
                "threshold": str(self.threshold),
                "percentile": str(percentile),
                "num_samples": len(normal_df),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return self.threshold

    def detect(
        self,
        data_df: pl.DataFrame,
        feature_cols: List[str]
    ) -> pl.DataFrame:
        """
        Detect anomalies in data.

        Args:
            data_df: DataFrame to check
            feature_cols: Feature column names

        Returns:
            DataFrame with anomaly scores and flags
        """
        if self.threshold is None:
            raise ValueError(
                "Threshold not set. Call fit_threshold() first or provide threshold."
            )

        # Compute reconstruction errors
        features = data_df.select(feature_cols).to_numpy()
        X = torch.tensor(features, dtype=torch.float32, device=self.device)

        errors = []
        batch_size = 256

        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            error = self.model.compute_reconstruction_error(batch, reduction="none")
            errors.append(error.cpu().numpy())

        import numpy as np

        all_errors = np.concatenate(errors)

        # Create result DataFrame
        result_df = data_df.with_columns([
            pl.Series("anomaly_score", all_errors.tolist()),
            pl.Series(
                "is_anomaly",
                (all_errors > float(self.threshold)).tolist()
            )
        ])

        num_anomalies = result_df.filter(pl.col("is_anomaly")).shape[0]

        logger.info(
            "Detected anomalies",
            extra={
                "num_samples": len(result_df),
                "num_anomalies": num_anomalies,
                "anomaly_rate": f"{num_anomalies / len(result_df):.4f}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return result_df


class AutoencoderTrainer:
    """Trainer for autoencoder model."""

    def __init__(
        self,
        model: AutoencoderModel,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Autoencoder model
            device: Computing device
        """
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        # Create loss function
        self.criterion = AutoencoderLoss(
            variational=model.config.variational
        )

        logger.info(
            "Initialized autoencoder trainer",
            extra={
                "device": str(self.device),
                "variational": model.config.variational,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    async def train_epoch(
        self,
        train_df: pl.DataFrame,
        feature_cols: List[str],
        optimizer: torch.optim.Optimizer,
        batch_size: int = 32
    ) -> Dict[str, Decimal]:
        """
        Train for one epoch.

        Args:
            train_df: Training data
            feature_cols: Feature column names
            optimizer: Optimizer
            batch_size: Batch size

        Returns:
            Training metrics
        """
        self.model.train()

        features = train_df.select(feature_cols).to_numpy()
        num_samples = len(features)

        total_loss = Decimal("0")
        total_recon_loss = Decimal("0")
        total_kl_loss = Decimal("0")
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            batch_features = features[i : i + batch_size]

            X = torch.tensor(
                batch_features,
                dtype=torch.float32,
                device=self.device
            )

            optimizer.zero_grad()

            # Forward pass
            output = self.model(X)

            if self.model.config.variational:
                reconstruction, z, mu, logvar = output
                loss, loss_dict = self.criterion(reconstruction, X, mu, logvar)
                total_kl_loss += Decimal(str(loss_dict["kl_loss"].item()))
            else:
                reconstruction, z = output
                loss, loss_dict = self.criterion(reconstruction, X)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += Decimal(str(loss.item()))
            total_recon_loss += Decimal(str(loss_dict["reconstruction_loss"].item()))
            num_batches += 1

        # Average losses
        avg_loss = total_loss / num_batches if num_batches > 0 else Decimal("0")
        avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else Decimal("0")

        metrics = {
            "total_loss": avg_loss,
            "reconstruction_loss": avg_recon_loss,
            "num_batches": Decimal(str(num_batches))
        }

        if self.model.config.variational:
            avg_kl_loss = total_kl_loss / num_batches if num_batches > 0 else Decimal("0")
            metrics["kl_loss"] = avg_kl_loss

        return metrics

    async def evaluate(
        self,
        eval_df: pl.DataFrame,
        feature_cols: List[str]
    ) -> Dict[str, Decimal]:
        """
        Evaluate model.

        Args:
            eval_df: Evaluation data
            feature_cols: Feature column names

        Returns:
            Evaluation metrics
        """
        self.model.eval()

        features = eval_df.select(feature_cols).to_numpy()
        X = torch.tensor(features, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            output = self.model(X, add_noise=False)

            if self.model.config.variational:
                reconstruction, z, mu, logvar = output
                loss, loss_dict = self.criterion(reconstruction, X, mu, logvar)
            else:
                reconstruction, z = output
                loss, loss_dict = self.criterion(reconstruction, X)

        metrics = {
            "eval_loss": Decimal(str(loss.item())),
            "reconstruction_loss": Decimal(str(loss_dict["reconstruction_loss"].item()))
        }

        if self.model.config.variational:
            metrics["kl_loss"] = Decimal(str(loss_dict["kl_loss"].item()))

        return metrics
