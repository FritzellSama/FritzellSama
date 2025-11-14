"""Generative Adversarial Network for market data generation and augmentation.

This module implements a GAN for generating synthetic market data, useful for
data augmentation in low-data regimes and stress testing trading strategies.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel

logger = get_logger(__name__)


class Generator(nn.Module):
    """Generator network for GAN.

    Transforms random noise into synthetic market data sequences.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        sequence_length: int
    ) -> None:
        """Initialize generator.

        Args:
            latent_dim: Dimension of latent noise vector
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            sequence_length: Length of generated sequences
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.output_dim = output_dim

        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.lstm = nn.LSTM(
            hidden_dim * 2,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.LeakyReLU(0.2)
        self.output_activation = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Latent noise tensor [batch_size, latent_dim]

        Returns:
            Generated sequences [batch_size, sequence_length, output_dim]
        """
        x = self.activation(self.fc1(z))
        x = self.activation(self.fc2(x))

        # Repeat for sequence length
        x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)

        # LSTM processing
        x, _ = self.lstm(x)

        # Output projection
        x = self.fc3(x)
        x = self.output_activation(x)

        return x


class Discriminator(nn.Module):
    """Discriminator network for GAN.

    Classifies sequences as real or generated.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sequence_length: int
    ) -> None:
        """Initialize discriminator.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            sequence_length: Length of input sequences
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.fc1 = nn.Linear(hidden_dim * sequence_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input sequences [batch_size, sequence_length, input_dim]

        Returns:
            Discriminator scores [batch_size, 1]
        """
        # LSTM processing
        lstm_out, _ = self.lstm(x)

        # Flatten
        lstm_out = lstm_out.reshape(lstm_out.size(0), -1)

        # FC layers
        x = self.activation(self.fc1(lstm_out))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))

        return x


class GANModel(BaseMLModel):
    """GAN model for market data generation.

    Uses adversarial training to learn market data distribution and
    generate realistic synthetic sequences for data augmentation.

    Attributes:
        config: Model configuration
        generator: Generator network
        discriminator: Discriminator network
        device: Torch device (CPU/GPU)

    Example:
        >>> config = {
        ...     "latent_dim": 100,
        ...     "hidden_dim": 256,
        ...     "sequence_length": 100,
        ...     "learning_rate_g": "0.0002",
        ...     "learning_rate_d": "0.0002",
        ...     "batch_size": 64
        ... }
        >>> model = GANModel(config)
        >>> model.train(real_data, labels=None)
        >>> synthetic_data = model.generate_samples(num_samples=1000)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize GAN model.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If config is invalid
        """
        super().__init__(config)

        self._validate_config()

        # Model parameters
        self.latent_dim = config["latent_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.sequence_length = config["sequence_length"]
        self.feature_dim = config.get("feature_dim", 5)  # OHLCV by default

        # Training parameters
        self.learning_rate_g = Decimal(str(config.get("learning_rate_g", "0.0002")))
        self.learning_rate_d = Decimal(str(config.get("learning_rate_d", "0.0002")))
        self.batch_size = config.get("batch_size", 64)
        self.num_epochs = config.get("num_epochs", 100)
        self.critic_iterations = config.get("critic_iterations", 5)

        # Device setup
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize networks
        self.generator = Generator(
            self.latent_dim,
            self.hidden_dim,
            self.feature_dim,
            self.sequence_length
        ).to(self.device)

        self.discriminator = Discriminator(
            self.feature_dim,
            self.hidden_dim,
            self.sequence_length
        ).to(self.device)

        # Optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=float(self.learning_rate_g),
            betas=(0.5, 0.999)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=float(self.learning_rate_d),
            betas=(0.5, 0.999)
        )

        # Loss function
        self.criterion = nn.BCELoss()

        # Training state
        self.training_history: List[Dict[str, float]] = []

        logger.info(
            "GAN model initialized",
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            device=str(self.device)
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If required config missing
        """
        required_keys = [
            "latent_dim",
            "hidden_dim",
            "sequence_length"
        ]

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def train(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> None:
        """Train GAN on real market data.

        Args:
            features: Real market data [num_samples, sequence_length, feature_dim]
            labels: Not used for GANs (kept for interface compatibility)

        Raises:
            ValueError: If features shape is invalid
        """
        try:
            if features.shape[1] != self.sequence_length:
                raise ValueError(
                    f"Expected sequence length {self.sequence_length}, "
                    f"got {features.shape[1]}"
                )

            if features.shape[2] != self.feature_dim:
                raise ValueError(
                    f"Expected feature dim {self.feature_dim}, "
                    f"got {features.shape[2]}"
                )

            logger.info(
                "Starting GAN training",
                num_samples=len(features),
                epochs=self.num_epochs
            )

            # Convert to tensor
            real_data = torch.FloatTensor(features).to(self.device)

            # Training loop
            for epoch in range(self.num_epochs):
                epoch_g_loss = Decimal("0")
                epoch_d_loss = Decimal("0")
                num_batches = 0

                # Batch training
                for i in range(0, len(real_data), self.batch_size):
                    batch_real = real_data[i:i + self.batch_size]

                    if len(batch_real) < self.batch_size:
                        continue

                    # Train discriminator
                    d_loss = self._train_discriminator_step(batch_real)
                    epoch_d_loss += Decimal(str(d_loss))

                    # Train generator (less frequently)
                    if num_batches % self.critic_iterations == 0:
                        g_loss = self._train_generator_step(len(batch_real))
                        epoch_g_loss += Decimal(str(g_loss))

                    num_batches += 1

                # Log epoch metrics
                avg_g_loss = float(epoch_g_loss / max(num_batches // self.critic_iterations, 1))
                avg_d_loss = float(epoch_d_loss / num_batches)

                self.training_history.append({
                    "epoch": epoch,
                    "g_loss": avg_g_loss,
                    "d_loss": avg_d_loss
                })

                if epoch % 10 == 0:
                    logger.info(
                        "Training progress",
                        epoch=epoch,
                        g_loss=avg_g_loss,
                        d_loss=avg_d_loss
                    )

            logger.info("GAN training completed", epochs=self.num_epochs)

        except Exception as e:
            logger.error("Failed to train GAN", error=str(e))
            raise

    def _train_discriminator_step(self, real_batch: torch.Tensor) -> float:
        """Single discriminator training step.

        Args:
            real_batch: Batch of real data

        Returns:
            Discriminator loss value
        """
        self.discriminator.zero_grad()

        batch_size = len(real_batch)

        # Labels
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)

        # Train on real data
        real_output = self.discriminator(real_batch)
        real_loss = self.criterion(real_output, real_labels)

        # Train on fake data
        noise = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.generator(noise)
        fake_output = self.discriminator(fake_data.detach())
        fake_loss = self.criterion(fake_output, fake_labels)

        # Combined loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.optimizer_d.step()

        return d_loss.item()

    def _train_generator_step(self, batch_size: int) -> float:
        """Single generator training step.

        Args:
            batch_size: Size of batch to generate

        Returns:
            Generator loss value
        """
        self.generator.zero_grad()

        # Generate fake data
        noise = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.generator(noise)

        # Try to fool discriminator
        fake_output = self.discriminator(fake_data)
        real_labels = torch.ones(batch_size, 1).to(self.device)

        g_loss = self.criterion(fake_output, real_labels)
        g_loss.backward()
        self.optimizer_g.step()

        return g_loss.item()

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate synthetic data (not used for prediction).

        Args:
            features: Not used

        Returns:
            Generated samples
        """
        num_samples = len(features) if len(features) > 0 else self.batch_size
        return self.generate_samples(num_samples)

    def generate_samples(self, num_samples: int) -> np.ndarray:
        """Generate synthetic market data samples.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Generated data [num_samples, sequence_length, feature_dim]
        """
        try:
            self.generator.eval()

            with torch.no_grad():
                noise = torch.randn(num_samples, self.latent_dim).to(self.device)
                generated = self.generator(noise)
                samples = generated.cpu().numpy()

            logger.info("Generated synthetic samples", num_samples=num_samples)

            return samples

        except Exception as e:
            logger.error("Failed to generate samples", error=str(e))
            raise

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate GAN quality metrics.

        Args:
            features: Real data for comparison
            labels: Not used

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Generate same number of samples
            generated = self.generate_samples(len(features))

            # Calculate distribution metrics
            metrics = {}

            for i in range(self.feature_dim):
                real_feature = features[:, :, i].flatten()
                gen_feature = generated[:, :, i].flatten()

                # Mean absolute difference
                metrics[f"feature_{i}_mean_diff"] = float(
                    np.abs(real_feature.mean() - gen_feature.mean())
                )

                # Std difference
                metrics[f"feature_{i}_std_diff"] = float(
                    np.abs(real_feature.std() - gen_feature.std())
                )

            logger.info("GAN evaluation completed", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("Failed to evaluate GAN", error=str(e))
            raise

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Directory path to save model
        """
        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)

            # Save generator
            torch.save(
                self.generator.state_dict(),
                save_path / "generator.pt"
            )

            # Save discriminator
            torch.save(
                self.discriminator.state_dict(),
                save_path / "discriminator.pt"
            )

            # Save config and history
            torch.save(
                {
                    "config": self.config,
                    "training_history": self.training_history
                },
                save_path / "metadata.pt"
            )

            logger.info("GAN model saved", path=path)

        except Exception as e:
            logger.error("Failed to save GAN model", error=str(e))
            raise

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Directory path to load model from
        """
        try:
            load_path = Path(path)

            # Load generator
            self.generator.load_state_dict(
                torch.load(load_path / "generator.pt", map_location=self.device)
            )

            # Load discriminator
            self.discriminator.load_state_dict(
                torch.load(load_path / "discriminator.pt", map_location=self.device)
            )

            # Load metadata
            metadata = torch.load(load_path / "metadata.pt", map_location=self.device)
            self.training_history = metadata["training_history"]

            logger.info("GAN model loaded", path=path)

        except Exception as e:
            logger.error("Failed to load GAN model", error=str(e))
            raise
