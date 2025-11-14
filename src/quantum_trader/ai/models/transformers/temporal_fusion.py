"""
Temporal Fusion Transformer for Trading

Production-ready Temporal Fusion Transformer implementation for multi-horizon
time series forecasting with interpretable attention mechanisms.
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


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for feature selection."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super(GatedLinearUnit, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gate = self.sigmoid(self.fc2(x))
        output = self.fc1(x) * gate
        return self.dropout(output)


class VariableSelectionNetwork(nn.Module):
    """Variable selection network for feature importance."""

    def __init__(self, input_size: int, num_features: int, hidden_size: int, dropout: float = 0.1):
        super(VariableSelectionNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features

        # Feature-specific GLUs
        self.feature_glus = nn.ModuleList([
            GatedLinearUnit(input_size, hidden_size, dropout)
            for _ in range(num_features)
        ])

        # Variable selection weights
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(num_features * hidden_size, num_features)

    def forward(self, x):
        # x shape: (batch, num_features, input_size)
        batch_size = x.size(0)

        # Apply GLU to each feature
        transformed_features = []
        for i in range(self.num_features):
            transformed = self.feature_glus[i](x[:, i, :])
            transformed_features.append(transformed)

        # Stack features
        stacked = torch.stack(transformed_features, dim=1)  # (batch, num_features, hidden)

        # Calculate selection weights
        flattened = stacked.view(batch_size, -1)
        weights = self.softmax(self.fc(flattened))  # (batch, num_features)

        # Apply weights
        weights_expanded = weights.unsqueeze(2).expand_as(stacked)
        selected = torch.sum(stacked * weights_expanded, dim=1)  # (batch, hidden)

        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with interpretability."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # x shape: (batch, sequence, hidden)
        batch_size, seq_len, _ = x.size()

        # Linear transformations
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_size ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        output = self.output(attended)

        return output, attention_weights


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer architecture."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.1
    ):
        super(TemporalFusionTransformer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Variable selection
        self.variable_selection = VariableSelectionNetwork(
            input_size, input_size, hidden_size, dropout
        )

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Multi-head attention
        self.attention = InterpretableMultiHeadAttention(hidden_size, num_heads, dropout)

        # Gate for skip connection
        self.gate = GatedLinearUnit(hidden_size, hidden_size, dropout)

        # Output layers
        self.fc_out = nn.Linear(hidden_size, output_size)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x shape: (batch, sequence, features)
        batch_size, seq_len, num_features = x.size()

        # Reshape for variable selection
        x_reshaped = x.view(batch_size * seq_len, num_features, -1)

        # Variable selection (apply to each timestep)
        selected_features = []
        feature_weights = []

        for t in range(seq_len):
            timestep_data = x[:, t, :].unsqueeze(2)  # (batch, features, 1)
            selected, weights = self.variable_selection(timestep_data)
            selected_features.append(selected)
            feature_weights.append(weights)

        # Stack temporal features
        temporal_features = torch.stack(selected_features, dim=1)  # (batch, seq, hidden)

        # LSTM processing
        lstm_out, _ = self.lstm(temporal_features)

        # Self-attention
        attended, attention_weights = self.attention(lstm_out)

        # Gated skip connection
        gated = self.gate(attended)
        combined = self.layer_norm(gated + lstm_out)

        # Take last timestep for prediction
        final_features = combined[:, -1, :]

        # Output projection
        output = self.fc_out(final_features)

        return output, attention_weights, feature_weights


class TemporalFusionModel(BaseMLModel):
    """
    Temporal Fusion Transformer for trading.

    Multi-horizon forecasting with interpretable attention mechanisms
    for understanding temporal dependencies and feature importance.

    Attributes:
        config: Configuration dictionary
        model: PyTorch TFT model
        device: Computation device
        is_trained: Training status

    Example:
        >>> config = {"model": {"tft": {"hidden_size": 128, "num_heads": 4}}}
        >>> model = TemporalFusionModel(config)
        >>> model.train(features, labels)
        >>> predictions, attention, importance = model.predict_with_interpretation(new_features)
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialize Temporal Fusion model.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__(config)
        self.config = config
        self._validate_config()

        tft_config = self.config.get("model", {}).get("tft", {})

        # Model architecture parameters
        self.hidden_size: int = tft_config.get("hidden_size", int(os.getenv("TFT_HIDDEN_SIZE", "128")))
        self.num_heads: int = tft_config.get("num_heads", int(os.getenv("TFT_NUM_HEADS", "4")))
        self.num_layers: int = tft_config.get("num_layers", int(os.getenv("TFT_NUM_LAYERS", "2")))
        self.dropout: float = float(tft_config.get("dropout", os.getenv("TFT_DROPOUT", "0.1")))
        self.output_size: int = tft_config.get("output_size", int(os.getenv("TFT_OUTPUT_SIZE", "1")))

        # Training parameters
        self.learning_rate: float = float(tft_config.get("learning_rate", os.getenv("TFT_LEARNING_RATE", "0.001")))
        self.batch_size: int = tft_config.get("batch_size", int(os.getenv("TFT_BATCH_SIZE", "32")))
        self.num_epochs: int = tft_config.get("num_epochs", int(os.getenv("TFT_NUM_EPOCHS", "100")))
        self.patience: int = tft_config.get("patience", int(os.getenv("TFT_PATIENCE", "15")))
        self.gradient_clip: float = float(tft_config.get("gradient_clip", os.getenv("TFT_GRADIENT_CLIP", "1.0")))

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model and optimizer
        self.model: Optional[TemporalFusionTransformer] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion: nn.Module = nn.MSELoss()
        self.is_trained: bool = False

        # Store attention weights for interpretation
        self.last_attention_weights: Optional[torch.Tensor] = None
        self.last_feature_weights: Optional[List[torch.Tensor]] = None

        logger.info(
            "tft_model_initialized",
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            device=str(self.device)
        )

    def _validate_config(self) -> None:
        """
        Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        tft_config = self.config.get("model", {}).get("tft", {})

        if tft_config:
            hidden_size = tft_config.get("hidden_size", 128)
            if hidden_size <= 0:
                raise ValueError("hidden_size must be positive")

            num_heads = tft_config.get("num_heads", 4)
            if hidden_size % num_heads != 0:
                raise ValueError("hidden_size must be divisible by num_heads")

    def _initialize_model(self, input_size: int) -> None:
        """Initialize model architecture."""
        self.model = TemporalFusionTransformer(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            output_size=self.output_size,
            dropout=self.dropout
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        logger.info("tft_network_created", input_size=input_size, output_size=self.output_size)

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Train TFT model.

        Args:
            features: Training features (n_samples, sequence_length, n_features)
            labels: Training labels (n_samples, output_size)

        Raises:
            ValueError: If inputs are invalid
        """
        try:
            # Validate inputs
            if not isinstance(features, np.ndarray):
                raise ValueError("features must be numpy array")
            if not isinstance(labels, np.ndarray):
                raise ValueError("labels must be numpy array")

            if len(features) == 0 or len(labels) == 0:
                raise ValueError("Cannot train on empty data")

            if len(features) != len(labels):
                raise ValueError("features and labels must have same length")

            # Ensure 3D features
            if len(features.shape) == 2:
                features = features[:, :, np.newaxis]

            n_samples, seq_length, n_features = features.shape

            logger.info(
                "tft_training_started",
                n_samples=n_samples,
                seq_length=seq_length,
                n_features=n_features
            )

            # Initialize model if not done
            if self.model is None:
                self._initialize_model(n_features)

            # Prepare data
            X_tensor = torch.FloatTensor(features).to(self.device)
            y_tensor = torch.FloatTensor(labels).to(self.device)

            if len(y_tensor.shape) == 1:
                y_tensor = y_tensor.unsqueeze(1)

            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Training loop
            best_loss = float('inf')
            patience_counter = 0

            for epoch in range(self.num_epochs):
                self.model.train()
                epoch_loss = 0.0

                for batch_X, batch_y in dataloader:
                    self.optimizer.zero_grad()

                    outputs, _, _ = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)

                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                    self.optimizer.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(dataloader)

                # Early stopping check
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    logger.debug(f"tft_training_epoch", epoch=epoch + 1, loss=avg_loss)

                if patience_counter >= self.patience:
                    logger.info("tft_early_stopping", epoch=epoch + 1, best_loss=best_loss)
                    break

            self.is_trained = True

            logger.info(
                "tft_training_completed",
                final_loss=best_loss,
                epochs_trained=epoch + 1
            )

        except Exception as e:
            logger.error("tft_training_failed", error=str(e))
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            features: Input features (n_samples, sequence_length, n_features)

        Returns:
            Predictions array

        Raises:
            ValueError: If model not trained or inputs invalid
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")

            if not isinstance(features, np.ndarray):
                raise ValueError("features must be numpy array")

            # Ensure 3D features
            if len(features.shape) == 2:
                features = features[:, :, np.newaxis]

            X_tensor = torch.FloatTensor(features).to(self.device)

            self.model.eval()
            with torch.no_grad():
                predictions, attention_weights, feature_weights = self.model(X_tensor)

                # Store for interpretation
                self.last_attention_weights = attention_weights
                self.last_feature_weights = feature_weights

            predictions_np = predictions.cpu().numpy().squeeze()

            logger.debug("tft_predictions_generated", n_samples=len(predictions_np))

            return predictions_np

        except Exception as e:
            logger.error("tft_prediction_failed", error=str(e))
            raise

    def predict_with_interpretation(
        self,
        features: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[np.ndarray]]]:
        """
        Generate predictions with interpretation.

        Args:
            features: Input features

        Returns:
            Tuple of (predictions, attention_weights, feature_importance_weights)
        """
        predictions = self.predict(features)

        attention_np = None
        if self.last_attention_weights is not None:
            attention_np = self.last_attention_weights.cpu().numpy()

        feature_weights_np = None
        if self.last_feature_weights is not None:
            feature_weights_np = [fw.cpu().numpy() for fw in self.last_feature_weights]

        return predictions, attention_np, feature_weights_np

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            features: Test features
            labels: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")

            predictions = self.predict(features)

            # Ensure same shape
            if len(predictions.shape) == 1 and len(labels.shape) == 1:
                pass
            elif len(predictions.shape) != len(labels.shape):
                if len(labels.shape) == 1:
                    labels = labels.reshape(-1, 1)
                if len(predictions.shape) == 1:
                    predictions = predictions.reshape(-1, 1)

            # Calculate metrics
            mse = np.mean((labels - predictions) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(labels - predictions))

            # R² score
            ss_res = np.sum((labels - predictions) ** 2)
            ss_tot = np.sum((labels - np.mean(labels)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))

            # MAPE
            mape = np.mean(np.abs((labels - predictions) / (labels + 1e-10))) * 100

            metrics = {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "r2_score": float(r2),
                "mape": float(mape)
            }

            logger.info("tft_evaluation_completed", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("tft_evaluation_failed", error=str(e))
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
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "output_size": self.output_size,
                "input_size": self.model.input_size,
                "is_trained": self.is_trained
            }

            torch.save(save_dict, path)
            logger.info("tft_model_saved", path=path)

        except Exception as e:
            logger.error("tft_save_failed", path=path, error=str(e))
            raise

    def load(self, path: str) -> None:
        """Load model from disk."""
        try:
            if not Path(path).exists():
                raise FileNotFoundError(f"Model file not found: {path}")

            checkpoint = torch.load(path, map_location=self.device)

            # Restore configuration
            self.hidden_size = checkpoint["hidden_size"]
            self.num_heads = checkpoint["num_heads"]
            self.num_layers = checkpoint["num_layers"]
            self.dropout = checkpoint["dropout"]
            self.output_size = checkpoint["output_size"]

            # Recreate model
            self._initialize_model(checkpoint["input_size"])

            # Load states
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.is_trained = checkpoint["is_trained"]

            logger.info("tft_model_loaded", path=path)

        except Exception as e:
            logger.error("tft_load_failed", path=path, error=str(e))
            raise
