"""Few-shot learning model for rapid adaptation to new market regimes.

This module implements a few-shot learning model that can quickly adapt
to new market conditions with minimal training examples. Uses metric learning
and prototype networks for efficient learning from limited data.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import polars as pl
from structlog import get_logger
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

logger = get_logger(__name__)


class BaseMLModel(ABC):
    """Abstract base for all ML models."""

    def __init__(self, config: Dict) -> None:
        """Initialize base model.

        Args:
            config: Model configuration dictionary
        """
        self.config = config

    @abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the model.

        Args:
            features: Training features
            labels: Training labels
        """
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            features: Input features

        Returns:
            Predictions
        """
        pass

    @abstractmethod
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            features: Evaluation features
            labels: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to load model from
        """
        pass


class PrototypeNetwork(nn.Module):
    """Prototype network for few-shot learning.

    Learns an embedding space where examples from the same class
    cluster together, enabling classification with few examples.

    Attributes:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        embedding_dim: Output embedding dimension
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        embedding_dim: int,
        dropout: float = 0.1
    ) -> None:
        """Initialize prototype network.

        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            embedding_dim: Embedding dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Build encoder network
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

        # Final embedding layer
        layers.append(nn.Linear(prev_dim, embedding_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Embedding tensor (batch_size, embedding_dim)
        """
        return self.encoder(x)

    def compute_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        n_classes: int
    ) -> torch.Tensor:
        """Compute class prototypes from support set.

        Args:
            support_embeddings: Support set embeddings (n_support, embedding_dim)
            support_labels: Support set labels (n_support,)
            n_classes: Number of classes

        Returns:
            Prototypes tensor (n_classes, embedding_dim)
        """
        prototypes = []

        for class_idx in range(n_classes):
            class_mask = support_labels == class_idx
            class_embeddings = support_embeddings[class_mask]

            if class_embeddings.size(0) > 0:
                prototype = class_embeddings.mean(dim=0)
            else:
                # Handle empty class
                prototype = torch.zeros(self.embedding_dim, device=support_embeddings.device)

            prototypes.append(prototype)

        return torch.stack(prototypes)


class FewShotLearner(BaseMLModel):
    """Few-shot learning model for trading signals.

    Implements a metric learning approach that can adapt to new market
    regimes with only a few training examples. Uses prototype networks
    to learn an embedding space optimized for classification.

    Attributes:
        config: Model configuration
        network: Prototype network
        scaler: Feature scaler
        device: PyTorch device (CPU/GPU)

    Example:
        >>> config = {
        ...     'input_dim': 50,
        ...     'hidden_dims': [128, 64],
        ...     'embedding_dim': 32,
        ...     'learning_rate': 0.001,
        ...     'n_episodes': 1000,
        ...     'n_way': 3,
        ...     'k_shot': 5,
        ...     'n_query': 10
        ... }
        >>> model = FewShotLearner(config)
        >>> model.train(features, labels)
        >>> predictions = model.predict(new_features)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize few-shot learner.

        Args:
            config: Configuration dictionary with keys:
                - input_dim: Input feature dimension
                - hidden_dims: List of hidden layer sizes
                - embedding_dim: Embedding dimension
                - learning_rate: Learning rate
                - n_episodes: Number of training episodes
                - n_way: Number of classes per episode
                - k_shot: Number of examples per class in support set
                - n_query: Number of query examples per class
                - dropout: Dropout probability
                - device: Device ('cpu' or 'cuda')

        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__(config)
        self._validate_config()

        # Model parameters
        self.input_dim = self.config['input_dim']
        self.hidden_dims = self.config.get('hidden_dims', [128, 64])
        self.embedding_dim = self.config.get('embedding_dim', 32)
        self.learning_rate = Decimal(str(self.config.get('learning_rate', 0.001)))
        self.dropout = self.config.get('dropout', 0.1)

        # Training parameters
        self.n_episodes = self.config.get('n_episodes', 1000)
        self.n_way = self.config.get('n_way', 3)
        self.k_shot = self.config.get('k_shot', 5)
        self.n_query = self.config.get('n_query', 10)

        # Device configuration
        device_str = self.config.get('device', 'cpu')
        self.device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

        # Initialize network
        self.network = PrototypeNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            embedding_dim=self.embedding_dim,
            dropout=self.dropout
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=float(self.learning_rate)
        )

        # Scaler for feature normalization
        self.scaler = StandardScaler()

        # Training state
        self.is_fitted = False
        self.training_history: List[Dict[str, Any]] = []
        self.prototypes: Optional[torch.Tensor] = None
        self.n_classes: int = 0

        logger.info(
            "few_shot_learner_initialized",
            input_dim=self.input_dim,
            embedding_dim=self.embedding_dim,
            device=str(self.device)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = ['input_dim']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if self.config['input_dim'] < 1:
            raise ValueError("input_dim must be positive")

        if 'embedding_dim' in self.config and self.config['embedding_dim'] < 1:
            raise ValueError("embedding_dim must be positive")

        if 'n_episodes' in self.config and self.config['n_episodes'] < 1:
            raise ValueError("n_episodes must be positive")

        if 'k_shot' in self.config and self.config['k_shot'] < 1:
            raise ValueError("k_shot must be positive")

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the few-shot learning model.

        Uses episodic training where each episode samples n_way classes
        with k_shot support examples and n_query query examples per class.

        Args:
            features: Training features (n_samples, n_features)
            labels: Training labels (n_samples,)

        Raises:
            ValueError: If input data is invalid
        """
        try:
            if features.shape[0] != labels.shape[0]:
                raise ValueError("Features and labels must have same length")

            if features.shape[1] != self.input_dim:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self.input_dim}, "
                    f"got {features.shape[1]}"
                )

            # Normalize features
            features_scaled = self.scaler.fit_transform(features)

            # Get unique classes
            unique_labels = np.unique(labels)
            self.n_classes = len(unique_labels)

            if self.n_classes < self.n_way:
                raise ValueError(
                    f"Not enough classes: need {self.n_way}, got {self.n_classes}"
                )

            # Convert to tensors
            features_tensor = torch.from_numpy(features_scaled).float()
            labels_tensor = torch.from_numpy(labels).long()

            # Episodic training
            self.network.train()
            self.training_history = []

            logger.info(
                "few_shot_training_started",
                n_episodes=self.n_episodes,
                n_way=self.n_way,
                k_shot=self.k_shot
            )

            for episode in range(self.n_episodes):
                # Sample episode
                support_x, support_y, query_x, query_y = self._sample_episode(
                    features_tensor,
                    labels_tensor,
                    unique_labels
                )

                # Move to device
                support_x = support_x.to(self.device)
                support_y = support_y.to(self.device)
                query_x = query_x.to(self.device)
                query_y = query_y.to(self.device)

                # Forward pass
                support_embeddings = self.network(support_x)
                query_embeddings = self.network(query_x)

                # Compute prototypes
                prototypes = self.network.compute_prototypes(
                    support_embeddings,
                    support_y,
                    self.n_way
                )

                # Compute distances to prototypes
                distances = self._compute_distances(query_embeddings, prototypes)

                # Compute loss (negative log-likelihood)
                log_probs = F.log_softmax(-distances, dim=1)
                loss = F.nll_loss(log_probs, query_y)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track metrics
                if episode % 100 == 0:
                    accuracy = self._compute_accuracy(distances, query_y)

                    self.training_history.append({
                        'episode': episode,
                        'loss': float(loss.item()),
                        'accuracy': float(accuracy),
                        'timestamp': datetime.now(timezone.utc)
                    })

                    logger.info(
                        "training_progress",
                        episode=episode,
                        loss=f"{loss.item():.4f}",
                        accuracy=f"{accuracy:.4f}"
                    )

            self.is_fitted = True

            logger.info(
                "few_shot_training_completed",
                final_loss=f"{loss.item():.4f}",
                episodes=self.n_episodes
            )

        except Exception as e:
            logger.error("few_shot_training_failed", error=str(e))
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
            features: Input features (n_samples, n_features)

        Returns:
            Predicted class labels (n_samples,)

        Raises:
            ValueError: If model not trained or input invalid
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        try:
            if features.shape[1] != self.input_dim:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self.input_dim}, "
                    f"got {features.shape[1]}"
                )

            # Normalize features
            features_scaled = self.scaler.transform(features)

            # Convert to tensor
            features_tensor = torch.from_numpy(features_scaled).float().to(self.device)

            # Get embeddings
            self.network.eval()
            with torch.no_grad():
                embeddings = self.network(features_tensor)

                # If prototypes are stored, use them for prediction
                if self.prototypes is not None:
                    distances = self._compute_distances(embeddings, self.prototypes)
                    predictions = torch.argmin(distances, dim=1)
                else:
                    # Need support set for prediction
                    raise ValueError(
                        "No prototypes stored. Call adapt() with support set first."
                    )

            predictions_np = predictions.cpu().numpy()

            logger.info(
                "predictions_made",
                n_samples=features.shape[0],
                unique_predictions=len(np.unique(predictions_np))
            )

            return predictions_np

        except Exception as e:
            logger.error("prediction_failed", error=str(e))
            raise

    def adapt(self, support_features: np.ndarray, support_labels: np.ndarray) -> None:
        """Adapt model to new classes using support set.

        Computes prototypes from the support set for quick adaptation
        to new market regimes or trading conditions.

        Args:
            support_features: Support set features (n_support, n_features)
            support_labels: Support set labels (n_support,)

        Example:
            >>> # Adapt to new market regime with few examples
            >>> model.adapt(regime_features, regime_labels)
            >>> predictions = model.predict(new_data)
        """
        try:
            # Normalize features
            support_scaled = self.scaler.transform(support_features)

            # Convert to tensors
            support_tensor = torch.from_numpy(support_scaled).float().to(self.device)
            labels_tensor = torch.from_numpy(support_labels).long().to(self.device)

            # Get embeddings
            self.network.eval()
            with torch.no_grad():
                support_embeddings = self.network(support_tensor)

                # Compute and store prototypes
                n_classes = len(np.unique(support_labels))
                self.prototypes = self.network.compute_prototypes(
                    support_embeddings,
                    labels_tensor,
                    n_classes
                )

            logger.info(
                "model_adapted",
                n_support=support_features.shape[0],
                n_classes=n_classes
            )

        except Exception as e:
            logger.error("adaptation_failed", error=str(e))
            raise

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            features: Evaluation features (n_samples, n_features)
            labels: True labels (n_samples,)

        Returns:
            Dictionary containing evaluation metrics:
                - accuracy: Classification accuracy
                - precision: Precision score
                - recall: Recall score
                - f1_score: F1 score

        Raises:
            ValueError: If model not trained
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")

        try:
            predictions = self.predict(features)

            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, average='weighted', zero_division=0)
            recall = recall_score(labels, predictions, average='weighted', zero_division=0)
            f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'n_samples': features.shape[0],
                'n_classes': len(np.unique(labels))
            }

            logger.info(
                "model_evaluated",
                accuracy=f"{accuracy:.4f}",
                f1_score=f"{f1:.4f}"
            )

            return metrics

        except Exception as e:
            logger.error("evaluation_failed", error=str(e))
            raise

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save model

        Raises:
            ValueError: If model not trained
        """
        if not self.is_fitted:
            raise ValueError("Cannot save untrained model")

        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save model state
            state = {
                'config': self.config,
                'network_state': self.network.state_dict(),
                'scaler': self.scaler,
                'n_classes': self.n_classes,
                'prototypes': self.prototypes.cpu() if self.prototypes is not None else None,
                'training_history': self.training_history,
                'is_fitted': self.is_fitted
            }

            with open(save_path, 'wb') as f:
                pickle.dump(state, f)

            logger.info("model_saved", path=path)

        except Exception as e:
            logger.error("model_save_failed", error=str(e))
            raise

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to load model from

        Raises:
            FileNotFoundError: If model file not found
        """
        try:
            load_path = Path(path)
            if not load_path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")

            with open(load_path, 'rb') as f:
                state = pickle.load(f)

            # Restore state
            self.config = state['config']
            self.network.load_state_dict(state['network_state'])
            self.scaler = state['scaler']
            self.n_classes = state['n_classes']
            self.prototypes = state['prototypes']
            if self.prototypes is not None:
                self.prototypes = self.prototypes.to(self.device)
            self.training_history = state['training_history']
            self.is_fitted = state['is_fitted']

            logger.info("model_loaded", path=path)

        except Exception as e:
            logger.error("model_load_failed", error=str(e))
            raise

    def _sample_episode(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        unique_labels: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a training episode.

        Args:
            features: All features
            labels: All labels
            unique_labels: Array of unique label values

        Returns:
            Tuple of (support_x, support_y, query_x, query_y)
        """
        # Sample n_way classes
        episode_classes = np.random.choice(
            unique_labels,
            size=self.n_way,
            replace=False
        )

        support_x_list = []
        support_y_list = []
        query_x_list = []
        query_y_list = []

        for class_idx, class_label in enumerate(episode_classes):
            # Get all examples from this class
            class_mask = labels == class_label
            class_features = features[class_mask]

            # Sample k_shot + n_query examples
            n_samples = min(self.k_shot + self.n_query, class_features.size(0))
            indices = torch.randperm(class_features.size(0))[:n_samples]
            sampled_features = class_features[indices]

            # Split into support and query
            support_x_list.append(sampled_features[:self.k_shot])
            support_y_list.append(torch.full((self.k_shot,), class_idx))

            if n_samples > self.k_shot:
                n_query_actual = n_samples - self.k_shot
                query_x_list.append(sampled_features[self.k_shot:self.k_shot + n_query_actual])
                query_y_list.append(torch.full((n_query_actual,), class_idx))

        support_x = torch.cat(support_x_list, dim=0)
        support_y = torch.cat(support_y_list, dim=0)
        query_x = torch.cat(query_x_list, dim=0)
        query_y = torch.cat(query_y_list, dim=0)

        return support_x, support_y, query_x, query_y

    def _compute_distances(
        self,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """Compute Euclidean distances between embeddings and prototypes.

        Args:
            embeddings: Query embeddings (n_query, embedding_dim)
            prototypes: Class prototypes (n_classes, embedding_dim)

        Returns:
            Distance matrix (n_query, n_classes)
        """
        # Expand dimensions for broadcasting
        embeddings_expanded = embeddings.unsqueeze(1)  # (n_query, 1, embedding_dim)
        prototypes_expanded = prototypes.unsqueeze(0)  # (1, n_classes, embedding_dim)

        # Compute squared Euclidean distances
        distances = torch.sum((embeddings_expanded - prototypes_expanded) ** 2, dim=2)

        return distances

    @staticmethod
    def _compute_accuracy(distances: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute classification accuracy.

        Args:
            distances: Distance matrix (n_query, n_classes)
            labels: True labels (n_query,)

        Returns:
            Accuracy as float
        """
        predictions = torch.argmin(distances, dim=1)
        accuracy = (predictions == labels).float().mean()
        return accuracy.item()
