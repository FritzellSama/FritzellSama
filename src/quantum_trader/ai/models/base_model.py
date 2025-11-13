"""
Base Model for All AI/ML Models.

Provides abstract base class and common functionality for all models
in the trading system.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
import torch

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for model tracking."""

    model_id: str
    model_type: str
    version: str
    created_at: datetime
    last_trained: Optional[datetime] = None
    last_predicted: Optional[datetime] = None
    total_predictions: int = 0
    training_samples: int = 0
    features: List[str] = None
    metrics: Dict[str, Decimal] = None

    def __post_init__(self):
        """Initialize optional fields."""
        if self.features is None:
            self.features = []
        if self.metrics is None:
            self.metrics = {}


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(
        self,
        model_id: str,
        model_type: str,
        version: str = "1.0.0"
    ):
        """
        Initialize base model.

        Args:
            model_id: Unique model identifier
            model_type: Type of model
            version: Model version
        """
        self.metadata = ModelMetadata(
            model_id=model_id,
            model_type=model_type,
            version=version,
            created_at=datetime.now(timezone.utc)
        )

        logger.info(
            "Initialized base model",
            extra={
                "model_id": model_id,
                "model_type": model_type,
                "version": version,
                "timestamp": self.metadata.created_at.isoformat()
            }
        )

    @abstractmethod
    async def train(
        self,
        train_df: pl.DataFrame,
        feature_cols: List[str],
        label_col: str,
        **kwargs
    ) -> Dict[str, Decimal]:
        """
        Train the model.

        Args:
            train_df: Training data
            feature_cols: Feature column names
            label_col: Label column name
            **kwargs: Additional training parameters

        Returns:
            Training metrics
        """
        pass

    @abstractmethod
    async def predict(
        self,
        data_df: pl.DataFrame,
        feature_cols: List[str]
    ) -> pl.DataFrame:
        """
        Make predictions.

        Args:
            data_df: Input data
            feature_cols: Feature column names

        Returns:
            DataFrame with predictions
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model from disk.

        Args:
            path: Path to model file
        """
        pass

    async def evaluate(
        self,
        test_df: pl.DataFrame,
        feature_cols: List[str],
        label_col: str
    ) -> Dict[str, Decimal]:
        """
        Evaluate model performance.

        Args:
            test_df: Test data
            feature_cols: Feature column names
            label_col: Label column name

        Returns:
            Evaluation metrics
        """
        # Get predictions
        pred_df = await self.predict(test_df, feature_cols)

        # Compute metrics
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score
        )

        y_true = test_df.select(label_col).to_numpy().ravel()
        y_pred = pred_df.select("prediction").to_numpy().ravel()

        metrics = {
            "accuracy": Decimal(str(accuracy_score(y_true, y_pred))),
            "precision": Decimal(str(precision_score(y_true, y_pred, average="weighted", zero_division=0))),
            "recall": Decimal(str(recall_score(y_true, y_pred, average="weighted", zero_division=0))),
            "f1": Decimal(str(f1_score(y_true, y_pred, average="weighted", zero_division=0)))
        }

        # Update metadata
        self.metadata.metrics = metrics

        logger.info(
            "Model evaluation completed",
            extra={
                "model_id": self.metadata.model_id,
                "metrics": {k: str(v) for k, v in metrics.items()},
                "num_samples": len(test_df),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return metrics

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.

        Returns:
            Dictionary with metadata
        """
        return {
            "model_id": self.metadata.model_id,
            "model_type": self.metadata.model_type,
            "version": self.metadata.version,
            "created_at": self.metadata.created_at.isoformat(),
            "last_trained": self.metadata.last_trained.isoformat() if self.metadata.last_trained else None,
            "last_predicted": self.metadata.last_predicted.isoformat() if self.metadata.last_predicted else None,
            "total_predictions": self.metadata.total_predictions,
            "training_samples": self.metadata.training_samples,
            "features": self.metadata.features,
            "metrics": {k: str(v) for k, v in self.metadata.metrics.items()}
        }

    def update_training_metadata(
        self,
        num_samples: int,
        features: List[str],
        metrics: Dict[str, Decimal]
    ) -> None:
        """
        Update metadata after training.

        Args:
            num_samples: Number of training samples
            features: Feature names
            metrics: Training metrics
        """
        self.metadata.last_trained = datetime.now(timezone.utc)
        self.metadata.training_samples = num_samples
        self.metadata.features = features
        self.metadata.metrics = metrics

    def update_prediction_metadata(self, num_predictions: int) -> None:
        """
        Update metadata after prediction.

        Args:
            num_predictions: Number of predictions made
        """
        self.metadata.last_predicted = datetime.now(timezone.utc)
        self.metadata.total_predictions += num_predictions


class PyTorchModel(BaseModel):
    """Base class for PyTorch models."""

    def __init__(
        self,
        model_id: str,
        model_type: str,
        network: torch.nn.Module,
        device: Optional[torch.device] = None,
        version: str = "1.0.0"
    ):
        """
        Initialize PyTorch model.

        Args:
            model_id: Model identifier
            model_type: Model type
            network: PyTorch network
            device: Computing device
            version: Model version
        """
        super().__init__(model_id, model_type, version)

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.network = network.to(self.device)

    async def train(
        self,
        train_df: pl.DataFrame,
        feature_cols: List[str],
        label_col: str,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        epochs: int = 10,
        batch_size: int = 32
    ) -> Dict[str, Decimal]:
        """Train PyTorch model."""
        self.network.train()

        features = train_df.select(feature_cols).to_numpy()
        labels = train_df.select(label_col).to_numpy()

        total_loss = Decimal("0")
        num_batches = 0

        for epoch in range(epochs):
            epoch_loss = Decimal("0")

            for i in range(0, len(features), batch_size):
                batch_features = features[i : i + batch_size]
                batch_labels = labels[i : i + batch_size]

                X = torch.tensor(
                    batch_features,
                    dtype=torch.float32,
                    device=self.device
                )
                y = torch.tensor(
                    batch_labels,
                    dtype=torch.long,
                    device=self.device
                ).squeeze()

                optimizer.zero_grad()
                outputs = self.network(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                epoch_loss += Decimal(str(loss.item()))
                num_batches += 1

            total_loss += epoch_loss

        avg_loss = total_loss / (epochs * num_batches) if num_batches > 0 else Decimal("0")

        metrics = {
            "avg_loss": avg_loss,
            "epochs": Decimal(str(epochs))
        }

        self.update_training_metadata(len(train_df), feature_cols, metrics)

        logger.info(
            "PyTorch model training completed",
            extra={
                "model_id": self.metadata.model_id,
                "epochs": epochs,
                "avg_loss": str(avg_loss),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return metrics

    async def predict(
        self,
        data_df: pl.DataFrame,
        feature_cols: List[str]
    ) -> pl.DataFrame:
        """Make predictions with PyTorch model."""
        self.network.eval()

        features = data_df.select(feature_cols).to_numpy()
        X = torch.tensor(features, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            outputs = self.network(X)

            if outputs.dim() > 1 and outputs.size(1) > 1:
                probs = torch.softmax(outputs, dim=-1)
                predictions = probs.argmax(dim=-1)
                confidence = probs.max(dim=-1)[0]
            else:
                predictions = (torch.sigmoid(outputs) > 0.5).long().squeeze()
                confidence = torch.sigmoid(outputs).squeeze()

        result_df = data_df.with_columns([
            pl.Series("prediction", predictions.cpu().numpy().tolist()),
            pl.Series("confidence", confidence.cpu().numpy().tolist())
        ])

        self.update_prediction_metadata(len(result_df))

        return result_df

    def save(self, path: str) -> None:
        """Save PyTorch model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "network": self.network.state_dict(),
            "metadata": self.metadata,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        torch.save(checkpoint, path)

        logger.info(
            "Saved PyTorch model",
            extra={
                "model_id": self.metadata.model_id,
                "path": path,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    def load(self, path: str) -> None:
        """Load PyTorch model."""
        checkpoint = torch.load(path, map_location=self.device)

        self.network.load_state_dict(checkpoint["network"])
        self.metadata = checkpoint["metadata"]

        logger.info(
            "Loaded PyTorch model",
            extra={
                "model_id": self.metadata.model_id,
                "path": path,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


class SklearnModel(BaseModel):
    """Base class for scikit-learn models."""

    def __init__(
        self,
        model_id: str,
        model_type: str,
        estimator: Any,
        version: str = "1.0.0"
    ):
        """
        Initialize sklearn model.

        Args:
            model_id: Model identifier
            model_type: Model type
            estimator: Sklearn estimator
            version: Model version
        """
        super().__init__(model_id, model_type, version)
        self.estimator = estimator

    async def train(
        self,
        train_df: pl.DataFrame,
        feature_cols: List[str],
        label_col: str,
        **kwargs
    ) -> Dict[str, Decimal]:
        """Train sklearn model."""
        X = train_df.select(feature_cols).to_numpy()
        y = train_df.select(label_col).to_numpy().ravel()

        self.estimator.fit(X, y)

        # Compute training score
        train_score = self.estimator.score(X, y)

        metrics = {
            "train_score": Decimal(str(train_score))
        }

        self.update_training_metadata(len(train_df), feature_cols, metrics)

        logger.info(
            "Sklearn model training completed",
            extra={
                "model_id": self.metadata.model_id,
                "train_score": str(train_score),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return metrics

    async def predict(
        self,
        data_df: pl.DataFrame,
        feature_cols: List[str]
    ) -> pl.DataFrame:
        """Make predictions with sklearn model."""
        X = data_df.select(feature_cols).to_numpy()

        predictions = self.estimator.predict(X)

        # Get probabilities if available
        if hasattr(self.estimator, "predict_proba"):
            probabilities = self.estimator.predict_proba(X)
            confidence = probabilities.max(axis=1)
        else:
            confidence = [1.0] * len(predictions)

        result_df = data_df.with_columns([
            pl.Series("prediction", predictions.tolist()),
            pl.Series("confidence", confidence.tolist() if isinstance(confidence, list) else confidence.tolist())
        ])

        self.update_prediction_metadata(len(result_df))

        return result_df

    def save(self, path: str) -> None:
        """Save sklearn model."""
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "estimator": self.estimator,
            "metadata": self.metadata
        }

        joblib.dump(checkpoint, path)

        logger.info(
            "Saved sklearn model",
            extra={
                "model_id": self.metadata.model_id,
                "path": path,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    def load(self, path: str) -> None:
        """Load sklearn model."""
        import joblib

        checkpoint = joblib.load(path)

        self.estimator = checkpoint["estimator"]
        self.metadata = checkpoint["metadata"]

        logger.info(
            "Loaded sklearn model",
            extra={
                "model_id": self.metadata.model_id,
                "path": path,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
