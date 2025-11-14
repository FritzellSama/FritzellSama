"""
Active Learning Module for Online Model Improvement.

This module implements active learning strategies to intelligently select
the most informative samples for labeling and model updates in live trading.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class QueryStrategy(Enum):
    """Active learning query strategies."""

    UNCERTAINTY = "uncertainty"  # Select samples with highest uncertainty
    MARGIN = "margin"  # Select samples with smallest margin
    ENTROPY = "entropy"  # Select samples with highest entropy
    COMMITTEE = "committee"  # Query by committee
    EXPECTED_ERROR = "expected_error"  # Expected error reduction
    DENSITY_WEIGHTED = "density_weighted"  # Density-weighted uncertainty


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning."""

    query_strategy: str  # Query strategy to use
    batch_size: int  # Samples per query batch
    uncertainty_threshold: Decimal  # Threshold for uncertain samples
    confidence_threshold: Decimal  # Minimum confidence for auto-labeling
    max_pool_size: int  # Maximum unlabeled pool size
    committee_size: int  # Number of models in committee
    update_frequency: int  # Steps between model updates
    min_samples_for_update: int  # Minimum samples before update
    diversity_factor: Decimal  # Weight for diversity in selection
    enable_pseudo_labeling: bool  # Use pseudo-labeling for confident predictions
    pseudo_label_threshold: Decimal  # Confidence threshold for pseudo-labels
    model_path: str  # Path to save/load models


@dataclass
class Sample:
    """Unlabeled sample for active learning."""

    features: np.ndarray
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Optional[Decimal] = None
    label: Optional[int] = None
    is_pseudo_labeled: bool = False


class UncertaintyEstimator:
    """Estimates prediction uncertainty for active learning."""

    def __init__(self, config: ActiveLearningConfig) -> None:
        """Initialize uncertainty estimator.

        Args:
            config: Active learning configuration
        """
        self.config = config
        logger.info("Uncertainty estimator initialized")

    def estimate_uncertainty(
        self,
        predictions: np.ndarray,
        strategy: QueryStrategy
    ) -> np.ndarray:
        """Estimate uncertainty for predictions.

        Args:
            predictions: Model predictions (probabilities)
            strategy: Query strategy to use

        Returns:
            Uncertainty scores for each prediction
        """
        try:
            if strategy == QueryStrategy.UNCERTAINTY:
                return self._uncertainty_sampling(predictions)
            elif strategy == QueryStrategy.MARGIN:
                return self._margin_sampling(predictions)
            elif strategy == QueryStrategy.ENTROPY:
                return self._entropy_sampling(predictions)
            else:
                logger.warning(
                    "Unknown strategy, using uncertainty",
                    strategy=strategy
                )
                return self._uncertainty_sampling(predictions)

        except Exception as e:
            logger.error("Uncertainty estimation failed", error=str(e))
            raise

    def _uncertainty_sampling(self, predictions: np.ndarray) -> np.ndarray:
        """Uncertainty sampling: 1 - max(p).

        Args:
            predictions: Probability predictions

        Returns:
            Uncertainty scores
        """
        max_probs = np.max(predictions, axis=1)
        return 1.0 - max_probs

    def _margin_sampling(self, predictions: np.ndarray) -> np.ndarray:
        """Margin sampling: difference between top 2 predictions.

        Args:
            predictions: Probability predictions

        Returns:
            Margin scores (lower = more uncertain)
        """
        if predictions.shape[1] < 2:
            return np.zeros(len(predictions))

        sorted_preds = np.sort(predictions, axis=1)
        margins = sorted_preds[:, -1] - sorted_preds[:, -2]
        return 1.0 - margins  # Invert so higher = more uncertain

    def _entropy_sampling(self, predictions: np.ndarray) -> np.ndarray:
        """Entropy sampling: H(p).

        Args:
            predictions: Probability predictions

        Returns:
            Entropy scores
        """
        epsilon = 1e-10
        entropy = -np.sum(
            predictions * np.log(predictions + epsilon),
            axis=1
        )
        # Normalize by max entropy
        max_entropy = np.log(predictions.shape[1])
        return entropy / max_entropy


class QueryByCommittee:
    """Query by committee for active learning."""

    def __init__(
        self,
        models: List[nn.Module],
        device: str = "cpu"
    ) -> None:
        """Initialize query by committee.

        Args:
            models: List of committee models
            device: Device to run models on
        """
        self.models = models
        self.device = device
        logger.info("Query by committee initialized", num_models=len(models))

    async def get_committee_predictions(
        self,
        features: np.ndarray
    ) -> np.ndarray:
        """Get predictions from all committee members.

        Args:
            features: Input features

        Returns:
            Array of shape (n_models, n_samples, n_classes)
        """
        try:
            features_tensor = torch.FloatTensor(features).to(self.device)
            predictions = []

            for model in self.models:
                model.eval()
                with torch.no_grad():
                    pred = model(features_tensor)
                    pred_probs = torch.softmax(pred, dim=-1)
                    predictions.append(pred_probs.cpu().numpy())

            return np.array(predictions)

        except Exception as e:
            logger.error("Committee predictions failed", error=str(e))
            raise

    def calculate_disagreement(
        self,
        committee_predictions: np.ndarray
    ) -> np.ndarray:
        """Calculate disagreement among committee members.

        Args:
            committee_predictions: Predictions from all models

        Returns:
            Disagreement scores
        """
        try:
            # Vote entropy
            avg_predictions = np.mean(committee_predictions, axis=0)
            epsilon = 1e-10
            entropy = -np.sum(
                avg_predictions * np.log(avg_predictions + epsilon),
                axis=1
            )

            return entropy

        except Exception as e:
            logger.error("Disagreement calculation failed", error=str(e))
            raise


class ActiveLearner:
    """Active learning system for intelligent sample selection."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any]
    ) -> None:
        """Initialize active learner.

        Args:
            model: Base model for predictions
            config: Active learning configuration

        Example:
            >>> config = {
            ...     "query_strategy": "uncertainty",
            ...     "batch_size": 32,
            ...     "uncertainty_threshold": "0.7"
            ... }
            >>> learner = ActiveLearner(model, config)
        """
        self.model = model
        self.config = self._build_config(config)
        self.uncertainty_estimator = UncertaintyEstimator(self.config)

        # Unlabeled sample pool
        self.unlabeled_pool: List[Sample] = []

        # Labeled samples for training
        self.labeled_samples: List[Sample] = []

        # Committee for query by committee
        self.committee: Optional[QueryByCommittee] = None
        if self.config.query_strategy == "committee":
            self._initialize_committee()

        # Statistics
        self.stats = {
            "queries": 0,
            "samples_labeled": 0,
            "pseudo_labels": 0,
            "model_updates": 0
        }

        logger.info(
            "Active learner initialized",
            strategy=self.config.query_strategy,
            batch_size=self.config.batch_size
        )

    def _build_config(self, config: Dict[str, Any]) -> ActiveLearningConfig:
        """Build ActiveLearningConfig from dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            ActiveLearningConfig instance
        """
        return ActiveLearningConfig(
            query_strategy=config.get("query_strategy", "uncertainty"),
            batch_size=config.get("batch_size", 32),
            uncertainty_threshold=Decimal(str(config.get("uncertainty_threshold", "0.7"))),
            confidence_threshold=Decimal(str(config.get("confidence_threshold", "0.95"))),
            max_pool_size=config.get("max_pool_size", 10000),
            committee_size=config.get("committee_size", 5),
            update_frequency=config.get("update_frequency", 100),
            min_samples_for_update=config.get("min_samples_for_update", 50),
            diversity_factor=Decimal(str(config.get("diversity_factor", "0.1"))),
            enable_pseudo_labeling=config.get("enable_pseudo_labeling", True),
            pseudo_label_threshold=Decimal(str(config.get("pseudo_label_threshold", "0.95"))),
            model_path=config.get("model_path", "/tmp/active_learning_model.pt")
        )

    def _initialize_committee(self) -> None:
        """Initialize committee for query by committee."""
        try:
            # Create committee models (simplified - would bootstrap from data in production)
            committee_models = []
            for i in range(self.config.committee_size):
                # Deep copy of base model
                import copy
                model_copy = copy.deepcopy(self.model)
                committee_models.append(model_copy)

            self.committee = QueryByCommittee(committee_models)
            logger.info("Committee initialized", size=self.config.committee_size)

        except Exception as e:
            logger.error("Committee initialization failed", error=str(e))
            raise

    async def add_to_pool(
        self,
        features: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add samples to unlabeled pool.

        Args:
            features: Feature array
            metadata: Optional metadata
        """
        try:
            sample = Sample(
                features=features,
                timestamp=datetime.utcnow(),
                metadata=metadata or {}
            )

            self.unlabeled_pool.append(sample)

            # Maintain pool size limit
            if len(self.unlabeled_pool) > self.config.max_pool_size:
                # Remove oldest samples
                self.unlabeled_pool = self.unlabeled_pool[-self.config.max_pool_size:]

            logger.debug(
                "Sample added to pool",
                pool_size=len(self.unlabeled_pool)
            )

        except Exception as e:
            logger.error("Failed to add sample to pool", error=str(e))
            raise

    async def query_samples(self) -> List[Sample]:
        """Query most informative samples from pool.

        Returns:
            List of samples to label

        Example:
            >>> samples = await learner.query_samples()
            >>> for sample in samples:
            ...     label = await get_label(sample)
            ...     await learner.add_labeled_sample(sample, label)
        """
        try:
            if len(self.unlabeled_pool) < self.config.batch_size:
                logger.warning(
                    "Pool too small for query",
                    pool_size=len(self.unlabeled_pool),
                    batch_size=self.config.batch_size
                )
                return []

            # Get predictions for all samples
            features = np.array([s.features for s in self.unlabeled_pool])
            predictions = await self._get_predictions(features)

            # Calculate uncertainty
            strategy = QueryStrategy(self.config.query_strategy)
            if strategy == QueryStrategy.COMMITTEE and self.committee:
                committee_preds = await self.committee.get_committee_predictions(features)
                uncertainties = self.committee.calculate_disagreement(committee_preds)
            else:
                uncertainties = self.uncertainty_estimator.estimate_uncertainty(
                    predictions,
                    strategy
                )

            # Update sample uncertainties
            for sample, uncertainty in zip(self.unlabeled_pool, uncertainties):
                sample.uncertainty = Decimal(str(uncertainty))

            # Select top-k uncertain samples
            sorted_indices = np.argsort(uncertainties)[::-1]
            selected_indices = sorted_indices[:self.config.batch_size]

            selected_samples = [self.unlabeled_pool[i] for i in selected_indices]

            # Handle pseudo-labeling for confident predictions
            if self.config.enable_pseudo_labeling:
                await self._apply_pseudo_labeling(predictions, uncertainties)

            self.stats["queries"] += 1

            logger.info(
                "Samples queried",
                num_samples=len(selected_samples),
                avg_uncertainty=float(np.mean([float(s.uncertainty) for s in selected_samples]))
            )

            return selected_samples

        except Exception as e:
            logger.error("Sample query failed", error=str(e))
            raise

    async def _get_predictions(self, features: np.ndarray) -> np.ndarray:
        """Get model predictions for features.

        Args:
            features: Input features

        Returns:
            Prediction probabilities
        """
        try:
            self.model.eval()
            features_tensor = torch.FloatTensor(features)

            with torch.no_grad():
                outputs = self.model(features_tensor)
                predictions = torch.softmax(outputs, dim=-1).numpy()

            return predictions

        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            raise

    async def _apply_pseudo_labeling(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray
    ) -> None:
        """Apply pseudo-labels to confident predictions.

        Args:
            predictions: Model predictions
            uncertainties: Uncertainty scores
        """
        try:
            confident_threshold = float(self.config.pseudo_label_threshold)
            max_probs = np.max(predictions, axis=1)
            confident_mask = max_probs >= confident_threshold

            pseudo_labeled = []
            for i, is_confident in enumerate(confident_mask):
                if is_confident and i < len(self.unlabeled_pool):
                    sample = self.unlabeled_pool[i]
                    pseudo_label = int(np.argmax(predictions[i]))

                    sample.label = pseudo_label
                    sample.is_pseudo_labeled = True
                    self.labeled_samples.append(sample)
                    pseudo_labeled.append(i)

            # Remove pseudo-labeled samples from pool
            for idx in sorted(pseudo_labeled, reverse=True):
                del self.unlabeled_pool[idx]

            if pseudo_labeled:
                self.stats["pseudo_labels"] += len(pseudo_labeled)
                logger.info(
                    "Pseudo-labels applied",
                    count=len(pseudo_labeled)
                )

        except Exception as e:
            logger.error("Pseudo-labeling failed", error=str(e))

    async def add_labeled_sample(
        self,
        sample: Sample,
        label: int
    ) -> None:
        """Add labeled sample to training set.

        Args:
            sample: Sample to add
            label: Ground truth label
        """
        try:
            sample.label = label
            sample.is_pseudo_labeled = False
            self.labeled_samples.append(sample)

            # Remove from unlabeled pool
            if sample in self.unlabeled_pool:
                self.unlabeled_pool.remove(sample)

            self.stats["samples_labeled"] += 1

            logger.debug(
                "Labeled sample added",
                total_labeled=len(self.labeled_samples)
            )

            # Check if we should update model
            if len(self.labeled_samples) >= self.config.min_samples_for_update:
                if self.stats["samples_labeled"] % self.config.update_frequency == 0:
                    await self.update_model()

        except Exception as e:
            logger.error("Failed to add labeled sample", error=str(e))
            raise

    async def update_model(self) -> Dict[str, float]:
        """Update model with labeled samples.

        Returns:
            Training metrics
        """
        try:
            if len(self.labeled_samples) < self.config.min_samples_for_update:
                logger.warning(
                    "Not enough samples for update",
                    current=len(self.labeled_samples),
                    required=self.config.min_samples_for_update
                )
                return {}

            # Prepare training data
            features = np.array([s.features for s in self.labeled_samples])
            labels = np.array([s.label for s in self.labeled_samples])

            # Train model (simplified - would use proper training loop)
            self.model.train()
            features_tensor = torch.FloatTensor(features)
            labels_tensor = torch.LongTensor(labels)

            # Note: Would use proper optimizer and training loop in production
            criterion = nn.CrossEntropyLoss()
            outputs = self.model(features_tensor)
            loss = criterion(outputs, labels_tensor)

            self.stats["model_updates"] += 1

            logger.info(
                "Model updated",
                num_samples=len(self.labeled_samples),
                loss=loss.item()
            )

            return {"loss": loss.item(), "num_samples": len(self.labeled_samples)}

        except Exception as e:
            logger.error("Model update failed", error=str(e))
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get active learning statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            **self.stats,
            "pool_size": len(self.unlabeled_pool),
            "labeled_size": len(self.labeled_samples),
            "pseudo_label_ratio": (
                self.stats["pseudo_labels"] / max(self.stats["samples_labeled"], 1)
            )
        }

    def save(self, path: Optional[str] = None) -> None:
        """Save active learner state.

        Args:
            path: Path to save (uses config path if None)
        """
        try:
            save_path = path or self.config.model_path
            state = {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "stats": self.stats,
                "labeled_samples": self.labeled_samples
            }
            torch.save(state, save_path)
            logger.info("Active learner saved", path=save_path)

        except Exception as e:
            logger.error("Save failed", error=str(e))
            raise

    def load(self, path: Optional[str] = None) -> None:
        """Load active learner state.

        Args:
            path: Path to load from (uses config path if None)
        """
        try:
            load_path = path or self.config.model_path
            state = torch.load(load_path)

            self.model.load_state_dict(state["model_state_dict"])
            self.stats = state["stats"]
            self.labeled_samples = state["labeled_samples"]

            logger.info("Active learner loaded", path=load_path)

        except Exception as e:
            logger.error("Load failed", error=str(e))
            raise

    async def get_pool_summary(self) -> pl.DataFrame:
        """Get summary of unlabeled pool.

        Returns:
            DataFrame with pool statistics
        """
        try:
            if not self.unlabeled_pool:
                return pl.DataFrame()

            data = {
                "timestamp": [s.timestamp for s in self.unlabeled_pool],
                "uncertainty": [float(s.uncertainty) if s.uncertainty else 0.0
                                for s in self.unlabeled_pool],
                "is_pseudo_labeled": [s.is_pseudo_labeled for s in self.unlabeled_pool]
            }

            return pl.DataFrame(data)

        except Exception as e:
            logger.error("Pool summary failed", error=str(e))
            raise
