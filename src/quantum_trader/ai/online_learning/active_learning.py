"""
Active Learning Implementation for Online Model Training.

Implements query strategies to intelligently select samples for labeling
in online learning scenarios to maximize model performance.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)


class QueryStrategy(str, Enum):
    """Available query strategies for active learning."""

    UNCERTAINTY = "uncertainty"
    MARGIN = "margin"
    ENTROPY = "entropy"
    QUERY_BY_COMMITTEE = "qbc"
    DIVERSITY = "diversity"
    EXPECTED_MODEL_CHANGE = "emc"
    HYBRID = "hybrid"


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning."""

    query_strategy: QueryStrategy = QueryStrategy.UNCERTAINTY
    batch_size: int = 32
    uncertainty_threshold: Decimal = Decimal("0.7")
    committee_size: int = 5
    diversity_weight: Decimal = Decimal("0.3")
    max_pool_size: int = 10000
    min_confidence: Decimal = Decimal("0.6")
    retraining_frequency: int = 100


@dataclass
class QueryResult:
    """Result from active learning query."""

    indices: List[int]
    scores: List[Decimal]
    features: torch.Tensor
    metadata: Dict[str, Any]
    timestamp: datetime


class UncertaintySampler:
    """Samples based on prediction uncertainty."""

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize uncertainty sampler.

        Args:
            model: PyTorch model for predictions
            device: Computing device
        """
        self.model = model
        self.device = device

    async def query(
        self,
        pool: torch.Tensor,
        batch_size: int
    ) -> Tuple[List[int], List[Decimal]]:
        """
        Query samples with highest prediction uncertainty.

        Args:
            pool: Unlabeled pool of samples
            batch_size: Number of samples to query

        Returns:
            Tuple of (sample indices, uncertainty scores)
        """
        self.model.eval()

        with torch.no_grad():
            pool_device = pool.to(self.device)
            predictions = self.model(pool_device)

            # Compute uncertainty (1 - max probability)
            probs = torch.softmax(predictions, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)
            uncertainties = 1.0 - max_probs

        # Select top uncertain samples
        top_k = min(batch_size, len(uncertainties))
        scores, indices = torch.topk(uncertainties, top_k)

        indices_list = indices.cpu().tolist()
        scores_list = [Decimal(str(s.item())) for s in scores]

        logger.info(
            "Uncertainty sampling completed",
            extra={
                "num_samples": len(indices_list),
                "mean_uncertainty": str(sum(scores_list) / len(scores_list)),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return indices_list, scores_list


class MarginSampler:
    """Samples based on margin between top predictions."""

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize margin sampler.

        Args:
            model: PyTorch model for predictions
            device: Computing device
        """
        self.model = model
        self.device = device

    async def query(
        self,
        pool: torch.Tensor,
        batch_size: int
    ) -> Tuple[List[int], List[Decimal]]:
        """
        Query samples with smallest margin between top two predictions.

        Args:
            pool: Unlabeled pool of samples
            batch_size: Number of samples to query

        Returns:
            Tuple of (sample indices, margin scores)
        """
        self.model.eval()

        with torch.no_grad():
            pool_device = pool.to(self.device)
            predictions = self.model(pool_device)
            probs = torch.softmax(predictions, dim=-1)

            # Compute margin (difference between top 2 predictions)
            top2_probs, _ = torch.topk(probs, 2, dim=-1)
            margins = top2_probs[:, 0] - top2_probs[:, 1]

        # Select samples with smallest margins (most uncertain)
        top_k = min(batch_size, len(margins))
        scores, indices = torch.topk(margins, top_k, largest=False)

        indices_list = indices.cpu().tolist()
        scores_list = [Decimal(str(s.item())) for s in scores]

        logger.info(
            "Margin sampling completed",
            extra={
                "num_samples": len(indices_list),
                "mean_margin": str(sum(scores_list) / len(scores_list)),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return indices_list, scores_list


class EntropySampler:
    """Samples based on prediction entropy."""

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize entropy sampler.

        Args:
            model: PyTorch model for predictions
            device: Computing device
        """
        self.model = model
        self.device = device

    async def query(
        self,
        pool: torch.Tensor,
        batch_size: int
    ) -> Tuple[List[int], List[Decimal]]:
        """
        Query samples with highest prediction entropy.

        Args:
            pool: Unlabeled pool of samples
            batch_size: Number of samples to query

        Returns:
            Tuple of (sample indices, entropy scores)
        """
        self.model.eval()

        with torch.no_grad():
            pool_device = pool.to(self.device)
            predictions = self.model(pool_device)
            probs = torch.softmax(predictions, dim=-1)

            # Compute entropy
            log_probs = torch.log(probs + 1e-10)
            entropies = -torch.sum(probs * log_probs, dim=-1)

        # Select top entropy samples
        top_k = min(batch_size, len(entropies))
        scores, indices = torch.topk(entropies, top_k)

        indices_list = indices.cpu().tolist()
        scores_list = [Decimal(str(s.item())) for s in scores]

        logger.info(
            "Entropy sampling completed",
            extra={
                "num_samples": len(indices_list),
                "mean_entropy": str(sum(scores_list) / len(scores_list)),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return indices_list, scores_list


class QueryByCommittee:
    """Query by Committee (QBC) sampling strategy."""

    def __init__(self, models: List[nn.Module], device: torch.device):
        """
        Initialize QBC sampler.

        Args:
            models: Committee of models
            device: Computing device
        """
        self.models = models
        self.device = device

    async def query(
        self,
        pool: torch.Tensor,
        batch_size: int
    ) -> Tuple[List[int], List[Decimal]]:
        """
        Query samples with highest disagreement among committee.

        Args:
            pool: Unlabeled pool of samples
            batch_size: Number of samples to query

        Returns:
            Tuple of (sample indices, disagreement scores)
        """
        all_predictions = []
        pool_device = pool.to(self.device)

        for model in self.models:
            model.eval()
            with torch.no_grad():
                predictions = model(pool_device)
                probs = torch.softmax(predictions, dim=-1)
                all_predictions.append(probs)

        # Stack predictions from all models
        stacked_preds = torch.stack(all_predictions)

        # Compute variance across committee (disagreement)
        disagreements = torch.var(stacked_preds, dim=0).mean(dim=-1)

        # Select top disagreement samples
        top_k = min(batch_size, len(disagreements))
        scores, indices = torch.topk(disagreements, top_k)

        indices_list = indices.cpu().tolist()
        scores_list = [Decimal(str(s.item())) for s in scores]

        logger.info(
            "QBC sampling completed",
            extra={
                "num_samples": len(indices_list),
                "committee_size": len(self.models),
                "mean_disagreement": str(sum(scores_list) / len(scores_list)),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return indices_list, scores_list


class DiversitySampler:
    """Samples based on feature diversity."""

    async def query(
        self,
        pool: torch.Tensor,
        batch_size: int
    ) -> Tuple[List[int], List[Decimal]]:
        """
        Query diverse samples using clustering.

        Args:
            pool: Unlabeled pool of samples
            batch_size: Number of samples to query

        Returns:
            Tuple of (sample indices, diversity scores)
        """
        pool_np = pool.cpu().numpy()

        # Use K-means clustering for diversity
        n_clusters = min(batch_size, len(pool_np))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(pool_np)

        # Find samples closest to cluster centers
        distances = pairwise_distances(pool_np, kmeans.cluster_centers_)
        min_distances = np.min(distances, axis=1)

        # Select samples closest to centers (most representative)
        indices_list = []
        scores_list = []

        for i in range(n_clusters):
            cluster_mask = kmeans.labels_ == i
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) > 0:
                # Find closest sample to center
                cluster_distances = distances[cluster_indices, i]
                closest_idx = cluster_indices[np.argmin(cluster_distances)]

                indices_list.append(int(closest_idx))
                scores_list.append(Decimal(str(min_distances[closest_idx])))

        logger.info(
            "Diversity sampling completed",
            extra={
                "num_samples": len(indices_list),
                "num_clusters": n_clusters,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return indices_list, scores_list


class ActiveLearner:
    """Main active learning coordinator."""

    def __init__(
        self,
        model: nn.Module,
        config: ActiveLearningConfig,
        device: Optional[torch.device] = None
    ):
        """
        Initialize active learner.

        Args:
            model: Base model for active learning
            config: Active learning configuration
            device: Computing device
        """
        self.model = model
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize samplers
        self.uncertainty_sampler = UncertaintySampler(model, self.device)
        self.margin_sampler = MarginSampler(model, self.device)
        self.entropy_sampler = EntropySampler(model, self.device)
        self.diversity_sampler = DiversitySampler()

        # Metrics
        self.queries_made = 0
        self.samples_labeled = 0

        logger.info(
            "Initialized active learner",
            extra={
                "strategy": config.query_strategy.value,
                "batch_size": config.batch_size,
                "device": str(self.device),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    async def query(
        self,
        pool_df: pl.DataFrame,
        feature_cols: List[str]
    ) -> QueryResult:
        """
        Query samples for labeling from unlabeled pool.

        Args:
            pool_df: DataFrame with unlabeled samples
            feature_cols: Column names for features

        Returns:
            QueryResult with selected samples
        """
        # Extract features
        features_np = pool_df.select(feature_cols).to_numpy()
        features = torch.tensor(features_np, dtype=torch.float32)

        # Apply query strategy
        if self.config.query_strategy == QueryStrategy.UNCERTAINTY:
            indices, scores = await self.uncertainty_sampler.query(
                features,
                self.config.batch_size
            )
        elif self.config.query_strategy == QueryStrategy.MARGIN:
            indices, scores = await self.margin_sampler.query(
                features,
                self.config.batch_size
            )
        elif self.config.query_strategy == QueryStrategy.ENTROPY:
            indices, scores = await self.entropy_sampler.query(
                features,
                self.config.batch_size
            )
        elif self.config.query_strategy == QueryStrategy.DIVERSITY:
            indices, scores = await self.diversity_sampler.query(
                features,
                self.config.batch_size
            )
        elif self.config.query_strategy == QueryStrategy.HYBRID:
            # Combine uncertainty and diversity
            uncertainty_indices, uncertainty_scores = \
                await self.uncertainty_sampler.query(
                    features,
                    self.config.batch_size * 2
                )

            # Get features of uncertain samples
            uncertain_features = features[uncertainty_indices]

            diversity_indices, diversity_scores = \
                await self.diversity_sampler.query(
                    uncertain_features,
                    self.config.batch_size
                )

            # Map back to original indices
            indices = [uncertainty_indices[i] for i in diversity_indices]
            scores = diversity_scores
        else:
            raise ValueError(f"Unknown query strategy: {self.config.query_strategy}")

        # Create result
        result = QueryResult(
            indices=indices,
            scores=scores,
            features=features[indices],
            metadata={
                "strategy": self.config.query_strategy.value,
                "pool_size": len(pool_df),
                "batch_size": len(indices)
            },
            timestamp=datetime.now(timezone.utc)
        )

        self.queries_made += 1
        self.samples_labeled += len(indices)

        logger.info(
            "Query completed",
            extra={
                "num_samples": len(indices),
                "total_queries": self.queries_made,
                "total_labeled": self.samples_labeled,
                "strategy": self.config.query_strategy.value,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return result

    async def update_model(
        self,
        train_df: pl.DataFrame,
        feature_cols: List[str],
        label_col: str,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epochs: int = 10
    ) -> Dict[str, Decimal]:
        """
        Update model with newly labeled samples.

        Args:
            train_df: Training data with labels
            feature_cols: Feature column names
            label_col: Label column name
            optimizer: Optimizer for training
            criterion: Loss criterion
            epochs: Number of training epochs

        Returns:
            Training metrics
        """
        # Prepare data
        X = torch.tensor(
            train_df.select(feature_cols).to_numpy(),
            dtype=torch.float32,
            device=self.device
        )
        y = torch.tensor(
            train_df.select(label_col).to_numpy().flatten(),
            dtype=torch.long,
            device=self.device
        )

        # Training loop
        self.model.train()
        total_loss = Decimal("0")

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(X)
            loss = criterion(outputs, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += Decimal(str(loss.item()))

        avg_loss = total_loss / epochs

        logger.info(
            "Model updated",
            extra={
                "num_samples": len(train_df),
                "epochs": epochs,
                "avg_loss": str(avg_loss),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return {
            "avg_loss": avg_loss,
            "num_samples": Decimal(str(len(train_df))),
            "epochs": Decimal(str(epochs))
        }

    def get_metrics(self) -> pl.DataFrame:
        """
        Get active learning metrics.

        Returns:
            DataFrame with metrics
        """
        metrics_data = {
            "queries_made": [self.queries_made],
            "samples_labeled": [self.samples_labeled],
            "strategy": [self.config.query_strategy.value],
            "timestamp": [datetime.now(timezone.utc).isoformat()]
        }

        return pl.DataFrame(metrics_data)
