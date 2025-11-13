"""
Attention Visualization for Model Explainability.

Provides tools to visualize and interpret attention weights from
attention-based models for trading decisions.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch

logger = logging.getLogger(__name__)


@dataclass
class AttentionVisualization:
    """Visualization data for attention weights."""

    attention_weights: np.ndarray
    timesteps: List[datetime]
    feature_names: List[str]
    layer_idx: int
    head_idx: Optional[int] = None
    aggregation: str = "mean"
    metadata: Dict[str, str] = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AttentionAnalysis:
    """Analysis results for attention patterns."""

    top_features: List[Tuple[str, Decimal]]
    top_timesteps: List[Tuple[datetime, Decimal]]
    attention_entropy: Decimal
    attention_sparsity: Decimal
    temporal_focus: str
    feature_importance: Dict[str, Decimal]
    timestamp: datetime


class AttentionVisualizer:
    """Visualizer for attention-based models."""

    def __init__(self):
        """Initialize attention visualizer."""
        logger.info(
            "Initialized attention visualizer",
            extra={"timestamp": datetime.now(timezone.utc).isoformat()}
        )

    def extract_attention_weights(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        layer_idx: int = -1,
        head_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Extract attention weights from model.

        Args:
            model: Attention-based model
            input_data: Input tensor
            layer_idx: Layer index (-1 for last layer)
            head_idx: Specific head index (None for all heads)

        Returns:
            Attention weights tensor
        """
        model.eval()

        with torch.no_grad():
            # Forward pass to get attention weights
            if hasattr(model, "get_attention_weights"):
                attention_weights = model.get_attention_weights(input_data, layer_idx)
            else:
                # Generic extraction
                _, attention_weights = model(input_data)

                if isinstance(attention_weights, list):
                    attention_weights = attention_weights[layer_idx]

        # Extract specific head if requested
        if head_idx is not None and attention_weights.dim() > 3:
            attention_weights = attention_weights[:, head_idx, :, :]

        logger.info(
            "Extracted attention weights",
            extra={
                "layer_idx": layer_idx,
                "head_idx": head_idx,
                "shape": list(attention_weights.shape),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return attention_weights

    def aggregate_attention(
        self,
        attention_weights: torch.Tensor,
        method: str = "mean"
    ) -> np.ndarray:
        """
        Aggregate attention weights across heads/layers.

        Args:
            attention_weights: Raw attention weights
            method: Aggregation method (mean, max, sum)

        Returns:
            Aggregated attention weights as numpy array
        """
        if attention_weights.dim() == 4:
            # Shape: (batch, heads, seq_len, seq_len)
            if method == "mean":
                aggregated = attention_weights.mean(dim=1)
            elif method == "max":
                aggregated = attention_weights.max(dim=1)[0]
            elif method == "sum":
                aggregated = attention_weights.sum(dim=1)
            else:
                raise ValueError(f"Unknown aggregation method: {method}")
        else:
            aggregated = attention_weights

        # Average over batch if needed
        if aggregated.dim() == 3:
            aggregated = aggregated.mean(dim=0)

        return aggregated.cpu().numpy()

    def create_visualization(
        self,
        attention_weights: torch.Tensor,
        timesteps: List[datetime],
        feature_names: List[str],
        layer_idx: int = 0,
        head_idx: Optional[int] = None,
        aggregation: str = "mean"
    ) -> AttentionVisualization:
        """
        Create visualization data structure.

        Args:
            attention_weights: Attention weights tensor
            timesteps: List of timesteps
            feature_names: List of feature names
            layer_idx: Layer index
            head_idx: Head index
            aggregation: Aggregation method

        Returns:
            AttentionVisualization object
        """
        # Aggregate weights
        aggregated_weights = self.aggregate_attention(attention_weights, aggregation)

        # Create visualization
        viz = AttentionVisualization(
            attention_weights=aggregated_weights,
            timesteps=timesteps,
            feature_names=feature_names,
            layer_idx=layer_idx,
            head_idx=head_idx,
            aggregation=aggregation,
            metadata={
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        logger.info(
            "Created attention visualization",
            extra={
                "shape": aggregated_weights.shape,
                "layer_idx": layer_idx,
                "aggregation": aggregation,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return viz

    def analyze_attention(
        self,
        visualization: AttentionVisualization,
        top_k: int = 10
    ) -> AttentionAnalysis:
        """
        Analyze attention patterns.

        Args:
            visualization: Attention visualization data
            top_k: Number of top features/timesteps to identify

        Returns:
            AttentionAnalysis with insights
        """
        weights = visualization.attention_weights

        # Compute feature importance (average attention received)
        if weights.ndim == 2:
            # Shape: (seq_len, seq_len)
            feature_importance = weights.mean(axis=0)
        else:
            feature_importance = weights.mean()

        # Get top features
        if isinstance(feature_importance, np.ndarray) and feature_importance.size > 1:
            top_feature_indices = np.argsort(feature_importance)[-top_k:][::-1]
            top_features = [
                (
                    visualization.feature_names[idx]
                    if idx < len(visualization.feature_names)
                    else f"feature_{idx}",
                    Decimal(str(feature_importance[idx]))
                )
                for idx in top_feature_indices
            ]
        else:
            top_features = []

        # Get top timesteps
        if weights.ndim == 2:
            timestep_importance = weights.mean(axis=1)
            top_timestep_indices = np.argsort(timestep_importance)[-top_k:][::-1]
            top_timesteps = [
                (
                    visualization.timesteps[idx]
                    if idx < len(visualization.timesteps)
                    else datetime.now(timezone.utc),
                    Decimal(str(timestep_importance[idx]))
                )
                for idx in top_timestep_indices
            ]
        else:
            top_timesteps = []

        # Compute attention entropy
        flat_weights = weights.flatten()
        flat_weights = flat_weights / (flat_weights.sum() + 1e-10)
        entropy = -np.sum(flat_weights * np.log(flat_weights + 1e-10))
        attention_entropy = Decimal(str(entropy))

        # Compute attention sparsity (Gini coefficient)
        sorted_weights = np.sort(flat_weights)
        n = len(sorted_weights)
        index = np.arange(1, n + 1)
        sparsity = (2 * np.sum(index * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n
        attention_sparsity = Decimal(str(sparsity))

        # Determine temporal focus
        if weights.ndim == 2 and len(weights) > 0:
            recent_attention = weights[-3:, :].mean() if len(weights) >= 3 else weights.mean()
            historical_attention = weights[:-3, :].mean() if len(weights) > 3 else Decimal("0")

            if recent_attention > historical_attention * 1.2:
                temporal_focus = "recent"
            elif historical_attention > recent_attention * 1.2:
                temporal_focus = "historical"
            else:
                temporal_focus = "balanced"
        else:
            temporal_focus = "unknown"

        # Create feature importance dict
        feature_importance_dict = {}
        if isinstance(feature_importance, np.ndarray) and feature_importance.size > 1:
            for idx, importance in enumerate(feature_importance):
                feature_name = (
                    visualization.feature_names[idx]
                    if idx < len(visualization.feature_names)
                    else f"feature_{idx}"
                )
                feature_importance_dict[feature_name] = Decimal(str(importance))

        analysis = AttentionAnalysis(
            top_features=top_features,
            top_timesteps=top_timesteps,
            attention_entropy=attention_entropy,
            attention_sparsity=attention_sparsity,
            temporal_focus=temporal_focus,
            feature_importance=feature_importance_dict,
            timestamp=datetime.now(timezone.utc)
        )

        logger.info(
            "Analyzed attention patterns",
            extra={
                "entropy": str(attention_entropy),
                "sparsity": str(attention_sparsity),
                "temporal_focus": temporal_focus,
                "num_top_features": len(top_features),
                "timestamp": analysis.timestamp.isoformat()
            }
        )

        return analysis

    def to_dataframe(
        self,
        visualization: AttentionVisualization
    ) -> pl.DataFrame:
        """
        Convert visualization to DataFrame.

        Args:
            visualization: Attention visualization

        Returns:
            DataFrame with attention data
        """
        weights = visualization.attention_weights

        if weights.ndim == 2:
            # Create DataFrame with timesteps as rows, features as columns
            data = {}

            # Add timestep column
            data["timestep"] = [
                ts.isoformat() for ts in visualization.timesteps[: weights.shape[0]]
            ]

            # Add attention weights for each feature
            for i, feature_name in enumerate(visualization.feature_names):
                if i < weights.shape[1]:
                    data[feature_name] = weights[:, i].tolist()

            df = pl.DataFrame(data)
        else:
            # Flatten to simple DataFrame
            df = pl.DataFrame({
                "attention_weight": weights.flatten().tolist()
            })

        logger.info(
            "Converted visualization to DataFrame",
            extra={
                "shape": df.shape,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return df

    def get_feature_attention_summary(
        self,
        analysis: AttentionAnalysis,
        threshold: Decimal = Decimal("0.05")
    ) -> pl.DataFrame:
        """
        Get summary of feature attention above threshold.

        Args:
            analysis: Attention analysis
            threshold: Minimum attention threshold

        Returns:
            DataFrame with feature attention summary
        """
        # Filter features above threshold
        filtered_features = {
            feature: importance
            for feature, importance in analysis.feature_importance.items()
            if importance >= threshold
        }

        # Sort by importance
        sorted_features = sorted(
            filtered_features.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Create DataFrame
        data = {
            "feature": [f for f, _ in sorted_features],
            "importance": [str(imp) for _, imp in sorted_features],
        }

        df = pl.DataFrame(data)

        logger.info(
            "Generated feature attention summary",
            extra={
                "num_features": len(df),
                "threshold": str(threshold),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return df

    def explain_prediction(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        timesteps: List[datetime],
        feature_names: List[str],
        layer_idx: int = -1
    ) -> Dict[str, any]:
        """
        Explain a prediction using attention visualization.

        Args:
            model: Attention model
            input_data: Input data
            timesteps: Timesteps for input
            feature_names: Feature names
            layer_idx: Layer to visualize

        Returns:
            Dictionary with explanation
        """
        # Extract attention weights
        attention_weights = self.extract_attention_weights(
            model,
            input_data,
            layer_idx
        )

        # Create visualization
        viz = self.create_visualization(
            attention_weights,
            timesteps,
            feature_names,
            layer_idx
        )

        # Analyze patterns
        analysis = self.analyze_attention(viz)

        # Get prediction
        model.eval()
        with torch.no_grad():
            if hasattr(model, "predict"):
                prediction = model.predict(input_data)
            else:
                prediction, _ = model(input_data)

        explanation = {
            "prediction": prediction.cpu().numpy().tolist(),
            "top_features": [
                {"feature": f, "importance": str(imp)}
                for f, imp in analysis.top_features
            ],
            "top_timesteps": [
                {"timestep": ts.isoformat(), "importance": str(imp)}
                for ts, imp in analysis.top_timesteps
            ],
            "attention_entropy": str(analysis.attention_entropy),
            "attention_sparsity": str(analysis.attention_sparsity),
            "temporal_focus": analysis.temporal_focus,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        logger.info(
            "Generated prediction explanation",
            extra={
                "num_top_features": len(analysis.top_features),
                "temporal_focus": analysis.temporal_focus,
                "timestamp": explanation["timestamp"]
            }
        )

        return explanation
