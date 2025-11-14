"""
Attention Visualizer for Model Explainability.

This module provides tools to visualize and analyze attention weights
from attention-based models to understand trading decisions.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for attention visualization."""

    output_dir: str  # Directory for saving visualizations
    figure_size: Tuple[int, int]  # Figure size (width, height)
    dpi: int  # Dots per inch for saved figures
    colormap: str  # Matplotlib colormap
    show_values: bool  # Show numeric values on heatmap
    normalize: bool  # Normalize attention weights
    top_k_features: int  # Show top K features
    min_attention_threshold: Decimal  # Minimum attention to display


@dataclass
class AttentionAnalysis:
    """Results from attention analysis."""

    top_features: List[Tuple[int, Decimal]]  # (feature_idx, attention_score)
    avg_attention_per_head: List[Decimal]  # Average attention per head
    entropy: Decimal  # Attention entropy (diversity)
    sparsity: Decimal  # Attention sparsity
    timestamp: datetime
    layer_statistics: Dict[str, Any] = field(default_factory=dict)


class AttentionVisualizer:
    """Visualizer for attention weights and explanations."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize attention visualizer.

        Args:
            config: Visualization configuration

        Example:
            >>> config = {
            ...     "output_dir": "/tmp/visualizations",
            ...     "top_k_features": 10,
            ...     "colormap": "viridis"
            ... }
            >>> visualizer = AttentionVisualizer(config)
        """
        self.config = self._build_config(config)

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Cache for attention weights
        self.attention_cache: Dict[str, np.ndarray] = {}

        # Feature names (if provided)
        self.feature_names: Optional[List[str]] = config.get("feature_names")

        logger.info(
            "Attention visualizer initialized",
            output_dir=self.config.output_dir
        )

    def _build_config(self, config: Dict[str, Any]) -> VisualizationConfig:
        """Build VisualizationConfig from dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            VisualizationConfig instance
        """
        return VisualizationConfig(
            output_dir=config.get("output_dir", "/tmp/visualizations"),
            figure_size=tuple(config.get("figure_size", [12, 8])),
            dpi=config.get("dpi", 100),
            colormap=config.get("colormap", "viridis"),
            show_values=config.get("show_values", False),
            normalize=config.get("normalize", True),
            top_k_features=config.get("top_k_features", 10),
            min_attention_threshold=Decimal(str(config.get("min_attention_threshold", "0.01")))
        )

    async def analyze_attention(
        self,
        attention_weights: np.ndarray,
        layer_idx: int = 0
    ) -> AttentionAnalysis:
        """Analyze attention weights to extract insights.

        Args:
            attention_weights: Attention weight array [batch, heads, seq, seq]
            layer_idx: Layer index for analysis

        Returns:
            AttentionAnalysis with insights

        Example:
            >>> weights = model.get_attention_weights()[0]
            >>> analysis = await visualizer.analyze_attention(weights)
            >>> print(f"Top features: {analysis.top_features}")
        """
        try:
            # Average over batch and heads
            if len(attention_weights.shape) == 4:
                # [batch, heads, seq, seq] -> [seq, seq]
                avg_attention = attention_weights.mean(axis=(0, 1))
            elif len(attention_weights.shape) == 3:
                # [heads, seq, seq] -> [seq, seq]
                avg_attention = attention_weights.mean(axis=0)
            else:
                avg_attention = attention_weights

            # Compute attention to each position (sum over source positions)
            attention_per_position = avg_attention.sum(axis=0)

            # Get top features
            top_indices = np.argsort(attention_per_position)[::-1]
            top_features = [
                (int(idx), Decimal(str(attention_per_position[idx])))
                for idx in top_indices[:self.config.top_k_features]
            ]

            # Compute per-head statistics
            if len(attention_weights.shape) >= 3:
                num_heads = attention_weights.shape[1] if len(attention_weights.shape) == 4 else attention_weights.shape[0]
                head_attention = attention_weights.mean(axis=(0, 2, 3)) if len(attention_weights.shape) == 4 else attention_weights.mean(axis=(1, 2))
                avg_attention_per_head = [Decimal(str(a)) for a in head_attention]
            else:
                avg_attention_per_head = [Decimal("1.0")]

            # Compute entropy (measure of attention diversity)
            entropy = self._compute_entropy(attention_per_position)

            # Compute sparsity (measure of attention concentration)
            sparsity = self._compute_sparsity(attention_per_position)

            # Layer statistics
            layer_stats = {
                "mean_attention": float(avg_attention.mean()),
                "std_attention": float(avg_attention.std()),
                "max_attention": float(avg_attention.max()),
                "min_attention": float(avg_attention.min())
            }

            logger.info(
                "Attention analysis completed",
                layer=layer_idx,
                entropy=float(entropy),
                sparsity=float(sparsity)
            )

            return AttentionAnalysis(
                top_features=top_features,
                avg_attention_per_head=avg_attention_per_head,
                entropy=entropy,
                sparsity=sparsity,
                timestamp=datetime.utcnow(),
                layer_statistics=layer_stats
            )

        except Exception as e:
            logger.error("Attention analysis failed", error=str(e))
            raise

    def _compute_entropy(self, attention: np.ndarray) -> Decimal:
        """Compute entropy of attention distribution.

        Args:
            attention: Attention weights

        Returns:
            Entropy value
        """
        # Normalize to probability distribution
        attention = attention / (attention.sum() + 1e-10)

        # Compute entropy
        entropy = -np.sum(attention * np.log(attention + 1e-10))

        return Decimal(str(entropy))

    def _compute_sparsity(self, attention: np.ndarray) -> Decimal:
        """Compute sparsity of attention distribution.

        Args:
            attention: Attention weights

        Returns:
            Sparsity value (0 = uniform, 1 = concentrated)
        """
        # Gini coefficient for sparsity
        sorted_attention = np.sort(attention)
        n = len(sorted_attention)
        cumsum = np.cumsum(sorted_attention)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_attention)) / (n * cumsum[-1]) - (n + 1) / n

        return Decimal(str(gini))

    async def visualize_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        title: str = "Attention Weights",
        save_path: Optional[str] = None
    ) -> str:
        """Create heatmap visualization of attention weights.

        Args:
            attention_weights: Attention weight array
            title: Plot title
            save_path: Path to save figure (auto-generated if None)

        Returns:
            Path to saved visualization

        Example:
            >>> path = await visualizer.visualize_attention_heatmap(
            ...     weights,
            ...     title="Layer 1 Attention"
            ... )
        """
        try:
            # Generate visualization data
            viz_data = self._prepare_heatmap_data(attention_weights)

            # Create save path
            if save_path is None:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                save_path = f"{self.config.output_dir}/attention_heatmap_{timestamp}.json"

            # Save visualization data as JSON (for later rendering)
            import json
            viz_config = {
                "type": "heatmap",
                "title": title,
                "data": viz_data.tolist(),
                "colormap": self.config.colormap,
                "show_values": self.config.show_values,
                "timestamp": datetime.utcnow().isoformat()
            }

            with open(save_path, 'w') as f:
                json.dump(viz_config, f, indent=2)

            logger.info("Heatmap visualization saved", path=save_path)
            return save_path

        except Exception as e:
            logger.error("Heatmap visualization failed", error=str(e))
            raise

    def _prepare_heatmap_data(self, attention_weights: np.ndarray) -> np.ndarray:
        """Prepare attention data for heatmap.

        Args:
            attention_weights: Raw attention weights

        Returns:
            Processed data for visualization
        """
        # Average over batch and heads if needed
        if len(attention_weights.shape) == 4:
            data = attention_weights.mean(axis=(0, 1))
        elif len(attention_weights.shape) == 3:
            data = attention_weights.mean(axis=0)
        else:
            data = attention_weights

        # Normalize if requested
        if self.config.normalize:
            data = data / (data.max() + 1e-10)

        return data

    async def explain_prediction(
        self,
        attention_weights: List[np.ndarray],
        features: np.ndarray,
        prediction: int,
        confidence: Decimal
    ) -> Dict[str, Any]:
        """Generate explanation for a model prediction.

        Args:
            attention_weights: List of attention weights from all layers
            features: Input features
            prediction: Model prediction
            confidence: Prediction confidence

        Returns:
            Explanation dictionary

        Example:
            >>> explanation = await visualizer.explain_prediction(
            ...     attention_weights,
            ...     features,
            ...     prediction=1,
            ...     confidence=Decimal("0.85")
            ... )
        """
        try:
            # Analyze each layer
            layer_analyses = []
            for idx, weights in enumerate(attention_weights):
                analysis = await self.analyze_attention(weights, idx)
                layer_analyses.append(analysis)

            # Aggregate top features across layers
            all_top_features: Dict[int, Decimal] = {}
            for analysis in layer_analyses:
                for feat_idx, score in analysis.top_features:
                    if feat_idx in all_top_features:
                        all_top_features[feat_idx] += score
                    else:
                        all_top_features[feat_idx] = score

            # Sort by aggregated importance
            sorted_features = sorted(
                all_top_features.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.config.top_k_features]

            # Create feature importance explanation
            feature_importance = []
            for feat_idx, importance in sorted_features:
                feature_name = (
                    self.feature_names[feat_idx]
                    if self.feature_names and feat_idx < len(self.feature_names)
                    else f"feature_{feat_idx}"
                )
                feature_value = float(features[feat_idx]) if feat_idx < len(features) else 0.0

                feature_importance.append({
                    "feature_index": feat_idx,
                    "feature_name": feature_name,
                    "feature_value": feature_value,
                    "importance_score": float(importance),
                    "normalized_importance": float(importance / sorted_features[0][1])
                })

            # Compute overall attention statistics
            avg_entropy = Decimal(str(np.mean([float(a.entropy) for a in layer_analyses])))
            avg_sparsity = Decimal(str(np.mean([float(a.sparsity) for a in layer_analyses])))

            explanation = {
                "prediction": prediction,
                "confidence": float(confidence),
                "feature_importance": feature_importance,
                "layer_analyses": [
                    {
                        "layer": idx,
                        "entropy": float(analysis.entropy),
                        "sparsity": float(analysis.sparsity),
                        "statistics": analysis.layer_statistics
                    }
                    for idx, analysis in enumerate(layer_analyses)
                ],
                "overall_metrics": {
                    "avg_entropy": float(avg_entropy),
                    "avg_sparsity": float(avg_sparsity),
                    "num_layers": len(layer_analyses)
                },
                "timestamp": datetime.utcnow().isoformat()
            }

            logger.info(
                "Prediction explanation generated",
                prediction=prediction,
                top_feature=feature_importance[0]["feature_name"] if feature_importance else None
            )

            return explanation

        except Exception as e:
            logger.error("Explanation generation failed", error=str(e))
            raise

    async def generate_feature_importance_report(
        self,
        attention_weights_history: List[List[np.ndarray]],
        feature_names: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """Generate feature importance report from attention history.

        Args:
            attention_weights_history: List of attention weights from multiple predictions
            feature_names: Optional feature names

        Returns:
            DataFrame with feature importance statistics

        Example:
            >>> report = await visualizer.generate_feature_importance_report(
            ...     attention_history,
            ...     feature_names=["price", "volume", "rsi"]
            ... )
        """
        try:
            if feature_names:
                self.feature_names = feature_names

            # Aggregate attention across all predictions
            feature_importance_scores: Dict[int, List[float]] = {}

            for attention_weights in attention_weights_history:
                for layer_weights in attention_weights:
                    analysis = await self.analyze_attention(layer_weights)

                    for feat_idx, score in analysis.top_features:
                        if feat_idx not in feature_importance_scores:
                            feature_importance_scores[feat_idx] = []
                        feature_importance_scores[feat_idx].append(float(score))

            # Compute statistics
            report_data = []
            for feat_idx, scores in feature_importance_scores.items():
                feature_name = (
                    self.feature_names[feat_idx]
                    if self.feature_names and feat_idx < len(self.feature_names)
                    else f"feature_{feat_idx}"
                )

                report_data.append({
                    "feature_index": feat_idx,
                    "feature_name": feature_name,
                    "mean_importance": np.mean(scores),
                    "std_importance": np.std(scores),
                    "max_importance": np.max(scores),
                    "min_importance": np.min(scores),
                    "num_appearances": len(scores)
                })

            # Create DataFrame
            df = pl.DataFrame(report_data)
            df = df.sort("mean_importance", descending=True)

            logger.info(
                "Feature importance report generated",
                num_features=len(report_data),
                top_feature=report_data[0]["feature_name"] if report_data else None
            )

            return df

        except Exception as e:
            logger.error("Report generation failed", error=str(e))
            raise

    async def visualize_attention_flow(
        self,
        attention_weights: List[np.ndarray],
        layer_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """Visualize attention flow across layers.

        Args:
            attention_weights: Attention weights from all layers
            layer_names: Optional layer names
            save_path: Path to save visualization

        Returns:
            Path to saved visualization
        """
        try:
            # Analyze each layer
            flow_data = []
            for idx, weights in enumerate(attention_weights):
                analysis = await self.analyze_attention(weights, idx)

                layer_name = (
                    layer_names[idx]
                    if layer_names and idx < len(layer_names)
                    else f"layer_{idx}"
                )

                flow_data.append({
                    "layer": layer_name,
                    "entropy": float(analysis.entropy),
                    "sparsity": float(analysis.sparsity),
                    "top_features": [
                        {"index": idx, "score": float(score)}
                        for idx, score in analysis.top_features[:5]
                    ]
                })

            # Create save path
            if save_path is None:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                save_path = f"{self.config.output_dir}/attention_flow_{timestamp}.json"

            # Save flow data
            import json
            with open(save_path, 'w') as f:
                json.dump({
                    "type": "attention_flow",
                    "layers": flow_data,
                    "timestamp": datetime.utcnow().isoformat()
                }, f, indent=2)

            logger.info("Attention flow visualization saved", path=save_path)
            return save_path

        except Exception as e:
            logger.error("Flow visualization failed", error=str(e))
            raise

    def get_cached_attention(self, cache_key: str) -> Optional[np.ndarray]:
        """Get cached attention weights.

        Args:
            cache_key: Cache key

        Returns:
            Cached attention weights or None
        """
        return self.attention_cache.get(cache_key)

    def cache_attention(self, cache_key: str, attention: np.ndarray) -> None:
        """Cache attention weights.

        Args:
            cache_key: Cache key
            attention: Attention weights
        """
        self.attention_cache[cache_key] = attention

    def clear_cache(self) -> None:
        """Clear attention cache."""
        self.attention_cache.clear()
        logger.info("Attention cache cleared")

    async def generate_summary_statistics(
        self,
        attention_weights_history: List[List[np.ndarray]]
    ) -> Dict[str, Any]:
        """Generate summary statistics for attention patterns.

        Args:
            attention_weights_history: History of attention weights

        Returns:
            Dictionary with summary statistics
        """
        try:
            all_entropies = []
            all_sparsities = []

            for attention_weights in attention_weights_history:
                for weights in attention_weights:
                    analysis = await self.analyze_attention(weights)
                    all_entropies.append(float(analysis.entropy))
                    all_sparsities.append(float(analysis.sparsity))

            summary = {
                "num_samples": len(attention_weights_history),
                "entropy": {
                    "mean": np.mean(all_entropies),
                    "std": np.std(all_entropies),
                    "min": np.min(all_entropies),
                    "max": np.max(all_entropies)
                },
                "sparsity": {
                    "mean": np.mean(all_sparsities),
                    "std": np.std(all_sparsities),
                    "min": np.min(all_sparsities),
                    "max": np.max(all_sparsities)
                },
                "timestamp": datetime.utcnow().isoformat()
            }

            logger.info("Summary statistics generated", num_samples=len(attention_weights_history))
            return summary

        except Exception as e:
            logger.error("Summary generation failed", error=str(e))
            raise
