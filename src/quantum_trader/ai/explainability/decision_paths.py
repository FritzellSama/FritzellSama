"""Decision path tracking and explainability for AI trading decisions.

This module provides tools for tracking and explaining the decision-making
process of AI models, enabling transparency and auditability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class DecisionNode:
    """Represents a single node in a decision path.

    Attributes:
        node_id: Unique identifier for this node
        node_type: Type of decision node (feature, condition, action)
        timestamp: When this node was evaluated
        feature_name: Name of feature being evaluated (if applicable)
        feature_value: Value of the feature
        condition: Condition being tested
        condition_result: Result of condition evaluation
        importance_score: Importance score of this node (0.0 to 1.0)
        metadata: Additional metadata
    """

    node_id: str
    node_type: str
    timestamp: datetime
    feature_name: Optional[str] = None
    feature_value: Optional[Decimal] = None
    condition: Optional[str] = None
    condition_result: Optional[bool] = None
    importance_score: Decimal = Decimal("0.0")
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionPath:
    """Represents a complete decision path from input to output.

    Attributes:
        path_id: Unique identifier for this path
        start_timestamp: When decision process started
        end_timestamp: When decision was made
        input_features: Input features used
        nodes: List of decision nodes in order
        final_decision: Final decision/prediction made
        confidence: Confidence in the decision
        model_name: Name of model that made decision
        metadata: Additional metadata
    """

    path_id: str
    start_timestamp: datetime
    end_timestamp: datetime
    input_features: Dict[str, Decimal]
    nodes: List[DecisionNode]
    final_decision: Any
    confidence: Decimal
    model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecisionPathTracker:
    """Tracks and records decision paths for explainability.

    Maintains a history of decision paths for analysis and debugging.

    Attributes:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize decision path tracker.

        Args:
            config: Configuration with parameters:
                - max_paths: Maximum paths to store (default: 10000)
                - track_importance: Whether to track feature importance (default: True)
                - persistence_enabled: Whether to persist paths (default: True)
                - min_confidence_threshold: Min confidence to record (default: 0.0)

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.max_paths = int(config.get("max_paths", 10000))
        self.track_importance = config.get("track_importance", True)
        self.persistence_enabled = config.get("persistence_enabled", True)
        self.min_confidence_threshold = Decimal(
            str(config.get("min_confidence_threshold", "0.0"))
        )

        # Storage
        self.paths: List[DecisionPath] = []
        self.current_path: Optional[DecisionPath] = None

        logger.info(
            "decision_path_tracker_initialized",
            max_paths=self.max_paths,
            track_importance=self.track_importance,
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(self.config)}")

    def start_path(
        self,
        path_id: str,
        input_features: Dict[str, Decimal],
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Start tracking a new decision path.

        Args:
            path_id: Unique identifier for this path
            input_features: Input features dictionary
            model_name: Name of the model making the decision
            metadata: Optional additional metadata

        Raises:
            ValueError: If path already started
        """
        try:
            if self.current_path is not None:
                logger.warning(
                    "path_already_in_progress",
                    current_path_id=self.current_path.path_id,
                    new_path_id=path_id,
                )
                self.end_path(final_decision="aborted", confidence=Decimal("0.0"))

            self.current_path = DecisionPath(
                path_id=path_id,
                start_timestamp=datetime.utcnow(),
                end_timestamp=datetime.utcnow(),  # Will be updated
                input_features=input_features,
                nodes=[],
                final_decision=None,
                confidence=Decimal("0.0"),
                model_name=model_name,
                metadata=metadata or {},
            )

            logger.debug("decision_path_started", path_id=path_id, model_name=model_name)

        except Exception as e:
            logger.error("start_path_failed", error=str(e))
            raise

    def add_node(
        self,
        node_type: str,
        feature_name: Optional[str] = None,
        feature_value: Optional[Decimal] = None,
        condition: Optional[str] = None,
        condition_result: Optional[bool] = None,
        importance_score: Optional[Decimal] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a node to the current decision path.

        Args:
            node_type: Type of node (feature, condition, action)
            feature_name: Name of feature being evaluated
            feature_value: Value of the feature
            condition: Condition being tested
            condition_result: Result of condition evaluation
            importance_score: Importance score (0.0 to 1.0)
            metadata: Additional metadata

        Raises:
            ValueError: If no path is currently being tracked
        """
        try:
            if self.current_path is None:
                raise ValueError("No decision path currently being tracked")

            node_id = f"{self.current_path.path_id}_node_{len(self.current_path.nodes)}"

            node = DecisionNode(
                node_id=node_id,
                node_type=node_type,
                timestamp=datetime.utcnow(),
                feature_name=feature_name,
                feature_value=feature_value,
                condition=condition,
                condition_result=condition_result,
                importance_score=importance_score or Decimal("0.0"),
                metadata=metadata or {},
            )

            self.current_path.nodes.append(node)

            logger.debug(
                "node_added",
                path_id=self.current_path.path_id,
                node_id=node_id,
                node_type=node_type,
            )

        except Exception as e:
            logger.error("add_node_failed", error=str(e))
            raise

    def end_path(
        self,
        final_decision: Any,
        confidence: Decimal,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DecisionPath:
        """End the current decision path and store it.

        Args:
            final_decision: Final decision made
            confidence: Confidence in the decision (0.0 to 1.0)
            metadata: Optional additional metadata

        Returns:
            The completed decision path

        Raises:
            ValueError: If no path is currently being tracked
        """
        try:
            if self.current_path is None:
                raise ValueError("No decision path currently being tracked")

            # Update path with final information
            self.current_path.end_timestamp = datetime.utcnow()
            self.current_path.final_decision = final_decision
            self.current_path.confidence = confidence

            if metadata:
                self.current_path.metadata.update(metadata)

            # Store path if it meets threshold
            if confidence >= self.min_confidence_threshold:
                self.paths.append(self.current_path)

                # Limit storage size
                if len(self.paths) > self.max_paths:
                    self.paths.pop(0)

                logger.info(
                    "decision_path_completed",
                    path_id=self.current_path.path_id,
                    num_nodes=len(self.current_path.nodes),
                    confidence=str(confidence),
                    decision=str(final_decision),
                )
            else:
                logger.debug(
                    "path_below_confidence_threshold",
                    path_id=self.current_path.path_id,
                    confidence=str(confidence),
                    threshold=str(self.min_confidence_threshold),
                )

            completed_path = self.current_path
            self.current_path = None

            return completed_path

        except Exception as e:
            logger.error("end_path_failed", error=str(e))
            raise

    def get_path(self, path_id: str) -> Optional[DecisionPath]:
        """Retrieve a decision path by ID.

        Args:
            path_id: Path identifier

        Returns:
            DecisionPath if found, None otherwise
        """
        try:
            for path in self.paths:
                if path.path_id == path_id:
                    return path
            return None

        except Exception as e:
            logger.error("get_path_failed", error=str(e))
            raise

    def get_recent_paths(self, count: int = 10) -> List[DecisionPath]:
        """Get the most recent decision paths.

        Args:
            count: Number of paths to retrieve

        Returns:
            List of recent decision paths
        """
        try:
            return self.paths[-count:]

        except Exception as e:
            logger.error("get_recent_paths_failed", error=str(e))
            raise

    def analyze_feature_importance(
        self, num_paths: Optional[int] = None
    ) -> Dict[str, Decimal]:
        """Analyze feature importance across decision paths.

        Args:
            num_paths: Number of recent paths to analyze (None = all)

        Returns:
            Dictionary mapping feature names to importance scores

        Raises:
            ValueError: If analysis fails
        """
        try:
            paths_to_analyze = self.paths[-num_paths:] if num_paths else self.paths

            if len(paths_to_analyze) == 0:
                logger.warning("no_paths_to_analyze")
                return {}

            feature_importance: Dict[str, List[Decimal]] = {}

            # Collect importance scores for each feature
            for path in paths_to_analyze:
                for node in path.nodes:
                    if node.feature_name and self.track_importance:
                        if node.feature_name not in feature_importance:
                            feature_importance[node.feature_name] = []
                        feature_importance[node.feature_name].append(
                            node.importance_score
                        )

            # Calculate average importance
            avg_importance = {}
            for feature, scores in feature_importance.items():
                avg_importance[feature] = sum(scores) / Decimal(str(len(scores)))

            # Sort by importance
            sorted_importance = dict(
                sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            )

            logger.info(
                "feature_importance_analyzed",
                num_features=len(sorted_importance),
                num_paths=len(paths_to_analyze),
            )

            return sorted_importance

        except Exception as e:
            logger.error("feature_importance_analysis_failed", error=str(e))
            raise

    def export_to_dataframe(self) -> pl.DataFrame:
        """Export decision paths to Polars DataFrame.

        Returns:
            DataFrame with all decision paths

        Raises:
            ValueError: If export fails
        """
        try:
            if len(self.paths) == 0:
                logger.warning("no_paths_to_export")
                return pl.DataFrame()

            # Flatten paths into records
            records = []
            for path in self.paths:
                for node in path.nodes:
                    record = {
                        "path_id": path.path_id,
                        "model_name": path.model_name,
                        "start_timestamp": path.start_timestamp,
                        "end_timestamp": path.end_timestamp,
                        "node_id": node.node_id,
                        "node_type": node.node_type,
                        "node_timestamp": node.timestamp,
                        "feature_name": node.feature_name,
                        "feature_value": node.feature_value,
                        "condition": node.condition,
                        "condition_result": node.condition_result,
                        "importance_score": node.importance_score,
                        "final_decision": str(path.final_decision),
                        "confidence": path.confidence,
                    }
                    records.append(record)

            df = pl.DataFrame(records)

            logger.info("paths_exported_to_dataframe", rows=len(df))

            return df

        except Exception as e:
            logger.error("export_to_dataframe_failed", error=str(e))
            raise

    def clear_paths(self) -> None:
        """Clear all stored paths."""
        try:
            num_cleared = len(self.paths)
            self.paths.clear()
            self.current_path = None

            logger.info("paths_cleared", num_paths=num_cleared)

        except Exception as e:
            logger.error("clear_paths_failed", error=str(e))
            raise


class DecisionExplainer:
    """Explains individual decisions using decision paths.

    Provides human-readable explanations of AI decisions.

    Attributes:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize decision explainer.

        Args:
            config: Configuration with parameters:
                - max_features_in_explanation: Max features to include (default: 5)
                - min_importance_threshold: Min importance to include (default: 0.1)

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.max_features = int(config.get("max_features_in_explanation", 5))
        self.min_importance = Decimal(str(config.get("min_importance_threshold", "0.1")))

        logger.info(
            "decision_explainer_initialized",
            max_features=self.max_features,
            min_importance=str(self.min_importance),
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(self.config)}")

    def explain_decision(self, path: DecisionPath) -> Dict[str, Any]:
        """Generate explanation for a decision path.

        Args:
            path: Decision path to explain

        Returns:
            Dictionary with explanation components

        Raises:
            ValueError: If explanation generation fails
        """
        try:
            logger.info("generating_decision_explanation", path_id=path.path_id)

            # Extract key features by importance
            feature_impacts = []
            for node in path.nodes:
                if (
                    node.feature_name
                    and node.importance_score >= self.min_importance
                ):
                    feature_impacts.append(
                        {
                            "feature": node.feature_name,
                            "value": node.feature_value,
                            "importance": node.importance_score,
                            "condition": node.condition,
                            "result": node.condition_result,
                        }
                    )

            # Sort by importance and limit
            feature_impacts.sort(key=lambda x: x["importance"], reverse=True)
            top_features = feature_impacts[: self.max_features]

            # Generate textual explanation
            explanation_text = self._generate_explanation_text(path, top_features)

            explanation = {
                "path_id": path.path_id,
                "model_name": path.model_name,
                "decision": path.final_decision,
                "confidence": path.confidence,
                "explanation_text": explanation_text,
                "top_features": top_features,
                "num_nodes": len(path.nodes),
                "duration_ms": (
                    path.end_timestamp - path.start_timestamp
                ).total_seconds()
                * 1000,
            }

            logger.info(
                "explanation_generated",
                path_id=path.path_id,
                num_features=len(top_features),
            )

            return explanation

        except Exception as e:
            logger.error("explain_decision_failed", error=str(e))
            raise

    def _generate_explanation_text(
        self, path: DecisionPath, top_features: List[Dict[str, Any]]
    ) -> str:
        """Generate human-readable explanation text.

        Args:
            path: Decision path
            top_features: List of top feature impacts

        Returns:
            Explanation text
        """
        try:
            parts = [
                f"Model '{path.model_name}' made decision: {path.final_decision}",
                f"Confidence: {float(path.confidence):.2%}",
                "",
                "Key factors:",
            ]

            for i, feature in enumerate(top_features, 1):
                importance_pct = float(feature["importance"]) * 100
                parts.append(
                    f"{i}. {feature['feature']}: {feature['value']} "
                    f"(importance: {importance_pct:.1f}%)"
                )

                if feature["condition"] and feature["result"] is not None:
                    result_text = "satisfied" if feature["result"] else "not satisfied"
                    parts.append(f"   Condition '{feature['condition']}' was {result_text}")

            explanation = "\n".join(parts)
            return explanation

        except Exception as e:
            logger.error("explanation_text_generation_failed", error=str(e))
            raise

    def compare_decisions(
        self, path1: DecisionPath, path2: DecisionPath
    ) -> Dict[str, Any]:
        """Compare two decision paths.

        Args:
            path1: First decision path
            path2: Second decision path

        Returns:
            Dictionary with comparison results

        Raises:
            ValueError: If comparison fails
        """
        try:
            logger.info(
                "comparing_decisions",
                path1_id=path1.path_id,
                path2_id=path2.path_id,
            )

            # Extract features from both paths
            features1 = {
                node.feature_name: node.feature_value
                for node in path1.nodes
                if node.feature_name
            }
            features2 = {
                node.feature_name: node.feature_value
                for node in path2.nodes
                if node.feature_name
            }

            # Find differences
            all_features = set(features1.keys()) | set(features2.keys())
            differences = []

            for feature in all_features:
                val1 = features1.get(feature)
                val2 = features2.get(feature)

                if val1 != val2:
                    differences.append(
                        {
                            "feature": feature,
                            "path1_value": val1,
                            "path2_value": val2,
                            "difference": (
                                abs(val1 - val2) if val1 and val2 else None
                            ),
                        }
                    )

            comparison = {
                "path1_id": path1.path_id,
                "path2_id": path2.path_id,
                "path1_decision": path1.final_decision,
                "path2_decision": path2.final_decision,
                "path1_confidence": path1.confidence,
                "path2_confidence": path2.confidence,
                "decisions_match": path1.final_decision == path2.final_decision,
                "confidence_difference": abs(path1.confidence - path2.confidence),
                "num_differences": len(differences),
                "differences": differences,
            }

            logger.info(
                "decisions_compared",
                num_differences=len(differences),
                decisions_match=comparison["decisions_match"],
            )

            return comparison

        except Exception as e:
            logger.error("compare_decisions_failed", error=str(e))
            raise


__all__ = [
    "DecisionNode",
    "DecisionPath",
    "DecisionPathTracker",
    "DecisionExplainer",
]
