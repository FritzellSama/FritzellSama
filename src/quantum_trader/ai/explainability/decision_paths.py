"""Decision Path Analysis for Model Explainability."""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

@dataclass
class DecisionPath:
    """Decision path through a model."""
    node_ids: List[int]
    feature_indices: List[int]
    feature_names: List[str]
    thresholds: List[Decimal]
    decisions: List[str]
    final_prediction: int
    confidence: Decimal

class DecisionPathAnalyzer:
    """Analyzer for decision tree paths."""

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

        logger.info("Initialized decision path analyzer", extra={"timestamp": datetime.now(timezone.utc).isoformat()})

    async def extract_path(self, sample: np.ndarray) -> DecisionPath:
        """Extract decision path for a sample."""
        if isinstance(self.model, DecisionTreeClassifier):
            # Get decision path
            path = self.model.decision_path([sample])
            node_ids = path.indices.tolist()

            # Get features and thresholds
            tree = self.model.tree_
            feature_indices = []
            thresholds = []
            decisions = []
            feature_name_list = []

            for node_id in node_ids[:-1]:  # Exclude leaf
                feature_idx = tree.feature[node_id]
                threshold = tree.threshold[node_id]

                feature_indices.append(feature_idx)
                thresholds.append(Decimal(str(threshold)))
                feature_name_list.append(self.feature_names[feature_idx])

                if sample[feature_idx] <= threshold:
                    decisions.append("<=")
                else:
                    decisions.append(">")

            # Get prediction
            prediction = self.model.predict([sample])[0]
            proba = self.model.predict_proba([sample])[0]
            confidence = Decimal(str(proba.max()))

            path = DecisionPath(
                node_ids=node_ids,
                feature_indices=feature_indices,
                feature_names=feature_name_list,
                thresholds=thresholds,
                decisions=decisions,
                final_prediction=int(prediction),
                confidence=confidence
            )

            logger.info("Extracted decision path", extra={"path_length": len(node_ids), "timestamp": datetime.now(timezone.utc).isoformat()})

            return path
        else:
            raise ValueError("Model type not supported for path extraction")

    def to_dataframe(self, path: DecisionPath) -> pl.DataFrame:
        """Convert path to DataFrame."""
        data = {
            "step": list(range(len(path.feature_names))),
            "feature": path.feature_names,
            "threshold": [str(t) for t in path.thresholds],
            "decision": path.decisions
        }

        return pl.DataFrame(data)
