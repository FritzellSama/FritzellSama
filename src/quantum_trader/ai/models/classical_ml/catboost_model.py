"""
CatBoost Model for Gradient Boosting.

Implements CatBoost classifier/regressor with optimized hyperparameters
for financial time series prediction.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from quantum_trader.ai.models.base_model import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class CatBoostConfig:
    """Configuration for CatBoost model."""

    task_type: str = "classification"  # classification or regression
    iterations: int = 1000
    learning_rate: Decimal = Decimal("0.03")
    depth: int = 6
    l2_leaf_reg: Decimal = Decimal("3.0")
    random_strength: Decimal = Decimal("1.0")
    bagging_temperature: Decimal = Decimal("1.0")
    border_count: int = 254
    early_stopping_rounds: int = 50
    use_best_model: bool = True
    eval_metric: str = "Logloss"  # or RMSE for regression
    verbose: int = 100
    task_type_device: str = "GPU"  # CPU or GPU
    cat_features: Optional[List[str]] = None


class CatBoostModel(BaseModel):
    """CatBoost model wrapper."""

    def __init__(
        self,
        model_id: str,
        config: CatBoostConfig,
        version: str = "1.0.0"
    ):
        """
        Initialize CatBoost model.

        Args:
            model_id: Model identifier
            config: CatBoost configuration
            version: Model version
        """
        super().__init__(
            model_id=model_id,
            model_type="catboost",
            version=version
        )

        self.config = config

        # Initialize CatBoost model
        if config.task_type == "classification":
            self.model = CatBoostClassifier(
                iterations=config.iterations,
                learning_rate=float(config.learning_rate),
                depth=config.depth,
                l2_leaf_reg=float(config.l2_leaf_reg),
                random_strength=float(config.random_strength),
                bagging_temperature=float(config.bagging_temperature),
                border_count=config.border_count,
                early_stopping_rounds=config.early_stopping_rounds,
                use_best_model=config.use_best_model,
                eval_metric=config.eval_metric,
                verbose=config.verbose,
                task_type=config.task_type_device,
                random_seed=42
            )
        else:
            self.model = CatBoostRegressor(
                iterations=config.iterations,
                learning_rate=float(config.learning_rate),
                depth=config.depth,
                l2_leaf_reg=float(config.l2_leaf_reg),
                random_strength=float(config.random_strength),
                bagging_temperature=float(config.bagging_temperature),
                border_count=config.border_count,
                early_stopping_rounds=config.early_stopping_rounds,
                use_best_model=config.use_best_model,
                eval_metric=config.eval_metric,
                verbose=config.verbose,
                task_type=config.task_type_device,
                random_seed=42
            )

        logger.info(
            "Initialized CatBoost model",
            extra={
                "model_id": model_id,
                "task_type": config.task_type,
                "iterations": config.iterations,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    async def train(
        self,
        train_df: pl.DataFrame,
        feature_cols: List[str],
        label_col: str,
        eval_df: Optional[pl.DataFrame] = None
    ) -> Dict[str, Decimal]:
        """
        Train CatBoost model.

        Args:
            train_df: Training data
            feature_cols: Feature column names
            label_col: Label column name
            eval_df: Optional evaluation data

        Returns:
            Training metrics
        """
        # Prepare training data
        X_train = train_df.select(feature_cols).to_numpy()
        y_train = train_df.select(label_col).to_numpy().ravel()

        # Identify categorical features
        cat_feature_indices = []
        if self.config.cat_features:
            for cat_feat in self.config.cat_features:
                if cat_feat in feature_cols:
                    cat_feature_indices.append(feature_cols.index(cat_feat))

        train_pool = Pool(
            X_train,
            y_train,
            cat_features=cat_feature_indices if cat_feature_indices else None
        )

        # Prepare evaluation data if provided
        eval_pool = None
        if eval_df is not None:
            X_eval = eval_df.select(feature_cols).to_numpy()
            y_eval = eval_df.select(label_col).to_numpy().ravel()

            eval_pool = Pool(
                X_eval,
                y_eval,
                cat_features=cat_feature_indices if cat_feature_indices else None
            )

        # Train model
        if eval_pool:
            self.model.fit(train_pool, eval_set=eval_pool)
        else:
            self.model.fit(train_pool)

        # Get metrics
        if self.config.task_type == "classification":
            train_score = self.model.score(X_train, y_train)
            metric_name = "accuracy"
        else:
            from sklearn.metrics import r2_score
            predictions = self.model.predict(X_train)
            train_score = r2_score(y_train, predictions)
            metric_name = "r2_score"

        metrics = {
            metric_name: Decimal(str(train_score)),
            "best_iteration": Decimal(str(self.model.get_best_iteration())),
            "num_trees": Decimal(str(self.model.tree_count_))
        }

        self.update_training_metadata(len(train_df), feature_cols, metrics)

        logger.info(
            "CatBoost training completed",
            extra={
                "model_id": self.metadata.model_id,
                "metrics": {k: str(v) for k, v in metrics.items()},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return metrics

    async def predict(
        self,
        data_df: pl.DataFrame,
        feature_cols: List[str]
    ) -> pl.DataFrame:
        """
        Make predictions with CatBoost model.

        Args:
            data_df: Input data
            feature_cols: Feature column names

        Returns:
            DataFrame with predictions
        """
        X = data_df.select(feature_cols).to_numpy()

        if self.config.task_type == "classification":
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            confidence = probabilities.max(axis=1)
        else:
            predictions = self.model.predict(X)
            confidence = [Decimal("1.0")] * len(predictions)

        result_df = data_df.with_columns([
            pl.Series("prediction", predictions.tolist()),
            pl.Series("confidence", confidence.tolist() if isinstance(confidence, list) else confidence.tolist())
        ])

        self.update_prediction_metadata(len(result_df))

        logger.info(
            "CatBoost predictions completed",
            extra={
                "model_id": self.metadata.model_id,
                "num_predictions": len(result_df),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return result_df

    def get_feature_importance(self) -> pl.DataFrame:
        """
        Get feature importance from trained model.

        Returns:
            DataFrame with feature importance
        """
        if not self.model.is_fitted():
            raise ValueError("Model not trained. Call train() first.")

        importance = self.model.get_feature_importance()
        feature_names = self.metadata.features

        # Create DataFrame
        importance_df = pl.DataFrame({
            "feature": feature_names,
            "importance": importance.tolist()
        }).sort("importance", descending=True)

        logger.info(
            "Retrieved feature importance",
            extra={
                "model_id": self.metadata.model_id,
                "num_features": len(importance_df),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return importance_df

    def get_model_info(self) -> Dict[str, any]:
        """
        Get detailed model information.

        Returns:
            Dictionary with model info
        """
        if not self.model.is_fitted():
            return {"fitted": False}

        info = {
            "fitted": True,
            "tree_count": self.model.tree_count_,
            "best_iteration": self.model.get_best_iteration(),
            "learning_rate": str(self.config.learning_rate),
            "depth": self.config.depth,
            "task_type": self.config.task_type
        }

        return info

    def save(self, path: str) -> None:
        """
        Save CatBoost model.

        Args:
            path: Path to save model
        """
        # Save CatBoost model
        model_path = path + ".cbm"
        self.model.save_model(model_path)

        # Save metadata
        import pickle
        metadata_path = path + ".meta"
        with open(metadata_path, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "config": self.config
            }, f)

        logger.info(
            "Saved CatBoost model",
            extra={
                "model_id": self.metadata.model_id,
                "path": path,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    def load(self, path: str) -> None:
        """
        Load CatBoost model.

        Args:
            path: Path to model file
        """
        # Load CatBoost model
        model_path = path + ".cbm"
        self.model.load_model(model_path)

        # Load metadata
        import pickle
        metadata_path = path + ".meta"
        with open(metadata_path, "rb") as f:
            data = pickle.load(f)
            self.metadata = data["metadata"]
            self.config = data["config"]

        logger.info(
            "Loaded CatBoost model",
            extra={
                "model_id": self.metadata.model_id,
                "path": path,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
