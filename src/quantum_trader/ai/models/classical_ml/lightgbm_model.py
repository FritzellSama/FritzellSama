"""LightGBM model for gradient boosting.

This module implements LightGBM for efficient gradient boosting on large-scale
tabular data with built-in categorical feature support.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import lightgbm as lgb
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel

logger = get_logger(__name__)


class LightGBMModel(BaseMLModel):
    """LightGBM gradient boosting model.

    Efficient gradient boosting implementation with support for
    categorical features, missing values, and large-scale datasets.

    Attributes:
        config: Model configuration
        model: LightGBM booster
        feature_importance: Feature importance scores

    Example:
        >>> config = {
        ...     "num_leaves": 31,
        ...     "max_depth": -1,
        ...     "learning_rate": "0.1",
        ...     "n_estimators": 100,
        ...     "objective": "regression",
        ...     "metric": "mse",
        ...     "boosting_type": "gbdt"
        ... }
        >>> model = LightGBMModel(config)
        >>> model.train(features, labels)
        >>> predictions = model.predict(test_features)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize LightGBM model.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If config is invalid
        """
        super().__init__(config)

        self._validate_config()

        # Model parameters
        self.num_leaves = config.get("num_leaves", 31)
        self.max_depth = config.get("max_depth", -1)
        self.learning_rate = Decimal(str(config.get("learning_rate", "0.1")))
        self.n_estimators = config.get("n_estimators", 100)
        self.objective = config.get("objective", "regression")
        self.metric = config.get("metric", "mse")
        self.boosting_type = config.get("boosting_type", "gbdt")

        # Regularization
        self.min_child_samples = config.get("min_child_samples", 20)
        self.subsample = Decimal(str(config.get("subsample", "1.0")))
        self.subsample_freq = config.get("subsample_freq", 0)
        self.colsample_bytree = Decimal(str(config.get("colsample_bytree", "1.0")))
        self.reg_alpha = Decimal(str(config.get("reg_alpha", "0.0")))
        self.reg_lambda = Decimal(str(config.get("reg_lambda", "0.0")))

        # Training parameters
        self.early_stopping_rounds = config.get("early_stopping_rounds", 50)
        self.verbose_eval = config.get("verbose_eval", 10)
        self.categorical_features = config.get("categorical_features", [])

        # Model state
        self.model: Optional[lgb.Booster] = None
        self.feature_importance: Optional[np.ndarray] = None
        self.training_history: List[Dict[str, Any]] = []

        logger.info(
            "LightGBM model initialized",
            num_leaves=self.num_leaves,
            learning_rate=float(self.learning_rate),
            n_estimators=self.n_estimators,
            objective=self.objective
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If required config missing
        """
        valid_objectives = [
            "regression",
            "binary",
            "multiclass",
            "cross_entropy",
            "regression_l1"
        ]

        objective = self.config.get("objective", "regression")

        if objective not in valid_objectives:
            raise ValueError(
                f"Invalid objective '{objective}'. "
                f"Must be one of {valid_objectives}"
            )

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None
    ) -> None:
        """Train LightGBM model.

        Args:
            features: Training features
            labels: Training labels
            val_features: Optional validation features
            val_labels: Optional validation labels

        Raises:
            ValueError: If input shape is invalid
        """
        try:
            logger.info(
                "Starting LightGBM training",
                num_samples=len(features),
                num_features=features.shape[1],
                n_estimators=self.n_estimators
            )

            # Create dataset
            train_data = lgb.Dataset(
                features,
                label=labels,
                categorical_feature=self.categorical_features
            )

            # Create validation dataset if provided
            valid_sets = [train_data]
            valid_names = ['train']

            if val_features is not None and val_labels is not None:
                val_data = lgb.Dataset(
                    val_features,
                    label=val_labels,
                    categorical_feature=self.categorical_features,
                    reference=train_data
                )
                valid_sets.append(val_data)
                valid_names.append('valid')

            # Set parameters
            params = {
                'objective': self.objective,
                'metric': self.metric,
                'boosting_type': self.boosting_type,
                'num_leaves': self.num_leaves,
                'max_depth': self.max_depth,
                'learning_rate': float(self.learning_rate),
                'min_child_samples': self.min_child_samples,
                'subsample': float(self.subsample),
                'subsample_freq': self.subsample_freq,
                'colsample_bytree': float(self.colsample_bytree),
                'reg_alpha': float(self.reg_alpha),
                'reg_lambda': float(self.reg_lambda),
                'verbose': -1
            }

            # Train
            callbacks = []

            if self.early_stopping_rounds and len(valid_sets) > 1:
                callbacks.append(
                    lgb.early_stopping(self.early_stopping_rounds)
                )

            if self.verbose_eval:
                callbacks.append(
                    lgb.log_evaluation(self.verbose_eval)
                )

            # Record training history
            evals_result = {}

            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=self.n_estimators,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
                categorical_feature=self.categorical_features,
                feval=None
            )

            # Extract feature importance
            self.feature_importance = self.model.feature_importance(
                importance_type='gain'
            )

            # Store training history
            self.training_history = [
                {
                    "iteration": i,
                    "train_metric": float(self.model.eval_train()[0][2]),
                }
                for i in range(self.model.num_trees())
            ]

            logger.info(
                "LightGBM training completed",
                num_trees=self.model.num_trees(),
                best_iteration=self.model.best_iteration if hasattr(self.model, 'best_iteration') else None
            )

        except Exception as e:
            logger.error("Failed to train LightGBM", error=str(e))
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions.

        Args:
            features: Input features

        Returns:
            Predictions

        Raises:
            ValueError: If model not trained
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained")

            predictions = self.model.predict(features)

            logger.info("Generated predictions", num_samples=len(predictions))

            return predictions.reshape(-1, 1)

        except Exception as e:
            logger.error("Failed to generate predictions", error=str(e))
            raise

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Generate probability predictions (for classification).

        Args:
            features: Input features

        Returns:
            Class probabilities

        Raises:
            ValueError: If model not trained or not classifier
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained")

            if self.objective not in ["binary", "multiclass"]:
                raise ValueError("predict_proba only available for classification")

            predictions = self.model.predict(features)

            logger.info("Generated probability predictions", num_samples=len(predictions))

            return predictions

        except Exception as e:
            logger.error("Failed to generate probability predictions", error=str(e))
            raise

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            features: Input features
            labels: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            predictions = self.predict(features).flatten()

            if self.objective == "regression":
                # Regression metrics
                mse = float(np.mean((predictions - labels) ** 2))
                mae = float(np.mean(np.abs(predictions - labels)))
                rmse = float(np.sqrt(mse))

                # R-squared
                ss_res = np.sum((labels - predictions) ** 2)
                ss_tot = np.sum((labels - np.mean(labels)) ** 2)
                r2 = float(1 - (ss_res / (ss_tot + 1e-10)))

                metrics = {
                    "mse": mse,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2
                }

            elif self.objective == "binary":
                # Binary classification metrics
                predictions_binary = (predictions > 0.5).astype(int)
                accuracy = float(np.mean(predictions_binary == labels))

                # Confusion matrix elements
                tp = np.sum((predictions_binary == 1) & (labels == 1))
                fp = np.sum((predictions_binary == 1) & (labels == 0))
                tn = np.sum((predictions_binary == 0) & (labels == 0))
                fn = np.sum((predictions_binary == 0) & (labels == 1))

                precision = float(tp / (tp + fp + 1e-10))
                recall = float(tp / (tp + fn + 1e-10))
                f1 = float(2 * precision * recall / (precision + recall + 1e-10))

                metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }

            else:
                # Default metrics
                metrics = {
                    "mse": float(np.mean((predictions - labels) ** 2))
                }

            logger.info("LightGBM evaluation completed", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("Failed to evaluate LightGBM", error=str(e))
            raise

    def get_feature_importance(
        self,
        importance_type: str = "gain"
    ) -> Dict[int, float]:
        """Get feature importance scores.

        Args:
            importance_type: Type of importance ('gain', 'split')

        Returns:
            Dictionary mapping feature index to importance

        Raises:
            ValueError: If model not trained
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained")

            importance = self.model.feature_importance(importance_type=importance_type)

            importance_dict = {
                i: float(imp)
                for i, imp in enumerate(importance)
            }

            logger.info(
                "Feature importance calculated",
                type=importance_type,
                num_features=len(importance_dict)
            )

            return importance_dict

        except Exception as e:
            logger.error("Failed to get feature importance", error=str(e))
            raise

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path to save model

        Raises:
            ValueError: If model not trained
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained")

            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save booster
            self.model.save_model(str(save_path))

            # Save metadata
            metadata_path = save_path.with_suffix('.metadata.npz')
            np.savez_compressed(
                metadata_path,
                config=self.config,
                feature_importance=self.feature_importance,
                training_history=self.training_history
            )

            logger.info("LightGBM model saved", path=path)

        except Exception as e:
            logger.error("Failed to save LightGBM model", error=str(e))
            raise

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: File path to load model from
        """
        try:
            # Load booster
            self.model = lgb.Booster(model_file=path)

            # Load metadata
            metadata_path = Path(path).with_suffix('.metadata.npz')

            if metadata_path.exists():
                data = np.load(metadata_path, allow_pickle=True)
                self.feature_importance = data["feature_importance"]
                self.training_history = data["training_history"].tolist()

            logger.info("LightGBM model loaded", path=path)

        except Exception as e:
            logger.error("Failed to load LightGBM model", error=str(e))
            raise
