"""
LightGBM Model for Trading Predictions.

This module implements a LightGBM gradient boosting model optimized
for financial time series prediction with proper error handling and
production-ready features.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import json

import numpy as np
import polars as pl
import lightgbm as lgb
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel
from quantum_trader.exceptions import ModelError, ValidationError

logger = get_logger(__name__)


class LightGBMModel(BaseMLModel):
    """LightGBM model for trading predictions.

    Production-ready gradient boosting model with hyperparameter
    optimization, early stopping, and proper validation.

    Attributes:
        config: Model configuration
        model: LightGBM model
        feature_names: Names of input features
        feature_importance: Feature importance scores
        best_iteration: Best iteration from training

    Example:
        >>> config = {
        ...     "objective": "regression",
        ...     "metric": "rmse",
        ...     "num_leaves": 31,
        ...     "learning_rate": 0.05,
        ...     "feature_fraction": 0.9,
        ...     "bagging_fraction": 0.8,
        ...     "bagging_freq": 5,
        ...     "max_depth": -1,
        ...     "min_data_in_leaf": 20,
        ...     "num_iterations": 1000,
        ...     "early_stopping_rounds": 50,
        ...     "verbose": -1
        ... }
        >>> model = LightGBMModel(config)
        >>> model.train(features, labels)
        >>> predictions = model.predict(test_features)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize LightGBM model.

        Args:
            config: Model configuration

        Raises:
            ValidationError: If configuration is invalid
        """
        super().__init__(config)

        self._validate_config()

        # LightGBM parameters
        self.objective = config.get("objective", "regression")
        self.metric = config.get("metric", "rmse")
        self.num_leaves = config.get("num_leaves", 31)
        self.learning_rate = Decimal(str(config.get("learning_rate", 0.05)))
        self.feature_fraction = Decimal(str(config.get("feature_fraction", 0.9)))
        self.bagging_fraction = Decimal(str(config.get("bagging_fraction", 0.8)))
        self.bagging_freq = config.get("bagging_freq", 5)
        self.max_depth = config.get("max_depth", -1)
        self.min_data_in_leaf = config.get("min_data_in_leaf", 20)
        self.num_iterations = config.get("num_iterations", 1000)
        self.early_stopping_rounds = config.get("early_stopping_rounds", 50)
        self.verbose = config.get("verbose", -1)
        self.num_threads = config.get("num_threads", -1)

        # Model state
        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []
        self.feature_importance: Dict[str, Decimal] = {}
        self.best_iteration: int = 0
        self._trained = False

        logger.info(
            "LightGBM model initialized",
            objective=self.objective,
            metric=self.metric,
            num_leaves=self.num_leaves,
            learning_rate=float(self.learning_rate)
        )

    def _validate_config(self) -> None:
        """Validate model configuration.

        Raises:
            ValidationError: If configuration is invalid
        """
        valid_objectives = ["regression", "binary", "multiclass", "lambdarank"]
        if self.config.get("objective") and self.config["objective"] not in valid_objectives:
            raise ValidationError(f"Invalid objective: {self.config['objective']}")

        if self.config.get("num_leaves", 1) < 2:
            raise ValidationError("num_leaves must be >= 2")

        if self.config.get("learning_rate", 0) <= 0:
            raise ValidationError("learning_rate must be > 0")

        if self.config.get("num_iterations", 0) < 1:
            raise ValidationError("num_iterations must be >= 1")

    def _get_params(self) -> Dict[str, Any]:
        """Get LightGBM parameters.

        Returns:
            Parameter dictionary
        """
        params = {
            "objective": self.objective,
            "metric": self.metric,
            "num_leaves": self.num_leaves,
            "learning_rate": float(self.learning_rate),
            "feature_fraction": float(self.feature_fraction),
            "bagging_fraction": float(self.bagging_fraction),
            "bagging_freq": self.bagging_freq,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "verbose": self.verbose,
            "num_threads": self.num_threads,
            "force_col_wise": True,
        }

        return params

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> None:
        """Train the LightGBM model.

        Args:
            features: Training features [N, num_features]
            labels: Training labels [N,]
            val_features: Validation features (optional)
            val_labels: Validation labels (optional)
            feature_names: Feature names (optional)

        Raises:
            ModelError: If training fails
            ValidationError: If data is invalid
        """
        try:
            logger.info("Starting LightGBM training", num_samples=len(features))

            # Validate data
            self._validate_training_data(features, labels)

            # Store feature names
            if feature_names is not None:
                self.feature_names = feature_names
            else:
                self.feature_names = [f"feature_{i}" for i in range(features.shape[1])]

            # Create dataset
            train_data = lgb.Dataset(
                features,
                label=labels,
                feature_name=self.feature_names,
                free_raw_data=False
            )

            # Create validation dataset
            valid_sets = [train_data]
            valid_names = ["train"]

            if val_features is not None and val_labels is not None:
                self._validate_training_data(val_features, val_labels)
                val_data = lgb.Dataset(
                    val_features,
                    label=val_labels,
                    feature_name=self.feature_names,
                    reference=train_data,
                    free_raw_data=False
                )
                valid_sets.append(val_data)
                valid_names.append("valid")

            # Train model
            params = self._get_params()

            callbacks = []
            if self.early_stopping_rounds > 0 and len(valid_sets) > 1:
                callbacks.append(
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False)
                )

            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=self.num_iterations,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks
            )

            # Store best iteration
            self.best_iteration = self.model.best_iteration

            # Calculate feature importance
            self._calculate_feature_importance()

            self._trained = True

            logger.info(
                "Training completed successfully",
                best_iteration=self.best_iteration,
                num_features=len(self.feature_names)
            )

        except Exception as e:
            logger.error("Training failed", error=str(e))
            raise ModelError(f"Training failed: {e}") from e

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions.

        Args:
            features: Input features [N, num_features]

        Returns:
            Predictions [N,]

        Raises:
            ModelError: If prediction fails
            ValidationError: If features are invalid
        """
        try:
            if self.model is None:
                raise ModelError("Model not trained")

            if not self._trained:
                logger.warning("Model not trained, predictions may be unreliable")

            # Validate features
            if features.shape[1] != len(self.feature_names):
                raise ValidationError(
                    f"Feature dimension mismatch: expected {len(self.feature_names)}, "
                    f"got {features.shape[1]}"
                )

            # Generate predictions
            predictions = self.model.predict(
                features,
                num_iteration=self.best_iteration
            )

            return predictions

        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            raise ModelError(f"Prediction failed: {e}") from e

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            features: Test features
            labels: Test labels

        Returns:
            Dictionary of evaluation metrics

        Raises:
            ModelError: If evaluation fails
        """
        try:
            if self.model is None:
                raise ModelError("Model not trained")

            # Generate predictions
            predictions = self.predict(features)

            # Calculate metrics
            mse = np.mean((predictions - labels) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - labels))

            # R-squared
            ss_res = np.sum((labels - predictions) ** 2)
            ss_tot = np.sum((labels - np.mean(labels)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((labels - predictions) / (labels + 1e-10))) * 100

            metrics = {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2),
                "mape": float(mape)
            }

            logger.info("Model evaluation completed", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("Evaluation failed", error=str(e))
            raise ModelError(f"Evaluation failed: {e}") from e

    def _calculate_feature_importance(self) -> None:
        """Calculate feature importance scores."""
        try:
            if self.model is None:
                return

            importance = self.model.feature_importance(importance_type="gain")

            for i, feat_name in enumerate(self.feature_names):
                self.feature_importance[feat_name] = Decimal(str(importance[i]))

            logger.info(
                "Feature importance calculated",
                num_features=len(self.feature_importance)
            )

        except Exception as e:
            logger.warning("Failed to calculate feature importance", error=str(e))

    def get_feature_importance(
        self,
        importance_type: str = "gain",
        top_k: Optional[int] = None
    ) -> Dict[str, Decimal]:
        """Get feature importance scores.

        Args:
            importance_type: Type of importance ('gain', 'split')
            top_k: Return only top-k features (optional)

        Returns:
            Dictionary of feature importances

        Raises:
            ModelError: If model not trained
        """
        try:
            if self.model is None:
                raise ModelError("Model not trained")

            importance = self.model.feature_importance(importance_type=importance_type)

            result = {}
            for i, feat_name in enumerate(self.feature_names):
                result[feat_name] = Decimal(str(importance[i]))

            # Sort by importance
            sorted_features = sorted(
                result.items(),
                key=lambda x: x[1],
                reverse=True
            )

            if top_k is not None:
                sorted_features = sorted_features[:top_k]

            return dict(sorted_features)

        except Exception as e:
            logger.error("Failed to get feature importance", error=str(e))
            raise ModelError(f"Failed to get feature importance: {e}") from e

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save model

        Raises:
            ModelError: If save fails
        """
        try:
            if self.model is None:
                raise ModelError("Model not trained")

            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save model
            self.model.save_model(str(save_path))

            # Save metadata
            metadata_path = save_path.with_suffix(".json")
            metadata = {
                "feature_names": self.feature_names,
                "feature_importance": {
                    k: float(v) for k, v in self.feature_importance.items()
                },
                "best_iteration": self.best_iteration,
                "config": self.config,
                "trained": self._trained
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info("Model saved", path=path)

        except Exception as e:
            logger.error("Failed to save model", error=str(e))
            raise ModelError(f"Failed to save model: {e}") from e

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to load model from

        Raises:
            ModelError: If load fails
        """
        try:
            load_path = Path(path)

            if not load_path.exists():
                raise ModelError(f"Model file not found: {path}")

            # Load model
            self.model = lgb.Booster(model_file=str(load_path))

            # Load metadata
            metadata_path = load_path.with_suffix(".json")
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                self.feature_names = metadata.get("feature_names", [])
                self.feature_importance = {
                    k: Decimal(str(v))
                    for k, v in metadata.get("feature_importance", {}).items()
                }
                self.best_iteration = metadata.get("best_iteration", 0)
                self._trained = metadata.get("trained", False)

            logger.info("Model loaded", path=path)

        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise ModelError(f"Failed to load model: {e}") from e

    def _validate_training_data(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> None:
        """Validate training data.

        Args:
            features: Feature array
            labels: Label array

        Raises:
            ValidationError: If data is invalid
        """
        if features is None or len(features) == 0:
            raise ValidationError("Features cannot be empty")

        if labels is None or len(labels) == 0:
            raise ValidationError("Labels cannot be empty")

        if len(features) != len(labels):
            raise ValidationError(
                f"Feature and label length mismatch: {len(features)} != {len(labels)}"
            )

        if features.ndim != 2:
            raise ValidationError(f"Features must be 2D array, got {features.ndim}D")

        if labels.ndim != 1:
            raise ValidationError(f"Labels must be 1D array, got {labels.ndim}D")

        # Check for NaN/Inf
        if np.any(np.isnan(features)):
            raise ValidationError("Features contain NaN values")

        if np.any(np.isinf(features)):
            raise ValidationError("Features contain Inf values")

        if np.any(np.isnan(labels)):
            raise ValidationError("Labels contain NaN values")

        if np.any(np.isinf(labels)):
            raise ValidationError("Labels contain Inf values")

    async def train_async(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> None:
        """Train model asynchronously.

        Args:
            features: Training features
            labels: Training labels
            val_features: Validation features (optional)
            val_labels: Validation labels (optional)
            feature_names: Feature names (optional)
        """
        await asyncio.to_thread(
            self.train,
            features,
            labels,
            val_features,
            val_labels,
            feature_names
        )

    async def predict_async(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions asynchronously.

        Args:
            features: Input features

        Returns:
            Predictions
        """
        return await asyncio.to_thread(self.predict, features)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary of model information
        """
        return {
            "trained": self._trained,
            "objective": self.objective,
            "metric": self.metric,
            "num_features": len(self.feature_names),
            "best_iteration": self.best_iteration,
            "num_leaves": self.num_leaves,
            "learning_rate": float(self.learning_rate)
        }
