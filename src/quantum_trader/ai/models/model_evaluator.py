"""
Model Evaluation Framework for Trading Models.

This module provides comprehensive model evaluation capabilities including
backtesting, walk-forward validation, and performance metrics calculation.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import polars as pl
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel
from quantum_trader.exceptions import EvaluationError, ValidationError

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Model evaluation metrics.

    Attributes:
        accuracy: Prediction accuracy
        precision: Precision score
        recall: Recall score
        f1_score: F1 score
        mse: Mean squared error
        rmse: Root mean squared error
        mae: Mean absolute error
        r2_score: R-squared score
        sharpe_ratio: Sharpe ratio (if applicable)
        max_drawdown: Maximum drawdown (if applicable)
        win_rate: Win rate (if applicable)
        profit_factor: Profit factor (if applicable)
        timestamp: Evaluation timestamp
        metadata: Additional metadata
    """
    accuracy: Optional[Decimal] = None
    precision: Optional[Decimal] = None
    recall: Optional[Decimal] = None
    f1_score: Optional[Decimal] = None
    mse: Optional[Decimal] = None
    rmse: Optional[Decimal] = None
    mae: Optional[Decimal] = None
    r2_score: Optional[Decimal] = None
    sharpe_ratio: Optional[Decimal] = None
    max_drawdown: Optional[Decimal] = None
    win_rate: Optional[Decimal] = None
    profit_factor: Optional[Decimal] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation.

    Attributes:
        test_size: Test set size ratio
        validation_type: Type of validation (holdout, cv, walk_forward)
        n_splits: Number of splits for cross-validation
        walk_forward_window: Window size for walk-forward validation
        metrics: List of metrics to calculate
        calculate_trading_metrics: Whether to calculate trading-specific metrics
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
    """
    test_size: Decimal = Decimal("0.2")
    validation_type: str = "holdout"
    n_splits: int = 5
    walk_forward_window: int = 100
    metrics: List[str] = None
    calculate_trading_metrics: bool = True
    risk_free_rate: Decimal = Decimal("0.02")

    def __post_init__(self) -> None:
        """Initialize default metrics."""
        if self.metrics is None:
            self.metrics = ["mse", "rmse", "mae", "r2_score"]


class ModelEvaluator:
    """Comprehensive model evaluation framework.

    Provides various evaluation strategies including holdout validation,
    cross-validation, walk-forward validation, and trading-specific metrics.

    Attributes:
        config: Evaluation configuration

    Example:
        >>> config = EvaluationConfig(
        ...     test_size=Decimal("0.2"),
        ...     validation_type="walk_forward",
        ...     walk_forward_window=100,
        ...     calculate_trading_metrics=True
        ... )
        >>> evaluator = ModelEvaluator(config)
        >>> metrics = evaluator.evaluate(model, features, labels)
    """

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize model evaluator.

        Args:
            config: Evaluation configuration

        Raises:
            ValidationError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        logger.info(
            "Model evaluator initialized",
            validation_type=config.validation_type,
            test_size=float(config.test_size)
        )

    def _validate_config(self) -> None:
        """Validate evaluator configuration.

        Raises:
            ValidationError: If configuration is invalid
        """
        if self.config.test_size <= 0 or self.config.test_size >= 1:
            raise ValidationError("test_size must be between 0 and 1")

        valid_types = ["holdout", "cv", "walk_forward"]
        if self.config.validation_type not in valid_types:
            raise ValidationError(f"Invalid validation_type: {self.config.validation_type}")

        if self.config.n_splits < 2:
            raise ValidationError("n_splits must be >= 2")

        if self.config.walk_forward_window < 1:
            raise ValidationError("walk_forward_window must be >= 1")

    def evaluate(
        self,
        model: BaseMLModel,
        features: np.ndarray,
        labels: np.ndarray,
        predictions: Optional[np.ndarray] = None
    ) -> EvaluationMetrics:
        """Evaluate model performance.

        Args:
            model: Model to evaluate
            features: Feature array
            labels: Label array
            predictions: Pre-computed predictions (optional)

        Returns:
            Evaluation metrics

        Raises:
            EvaluationError: If evaluation fails
        """
        try:
            logger.info(
                "Starting model evaluation",
                validation_type=self.config.validation_type,
                num_samples=len(features)
            )

            # Generate predictions if not provided
            if predictions is None:
                predictions = model.predict(features)

            # Calculate metrics
            metrics = EvaluationMetrics()

            # Regression metrics
            if "mse" in self.config.metrics:
                metrics.mse = self._calculate_mse(labels, predictions)

            if "rmse" in self.config.metrics:
                metrics.rmse = self._calculate_rmse(labels, predictions)

            if "mae" in self.config.metrics:
                metrics.mae = self._calculate_mae(labels, predictions)

            if "r2_score" in self.config.metrics:
                metrics.r2_score = self._calculate_r2(labels, predictions)

            # Classification metrics (if labels are binary/categorical)
            if self._is_classification(labels):
                if "accuracy" in self.config.metrics:
                    metrics.accuracy = self._calculate_accuracy(labels, predictions)

                if "precision" in self.config.metrics:
                    metrics.precision = self._calculate_precision(labels, predictions)

                if "recall" in self.config.metrics:
                    metrics.recall = self._calculate_recall(labels, predictions)

                if "f1_score" in self.config.metrics:
                    metrics.f1_score = self._calculate_f1(labels, predictions)

            # Trading-specific metrics
            if self.config.calculate_trading_metrics:
                trading_metrics = self._calculate_trading_metrics(labels, predictions)
                metrics.sharpe_ratio = trading_metrics.get("sharpe_ratio")
                metrics.max_drawdown = trading_metrics.get("max_drawdown")
                metrics.win_rate = trading_metrics.get("win_rate")
                metrics.profit_factor = trading_metrics.get("profit_factor")

            metrics.metadata = {
                "num_samples": len(labels),
                "validation_type": self.config.validation_type
            }

            logger.info(
                "Model evaluation completed",
                mse=float(metrics.mse) if metrics.mse else None,
                rmse=float(metrics.rmse) if metrics.rmse else None,
                mae=float(metrics.mae) if metrics.mae else None
            )

            return metrics

        except Exception as e:
            logger.error("Evaluation failed", error=str(e))
            raise EvaluationError(f"Evaluation failed: {e}") from e

    def walk_forward_validation(
        self,
        model: BaseMLModel,
        features: np.ndarray,
        labels: np.ndarray,
        retrain: bool = True
    ) -> List[EvaluationMetrics]:
        """Perform walk-forward validation.

        Args:
            model: Model to evaluate
            features: Feature array
            labels: Label array
            retrain: Whether to retrain model at each step

        Returns:
            List of evaluation metrics for each window

        Raises:
            EvaluationError: If validation fails
        """
        try:
            logger.info(
                "Starting walk-forward validation",
                window_size=self.config.walk_forward_window,
                num_samples=len(features)
            )

            metrics_list = []
            window_size = self.config.walk_forward_window

            for i in range(window_size, len(features), window_size // 2):
                # Training window
                train_features = features[max(0, i - window_size):i]
                train_labels = labels[max(0, i - window_size):i]

                # Test window
                test_end = min(i + window_size // 2, len(features))
                test_features = features[i:test_end]
                test_labels = labels[i:test_end]

                if len(test_features) == 0:
                    break

                # Retrain if needed
                if retrain:
                    model.train(train_features, train_labels)

                # Evaluate
                predictions = model.predict(test_features)
                metrics = self.evaluate(model, test_features, test_labels, predictions)
                metrics.metadata["window_start"] = i
                metrics.metadata["window_end"] = test_end

                metrics_list.append(metrics)

            logger.info(
                "Walk-forward validation completed",
                num_windows=len(metrics_list)
            )

            return metrics_list

        except Exception as e:
            logger.error("Walk-forward validation failed", error=str(e))
            raise EvaluationError(f"Walk-forward validation failed: {e}") from e

    def cross_validate(
        self,
        model: BaseMLModel,
        features: np.ndarray,
        labels: np.ndarray
    ) -> List[EvaluationMetrics]:
        """Perform cross-validation.

        Args:
            model: Model to evaluate
            features: Feature array
            labels: Label array

        Returns:
            List of evaluation metrics for each fold

        Raises:
            EvaluationError: If validation fails
        """
        try:
            logger.info(
                "Starting cross-validation",
                n_splits=self.config.n_splits,
                num_samples=len(features)
            )

            metrics_list = []
            fold_size = len(features) // self.config.n_splits

            for fold in range(self.config.n_splits):
                # Split data
                test_start = fold * fold_size
                test_end = (fold + 1) * fold_size if fold < self.config.n_splits - 1 else len(features)

                test_features = features[test_start:test_end]
                test_labels = labels[test_start:test_end]

                train_features = np.concatenate([
                    features[:test_start],
                    features[test_end:]
                ])
                train_labels = np.concatenate([
                    labels[:test_start],
                    labels[test_end:]
                ])

                # Train and evaluate
                model.train(train_features, train_labels)
                predictions = model.predict(test_features)
                metrics = self.evaluate(model, test_features, test_labels, predictions)
                metrics.metadata["fold"] = fold

                metrics_list.append(metrics)

            logger.info(
                "Cross-validation completed",
                num_folds=len(metrics_list)
            )

            return metrics_list

        except Exception as e:
            logger.error("Cross-validation failed", error=str(e))
            raise EvaluationError(f"Cross-validation failed: {e}") from e

    def _calculate_mse(
        self,
        labels: np.ndarray,
        predictions: np.ndarray
    ) -> Decimal:
        """Calculate mean squared error."""
        mse = np.mean((labels - predictions) ** 2)
        return Decimal(str(mse))

    def _calculate_rmse(
        self,
        labels: np.ndarray,
        predictions: np.ndarray
    ) -> Decimal:
        """Calculate root mean squared error."""
        rmse = np.sqrt(np.mean((labels - predictions) ** 2))
        return Decimal(str(rmse))

    def _calculate_mae(
        self,
        labels: np.ndarray,
        predictions: np.ndarray
    ) -> Decimal:
        """Calculate mean absolute error."""
        mae = np.mean(np.abs(labels - predictions))
        return Decimal(str(mae))

    def _calculate_r2(
        self,
        labels: np.ndarray,
        predictions: np.ndarray
    ) -> Decimal:
        """Calculate R-squared score."""
        ss_res = np.sum((labels - predictions) ** 2)
        ss_tot = np.sum((labels - np.mean(labels)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        return Decimal(str(r2))

    def _calculate_accuracy(
        self,
        labels: np.ndarray,
        predictions: np.ndarray
    ) -> Decimal:
        """Calculate accuracy."""
        predictions_binary = (predictions > 0.5).astype(int)
        labels_binary = (labels > 0.5).astype(int)
        accuracy = np.mean(predictions_binary == labels_binary)
        return Decimal(str(accuracy))

    def _calculate_precision(
        self,
        labels: np.ndarray,
        predictions: np.ndarray
    ) -> Decimal:
        """Calculate precision."""
        predictions_binary = (predictions > 0.5).astype(int)
        labels_binary = (labels > 0.5).astype(int)

        tp = np.sum((predictions_binary == 1) & (labels_binary == 1))
        fp = np.sum((predictions_binary == 1) & (labels_binary == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        return Decimal(str(precision))

    def _calculate_recall(
        self,
        labels: np.ndarray,
        predictions: np.ndarray
    ) -> Decimal:
        """Calculate recall."""
        predictions_binary = (predictions > 0.5).astype(int)
        labels_binary = (labels > 0.5).astype(int)

        tp = np.sum((predictions_binary == 1) & (labels_binary == 1))
        fn = np.sum((predictions_binary == 0) & (labels_binary == 1))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return Decimal(str(recall))

    def _calculate_f1(
        self,
        labels: np.ndarray,
        predictions: np.ndarray
    ) -> Decimal:
        """Calculate F1 score."""
        precision = float(self._calculate_precision(labels, predictions))
        recall = float(self._calculate_recall(labels, predictions))

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return Decimal(str(f1))

    def _calculate_trading_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray
    ) -> Dict[str, Decimal]:
        """Calculate trading-specific metrics."""
        metrics = {}

        # Treat predictions as returns
        returns = predictions.flatten()

        # Sharpe ratio
        if len(returns) > 0:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                sharpe = (mean_return - float(self.config.risk_free_rate) / 252) / std_return * np.sqrt(252)
                metrics["sharpe_ratio"] = Decimal(str(sharpe))

        # Maximum drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        metrics["max_drawdown"] = Decimal(str(max_drawdown))

        # Win rate
        wins = np.sum(returns > 0)
        total_trades = len(returns)
        win_rate = wins / total_trades if total_trades > 0 else 0
        metrics["win_rate"] = Decimal(str(win_rate))

        # Profit factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        metrics["profit_factor"] = Decimal(str(profit_factor))

        return metrics

    def _is_classification(self, labels: np.ndarray) -> bool:
        """Check if problem is classification."""
        unique_values = np.unique(labels)
        return len(unique_values) <= 10 and np.all(labels == labels.astype(int))

    async def evaluate_async(
        self,
        model: BaseMLModel,
        features: np.ndarray,
        labels: np.ndarray,
        predictions: Optional[np.ndarray] = None
    ) -> EvaluationMetrics:
        """Evaluate model asynchronously."""
        return await asyncio.to_thread(
            self.evaluate,
            model,
            features,
            labels,
            predictions
        )

    def aggregate_metrics(
        self,
        metrics_list: List[EvaluationMetrics]
    ) -> EvaluationMetrics:
        """Aggregate multiple evaluation metrics.

        Args:
            metrics_list: List of evaluation metrics

        Returns:
            Aggregated metrics

        Raises:
            ValidationError: If metrics list is empty
        """
        try:
            if not metrics_list:
                raise ValidationError("Metrics list cannot be empty")

            aggregated = EvaluationMetrics()

            # Aggregate each metric
            for metric_name in ["mse", "rmse", "mae", "r2_score", "accuracy",
                                "precision", "recall", "f1_score", "sharpe_ratio",
                                "max_drawdown", "win_rate", "profit_factor"]:
                values = [
                    getattr(m, metric_name)
                    for m in metrics_list
                    if getattr(m, metric_name) is not None
                ]

                if values:
                    mean_value = sum(values) / Decimal(str(len(values)))
                    setattr(aggregated, metric_name, mean_value)

            aggregated.metadata = {
                "num_evaluations": len(metrics_list),
                "aggregation_type": "mean"
            }

            logger.info(
                "Metrics aggregated",
                num_evaluations=len(metrics_list)
            )

            return aggregated

        except Exception as e:
            logger.error("Metric aggregation failed", error=str(e))
            raise EvaluationError(f"Metric aggregation failed: {e}") from e
