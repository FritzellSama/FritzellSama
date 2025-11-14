"""Comprehensive model evaluation system for trading models.

This module provides advanced evaluation metrics and backtesting capabilities
for assessing model performance in realistic trading scenarios.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel

logger = get_logger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation.

    Attributes:
        backtesting_enabled: Whether to run backtesting
        calculate_sharpe: Whether to calculate Sharpe ratio
        calculate_sortino: Whether to calculate Sortino ratio
        risk_free_rate: Risk-free rate for Sharpe/Sortino
        confidence_level: Confidence level for VaR
        bootstrap_samples: Number of bootstrap samples
        cross_validation_folds: Number of CV folds
        save_predictions: Whether to save predictions
        output_dir: Directory for saving results
    """

    backtesting_enabled: bool
    calculate_sharpe: bool
    calculate_sortino: bool
    risk_free_rate: Decimal
    confidence_level: Decimal
    bootstrap_samples: int
    cross_validation_folds: int
    save_predictions: bool
    output_dir: str


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics.

    Attributes:
        accuracy: Classification accuracy
        precision: Precision score
        recall: Recall score
        f1: F1 score
        auc_roc: Area under ROC curve
        mse: Mean squared error
        mae: Mean absolute error
        rmse: Root mean squared error
        r2: R-squared score
        sharpe_ratio: Sharpe ratio
        sortino_ratio: Sortino ratio
        max_drawdown: Maximum drawdown
        win_rate: Win rate for trades
        profit_factor: Profit factor
        var: Value at Risk
        cvar: Conditional Value at Risk
        calmar_ratio: Calmar ratio
        information_ratio: Information ratio
        timestamp: Evaluation timestamp
    """

    accuracy: Optional[Decimal] = None
    precision: Optional[Decimal] = None
    recall: Optional[Decimal] = None
    f1: Optional[Decimal] = None
    auc_roc: Optional[Decimal] = None
    mse: Optional[Decimal] = None
    mae: Optional[Decimal] = None
    rmse: Optional[Decimal] = None
    r2: Optional[Decimal] = None
    sharpe_ratio: Optional[Decimal] = None
    sortino_ratio: Optional[Decimal] = None
    max_drawdown: Optional[Decimal] = None
    win_rate: Optional[Decimal] = None
    profit_factor: Optional[Decimal] = None
    var: Optional[Decimal] = None
    cvar: Optional[Decimal] = None
    calmar_ratio: Optional[Decimal] = None
    information_ratio: Optional[Decimal] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ModelEvaluator:
    """Comprehensive model evaluation system.

    Provides extensive evaluation capabilities including:
    - Standard ML metrics (accuracy, precision, recall, etc.)
    - Trading-specific metrics (Sharpe, Sortino, drawdown)
    - Risk metrics (VaR, CVaR)
    - Cross-validation and bootstrap analysis
    - Prediction stability analysis

    Example:
        >>> config = EvaluationConfig(
        ...     backtesting_enabled=True,
        ...     calculate_sharpe=True,
        ...     calculate_sortino=True,
        ...     risk_free_rate=Decimal("0.02"),
        ...     confidence_level=Decimal("0.95"),
        ...     bootstrap_samples=1000,
        ...     cross_validation_folds=5,
        ...     save_predictions=True,
        ...     output_dir="./evaluation_results"
        ... )
        >>> evaluator = ModelEvaluator(config)
        >>> metrics = await evaluator.evaluate_model(
        ...     model=my_model,
        ...     test_features=X_test,
        ...     test_labels=y_test,
        ...     returns=returns_data
        ... )
    """

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize model evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config

        # Results storage
        self.evaluation_history: List[PerformanceMetrics] = []
        self.predictions_cache: Dict[str, np.ndarray] = {}

        logger.info("model_evaluator_initialized")

    async def evaluate_model(
        self,
        model: BaseMLModel,
        test_features: np.ndarray,
        test_labels: np.ndarray,
        returns: Optional[np.ndarray] = None,
        task_type: str = "regression"
    ) -> PerformanceMetrics:
        """Comprehensively evaluate model performance.

        Args:
            model: Model to evaluate
            test_features: Test features
            test_labels: Test labels
            returns: Optional returns data for financial metrics
            task_type: 'regression' or 'classification'

        Returns:
            Performance metrics
        """
        try:
            logger.info(
                "evaluation_started",
                task_type=task_type,
                n_samples=len(test_features)
            )

            # Generate predictions
            predictions = model.predict(test_features)

            # Calculate metrics based on task type
            if task_type == "classification":
                metrics = await self._evaluate_classification(
                    test_labels,
                    predictions
                )
            else:
                metrics = await self._evaluate_regression(
                    test_labels,
                    predictions
                )

            # Calculate financial metrics if returns provided
            if returns is not None:
                financial_metrics = await self._calculate_financial_metrics(
                    predictions,
                    returns
                )

                # Merge metrics
                for key, value in financial_metrics.items():
                    setattr(metrics, key, value)

            # Save predictions if enabled
            if self.config.save_predictions:
                await self._save_predictions(predictions, test_labels)

            # Store in history
            self.evaluation_history.append(metrics)

            logger.info(
                "evaluation_completed",
                task_type=task_type
            )

            return metrics

        except Exception as e:
            logger.error("evaluation_failed", error=str(e))
            raise

    async def _evaluate_classification(
        self,
        labels: np.ndarray,
        predictions: np.ndarray
    ) -> PerformanceMetrics:
        """Evaluate classification model.

        Args:
            labels: True labels
            predictions: Predicted labels/probabilities

        Returns:
            Classification metrics
        """
        try:
            # Convert probabilities to binary predictions if needed
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                pred_classes = np.argmax(predictions, axis=1)
                pred_probs = predictions
            else:
                pred_classes = (predictions > 0.5).astype(int).flatten()
                pred_probs = predictions.flatten()

            labels = labels.flatten().astype(int)

            # Calculate metrics
            accuracy = Decimal(str(accuracy_score(labels, pred_classes)))
            precision = Decimal(str(precision_score(
                labels,
                pred_classes,
                average='weighted',
                zero_division=0
            )))
            recall = Decimal(str(recall_score(
                labels,
                pred_classes,
                average='weighted',
                zero_division=0
            )))
            f1 = Decimal(str(f1_score(
                labels,
                pred_classes,
                average='weighted',
                zero_division=0
            )))

            # AUC-ROC (if binary classification)
            auc_roc = None
            if len(np.unique(labels)) == 2:
                try:
                    auc_roc = Decimal(str(roc_auc_score(labels, pred_probs)))
                except Exception as e:
                    logger.warning("auc_calculation_failed", error=str(e))

            return PerformanceMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                auc_roc=auc_roc
            )

        except Exception as e:
            logger.error("classification_evaluation_failed", error=str(e))
            raise

    async def _evaluate_regression(
        self,
        labels: np.ndarray,
        predictions: np.ndarray
    ) -> PerformanceMetrics:
        """Evaluate regression model.

        Args:
            labels: True values
            predictions: Predicted values

        Returns:
            Regression metrics
        """
        try:
            labels = labels.flatten()
            predictions = predictions.flatten()

            # Calculate metrics
            mse = Decimal(str(mean_squared_error(labels, predictions)))
            mae = Decimal(str(mean_absolute_error(labels, predictions)))
            rmse = Decimal(str(np.sqrt(float(mse))))
            r2 = Decimal(str(r2_score(labels, predictions)))

            return PerformanceMetrics(
                mse=mse,
                mae=mae,
                rmse=rmse,
                r2=r2
            )

        except Exception as e:
            logger.error("regression_evaluation_failed", error=str(e))
            raise

    async def _calculate_financial_metrics(
        self,
        predictions: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, Decimal]:
        """Calculate trading-specific financial metrics.

        Args:
            predictions: Model predictions (position sizing)
            returns: Actual returns

        Returns:
            Dictionary of financial metrics
        """
        try:
            predictions = predictions.flatten()
            returns = returns.flatten()

            # Align lengths
            min_length = min(len(predictions), len(returns))
            predictions = predictions[:min_length]
            returns = returns[:min_length]

            # Calculate strategy returns
            strategy_returns = predictions * returns

            # Sharpe ratio
            sharpe_ratio = None
            if self.config.calculate_sharpe:
                sharpe_ratio = self._calculate_sharpe_ratio(
                    strategy_returns,
                    self.config.risk_free_rate
                )

            # Sortino ratio
            sortino_ratio = None
            if self.config.calculate_sortino:
                sortino_ratio = self._calculate_sortino_ratio(
                    strategy_returns,
                    self.config.risk_free_rate
                )

            # Maximum drawdown
            max_drawdown = self._calculate_max_drawdown(strategy_returns)

            # Win rate
            win_rate = self._calculate_win_rate(strategy_returns)

            # Profit factor
            profit_factor = self._calculate_profit_factor(strategy_returns)

            # VaR and CVaR
            var = self._calculate_var(
                strategy_returns,
                self.config.confidence_level
            )

            cvar = self._calculate_cvar(
                strategy_returns,
                self.config.confidence_level
            )

            # Calmar ratio
            calmar_ratio = None
            if max_drawdown != Decimal("0"):
                annual_return = Decimal(str(np.mean(strategy_returns))) * Decimal("252")
                calmar_ratio = annual_return / abs(max_drawdown)

            # Information ratio (vs zero benchmark)
            information_ratio = self._calculate_information_ratio(strategy_returns)

            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'var': var,
                'cvar': cvar,
                'calmar_ratio': calmar_ratio,
                'information_ratio': information_ratio
            }

        except Exception as e:
            logger.error("financial_metrics_failed", error=str(e))
            return {}

    def _calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: Decimal
    ) -> Decimal:
        """Calculate Sharpe ratio.

        Args:
            returns: Strategy returns
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Sharpe ratio
        """
        try:
            mean_return = Decimal(str(np.mean(returns)))
            std_return = Decimal(str(np.std(returns)))

            if std_return == Decimal("0"):
                return Decimal("0")

            # Annualize (assuming daily returns)
            annual_return = mean_return * Decimal("252")
            annual_std = std_return * Decimal(str(np.sqrt(252)))

            sharpe = (annual_return - risk_free_rate) / annual_std

            return sharpe

        except Exception as e:
            logger.error("sharpe_calculation_failed", error=str(e))
            return Decimal("0")

    def _calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: Decimal
    ) -> Decimal:
        """Calculate Sortino ratio.

        Args:
            returns: Strategy returns
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Sortino ratio
        """
        try:
            mean_return = Decimal(str(np.mean(returns)))
            downside_returns = returns[returns < 0]

            if len(downside_returns) == 0:
                return Decimal("0")

            downside_std = Decimal(str(np.std(downside_returns)))

            if downside_std == Decimal("0"):
                return Decimal("0")

            # Annualize
            annual_return = mean_return * Decimal("252")
            annual_downside_std = downside_std * Decimal(str(np.sqrt(252)))

            sortino = (annual_return - risk_free_rate) / annual_downside_std

            return sortino

        except Exception as e:
            logger.error("sortino_calculation_failed", error=str(e))
            return Decimal("0")

    def _calculate_max_drawdown(self, returns: np.ndarray) -> Decimal:
        """Calculate maximum drawdown.

        Args:
            returns: Strategy returns

        Returns:
            Maximum drawdown (negative value)
        """
        try:
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = cumulative - running_max

            max_dd = Decimal(str(np.min(drawdowns)))

            return max_dd

        except Exception as e:
            logger.error("max_drawdown_failed", error=str(e))
            return Decimal("0")

    def _calculate_win_rate(self, returns: np.ndarray) -> Decimal:
        """Calculate win rate.

        Args:
            returns: Strategy returns

        Returns:
            Win rate (0 to 1)
        """
        try:
            wins = np.sum(returns > 0)
            total = len(returns)

            if total == 0:
                return Decimal("0")

            return Decimal(str(wins / total))

        except Exception as e:
            logger.error("win_rate_failed", error=str(e))
            return Decimal("0")

    def _calculate_profit_factor(self, returns: np.ndarray) -> Decimal:
        """Calculate profit factor.

        Args:
            returns: Strategy returns

        Returns:
            Profit factor
        """
        try:
            profits = np.sum(returns[returns > 0])
            losses = abs(np.sum(returns[returns < 0]))

            if losses == 0:
                return Decimal("0") if profits == 0 else Decimal("999")

            return Decimal(str(profits / losses))

        except Exception as e:
            logger.error("profit_factor_failed", error=str(e))
            return Decimal("0")

    def _calculate_var(
        self,
        returns: np.ndarray,
        confidence_level: Decimal
    ) -> Decimal:
        """Calculate Value at Risk.

        Args:
            returns: Strategy returns
            confidence_level: Confidence level (e.g., 0.95)

        Returns:
            VaR (negative value representing loss)
        """
        try:
            percentile = (1 - float(confidence_level)) * 100
            var = Decimal(str(np.percentile(returns, percentile)))

            return var

        except Exception as e:
            logger.error("var_calculation_failed", error=str(e))
            return Decimal("0")

    def _calculate_cvar(
        self,
        returns: np.ndarray,
        confidence_level: Decimal
    ) -> Decimal:
        """Calculate Conditional Value at Risk (Expected Shortfall).

        Args:
            returns: Strategy returns
            confidence_level: Confidence level (e.g., 0.95)

        Returns:
            CVaR (negative value representing expected loss)
        """
        try:
            var = float(self._calculate_var(returns, confidence_level))
            tail_losses = returns[returns <= var]

            if len(tail_losses) == 0:
                return Decimal(str(var))

            cvar = Decimal(str(np.mean(tail_losses)))

            return cvar

        except Exception as e:
            logger.error("cvar_calculation_failed", error=str(e))
            return Decimal("0")

    def _calculate_information_ratio(self, returns: np.ndarray) -> Decimal:
        """Calculate information ratio.

        Args:
            returns: Strategy returns (excess over benchmark)

        Returns:
            Information ratio
        """
        try:
            mean_return = Decimal(str(np.mean(returns)))
            std_return = Decimal(str(np.std(returns)))

            if std_return == Decimal("0"):
                return Decimal("0")

            # Annualize
            annual_return = mean_return * Decimal("252")
            annual_std = std_return * Decimal(str(np.sqrt(252)))

            ir = annual_return / annual_std

            return ir

        except Exception as e:
            logger.error("information_ratio_failed", error=str(e))
            return Decimal("0")

    async def cross_validate(
        self,
        model: BaseMLModel,
        features: np.ndarray,
        labels: np.ndarray,
        n_folds: Optional[int] = None
    ) -> Dict[str, List[Decimal]]:
        """Perform cross-validation.

        Args:
            model: Model to evaluate
            features: Full feature set
            labels: Full label set
            n_folds: Number of folds (uses config default if None)

        Returns:
            Dictionary of metric lists across folds
        """
        try:
            n_folds = n_folds or self.config.cross_validation_folds

            fold_size = len(features) // n_folds
            metrics_by_fold: Dict[str, List[Decimal]] = {
                'mse': [],
                'mae': [],
                'r2': []
            }

            for fold in range(n_folds):
                # Create train/val split
                val_start = fold * fold_size
                val_end = val_start + fold_size

                val_features = features[val_start:val_end]
                val_labels = labels[val_start:val_end]

                train_features = np.concatenate([
                    features[:val_start],
                    features[val_end:]
                ])
                train_labels = np.concatenate([
                    labels[:val_start],
                    labels[val_end:]
                ])

                # Train model
                model.train(train_features, train_labels)

                # Evaluate
                val_metrics = await self._evaluate_regression(val_labels, model.predict(val_features))

                metrics_by_fold['mse'].append(val_metrics.mse)
                metrics_by_fold['mae'].append(val_metrics.mae)
                metrics_by_fold['r2'].append(val_metrics.r2)

                logger.info(
                    "fold_completed",
                    fold=fold,
                    mse=str(val_metrics.mse)
                )

            return metrics_by_fold

        except Exception as e:
            logger.error("cross_validation_failed", error=str(e))
            return {}

    async def bootstrap_evaluation(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        n_samples: Optional[int] = None
    ) -> Dict[str, Tuple[Decimal, Decimal]]:
        """Perform bootstrap evaluation.

        Args:
            predictions: Model predictions
            labels: True labels
            n_samples: Number of bootstrap samples (uses config default if None)

        Returns:
            Dictionary mapping metrics to (mean, std) tuples
        """
        try:
            n_samples = n_samples or self.config.bootstrap_samples

            mse_samples = []
            mae_samples = []

            for _ in range(n_samples):
                # Bootstrap sample
                indices = np.random.choice(
                    len(predictions),
                    len(predictions),
                    replace=True
                )

                boot_pred = predictions[indices]
                boot_labels = labels[indices]

                # Calculate metrics
                mse = mean_squared_error(boot_labels, boot_pred)
                mae = mean_absolute_error(boot_labels, boot_pred)

                mse_samples.append(mse)
                mae_samples.append(mae)

            # Calculate statistics
            results = {
                'mse': (
                    Decimal(str(np.mean(mse_samples))),
                    Decimal(str(np.std(mse_samples)))
                ),
                'mae': (
                    Decimal(str(np.mean(mae_samples))),
                    Decimal(str(np.std(mae_samples)))
                )
            }

            logger.info("bootstrap_completed", n_samples=n_samples)

            return results

        except Exception as e:
            logger.error("bootstrap_failed", error=str(e))
            return {}

    async def _save_predictions(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> None:
        """Save predictions to disk.

        Args:
            predictions: Model predictions
            labels: True labels
        """
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"predictions_{timestamp}.parquet"

            # Create DataFrame
            df = pl.DataFrame({
                'prediction': predictions.flatten(),
                'label': labels.flatten(),
                'error': (predictions - labels).flatten()
            })

            await asyncio.to_thread(df.write_parquet, output_path)

            logger.info("predictions_saved", path=str(output_path))

        except Exception as e:
            logger.error("predictions_save_failed", error=str(e))

    def get_metrics_dataframe(self) -> pl.DataFrame:
        """Get evaluation history as Polars DataFrame.

        Returns:
            DataFrame with evaluation metrics
        """
        try:
            if not self.evaluation_history:
                return pl.DataFrame()

            data = {
                'timestamp': [m.timestamp for m in self.evaluation_history],
                'mse': [float(m.mse) if m.mse else None for m in self.evaluation_history],
                'mae': [float(m.mae) if m.mae else None for m in self.evaluation_history],
                'r2': [float(m.r2) if m.r2 else None for m in self.evaluation_history],
                'sharpe_ratio': [float(m.sharpe_ratio) if m.sharpe_ratio else None for m in self.evaluation_history],
                'max_drawdown': [float(m.max_drawdown) if m.max_drawdown else None for m in self.evaluation_history]
            }

            return pl.DataFrame(data)

        except Exception as e:
            logger.error("metrics_dataframe_creation_failed", error=str(e))
            return pl.DataFrame()

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of all evaluations.

        Returns:
            Summary statistics dictionary
        """
        try:
            if not self.evaluation_history:
                return {}

            metrics_df = self.get_metrics_dataframe()

            summary = {}

            for col in metrics_df.columns:
                if col != 'timestamp':
                    values = metrics_df.select(col).to_numpy().flatten()
                    values = values[~np.isnan(values)]

                    if len(values) > 0:
                        summary[col] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values)),
                            'median': float(np.median(values))
                        }

            return summary

        except Exception as e:
            logger.error("summary_statistics_failed", error=str(e))
            return {}
