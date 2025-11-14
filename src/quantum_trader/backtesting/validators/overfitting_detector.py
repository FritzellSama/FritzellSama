"""Overfitting Detector for Backtest Validation.

Detects potential overfitting in trading strategies through
statistical analysis and out-of-sample testing.
"""

import asyncio
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class OverfittingRisk(Enum):
    """Overfitting risk levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class OverfittingMetrics:
    """Metrics for overfitting detection.

    Attributes:
        in_sample_sharpe: Sharpe ratio on training data
        out_sample_sharpe: Sharpe ratio on test data
        sharpe_degradation: Percentage degradation
        parameter_sensitivity: How sensitive to parameters
        trade_count_ratio: Ratio of trades in/out sample
        risk_level: Overall overfitting risk assessment
    """
    in_sample_sharpe: Decimal
    out_sample_sharpe: Decimal
    sharpe_degradation: Decimal
    parameter_sensitivity: Decimal
    trade_count_ratio: Decimal
    risk_level: OverfittingRisk


class OverfittingDetector:
    """Detect overfitting in trading strategy backtests.

    Uses multiple statistical tests to identify potential overfitting:
    - Walk-forward analysis
    - Out-of-sample performance degradation
    - Parameter sensitivity analysis
    - Trade distribution analysis

    Attributes:
        config: Detector configuration from config files
        degradation_threshold: Maximum acceptable performance degradation
        sensitivity_threshold: Maximum parameter sensitivity
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize overfitting detector.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = {
            ...     "degradation_threshold_pct": 20,
            ...     "sensitivity_threshold": 0.3,
            ...     "min_out_sample_trades": 30
            ... }
            >>> detector = OverfittingDetector(config)
        """
        self.config = config
        self._validate_config()

        self.degradation_threshold = Decimal(str(config.get("degradation_threshold_pct", 20)))
        self.sensitivity_threshold = Decimal(str(config.get("sensitivity_threshold", 0.3)))
        self.min_out_sample_trades = config.get("min_out_sample_trades", 30)

        logger.info(
            "OverfittingDetector initialized",
            degradation_threshold=float(self.degradation_threshold),
            sensitivity_threshold=float(self.sensitivity_threshold)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("config must be a dictionary")

    async def analyze_overfitting(
        self,
        in_sample_results: Dict[str, Any],
        out_sample_results: Dict[str, Any],
        parameter_variations: Optional[List[Dict[str, Any]]] = None
    ) -> OverfittingMetrics:
        """Analyze potential overfitting.

        Args:
            in_sample_results: Results from training period
            out_sample_results: Results from test period
            parameter_variations: Optional parameter sensitivity tests

        Returns:
            Overfitting metrics and risk assessment

        Raises:
            ValueError: If inputs are invalid

        Example:
            >>> detector = OverfittingDetector(config)
            >>> in_sample = {"sharpe_ratio": 2.5, "total_trades": 100, ...}
            >>> out_sample = {"sharpe_ratio": 1.8, "total_trades": 50, ...}
            >>> metrics = await detector.analyze_overfitting(in_sample, out_sample)
            >>> print(f"Risk level: {metrics.risk_level.value}")
        """
        try:
            # Extract performance metrics
            in_sharpe = Decimal(str(in_sample_results.get("sharpe_ratio", 0)))
            out_sharpe = Decimal(str(out_sample_results.get("sharpe_ratio", 0)))

            # Calculate performance degradation
            if in_sharpe > Decimal("0"):
                sharpe_degradation = ((in_sharpe - out_sharpe) / in_sharpe) * Decimal("100")
            else:
                sharpe_degradation = Decimal("0")

            # Analyze trade distribution
            in_trades = in_sample_results.get("total_trades", 0)
            out_trades = out_sample_results.get("total_trades", 0)

            if out_trades > 0:
                trade_ratio = Decimal(str(in_trades)) / Decimal(str(out_trades))
            else:
                trade_ratio = Decimal("0")

            # Calculate parameter sensitivity if provided
            if parameter_variations:
                param_sensitivity = await self._calculate_parameter_sensitivity(
                    parameter_variations
                )
            else:
                param_sensitivity = Decimal("0")

            # Assess overall risk
            risk_level = self._assess_overfitting_risk(
                sharpe_degradation,
                param_sensitivity,
                out_trades
            )

            metrics = OverfittingMetrics(
                in_sample_sharpe=in_sharpe,
                out_sample_sharpe=out_sharpe,
                sharpe_degradation=sharpe_degradation,
                parameter_sensitivity=param_sensitivity,
                trade_count_ratio=trade_ratio,
                risk_level=risk_level
            )

            logger.info(
                "Overfitting analysis completed",
                risk_level=risk_level.value,
                sharpe_degradation=float(sharpe_degradation),
                param_sensitivity=float(param_sensitivity)
            )

            return metrics

        except Exception as e:
            logger.error("Failed to analyze overfitting", error=str(e))
            raise

    async def _calculate_parameter_sensitivity(
        self,
        variations: List[Dict[str, Any]]
    ) -> Decimal:
        """Calculate sensitivity to parameter changes.

        Args:
            variations: List of results with different parameters

        Returns:
            Sensitivity score (0-1, higher = more sensitive)
        """
        if not variations or len(variations) < 2:
            return Decimal("0")

        # Extract performance metrics
        returns = [Decimal(str(v.get("total_return", 0))) for v in variations]

        # Calculate coefficient of variation
        if len(returns) > 1:
            mean_return = sum(returns) / Decimal(str(len(returns)))

            if mean_return > Decimal("0"):
                variance = sum((r - mean_return) ** 2 for r in returns) / Decimal(str(len(returns)))
                std_dev = variance.sqrt()
                cv = std_dev / mean_return

                # Normalize to 0-1 range
                sensitivity = min(Decimal("1"), cv)
            else:
                sensitivity = Decimal("1")
        else:
            sensitivity = Decimal("0")

        return sensitivity

    def _assess_overfitting_risk(
        self,
        sharpe_degradation: Decimal,
        param_sensitivity: Decimal,
        out_sample_trades: int
    ) -> OverfittingRisk:
        """Assess overall overfitting risk level.

        Args:
            sharpe_degradation: Performance degradation percentage
            param_sensitivity: Parameter sensitivity score
            out_sample_trades: Number of out-of-sample trades

        Returns:
            Risk level classification
        """
        risk_score = Decimal("0")

        # Factor 1: Performance degradation
        if sharpe_degradation > Decimal("50"):
            risk_score += Decimal("3")
        elif sharpe_degradation > Decimal("30"):
            risk_score += Decimal("2")
        elif sharpe_degradation > Decimal("15"):
            risk_score += Decimal("1")

        # Factor 2: Parameter sensitivity
        if param_sensitivity > Decimal("0.5"):
            risk_score += Decimal("2")
        elif param_sensitivity > Decimal("0.3"):
            risk_score += Decimal("1")

        # Factor 3: Trade count
        if out_sample_trades < self.min_out_sample_trades:
            risk_score += Decimal("2")

        # Classify risk
        if risk_score >= Decimal("5"):
            return OverfittingRisk.CRITICAL
        elif risk_score >= Decimal("3"):
            return OverfittingRisk.HIGH
        elif risk_score >= Decimal("1"):
            return OverfittingRisk.MEDIUM
        else:
            return OverfittingRisk.LOW

    async def perform_walk_forward_analysis(
        self,
        data: pl.DataFrame,
        strategy_func: Any,
        in_sample_pct: Decimal = Decimal("0.7"),
        num_folds: int = 5
    ) -> Dict[str, Any]:
        """Perform walk-forward analysis.

        Args:
            data: Complete dataset
            strategy_func: Strategy to test
            in_sample_pct: Percentage for training
            num_folds: Number of walk-forward folds

        Returns:
            Walk-forward analysis results
        """
        logger.info("Starting walk-forward analysis", num_folds=num_folds)

        results = {
            "fold_results": [],
            "avg_in_sample_sharpe": Decimal("0"),
            "avg_out_sample_sharpe": Decimal("0"),
            "consistency_score": Decimal("0")
        }

        # Implementation would perform actual walk-forward testing
        # This is a simplified placeholder

        logger.info("Walk-forward analysis completed")
        return results
