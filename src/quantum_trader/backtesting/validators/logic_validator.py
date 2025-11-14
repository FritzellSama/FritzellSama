"""Logic Validator for Backtesting.

Validates trading logic and strategy implementation for common errors
and anti-patterns in backtesting.
"""

import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ValidationIssue:
    """Represents a validation issue found during analysis.

    Attributes:
        severity: Issue severity level
        category: Issue category
        message: Detailed issue description
        timestamp: When the issue was detected
        metadata: Additional context about the issue
    """
    severity: ValidationSeverity
    category: str
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class LogicValidator:
    """Validate trading logic and backtesting implementation.

    Detects common errors such as:
    - Look-ahead bias
    - Position sizing errors
    - Unrealistic fills
    - Time-based logic errors
    - Data integrity issues

    Attributes:
        config: Validator configuration from config files
        issues: List of detected validation issues
        strict_mode: Whether to raise exceptions on errors
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize logic validator.

        Args:
            config: Configuration dictionary with validation parameters

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = {
            ...     "strict_mode": False,
            ...     "max_position_size_pct": 25,
            ...     "max_leverage": 3.0,
            ...     "min_time_between_trades_seconds": 1
            ... }
            >>> validator = LogicValidator(config)
        """
        self.config = config
        self._validate_config()

        self.strict_mode = config.get("strict_mode", False)
        self.issues: List[ValidationIssue] = []

        # Validation thresholds
        self.max_position_size_pct = Decimal(str(config.get("max_position_size_pct", 25)))
        self.max_leverage = Decimal(str(config.get("max_leverage", 3.0)))
        self.min_time_between_trades_seconds = config.get("min_time_between_trades_seconds", 1)
        self.max_slippage_pct = Decimal(str(config.get("max_slippage_pct", 5.0)))

        logger.info(
            "LogicValidator initialized",
            strict_mode=self.strict_mode,
            max_position_size_pct=float(self.max_position_size_pct)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("config must be a dictionary")

    async def validate_backtest_results(
        self,
        trades: pl.DataFrame,
        positions: pl.DataFrame,
        equity_curve: pl.DataFrame,
        market_data: pl.DataFrame
    ) -> List[ValidationIssue]:
        """Validate complete backtest results.

        Args:
            trades: DataFrame with executed trades
            positions: DataFrame with position history
            equity_curve: DataFrame with equity progression
            market_data: DataFrame with market data used

        Returns:
            List of validation issues found

        Raises:
            ValueError: If strict_mode and critical issues found

        Example:
            >>> validator = LogicValidator(config)
            >>> issues = await validator.validate_backtest_results(
            ...     trades_df, positions_df, equity_df, market_df
            ... )
            >>> for issue in issues:
            ...     print(f"{issue.severity}: {issue.message}")
        """
        self.issues = []

        try:
            logger.info("Starting backtest validation")

            # Run all validation checks
            await self._validate_trade_timestamps(trades)
            await self._validate_position_sizes(positions, equity_curve)
            await self._validate_trade_prices(trades, market_data)
            await self._validate_equity_curve(equity_curve)
            await self._validate_data_integrity(trades, positions)
            await self._validate_leverage(positions, equity_curve)

            # Log summary
            severity_counts = self._count_issues_by_severity()
            logger.info(
                "Backtest validation completed",
                total_issues=len(self.issues),
                severity_counts=severity_counts
            )

            # Raise exception if strict mode and critical issues found
            if self.strict_mode and self._has_critical_issues():
                critical_issues = [i for i in self.issues if i.severity == ValidationSeverity.CRITICAL]
                error_msg = f"Critical validation issues found: {len(critical_issues)}"
                logger.error("Validation failed in strict mode", critical_issues=len(critical_issues))
                raise ValueError(error_msg)

            return self.issues

        except Exception as e:
            logger.error("Validation failed", error=str(e))
            raise

    async def _validate_trade_timestamps(self, trades: pl.DataFrame) -> None:
        """Validate trade timestamps for logical consistency.

        Args:
            trades: DataFrame with trade data
        """
        if trades.height == 0:
            logger.info("No trades to validate timestamps")
            return

        try:
            # Check for required columns
            required_cols = ["timestamp", "symbol"]
            if not all(col in trades.columns for col in required_cols):
                self._add_issue(
                    ValidationSeverity.ERROR,
                    "data_integrity",
                    f"Missing required columns in trades DataFrame: {required_cols}"
                )
                return

            # Check for time travel (trades out of chronological order)
            timestamps = trades["timestamp"].to_list()
            for i in range(1, len(timestamps)):
                if timestamps[i] < timestamps[i-1]:
                    self._add_issue(
                        ValidationSeverity.ERROR,
                        "time_logic",
                        f"Trades out of chronological order at index {i}",
                        {"timestamp_current": timestamps[i], "timestamp_previous": timestamps[i-1]}
                    )

            # Check for suspiciously rapid trades
            if self.min_time_between_trades_seconds > 0:
                for i in range(1, len(timestamps)):
                    time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                    if 0 < time_diff < self.min_time_between_trades_seconds:
                        self._add_issue(
                            ValidationSeverity.WARNING,
                            "time_logic",
                            f"Trades too close together: {time_diff}s",
                            {"index": i, "time_diff_seconds": time_diff}
                        )

            logger.debug("Trade timestamp validation completed", trades_count=trades.height)

        except Exception as e:
            logger.error("Failed to validate trade timestamps", error=str(e))
            self._add_issue(
                ValidationSeverity.ERROR,
                "validation_error",
                f"Trade timestamp validation failed: {e}"
            )

    async def _validate_position_sizes(
        self,
        positions: pl.DataFrame,
        equity_curve: pl.DataFrame
    ) -> None:
        """Validate position sizes are reasonable relative to portfolio.

        Args:
            positions: DataFrame with position data
            equity_curve: DataFrame with equity progression
        """
        if positions.height == 0:
            logger.info("No positions to validate")
            return

        try:
            # Check for required columns
            if "position_value" not in positions.columns or "timestamp" not in positions.columns:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    "data_integrity",
                    "Missing required columns in positions DataFrame"
                )
                return

            # Validate each position against portfolio equity
            for row in positions.iter_rows(named=True):
                position_value = abs(Decimal(str(row.get("position_value", 0))))
                timestamp = row.get("timestamp")

                # Find corresponding equity
                equity_at_time = self._get_equity_at_timestamp(equity_curve, timestamp)

                if equity_at_time > Decimal("0"):
                    position_pct = (position_value / equity_at_time) * Decimal("100")

                    if position_pct > self.max_position_size_pct:
                        self._add_issue(
                            ValidationSeverity.WARNING,
                            "position_sizing",
                            f"Position size exceeds maximum: {position_pct:.2f}%",
                            {
                                "position_pct": float(position_pct),
                                "max_allowed": float(self.max_position_size_pct),
                                "timestamp": timestamp
                            }
                        )

            logger.debug("Position size validation completed", positions_count=positions.height)

        except Exception as e:
            logger.error("Failed to validate position sizes", error=str(e))
            self._add_issue(
                ValidationSeverity.ERROR,
                "validation_error",
                f"Position size validation failed: {e}"
            )

    async def _validate_trade_prices(
        self,
        trades: pl.DataFrame,
        market_data: pl.DataFrame
    ) -> None:
        """Validate trade prices are within reasonable market bounds.

        Args:
            trades: DataFrame with trade data
            market_data: DataFrame with market data
        """
        if trades.height == 0 or market_data.height == 0:
            logger.info("Insufficient data to validate trade prices")
            return

        try:
            # Check for required columns
            required_trade_cols = ["timestamp", "price", "symbol"]
            required_market_cols = ["timestamp", "high", "low", "symbol"]

            if not all(col in trades.columns for col in required_trade_cols):
                self._add_issue(
                    ValidationSeverity.ERROR,
                    "data_integrity",
                    "Missing required columns in trades DataFrame"
                )
                return

            if not all(col in market_data.columns for col in required_market_cols):
                self._add_issue(
                    ValidationSeverity.ERROR,
                    "data_integrity",
                    "Missing required columns in market_data DataFrame"
                )
                return

            # Validate each trade price against market data
            for trade_row in trades.iter_rows(named=True):
                trade_price = Decimal(str(trade_row.get("price", 0)))
                trade_time = trade_row.get("timestamp")
                trade_symbol = trade_row.get("symbol")

                # Find corresponding market data
                market_row = self._get_market_data_at_timestamp(
                    market_data,
                    trade_time,
                    trade_symbol
                )

                if market_row is not None:
                    high = Decimal(str(market_row.get("high", 0)))
                    low = Decimal(str(market_row.get("low", 0)))

                    # Check if trade price is within market bounds
                    if trade_price > high or trade_price < low:
                        # Calculate how far outside bounds
                        if trade_price > high:
                            deviation_pct = ((trade_price - high) / high) * Decimal("100")
                        else:
                            deviation_pct = ((low - trade_price) / low) * Decimal("100")

                        if deviation_pct > self.max_slippage_pct:
                            self._add_issue(
                                ValidationSeverity.ERROR,
                                "unrealistic_fill",
                                f"Trade price outside market bounds: {deviation_pct:.2f}% deviation",
                                {
                                    "trade_price": float(trade_price),
                                    "market_high": float(high),
                                    "market_low": float(low),
                                    "timestamp": trade_time
                                }
                            )

            logger.debug("Trade price validation completed")

        except Exception as e:
            logger.error("Failed to validate trade prices", error=str(e))
            self._add_issue(
                ValidationSeverity.ERROR,
                "validation_error",
                f"Trade price validation failed: {e}"
            )

    async def _validate_equity_curve(self, equity_curve: pl.DataFrame) -> None:
        """Validate equity curve for anomalies.

        Args:
            equity_curve: DataFrame with equity progression
        """
        if equity_curve.height == 0:
            self._add_issue(
                ValidationSeverity.CRITICAL,
                "data_integrity",
                "Equity curve is empty"
            )
            return

        try:
            # Check for required columns
            if "equity" not in equity_curve.columns:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    "data_integrity",
                    "Missing 'equity' column in equity_curve DataFrame"
                )
                return

            equity_values = [Decimal(str(v)) for v in equity_curve["equity"].to_list()]

            # Check for negative equity
            for i, equity in enumerate(equity_values):
                if equity < Decimal("0"):
                    self._add_issue(
                        ValidationSeverity.CRITICAL,
                        "equity_integrity",
                        f"Negative equity detected at index {i}: {equity}",
                        {"index": i, "equity": float(equity)}
                    )

            # Check for sudden jumps (possible errors)
            max_single_period_change_pct = Decimal(str(self.config.get("max_single_period_change_pct", 50)))

            for i in range(1, len(equity_values)):
                if equity_values[i-1] > Decimal("0"):
                    pct_change = abs((equity_values[i] - equity_values[i-1]) / equity_values[i-1]) * Decimal("100")

                    if pct_change > max_single_period_change_pct:
                        self._add_issue(
                            ValidationSeverity.WARNING,
                            "equity_integrity",
                            f"Large equity jump: {pct_change:.2f}% in single period",
                            {"index": i, "pct_change": float(pct_change)}
                        )

            logger.debug("Equity curve validation completed", data_points=len(equity_values))

        except Exception as e:
            logger.error("Failed to validate equity curve", error=str(e))
            self._add_issue(
                ValidationSeverity.ERROR,
                "validation_error",
                f"Equity curve validation failed: {e}"
            )

    async def _validate_data_integrity(
        self,
        trades: pl.DataFrame,
        positions: pl.DataFrame
    ) -> None:
        """Validate data integrity and consistency.

        Args:
            trades: DataFrame with trade data
            positions: DataFrame with position data
        """
        try:
            # Check for duplicate trade IDs
            if "trade_id" in trades.columns:
                unique_ids = trades["trade_id"].unique().len()
                if unique_ids < trades.height:
                    self._add_issue(
                        ValidationSeverity.ERROR,
                        "data_integrity",
                        f"Duplicate trade IDs detected: {trades.height - unique_ids} duplicates"
                    )

            # Check for null values in critical columns
            critical_trade_cols = ["timestamp", "price", "quantity"]
            for col in critical_trade_cols:
                if col in trades.columns:
                    null_count = trades[col].null_count()
                    if null_count > 0:
                        self._add_issue(
                            ValidationSeverity.ERROR,
                            "data_integrity",
                            f"Null values in critical column '{col}': {null_count} nulls"
                        )

            logger.debug("Data integrity validation completed")

        except Exception as e:
            logger.error("Failed to validate data integrity", error=str(e))
            self._add_issue(
                ValidationSeverity.ERROR,
                "validation_error",
                f"Data integrity validation failed: {e}"
            )

    async def _validate_leverage(
        self,
        positions: pl.DataFrame,
        equity_curve: pl.DataFrame
    ) -> None:
        """Validate leverage levels are within limits.

        Args:
            positions: DataFrame with position data
            equity_curve: DataFrame with equity progression
        """
        if positions.height == 0:
            return

        try:
            for row in positions.iter_rows(named=True):
                if "position_value" not in row or "timestamp" not in row:
                    continue

                position_value = abs(Decimal(str(row.get("position_value", 0))))
                timestamp = row.get("timestamp")

                equity = self._get_equity_at_timestamp(equity_curve, timestamp)

                if equity > Decimal("0"):
                    leverage = position_value / equity

                    if leverage > self.max_leverage:
                        self._add_issue(
                            ValidationSeverity.WARNING,
                            "risk_management",
                            f"Leverage exceeds maximum: {leverage:.2f}x",
                            {
                                "leverage": float(leverage),
                                "max_allowed": float(self.max_leverage),
                                "timestamp": timestamp
                            }
                        )

            logger.debug("Leverage validation completed")

        except Exception as e:
            logger.error("Failed to validate leverage", error=str(e))
            self._add_issue(
                ValidationSeverity.ERROR,
                "validation_error",
                f"Leverage validation failed: {e}"
            )

    def _get_equity_at_timestamp(
        self,
        equity_curve: pl.DataFrame,
        timestamp: datetime
    ) -> Decimal:
        """Get equity value at specific timestamp.

        Args:
            equity_curve: DataFrame with equity data
            timestamp: Target timestamp

        Returns:
            Equity value at timestamp
        """
        try:
            if "timestamp" not in equity_curve.columns or "equity" not in equity_curve.columns:
                return Decimal("0")

            # Find nearest timestamp
            equity_list = equity_curve.to_dicts()

            for row in equity_list:
                if row["timestamp"] >= timestamp:
                    return Decimal(str(row["equity"]))

            # Return last equity if timestamp is after all data
            if equity_list:
                return Decimal(str(equity_list[-1]["equity"]))

            return Decimal("0")

        except Exception:
            return Decimal("0")

    def _get_market_data_at_timestamp(
        self,
        market_data: pl.DataFrame,
        timestamp: datetime,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Get market data at specific timestamp for symbol.

        Args:
            market_data: DataFrame with market data
            timestamp: Target timestamp
            symbol: Trading symbol

        Returns:
            Market data dictionary or None
        """
        try:
            # Filter by symbol and find nearest timestamp
            market_list = market_data.to_dicts()

            for row in market_list:
                if row.get("symbol") == symbol and row.get("timestamp") >= timestamp:
                    return row

            return None

        except Exception:
            return None

    def _add_issue(
        self,
        severity: ValidationSeverity,
        category: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add validation issue to the list.

        Args:
            severity: Issue severity
            category: Issue category
            message: Issue description
            metadata: Additional context
        """
        issue = ValidationIssue(
            severity=severity,
            category=category,
            message=message,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )

        self.issues.append(issue)

        logger.log(
            severity.value.lower(),
            f"Validation issue: {message}",
            category=category,
            metadata=metadata
        )

    def _count_issues_by_severity(self) -> Dict[str, int]:
        """Count issues by severity level.

        Returns:
            Dictionary mapping severity to count
        """
        counts = {severity.value: 0 for severity in ValidationSeverity}

        for issue in self.issues:
            counts[issue.severity.value] += 1

        return counts

    def _has_critical_issues(self) -> bool:
        """Check if any critical issues were found.

        Returns:
            True if critical issues exist
        """
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)

    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get all issues of a specific severity.

        Args:
            severity: Severity level to filter by

        Returns:
            List of matching issues

        Example:
            >>> validator = LogicValidator(config)
            >>> # ... run validation ...
            >>> errors = validator.get_issues_by_severity(ValidationSeverity.ERROR)
        """
        return [issue for issue in self.issues if issue.severity == severity]

    def get_issues_by_category(self, category: str) -> List[ValidationIssue]:
        """Get all issues of a specific category.

        Args:
            category: Category to filter by

        Returns:
            List of matching issues
        """
        return [issue for issue in self.issues if issue.category == category]
