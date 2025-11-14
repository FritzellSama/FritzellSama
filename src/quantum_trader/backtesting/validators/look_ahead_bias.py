"""Look-Ahead Bias Detector for Backtesting.

Detects and prevents look-ahead bias in backtesting strategies where
future data is inadvertently used to make past trading decisions.
"""

import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class BiasType(Enum):
    """Types of look-ahead bias."""
    FUTURE_DATA_ACCESS = "FUTURE_DATA_ACCESS"
    DATA_SNOOPING = "DATA_SNOOPING"
    SURVIVORSHIP_BIAS = "SURVIVORSHIP_BIAS"
    TIMESTAMP_MISMATCH = "TIMESTAMP_MISMATCH"


@dataclass
class BiasDetection:
    """Represents detected look-ahead bias.

    Attributes:
        bias_type: Type of bias detected
        timestamp: When bias occurred
        description: Detailed description
        severity: How severe the bias is (0-1)
        affected_trades: List of trade IDs affected
        metadata: Additional context
    """
    bias_type: BiasType
    timestamp: datetime
    description: str
    severity: Decimal
    affected_trades: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LookAheadBiasDetector:
    """Detect look-ahead bias in backtesting.

    Monitors data access patterns and trading decisions to identify
    cases where future information is being used inappropriately.

    Attributes:
        config: Detector configuration from config files
        detections: List of detected bias instances
        strict_mode: Whether to raise exceptions on detection
        data_access_log: Log of data access timestamps
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize look-ahead bias detector.

        Args:
            config: Configuration dictionary with detection parameters

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = {
            ...     "strict_mode": True,
            ...     "max_timestamp_tolerance_seconds": 1,
            ...     "enable_access_logging": True
            ... }
            >>> detector = LookAheadBiasDetector(config)
        """
        self.config = config
        self._validate_config()

        self.strict_mode = config.get("strict_mode", False)
        self.enable_access_logging = config.get("enable_access_logging", True)
        self.max_timestamp_tolerance_seconds = config.get("max_timestamp_tolerance_seconds", 1)

        self.detections: List[BiasDetection] = []
        self.data_access_log: List[Dict[str, Any]] = []

        # Track current simulation time
        self.current_simulation_time: Optional[datetime] = None

        # Track accessed data timestamps
        self.accessed_data_timestamps: Set[datetime] = set()

        logger.info(
            "LookAheadBiasDetector initialized",
            strict_mode=self.strict_mode,
            access_logging=self.enable_access_logging
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("config must be a dictionary")

    def set_simulation_time(self, timestamp: datetime) -> None:
        """Set current simulation time for bias checking.

        Args:
            timestamp: Current simulation timestamp

        Example:
            >>> detector = LookAheadBiasDetector(config)
            >>> detector.set_simulation_time(datetime(2024, 1, 1, 12, 0))
        """
        if self.current_simulation_time is not None:
            if timestamp < self.current_simulation_time:
                logger.warning(
                    "Simulation time moved backwards",
                    previous=self.current_simulation_time.isoformat(),
                    current=timestamp.isoformat()
                )

        self.current_simulation_time = timestamp
        logger.debug("Simulation time updated", timestamp=timestamp.isoformat())

    async def check_data_access(
        self,
        data: pl.DataFrame,
        access_timestamp: datetime,
        context: Optional[str] = None
    ) -> bool:
        """Check data access for look-ahead bias.

        Args:
            data: DataFrame being accessed
            access_timestamp: Timestamp of the data access
            context: Optional context description

        Returns:
            True if access is valid, False if bias detected

        Raises:
            ValueError: If strict_mode and bias detected

        Example:
            >>> detector = LookAheadBiasDetector(config)
            >>> detector.set_simulation_time(datetime(2024, 1, 1, 12, 0))
            >>> is_valid = await detector.check_data_access(
            ...     market_data_df,
            ...     datetime(2024, 1, 1, 11, 59),
            ...     "Strategy signal generation"
            ... )
        """
        if self.current_simulation_time is None:
            logger.warning("Simulation time not set, cannot check for look-ahead bias")
            return True

        try:
            # Log data access if enabled
            if self.enable_access_logging:
                self._log_data_access(data, access_timestamp, context)

            # Check if data is from the future
            bias_detected = await self._detect_future_data_access(
                data,
                access_timestamp,
                context
            )

            if bias_detected and self.strict_mode:
                error_msg = "Look-ahead bias detected in strict mode"
                logger.error(error_msg, context=context)
                raise ValueError(error_msg)

            return not bias_detected

        except Exception as e:
            logger.error("Failed to check data access", error=str(e))
            if self.strict_mode:
                raise
            return False

    async def _detect_future_data_access(
        self,
        data: pl.DataFrame,
        access_timestamp: datetime,
        context: Optional[str]
    ) -> bool:
        """Detect if data from the future is being accessed.

        Args:
            data: DataFrame being accessed
            access_timestamp: When data is being accessed
            context: Context description

        Returns:
            True if future data access detected
        """
        if self.current_simulation_time is None:
            return False

        bias_detected = False

        try:
            # Check if DataFrame has timestamp column
            if "timestamp" in data.columns:
                data_timestamps = data["timestamp"].to_list()

                # Check each timestamp
                for ts in data_timestamps:
                    if ts > self.current_simulation_time:
                        # Allow small tolerance for timestamp precision
                        time_diff = (ts - self.current_simulation_time).total_seconds()

                        if time_diff > self.max_timestamp_tolerance_seconds:
                            self._record_bias_detection(
                                BiasType.FUTURE_DATA_ACCESS,
                                self.current_simulation_time,
                                f"Future data accessed: {time_diff:.2f}s ahead of simulation time",
                                Decimal("0.8"),
                                {
                                    "context": context,
                                    "data_timestamp": ts.isoformat(),
                                    "simulation_time": self.current_simulation_time.isoformat(),
                                    "time_diff_seconds": time_diff
                                }
                            )
                            bias_detected = True

            # Check access timestamp vs simulation time
            if access_timestamp > self.current_simulation_time:
                time_diff = (access_timestamp - self.current_simulation_time).total_seconds()

                if time_diff > self.max_timestamp_tolerance_seconds:
                    self._record_bias_detection(
                        BiasType.TIMESTAMP_MISMATCH,
                        self.current_simulation_time,
                        f"Data access timestamp in future: {time_diff:.2f}s ahead",
                        Decimal("0.6"),
                        {
                            "context": context,
                            "access_timestamp": access_timestamp.isoformat(),
                            "simulation_time": self.current_simulation_time.isoformat()
                        }
                    )
                    bias_detected = True

        except Exception as e:
            logger.error("Error detecting future data access", error=str(e))

        return bias_detected

    async def check_indicator_calculation(
        self,
        indicator_name: str,
        calculation_data: pl.DataFrame,
        result_timestamp: datetime
    ) -> bool:
        """Check technical indicator calculation for look-ahead bias.

        Args:
            indicator_name: Name of the indicator
            calculation_data: Data used in calculation
            result_timestamp: Timestamp of the indicator result

        Returns:
            True if calculation is valid

        Example:
            >>> detector = LookAheadBiasDetector(config)
            >>> detector.set_simulation_time(datetime(2024, 1, 1, 12, 0))
            >>> is_valid = await detector.check_indicator_calculation(
            ...     "RSI",
            ...     price_data_df,
            ...     datetime(2024, 1, 1, 12, 0)
            ... )
        """
        if self.current_simulation_time is None:
            return True

        try:
            # Check if calculation uses future data
            if "timestamp" in calculation_data.columns:
                max_data_timestamp = calculation_data["timestamp"].max()

                if max_data_timestamp > result_timestamp:
                    self._record_bias_detection(
                        BiasType.FUTURE_DATA_ACCESS,
                        result_timestamp,
                        f"Indicator '{indicator_name}' uses future data in calculation",
                        Decimal("0.9"),
                        {
                            "indicator": indicator_name,
                            "result_timestamp": result_timestamp.isoformat(),
                            "max_data_timestamp": max_data_timestamp
                        }
                    )
                    return False

            return True

        except Exception as e:
            logger.error("Failed to check indicator calculation", error=str(e), indicator=indicator_name)
            return False

    async def check_trade_decision(
        self,
        decision_timestamp: datetime,
        decision_data: pl.DataFrame,
        trade_id: Optional[str] = None
    ) -> bool:
        """Check trading decision for look-ahead bias.

        Args:
            decision_timestamp: When decision was made
            decision_data: Data used to make decision
            trade_id: Optional trade identifier

        Returns:
            True if decision is valid

        Example:
            >>> detector = LookAheadBiasDetector(config)
            >>> detector.set_simulation_time(datetime(2024, 1, 1, 12, 0))
            >>> is_valid = await detector.check_trade_decision(
            ...     datetime(2024, 1, 1, 12, 0),
            ...     signal_data_df,
            ...     "TRADE_001"
            ... )
        """
        if self.current_simulation_time is None:
            return True

        try:
            # Verify decision timestamp is not in future
            if decision_timestamp > self.current_simulation_time:
                time_diff = (decision_timestamp - self.current_simulation_time).total_seconds()

                if time_diff > self.max_timestamp_tolerance_seconds:
                    self._record_bias_detection(
                        BiasType.TIMESTAMP_MISMATCH,
                        decision_timestamp,
                        f"Trade decision made in future: {time_diff:.2f}s ahead",
                        Decimal("1.0"),
                        {
                            "trade_id": trade_id,
                            "decision_timestamp": decision_timestamp.isoformat(),
                            "simulation_time": self.current_simulation_time.isoformat()
                        }
                    )
                    return False

            # Check if decision data contains future information
            if "timestamp" in decision_data.columns:
                future_data_points = decision_data.filter(
                    pl.col("timestamp") > decision_timestamp
                )

                if future_data_points.height > 0:
                    self._record_bias_detection(
                        BiasType.FUTURE_DATA_ACCESS,
                        decision_timestamp,
                        f"Trade decision uses {future_data_points.height} future data points",
                        Decimal("1.0"),
                        {"trade_id": trade_id, "future_points": future_data_points.height}
                    )
                    return False

            return True

        except Exception as e:
            logger.error("Failed to check trade decision", error=str(e), trade_id=trade_id)
            return False

    def _log_data_access(
        self,
        data: pl.DataFrame,
        access_timestamp: datetime,
        context: Optional[str]
    ) -> None:
        """Log data access for audit trail.

        Args:
            data: DataFrame being accessed
            access_timestamp: When data was accessed
            context: Context description
        """
        access_entry = {
            "access_timestamp": access_timestamp,
            "simulation_time": self.current_simulation_time,
            "data_shape": (data.height, data.width),
            "context": context,
            "logged_at": datetime.utcnow()
        }

        self.data_access_log.append(access_entry)

        # Track unique timestamps
        if "timestamp" in data.columns:
            for ts in data["timestamp"].to_list():
                self.accessed_data_timestamps.add(ts)

    def _record_bias_detection(
        self,
        bias_type: BiasType,
        timestamp: datetime,
        description: str,
        severity: Decimal,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record detected bias instance.

        Args:
            bias_type: Type of bias
            timestamp: When bias occurred
            description: Description
            severity: Severity level (0-1)
            metadata: Additional context
        """
        detection = BiasDetection(
            bias_type=bias_type,
            timestamp=timestamp,
            description=description,
            severity=severity,
            metadata=metadata or {}
        )

        self.detections.append(detection)

        logger.warning(
            "Look-ahead bias detected",
            bias_type=bias_type.value,
            description=description,
            severity=float(severity)
        )

    def get_detections(self) -> List[BiasDetection]:
        """Get all detected bias instances.

        Returns:
            List of bias detections

        Example:
            >>> detector = LookAheadBiasDetector(config)
            >>> # ... run backtest ...
            >>> biases = detector.get_detections()
            >>> for bias in biases:
            ...     print(f"{bias.bias_type}: {bias.description}")
        """
        return self.detections.copy()

    def get_detections_by_type(self, bias_type: BiasType) -> List[BiasDetection]:
        """Get detections of specific type.

        Args:
            bias_type: Type of bias to filter

        Returns:
            List of matching detections
        """
        return [d for d in self.detections if d.bias_type == bias_type]

    def get_high_severity_detections(self, threshold: Decimal = Decimal("0.7")) -> List[BiasDetection]:
        """Get high severity bias detections.

        Args:
            threshold: Minimum severity level (0-1)

        Returns:
            List of high severity detections
        """
        return [d for d in self.detections if d.severity >= threshold]

    def has_critical_bias(self, threshold: Decimal = Decimal("0.8")) -> bool:
        """Check if critical bias was detected.

        Args:
            threshold: Critical severity threshold

        Returns:
            True if critical bias exists
        """
        return any(d.severity >= threshold for d in self.detections)

    def get_bias_summary(self) -> Dict[str, Any]:
        """Get summary of detected biases.

        Returns:
            Dictionary with bias statistics

        Example:
            >>> detector = LookAheadBiasDetector(config)
            >>> # ... run backtest ...
            >>> summary = detector.get_bias_summary()
            >>> print(f"Total biases: {summary['total_detections']}")
        """
        by_type = {}
        for bias_type in BiasType:
            by_type[bias_type.value] = len(self.get_detections_by_type(bias_type))

        severity_distribution = {
            "low": len([d for d in self.detections if d.severity < Decimal("0.4")]),
            "medium": len([d for d in self.detections if Decimal("0.4") <= d.severity < Decimal("0.7")]),
            "high": len([d for d in self.detections if Decimal("0.7") <= d.severity < Decimal("0.9")]),
            "critical": len([d for d in self.detections if d.severity >= Decimal("0.9")])
        }

        summary = {
            "total_detections": len(self.detections),
            "by_type": by_type,
            "severity_distribution": severity_distribution,
            "has_critical_bias": self.has_critical_bias(),
            "data_access_log_size": len(self.data_access_log)
        }

        return summary

    def reset(self) -> None:
        """Reset detector state.

        Example:
            >>> detector = LookAheadBiasDetector(config)
            >>> # ... run backtest ...
            >>> detector.reset()  # Prepare for next backtest
        """
        self.detections = []
        self.data_access_log = []
        self.accessed_data_timestamps = set()
        self.current_simulation_time = None

        logger.info("LookAheadBiasDetector reset")

    async def analyze_backtest_results(
        self,
        trades: pl.DataFrame,
        market_data: pl.DataFrame
    ) -> Dict[str, Any]:
        """Analyze complete backtest for look-ahead bias.

        Args:
            trades: DataFrame with executed trades
            market_data: DataFrame with market data used

        Returns:
            Analysis results dictionary

        Example:
            >>> detector = LookAheadBiasDetector(config)
            >>> results = await detector.analyze_backtest_results(trades_df, market_df)
            >>> if results["bias_detected"]:
            ...     print(f"Warning: {results['total_biases']} biases detected")
        """
        try:
            logger.info("Starting backtest bias analysis")

            # Validate each trade against available market data
            for trade_row in trades.iter_rows(named=True):
                trade_time = trade_row.get("timestamp")
                trade_price = Decimal(str(trade_row.get("price", 0)))

                # Get market data available at trade time
                available_data = market_data.filter(
                    pl.col("timestamp") <= trade_time
                )

                # Check if trade could have been made with available data
                future_data = market_data.filter(
                    pl.col("timestamp") > trade_time
                )

                if future_data.height == 0:
                    continue

                # Check if trade price suggests future knowledge
                # (This is a simplified check - real implementation would be more sophisticated)
                self.set_simulation_time(trade_time)
                await self.check_trade_decision(
                    trade_time,
                    available_data,
                    trade_row.get("trade_id")
                )

            # Generate summary
            summary = self.get_bias_summary()
            summary["bias_detected"] = len(self.detections) > 0
            summary["analysis_timestamp"] = datetime.utcnow().isoformat()

            logger.info(
                "Backtest bias analysis completed",
                total_biases=summary["total_detections"],
                critical_bias=summary["has_critical_bias"]
            )

            return summary

        except Exception as e:
            logger.error("Failed to analyze backtest results", error=str(e))
            raise
