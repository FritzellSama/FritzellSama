"""Survivorship Bias Detector for Backtesting.

Detects and corrects for survivorship bias in backtesting data where
delisted or failed assets are excluded from historical datasets.
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


class BiasLevel(Enum):
    """Survivorship bias risk levels."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class SurvivorshipAnalysis:
    """Results of survivorship bias analysis.

    Attributes:
        bias_level: Overall bias risk level
        active_symbols: Number of symbols in dataset
        expected_symbols: Expected number if no bias
        missing_symbols_pct: Percentage of missing symbols
        delisting_events: Number of detected delistings
        data_completeness: Data completeness score (0-1)
        recommendations: List of recommendations
    """
    bias_level: BiasLevel
    active_symbols: int
    expected_symbols: int
    missing_symbols_pct: Decimal
    delisting_events: int
    data_completeness: Decimal
    recommendations: List[str] = field(default_factory=list)


class SurvivorshipBiasDetector:
    """Detect survivorship bias in backtesting datasets.

    Analyzes historical data for signs of survivorship bias:
    - Missing delisted symbols
    - Incomplete symbol histories
    - Abnormal symbol turnover
    - Data gaps and discontinuities

    Attributes:
        config: Detector configuration from config files
        expected_delisting_rate: Expected annual delisting rate
        min_data_coverage: Minimum required data coverage
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize survivorship bias detector.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = {
            ...     "expected_delisting_rate_pct": 5.0,
            ...     "min_data_coverage_pct": 90.0,
            ...     "lookback_years": 5
            ... }
            >>> detector = SurvivorshipBiasDetector(config)
        """
        self.config = config
        self._validate_config()

        self.expected_delisting_rate = Decimal(str(config.get("expected_delisting_rate_pct", 5.0)))
        self.min_data_coverage = Decimal(str(config.get("min_data_coverage_pct", 90.0)))
        self.lookback_years = config.get("lookback_years", 5)

        logger.info(
            "SurvivorshipBiasDetector initialized",
            expected_delisting_rate=float(self.expected_delisting_rate),
            min_coverage=float(self.min_data_coverage)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("config must be a dictionary")

    async def analyze_dataset(
        self,
        market_data: pl.DataFrame,
        start_date: datetime,
        end_date: datetime,
        expected_universe_size: Optional[int] = None
    ) -> SurvivorshipAnalysis:
        """Analyze dataset for survivorship bias.

        Args:
            market_data: Historical market data DataFrame
            start_date: Analysis start date
            end_date: Analysis end date
            expected_universe_size: Expected number of symbols (optional)

        Returns:
            Survivorship bias analysis results

        Raises:
            ValueError: If inputs are invalid

        Example:
            >>> detector = SurvivorshipBiasDetector(config)
            >>> analysis = await detector.analyze_dataset(
            ...     market_data_df,
            ...     datetime(2020, 1, 1),
            ...     datetime(2024, 1, 1),
            ...     expected_universe_size=500
            ... )
            >>> print(f"Bias level: {analysis.bias_level.value}")
        """
        try:
            self._validate_market_data(market_data)

            logger.info(
                "Analyzing dataset for survivorship bias",
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat()
            )

            # Count active symbols
            active_symbols = await self._count_active_symbols(market_data, end_date)

            # Detect delisting events
            delisting_events = await self._detect_delistings(market_data, start_date, end_date)

            # Calculate data completeness
            data_completeness = await self._calculate_data_completeness(
                market_data,
                start_date,
                end_date
            )

            # Estimate expected symbols if not provided
            if expected_universe_size is None:
                expected_symbols = await self._estimate_expected_universe(
                    active_symbols,
                    delisting_events,
                    start_date,
                    end_date
                )
            else:
                expected_symbols = expected_universe_size

            # Calculate missing symbols
            if expected_symbols > 0:
                missing_pct = ((expected_symbols - active_symbols) / expected_symbols) * Decimal("100")
            else:
                missing_pct = Decimal("0")

            # Assess bias level
            bias_level = self._assess_bias_level(
                missing_pct,
                delisting_events,
                data_completeness,
                active_symbols
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(
                bias_level,
                missing_pct,
                data_completeness
            )

            analysis = SurvivorshipAnalysis(
                bias_level=bias_level,
                active_symbols=active_symbols,
                expected_symbols=expected_symbols,
                missing_symbols_pct=missing_pct,
                delisting_events=delisting_events,
                data_completeness=data_completeness,
                recommendations=recommendations
            )

            logger.info(
                "Survivorship bias analysis completed",
                bias_level=bias_level.value,
                missing_pct=float(missing_pct)
            )

            return analysis

        except Exception as e:
            logger.error("Failed to analyze survivorship bias", error=str(e))
            raise

    def _validate_market_data(self, market_data: pl.DataFrame) -> None:
        """Validate market data DataFrame.

        Args:
            market_data: Market data to validate

        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(market_data, pl.DataFrame):
            raise ValueError("market_data must be a Polars DataFrame")

        required_cols = ["timestamp", "symbol"]
        missing_cols = [col for col in required_cols if col not in market_data.columns]

        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            logger.error("Invalid market data", missing_cols=missing_cols)
            raise ValueError(error_msg)

        if market_data.height == 0:
            raise ValueError("market_data cannot be empty")

    async def _count_active_symbols(
        self,
        market_data: pl.DataFrame,
        end_date: datetime
    ) -> int:
        """Count number of active symbols at end date.

        Args:
            market_data: Market data DataFrame
            end_date: End date for counting

        Returns:
            Number of active symbols
        """
        # Filter data up to end date
        active_data = market_data.filter(pl.col("timestamp") <= end_date)

        # Count unique symbols
        if active_data.height > 0:
            unique_symbols = active_data["symbol"].unique()
            return unique_symbols.len()

        return 0

    async def _detect_delistings(
        self,
        market_data: pl.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Detect delisting events in the data.

        Args:
            market_data: Market data DataFrame
            start_date: Analysis start date
            end_date: Analysis end date

        Returns:
            Number of detected delistings
        """
        # Get all symbols
        all_symbols = market_data["symbol"].unique().to_list()

        delisting_count = 0

        # Check each symbol for early termination
        for symbol in all_symbols:
            symbol_data = market_data.filter(pl.col("symbol") == symbol)

            if symbol_data.height > 0:
                last_timestamp = symbol_data["timestamp"].max()

                # If symbol data ends significantly before end_date, might be delisted
                days_before_end = (end_date - last_timestamp).days

                # Consider it a delisting if data ends > 30 days before end date
                delisting_threshold_days = self.config.get("delisting_threshold_days", 30)

                if days_before_end > delisting_threshold_days:
                    delisting_count += 1
                    logger.debug(
                        "Potential delisting detected",
                        symbol=symbol,
                        last_date=last_timestamp.isoformat(),
                        days_before_end=days_before_end
                    )

        return delisting_count

    async def _calculate_data_completeness(
        self,
        market_data: pl.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> Decimal:
        """Calculate data completeness score.

        Args:
            market_data: Market data DataFrame
            start_date: Start date
            end_date: End date

        Returns:
            Completeness score (0-1)
        """
        # Calculate expected data points
        total_days = (end_date - start_date).days

        if total_days <= 0:
            return Decimal("1")

        # Get symbols
        symbols = market_data["symbol"].unique().to_list()

        if not symbols:
            return Decimal("0")

        # Calculate average data coverage per symbol
        total_coverage = Decimal("0")

        for symbol in symbols:
            symbol_data = market_data.filter(pl.col("symbol") == symbol)
            symbol_days = symbol_data["timestamp"].unique().len()

            symbol_coverage = Decimal(str(symbol_days)) / Decimal(str(total_days))
            total_coverage += min(Decimal("1"), symbol_coverage)

        avg_coverage = total_coverage / Decimal(str(len(symbols)))

        return avg_coverage

    async def _estimate_expected_universe(
        self,
        active_symbols: int,
        delisting_events: int,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Estimate expected universe size accounting for delistings.

        Args:
            active_symbols: Current active symbols
            delisting_events: Detected delistings
            start_date: Start date
            end_date: End date

        Returns:
            Estimated expected universe size
        """
        years = (end_date - start_date).days / 365.25

        if years <= 0:
            return active_symbols

        # Expected delistings based on rate
        expected_delistings = int(
            float(active_symbols) *
            float(self.expected_delisting_rate / Decimal("100")) *
            years
        )

        # Estimate original universe
        estimated_universe = active_symbols + max(expected_delistings - delisting_events, 0)

        return estimated_universe

    def _assess_bias_level(
        self,
        missing_pct: Decimal,
        delisting_events: int,
        data_completeness: Decimal,
        active_symbols: int
    ) -> BiasLevel:
        """Assess overall survivorship bias level.

        Args:
            missing_pct: Percentage of missing symbols
            delisting_events: Number of delistings
            data_completeness: Data completeness score
            active_symbols: Number of active symbols

        Returns:
            Bias level classification
        """
        risk_score = Decimal("0")

        # Factor 1: Missing symbols
        if missing_pct > Decimal("30"):
            risk_score += Decimal("3")
        elif missing_pct > Decimal("15"):
            risk_score += Decimal("2")
        elif missing_pct > Decimal("5"):
            risk_score += Decimal("1")

        # Factor 2: Delisting detection
        if active_symbols > 0:
            delisting_rate = Decimal(str(delisting_events)) / Decimal(str(active_symbols)) * Decimal("100")

            if delisting_rate < Decimal("1"):
                risk_score += Decimal("2")  # Too few delistings = suspicious
            elif delisting_rate < Decimal("3"):
                risk_score += Decimal("1")

        # Factor 3: Data completeness
        if data_completeness < Decimal("0.7"):
            risk_score += Decimal("2")
        elif data_completeness < Decimal("0.9"):
            risk_score += Decimal("1")

        # Classify risk
        if risk_score >= Decimal("5"):
            return BiasLevel.CRITICAL
        elif risk_score >= Decimal("4"):
            return BiasLevel.HIGH
        elif risk_score >= Decimal("2"):
            return BiasLevel.MEDIUM
        elif risk_score >= Decimal("1"):
            return BiasLevel.LOW
        else:
            return BiasLevel.NONE

    def _generate_recommendations(
        self,
        bias_level: BiasLevel,
        missing_pct: Decimal,
        data_completeness: Decimal
    ) -> List[str]:
        """Generate recommendations based on analysis.

        Args:
            bias_level: Detected bias level
            missing_pct: Missing symbols percentage
            data_completeness: Data completeness score

        Returns:
            List of recommendations
        """
        recommendations = []

        if bias_level in [BiasLevel.HIGH, BiasLevel.CRITICAL]:
            recommendations.append(
                "CRITICAL: Obtain historical data including delisted symbols"
            )
            recommendations.append(
                "Consider adjusting backtest results for survivorship bias"
            )

        if missing_pct > Decimal("10"):
            recommendations.append(
                f"Dataset missing {missing_pct:.1f}% of expected symbols"
            )

        if data_completeness < Decimal("0.9"):
            recommendations.append(
                f"Data completeness is {data_completeness*100:.1f}% - fill data gaps"
            )

        if bias_level == BiasLevel.NONE:
            recommendations.append(
                "No significant survivorship bias detected"
            )

        return recommendations

    async def adjust_results_for_bias(
        self,
        original_return: Decimal,
        bias_level: BiasLevel,
        missing_pct: Decimal
    ) -> Decimal:
        """Adjust backtest returns for survivorship bias.

        Args:
            original_return: Original backtest return
            bias_level: Detected bias level
            missing_pct: Percentage of missing symbols

        Returns:
            Adjusted return estimate

        Example:
            >>> detector = SurvivorshipBiasDetector(config)
            >>> adjusted = await detector.adjust_results_for_bias(
            ...     Decimal("0.25"),
            ...     BiasLevel.MEDIUM,
            ...     Decimal("15")
            ... )
        """
        # Simplified adjustment - real implementation would be more sophisticated
        adjustment_factor = Decimal("1")

        if bias_level == BiasLevel.LOW:
            adjustment_factor = Decimal("0.98")
        elif bias_level == BiasLevel.MEDIUM:
            adjustment_factor = Decimal("0.95")
        elif bias_level == BiasLevel.HIGH:
            adjustment_factor = Decimal("0.90")
        elif bias_level == BiasLevel.CRITICAL:
            adjustment_factor = Decimal("0.85")

        # Further adjust based on missing percentage
        if missing_pct > Decimal("0"):
            additional_adjustment = Decimal("1") - (missing_pct / Decimal("200"))
            adjustment_factor *= max(Decimal("0.5"), additional_adjustment)

        adjusted_return = original_return * adjustment_factor

        logger.info(
            "Return adjusted for survivorship bias",
            original=float(original_return),
            adjusted=float(adjusted_return),
            factor=float(adjustment_factor)
        )

        return adjusted_return
