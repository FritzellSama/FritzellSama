#!/usr/bin/env python3
"""
Quantum Trader AI - Performance Analysis Script

Analyzes trading system performance metrics including:
- Execution latency and throughput
- Order fill rates and slippage
- System resource utilization
- API response times
- Database query performance

Usage:
    python scripts/analysis/performance_analysis.py --config config/environments/production.yaml
    python scripts/analysis/performance_analysis.py --start-date 2024-01-01 --end-date 2024-01-31
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
import structlog
import yaml
from pydantic import BaseModel, Field, validator

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.quantum_trader.core.config import ConfigManager
from src.quantum_trader.core.database import DatabaseHandler
from src.quantum_trader.core.exceptions import AnalysisError, ConfigurationError

logger = structlog.get_logger(__name__)


class PerformanceMetrics(BaseModel):
    """Performance metrics data model."""

    timestamp: datetime
    metric_type: str
    metric_name: str
    value: Decimal
    unit: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("timestamp")
    def ensure_utc(cls, v: datetime) -> datetime:
        """Ensure timestamp is UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


class PerformanceAnalyzer:
    """
    Analyzes trading system performance metrics.

    Attributes:
        config: Configuration manager
        db: Database handler
        start_date: Analysis start date
        end_date: Analysis end date
    """

    def __init__(
        self,
        config_path: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> None:
        """
        Initialize performance analyzer.

        Args:
            config_path: Path to configuration file
            start_date: Analysis start date (defaults to 24 hours ago)
            end_date: Analysis end date (defaults to now)

        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.config = ConfigManager(config_path)
        self.db: Optional[DatabaseHandler] = None

        # Set date range
        now = datetime.now(timezone.utc)
        self.end_date = end_date or now
        self.start_date = start_date or (now - timedelta(hours=24))

        # Load thresholds from config
        perf_config = self.config.get("performance_analysis", {})
        self.latency_threshold_ms = Decimal(
            str(perf_config.get("latency_threshold_ms", "10"))
        )
        self.min_fill_rate = Decimal(str(perf_config.get("min_fill_rate", "0.95")))
        self.max_slippage_bps = Decimal(
            str(perf_config.get("max_slippage_bps", "5"))
        )

        logger.info(
            "Performance analyzer initialized",
            start_date=self.start_date.isoformat(),
            end_date=self.end_date.isoformat(),
        )

    async def initialize(self) -> None:
        """Initialize database connection."""
        try:
            db_config = self.config.get("database", {})
            self.db = DatabaseHandler(db_config)
            await self.db.connect()
            logger.info("Database connection established")
        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
            raise ConfigurationError(f"Database initialization failed: {e}") from e

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.db:
            try:
                await self.db.disconnect()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error("Error during cleanup", error=str(e))

    async def analyze_execution_latency(self) -> pl.DataFrame:
        """
        Analyze order execution latency.

        Returns:
            DataFrame with latency statistics

        Raises:
            AnalysisError: If analysis fails
        """
        try:
            logger.info("Analyzing execution latency")

            query = """
                SELECT
                    DATE_TRUNC('minute', created_at) as time_bucket,
                    EXTRACT(EPOCH FROM (filled_at - created_at)) * 1000 as latency_ms,
                    exchange,
                    symbol,
                    order_type
                FROM orders
                WHERE created_at >= $1
                    AND created_at <= $2
                    AND status = 'FILLED'
                    AND filled_at IS NOT NULL
                ORDER BY created_at
            """

            result = await self.db.query(query, self.start_date, self.end_date)

            if not result:
                logger.warning("No order data found for latency analysis")
                return pl.DataFrame()

            # Convert to Polars DataFrame
            df = pl.DataFrame(result)

            # Calculate statistics using Decimal
            stats = df.group_by("time_bucket").agg(
                [
                    pl.col("latency_ms").mean().alias("avg_latency_ms"),
                    pl.col("latency_ms").median().alias("median_latency_ms"),
                    pl.col("latency_ms").quantile(0.95).alias("p95_latency_ms"),
                    pl.col("latency_ms").quantile(0.99).alias("p99_latency_ms"),
                    pl.col("latency_ms").max().alias("max_latency_ms"),
                    pl.col("latency_ms").count().alias("order_count"),
                ]
            )

            # Identify violations
            violations = stats.filter(
                pl.col("p95_latency_ms") > float(self.latency_threshold_ms)
            )

            if len(violations) > 0:
                logger.warning(
                    "Latency threshold violations detected",
                    violation_count=len(violations),
                    threshold_ms=str(self.latency_threshold_ms),
                )

            logger.info(
                "Execution latency analysis complete", total_buckets=len(stats)
            )

            return stats

        except Exception as e:
            logger.error("Execution latency analysis failed", error=str(e))
            raise AnalysisError(f"Latency analysis failed: {e}") from e

    async def analyze_fill_rates(self) -> pl.DataFrame:
        """
        Analyze order fill rates.

        Returns:
            DataFrame with fill rate statistics

        Raises:
            AnalysisError: If analysis fails
        """
        try:
            logger.info("Analyzing fill rates")

            query = """
                SELECT
                    exchange,
                    symbol,
                    order_type,
                    COUNT(*) as total_orders,
                    SUM(CASE WHEN status = 'FILLED' THEN 1 ELSE 0 END) as filled_orders,
                    SUM(CASE WHEN status = 'PARTIAL' THEN 1 ELSE 0 END) as partial_orders,
                    SUM(CASE WHEN status = 'CANCELLED' THEN 1 ELSE 0 END) as cancelled_orders,
                    SUM(CASE WHEN status = 'REJECTED' THEN 1 ELSE 0 END) as rejected_orders
                FROM orders
                WHERE created_at >= $1 AND created_at <= $2
                GROUP BY exchange, symbol, order_type
            """

            result = await self.db.query(query, self.start_date, self.end_date)

            if not result:
                logger.warning("No order data found for fill rate analysis")
                return pl.DataFrame()

            df = pl.DataFrame(result)

            # Calculate fill rates
            df = df.with_columns(
                [
                    (pl.col("filled_orders") / pl.col("total_orders")).alias(
                        "fill_rate"
                    ),
                    (
                        (pl.col("filled_orders") + pl.col("partial_orders"))
                        / pl.col("total_orders")
                    ).alias("fill_or_partial_rate"),
                ]
            )

            # Identify low fill rates
            low_fill_rates = df.filter(
                pl.col("fill_rate") < float(self.min_fill_rate)
            )

            if len(low_fill_rates) > 0:
                logger.warning(
                    "Low fill rates detected",
                    count=len(low_fill_rates),
                    min_fill_rate=str(self.min_fill_rate),
                )

            logger.info("Fill rate analysis complete", groups=len(df))

            return df

        except Exception as e:
            logger.error("Fill rate analysis failed", error=str(e))
            raise AnalysisError(f"Fill rate analysis failed: {e}") from e

    async def analyze_slippage(self) -> pl.DataFrame:
        """
        Analyze price slippage.

        Returns:
            DataFrame with slippage statistics

        Raises:
            AnalysisError: If analysis fails
        """
        try:
            logger.info("Analyzing slippage")

            query = """
                SELECT
                    exchange,
                    symbol,
                    side,
                    order_type,
                    price as expected_price,
                    filled_price as actual_price,
                    quantity,
                    created_at
                FROM orders
                WHERE created_at >= $1
                    AND created_at <= $2
                    AND status = 'FILLED'
                    AND filled_price IS NOT NULL
                    AND price IS NOT NULL
            """

            result = await self.db.query(query, self.start_date, self.end_date)

            if not result:
                logger.warning("No order data found for slippage analysis")
                return pl.DataFrame()

            df = pl.DataFrame(result)

            # Calculate slippage in basis points
            df = df.with_columns(
                [
                    (
                        (pl.col("actual_price") - pl.col("expected_price"))
                        / pl.col("expected_price")
                        * Decimal("10000")
                    ).alias("slippage_bps")
                ]
            )

            # Aggregate statistics
            stats = df.group_by(["exchange", "symbol", "side"]).agg(
                [
                    pl.col("slippage_bps").mean().alias("avg_slippage_bps"),
                    pl.col("slippage_bps").median().alias("median_slippage_bps"),
                    pl.col("slippage_bps").std().alias("std_slippage_bps"),
                    pl.col("slippage_bps").abs().max().alias("max_abs_slippage_bps"),
                    pl.col("quantity").sum().alias("total_quantity"),
                    pl.col("symbol").count().alias("order_count"),
                ]
            )

            # Identify excessive slippage
            excessive_slippage = stats.filter(
                pl.col("max_abs_slippage_bps") > float(self.max_slippage_bps)
            )

            if len(excessive_slippage) > 0:
                logger.warning(
                    "Excessive slippage detected",
                    count=len(excessive_slippage),
                    max_slippage_bps=str(self.max_slippage_bps),
                )

            logger.info("Slippage analysis complete", groups=len(stats))

            return stats

        except Exception as e:
            logger.error("Slippage analysis failed", error=str(e))
            raise AnalysisError(f"Slippage analysis failed: {e}") from e

    async def analyze_system_metrics(self) -> pl.DataFrame:
        """
        Analyze system resource utilization.

        Returns:
            DataFrame with system metrics

        Raises:
            AnalysisError: If analysis fails
        """
        try:
            logger.info("Analyzing system metrics")

            query = """
                SELECT
                    timestamp,
                    metric_name,
                    metric_value,
                    host,
                    component
                FROM system_metrics
                WHERE timestamp >= $1 AND timestamp <= $2
                ORDER BY timestamp
            """

            result = await self.db.query(query, self.start_date, self.end_date)

            if not result:
                logger.warning("No system metrics found")
                return pl.DataFrame()

            df = pl.DataFrame(result)

            # Aggregate by component and metric
            stats = df.group_by(["component", "metric_name"]).agg(
                [
                    pl.col("metric_value").mean().alias("avg_value"),
                    pl.col("metric_value").max().alias("max_value"),
                    pl.col("metric_value").min().alias("min_value"),
                    pl.col("metric_value").std().alias("std_value"),
                ]
            )

            logger.info("System metrics analysis complete", metrics=len(stats))

            return stats

        except Exception as e:
            logger.error("System metrics analysis failed", error=str(e))
            raise AnalysisError(f"System metrics analysis failed: {e}") from e

    async def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Returns:
            Performance report dictionary

        Raises:
            AnalysisError: If report generation fails
        """
        try:
            logger.info("Generating performance report")

            # Run all analyses
            latency_stats = await self.analyze_execution_latency()
            fill_rate_stats = await self.analyze_fill_rates()
            slippage_stats = await self.analyze_slippage()
            system_stats = await self.analyze_system_metrics()

            report = {
                "analysis_period": {
                    "start": self.start_date.isoformat(),
                    "end": self.end_date.isoformat(),
                },
                "latency": {
                    "data": latency_stats.to_dicts() if len(latency_stats) > 0 else [],
                    "threshold_ms": str(self.latency_threshold_ms),
                },
                "fill_rates": {
                    "data": fill_rate_stats.to_dicts()
                    if len(fill_rate_stats) > 0
                    else [],
                    "min_threshold": str(self.min_fill_rate),
                },
                "slippage": {
                    "data": slippage_stats.to_dicts() if len(slippage_stats) > 0 else [],
                    "max_threshold_bps": str(self.max_slippage_bps),
                },
                "system_metrics": {
                    "data": system_stats.to_dicts() if len(system_stats) > 0 else []
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

            logger.info("Performance report generated successfully")

            return report

        except Exception as e:
            logger.error("Report generation failed", error=str(e))
            raise AnalysisError(f"Report generation failed: {e}") from e

    async def save_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Save report to file.

        Args:
            report: Performance report
            output_path: Output file path

        Raises:
            AnalysisError: If save fails
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                yaml.dump(report, f, default_flow_style=False, sort_keys=False)

            logger.info("Report saved", path=str(output_file))

        except Exception as e:
            logger.error("Failed to save report", error=str(e))
            raise AnalysisError(f"Report save failed: {e}") from e


async def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Quantum Trader AI Performance Analysis"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--start-date",
        help="Start date (ISO format)",
    )
    parser.add_argument(
        "--end-date",
        help="End date (ISO format)",
    )
    parser.add_argument(
        "--output",
        default="reports/performance_analysis.yaml",
        help="Output report path",
    )

    args = parser.parse_args()

    # Parse dates
    start_date = (
        datetime.fromisoformat(args.start_date).replace(tzinfo=timezone.utc)
        if args.start_date
        else None
    )
    end_date = (
        datetime.fromisoformat(args.end_date).replace(tzinfo=timezone.utc)
        if args.end_date
        else None
    )

    # Run analysis
    analyzer = PerformanceAnalyzer(args.config, start_date, end_date)

    try:
        await analyzer.initialize()
        report = await analyzer.generate_report()
        await analyzer.save_report(report, args.output)

        print(f"✓ Performance analysis complete. Report saved to: {args.output}")

    except Exception as e:
        logger.error("Analysis failed", error=str(e))
        print(f"✗ Analysis failed: {e}")
        sys.exit(1)

    finally:
        await analyzer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
