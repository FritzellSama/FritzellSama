#!/usr/bin/env python3
"""
Quantum Trader AI - Risk Analysis Report Generator

Generates comprehensive risk reports including:
- Value at Risk (VaR) and Conditional VaR
- Position exposure and concentration
- Leverage analysis
- Drawdown metrics
- Correlation analysis
- Risk limit violations

Usage:
    python scripts/analysis/risk_report.py --config config/environments/production.yaml
    python scripts/analysis/risk_report.py --start-date 2024-01-01 --var-confidence 0.99
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


class RiskMetrics(BaseModel):
    """Risk metrics data model."""

    timestamp: datetime
    metric_name: str
    value: Decimal
    threshold: Optional[Decimal] = None
    status: str  # "OK", "WARNING", "CRITICAL"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("timestamp")
    def ensure_utc(cls, v: datetime) -> datetime:
        """Ensure timestamp is UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


class RiskReportGenerator:
    """
    Generates comprehensive risk analysis reports.

    Attributes:
        config: Configuration manager
        db: Database handler
        start_date: Analysis start date
        end_date: Analysis end date
        var_confidence: VaR confidence level
    """

    def __init__(
        self,
        config_path: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        var_confidence: Optional[Decimal] = None,
    ) -> None:
        """
        Initialize risk report generator.

        Args:
            config_path: Path to configuration file
            start_date: Analysis start date (defaults to 30 days ago)
            end_date: Analysis end date (defaults to now)
            var_confidence: VaR confidence level (defaults to 0.95)

        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.config = ConfigManager(config_path)
        self.db: Optional[DatabaseHandler] = None

        # Set date range
        now = datetime.now(timezone.utc)
        self.end_date = end_date or now
        self.start_date = start_date or (now - timedelta(days=30))

        # Load risk parameters from config
        risk_config = self.config.get("risk_management", {})
        self.var_confidence = var_confidence or Decimal(
            str(risk_config.get("var_confidence", "0.95"))
        )
        self.max_position_size = Decimal(
            str(risk_config.get("max_position_size_usd", "1000000"))
        )
        self.max_leverage = Decimal(str(risk_config.get("max_leverage", "10")))
        self.max_drawdown_pct = Decimal(
            str(risk_config.get("max_drawdown_pct", "20"))
        )
        self.max_concentration_pct = Decimal(
            str(risk_config.get("max_concentration_pct", "25"))
        )

        logger.info(
            "Risk report generator initialized",
            start_date=self.start_date.isoformat(),
            end_date=self.end_date.isoformat(),
            var_confidence=str(self.var_confidence),
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

    async def calculate_var(self) -> Dict[str, Any]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR).

        Returns:
            Dictionary with VaR metrics

        Raises:
            AnalysisError: If calculation fails
        """
        try:
            logger.info("Calculating VaR", confidence=str(self.var_confidence))

            # Get daily P&L data
            query = """
                SELECT
                    DATE_TRUNC('day', closed_at) as date,
                    SUM(realized_pnl - fees) as daily_pnl
                FROM trades
                WHERE closed_at >= $1 AND closed_at <= $2
                    AND status = 'CLOSED'
                GROUP BY DATE_TRUNC('day', closed_at)
                ORDER BY date
            """

            result = await self.db.query(query, self.start_date, self.end_date)

            if not result or len(result) < 2:
                logger.warning("Insufficient data for VaR calculation")
                return {}

            df = pl.DataFrame(result)

            # Calculate returns
            df = df.with_columns(
                [
                    pl.col("daily_pnl").pct_change().alias("return"),
                ]
            )

            # Remove first row with null return
            df = df.filter(pl.col("return").is_not_null())

            if len(df) < 2:
                logger.warning("Insufficient return data for VaR")
                return {}

            # Calculate VaR (percentile method)
            var_percentile = 1 - float(self.var_confidence)
            var_value = Decimal(str(df["return"].quantile(var_percentile)))

            # Calculate CVaR (average of returns below VaR)
            var_threshold = float(var_value)
            cvar_returns = df.filter(pl.col("return") <= var_threshold)
            cvar_value = (
                Decimal(str(cvar_returns["return"].mean()))
                if len(cvar_returns) > 0
                else Decimal("0")
            )

            # Calculate parametric VaR (assumes normal distribution)
            mean_return = Decimal(str(df["return"].mean()))
            std_return = Decimal(str(df["return"].std()))

            # Z-score for confidence level (approximation)
            z_scores = {
                Decimal("0.90"): Decimal("1.282"),
                Decimal("0.95"): Decimal("1.645"),
                Decimal("0.99"): Decimal("2.326"),
            }
            z_score = z_scores.get(self.var_confidence, Decimal("1.645"))
            parametric_var = mean_return - (z_score * std_return)

            var_metrics = {
                "confidence_level": str(self.var_confidence),
                "historical_var": str(var_value),
                "conditional_var": str(cvar_value),
                "parametric_var": str(parametric_var),
                "mean_daily_return": str(mean_return),
                "return_volatility": str(std_return),
                "observations": len(df),
            }

            logger.info(
                "VaR calculated",
                var=str(var_value),
                cvar=str(cvar_value),
            )

            return var_metrics

        except Exception as e:
            logger.error("VaR calculation failed", error=str(e))
            raise AnalysisError(f"VaR calculation failed: {e}") from e

    async def analyze_position_exposure(self) -> pl.DataFrame:
        """
        Analyze current position exposure and concentration.

        Returns:
            DataFrame with exposure metrics

        Raises:
            AnalysisError: If analysis fails
        """
        try:
            logger.info("Analyzing position exposure")

            query = """
                SELECT
                    exchange,
                    symbol,
                    side,
                    quantity,
                    entry_price,
                    current_price,
                    leverage,
                    quantity * current_price as position_value_usd,
                    (current_price - entry_price) * quantity *
                        CASE WHEN side = 'LONG' THEN 1 ELSE -1 END as unrealized_pnl,
                    updated_at
                FROM positions
                WHERE status = 'OPEN'
            """

            result = await self.db.query(query)

            if not result:
                logger.warning("No open positions found")
                return pl.DataFrame()

            df = pl.DataFrame(result)

            # Calculate total portfolio value
            total_value = Decimal(str(df["position_value_usd"].sum()))

            # Calculate concentration per symbol
            df = df.with_columns(
                [
                    (pl.col("position_value_usd") / float(total_value) * 100).alias(
                        "concentration_pct"
                    )
                ]
            )

            # Aggregate by symbol
            exposure = df.group_by(["exchange", "symbol"]).agg(
                [
                    pl.col("position_value_usd").sum().alias("total_exposure_usd"),
                    pl.col("concentration_pct").sum().alias("concentration_pct"),
                    pl.col("unrealized_pnl").sum().alias("unrealized_pnl"),
                    pl.col("leverage").max().alias("max_leverage"),
                    pl.col("symbol").count().alias("position_count"),
                ]
            )

            # Identify concentration violations
            violations = exposure.filter(
                pl.col("concentration_pct") > float(self.max_concentration_pct)
            )

            if len(violations) > 0:
                logger.warning(
                    "Concentration limit violations detected",
                    violation_count=len(violations),
                    max_concentration=str(self.max_concentration_pct),
                )

            # Identify position size violations
            size_violations = exposure.filter(
                pl.col("total_exposure_usd") > float(self.max_position_size)
            )

            if len(size_violations) > 0:
                logger.warning(
                    "Position size limit violations detected",
                    violation_count=len(size_violations),
                    max_position_size=str(self.max_position_size),
                )

            logger.info(
                "Position exposure analyzed",
                total_positions=len(df),
                total_value=str(total_value),
            )

            return exposure

        except Exception as e:
            logger.error("Position exposure analysis failed", error=str(e))
            raise AnalysisError(f"Exposure analysis failed: {e}") from e

    async def analyze_drawdown(self) -> Dict[str, Any]:
        """
        Analyze drawdown metrics.

        Returns:
            Dictionary with drawdown metrics

        Raises:
            AnalysisError: If analysis fails
        """
        try:
            logger.info("Analyzing drawdown")

            # Get cumulative P&L over time
            query = """
                SELECT
                    DATE_TRUNC('day', closed_at) as date,
                    SUM(realized_pnl - fees) OVER (ORDER BY DATE_TRUNC('day', closed_at)) as cumulative_pnl
                FROM trades
                WHERE closed_at >= $1 AND closed_at <= $2
                    AND status = 'CLOSED'
                GROUP BY DATE_TRUNC('day', closed_at)
                ORDER BY date
            """

            result = await self.db.query(query, self.start_date, self.end_date)

            if not result or len(result) < 2:
                logger.warning("Insufficient data for drawdown calculation")
                return {}

            df = pl.DataFrame(result)

            # Calculate running maximum
            df = df.with_columns(
                [pl.col("cumulative_pnl").cum_max().alias("running_max")]
            )

            # Calculate drawdown
            df = df.with_columns(
                [(pl.col("cumulative_pnl") - pl.col("running_max")).alias("drawdown")]
            )

            # Calculate drawdown percentage
            df = df.with_columns(
                [
                    (
                        pl.col("drawdown")
                        / pl.col("running_max").abs()
                        * 100
                    ).alias("drawdown_pct")
                ]
            )

            # Get maximum drawdown
            max_dd = Decimal(str(df["drawdown"].min()))
            max_dd_pct = Decimal(str(df["drawdown_pct"].min()))
            max_dd_date = df.filter(pl.col("drawdown") == float(max_dd))["date"][0]

            # Current drawdown
            current_dd = Decimal(str(df["drawdown"][-1]))
            current_dd_pct = Decimal(str(df["drawdown_pct"][-1]))

            # Check for drawdown violations
            if abs(current_dd_pct) > self.max_drawdown_pct:
                logger.error(
                    "Maximum drawdown limit exceeded",
                    current_drawdown=str(current_dd_pct),
                    max_allowed=str(self.max_drawdown_pct),
                )

            drawdown_metrics = {
                "max_drawdown": str(max_dd),
                "max_drawdown_pct": str(max_dd_pct),
                "max_drawdown_date": max_dd_date.isoformat(),
                "current_drawdown": str(current_dd),
                "current_drawdown_pct": str(current_dd_pct),
                "drawdown_limit": str(self.max_drawdown_pct),
                "limit_exceeded": abs(current_dd_pct) > self.max_drawdown_pct,
            }

            logger.info(
                "Drawdown analyzed",
                max_drawdown=str(max_dd_pct),
                current_drawdown=str(current_dd_pct),
            )

            return drawdown_metrics

        except Exception as e:
            logger.error("Drawdown analysis failed", error=str(e))
            raise AnalysisError(f"Drawdown analysis failed: {e}") from e

    async def analyze_leverage(self) -> pl.DataFrame:
        """
        Analyze leverage usage across positions.

        Returns:
            DataFrame with leverage metrics

        Raises:
            AnalysisError: If analysis fails
        """
        try:
            logger.info("Analyzing leverage")

            query = """
                SELECT
                    exchange,
                    symbol,
                    leverage,
                    quantity * current_price as position_value,
                    COUNT(*) as position_count
                FROM positions
                WHERE status = 'OPEN'
                GROUP BY exchange, symbol, leverage, quantity, current_price
            """

            result = await self.db.query(query)

            if not result:
                logger.warning("No positions found for leverage analysis")
                return pl.DataFrame()

            df = pl.DataFrame(result)

            # Aggregate leverage statistics
            stats = df.group_by(["exchange"]).agg(
                [
                    pl.col("leverage").mean().alias("avg_leverage"),
                    pl.col("leverage").max().alias("max_leverage"),
                    pl.col("position_value").sum().alias("total_exposure"),
                    pl.col("position_count").sum().alias("total_positions"),
                ]
            )

            # Identify leverage violations
            violations = df.filter(pl.col("leverage") > float(self.max_leverage))

            if len(violations) > 0:
                logger.warning(
                    "Leverage limit violations detected",
                    violation_count=len(violations),
                    max_leverage=str(self.max_leverage),
                )

            logger.info("Leverage analysis complete", exchanges=len(stats))

            return stats

        except Exception as e:
            logger.error("Leverage analysis failed", error=str(e))
            raise AnalysisError(f"Leverage analysis failed: {e}") from e

    async def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive risk report.

        Returns:
            Risk report dictionary

        Raises:
            AnalysisError: If report generation fails
        """
        try:
            logger.info("Generating risk report")

            # Calculate all risk metrics
            var_metrics = await self.calculate_var()
            exposure = await self.analyze_position_exposure()
            drawdown = await self.analyze_drawdown()
            leverage = await self.analyze_leverage()

            # Determine overall risk status
            risk_status = "OK"
            if drawdown.get("limit_exceeded", False):
                risk_status = "CRITICAL"
            elif len(exposure.filter(
                pl.col("concentration_pct") > float(self.max_concentration_pct)
            )) > 0:
                risk_status = "WARNING"

            report = {
                "report_metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "period_start": self.start_date.isoformat(),
                    "period_end": self.end_date.isoformat(),
                    "overall_risk_status": risk_status,
                },
                "var_metrics": var_metrics,
                "position_exposure": {
                    "data": exposure.to_dicts() if len(exposure) > 0 else [],
                    "max_concentration_limit": str(self.max_concentration_pct),
                    "max_position_size_limit": str(self.max_position_size),
                },
                "drawdown_metrics": drawdown,
                "leverage_metrics": {
                    "data": leverage.to_dicts() if len(leverage) > 0 else [],
                    "max_leverage_limit": str(self.max_leverage),
                },
            }

            logger.info("Risk report generated successfully", status=risk_status)

            return report

        except Exception as e:
            logger.error("Risk report generation failed", error=str(e))
            raise AnalysisError(f"Report generation failed: {e}") from e

    async def save_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Save report to file.

        Args:
            report: Risk report
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

    parser = argparse.ArgumentParser(description="Quantum Trader AI Risk Report")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--start-date", help="Start date (ISO format)")
    parser.add_argument("--end-date", help="End date (ISO format)")
    parser.add_argument(
        "--var-confidence",
        type=float,
        help="VaR confidence level (e.g., 0.95, 0.99)",
    )
    parser.add_argument(
        "--output", default="reports/risk_report.yaml", help="Output report path"
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
    var_confidence = Decimal(str(args.var_confidence)) if args.var_confidence else None

    # Run report
    generator = RiskReportGenerator(args.config, start_date, end_date, var_confidence)

    try:
        await generator.initialize()
        report = await generator.generate_report()
        await generator.save_report(report, args.output)

        print(f"✓ Risk report generated. Saved to: {args.output}")
        print(f"  Risk Status: {report['report_metadata']['overall_risk_status']}")

    except Exception as e:
        logger.error("Report generation failed", error=str(e))
        print(f"✗ Report generation failed: {e}")
        sys.exit(1)

    finally:
        await generator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
