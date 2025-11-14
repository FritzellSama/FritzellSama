#!/usr/bin/env python3
"""
Quantum Trader AI - Profit & Loss Report Generator

Generates comprehensive P&L reports including:
- Realized and unrealized P&L
- P&L by symbol, exchange, strategy
- Daily, weekly, monthly aggregations
- Performance attribution
- Fee analysis

Usage:
    python scripts/analysis/profit_report.py --config config/environments/production.yaml
    python scripts/analysis/profit_report.py --start-date 2024-01-01 --end-date 2024-01-31 --exchange BINANCE
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


class PnLEntry(BaseModel):
    """P&L entry data model."""

    timestamp: datetime
    symbol: str
    exchange: str
    strategy: Optional[str] = None
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    fees: Decimal
    net_pnl: Decimal
    trade_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("timestamp")
    def ensure_utc(cls, v: datetime) -> datetime:
        """Ensure timestamp is UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @validator("net_pnl", always=True)
    def calculate_net_pnl(cls, v: Decimal, values: Dict[str, Any]) -> Decimal:
        """Calculate net P&L."""
        realized = values.get("realized_pnl", Decimal("0"))
        unrealized = values.get("unrealized_pnl", Decimal("0"))
        fees = values.get("fees", Decimal("0"))
        return realized + unrealized - fees


class ProfitReportGenerator:
    """
    Generates comprehensive profit and loss reports.

    Attributes:
        config: Configuration manager
        db: Database handler
        start_date: Report start date
        end_date: Report end date
    """

    def __init__(
        self,
        config_path: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        exchange: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> None:
        """
        Initialize profit report generator.

        Args:
            config_path: Path to configuration file
            start_date: Report start date (defaults to 30 days ago)
            end_date: Report end date (defaults to now)
            exchange: Filter by exchange
            symbol: Filter by symbol

        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.config = ConfigManager(config_path)
        self.db: Optional[DatabaseHandler] = None

        # Set date range
        now = datetime.now(timezone.utc)
        self.end_date = end_date or now
        self.start_date = start_date or (now - timedelta(days=30))

        # Filters
        self.exchange = exchange
        self.symbol = symbol

        # Load precision from config
        report_config = self.config.get("profit_report", {})
        self.decimal_places = int(report_config.get("decimal_places", "8"))
        self.base_currency = report_config.get("base_currency", "USDT")

        logger.info(
            "Profit report generator initialized",
            start_date=self.start_date.isoformat(),
            end_date=self.end_date.isoformat(),
            exchange=self.exchange,
            symbol=self.symbol,
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

    async def calculate_realized_pnl(self) -> pl.DataFrame:
        """
        Calculate realized P&L from closed trades.

        Returns:
            DataFrame with realized P&L

        Raises:
            AnalysisError: If calculation fails
        """
        try:
            logger.info("Calculating realized P&L")

            # Build query with optional filters
            filters = ["t.closed_at >= $1", "t.closed_at <= $2", "t.status = 'CLOSED'"]
            params: List[Any] = [self.start_date, self.end_date]

            if self.exchange:
                filters.append(f"t.exchange = ${len(params) + 1}")
                params.append(self.exchange)

            if self.symbol:
                filters.append(f"t.symbol = ${len(params) + 1}")
                params.append(self.symbol)

            query = f"""
                SELECT
                    DATE_TRUNC('day', t.closed_at) as date,
                    t.exchange,
                    t.symbol,
                    t.strategy,
                    SUM(t.realized_pnl) as realized_pnl,
                    SUM(t.fees) as fees,
                    COUNT(*) as trade_count,
                    AVG(t.realized_pnl) as avg_pnl_per_trade,
                    STDDEV(t.realized_pnl) as pnl_stddev
                FROM trades t
                WHERE {' AND '.join(filters)}
                GROUP BY DATE_TRUNC('day', t.closed_at), t.exchange, t.symbol, t.strategy
                ORDER BY date, t.exchange, t.symbol
            """

            result = await self.db.query(query, *params)

            if not result:
                logger.warning("No realized P&L data found")
                return pl.DataFrame()

            df = pl.DataFrame(result)

            # Calculate net realized P&L
            df = df.with_columns(
                [(pl.col("realized_pnl") - pl.col("fees")).alias("net_realized_pnl")]
            )

            logger.info(
                "Realized P&L calculated",
                total_rows=len(df),
                total_pnl=str(df["net_realized_pnl"].sum()),
            )

            return df

        except Exception as e:
            logger.error("Realized P&L calculation failed", error=str(e))
            raise AnalysisError(f"Realized P&L calculation failed: {e}") from e

    async def calculate_unrealized_pnl(self) -> pl.DataFrame:
        """
        Calculate unrealized P&L from open positions.

        Returns:
            DataFrame with unrealized P&L

        Raises:
            AnalysisError: If calculation fails
        """
        try:
            logger.info("Calculating unrealized P&L")

            # Build query with optional filters
            filters = ["p.status = 'OPEN'"]
            params: List[Any] = []

            if self.exchange:
                filters.append(f"p.exchange = ${len(params) + 1}")
                params.append(self.exchange)

            if self.symbol:
                filters.append(f"p.symbol = ${len(params) + 1}")
                params.append(self.symbol)

            query = f"""
                SELECT
                    p.exchange,
                    p.symbol,
                    p.side,
                    p.entry_price,
                    p.current_price,
                    p.quantity,
                    p.leverage,
                    (p.current_price - p.entry_price) * p.quantity *
                        CASE WHEN p.side = 'LONG' THEN 1 ELSE -1 END as unrealized_pnl,
                    p.updated_at
                FROM positions p
                WHERE {' AND '.join(filters)}
            """

            result = await self.db.query(query, *params)

            if not result:
                logger.warning("No open positions found")
                return pl.DataFrame()

            df = pl.DataFrame(result)

            # Aggregate by exchange and symbol
            stats = df.group_by(["exchange", "symbol"]).agg(
                [
                    pl.col("unrealized_pnl").sum().alias("total_unrealized_pnl"),
                    pl.col("quantity").sum().alias("total_quantity"),
                    pl.col("symbol").count().alias("position_count"),
                ]
            )

            logger.info(
                "Unrealized P&L calculated",
                total_pnl=str(stats["total_unrealized_pnl"].sum()),
                position_count=len(df),
            )

            return stats

        except Exception as e:
            logger.error("Unrealized P&L calculation failed", error=str(e))
            raise AnalysisError(f"Unrealized P&L calculation failed: {e}") from e

    async def analyze_fees(self) -> pl.DataFrame:
        """
        Analyze trading fees breakdown.

        Returns:
            DataFrame with fee analysis

        Raises:
            AnalysisError: If analysis fails
        """
        try:
            logger.info("Analyzing fees")

            filters = ["t.created_at >= $1", "t.created_at <= $2"]
            params: List[Any] = [self.start_date, self.end_date]

            if self.exchange:
                filters.append(f"t.exchange = ${len(params) + 1}")
                params.append(self.exchange)

            if self.symbol:
                filters.append(f"t.symbol = ${len(params) + 1}")
                params.append(self.symbol)

            query = f"""
                SELECT
                    t.exchange,
                    t.symbol,
                    t.fee_type,
                    SUM(t.fees) as total_fees,
                    COUNT(*) as transaction_count,
                    AVG(t.fees) as avg_fee,
                    SUM(t.quantity * t.price) as total_volume
                FROM trades t
                WHERE {' AND '.join(filters)}
                GROUP BY t.exchange, t.symbol, t.fee_type
                ORDER BY total_fees DESC
            """

            result = await self.db.query(query, *params)

            if not result:
                logger.warning("No fee data found")
                return pl.DataFrame()

            df = pl.DataFrame(result)

            # Calculate fee percentage of volume
            df = df.with_columns(
                [(pl.col("total_fees") / pl.col("total_volume") * 100).alias("fee_pct")]
            )

            logger.info(
                "Fee analysis complete",
                total_fees=str(df["total_fees"].sum()),
                exchanges=df["exchange"].n_unique(),
            )

            return df

        except Exception as e:
            logger.error("Fee analysis failed", error=str(e))
            raise AnalysisError(f"Fee analysis failed: {e}") from e

    async def calculate_performance_metrics(
        self, realized_pnl_df: pl.DataFrame
    ) -> Dict[str, Decimal]:
        """
        Calculate performance metrics.

        Args:
            realized_pnl_df: DataFrame with realized P&L

        Returns:
            Dictionary of performance metrics

        Raises:
            AnalysisError: If calculation fails
        """
        try:
            logger.info("Calculating performance metrics")

            if len(realized_pnl_df) == 0:
                return {}

            # Total P&L
            total_pnl = Decimal(str(realized_pnl_df["net_realized_pnl"].sum()))
            total_fees = Decimal(str(realized_pnl_df["fees"].sum()))
            total_trades = int(realized_pnl_df["trade_count"].sum())

            # Win rate
            winning_days = len(
                realized_pnl_df.filter(pl.col("net_realized_pnl") > 0)
            )
            total_days = len(realized_pnl_df.group_by("date").count())
            win_rate = (
                Decimal(str(winning_days)) / Decimal(str(total_days))
                if total_days > 0
                else Decimal("0")
            )

            # Profit factor
            gross_profit = Decimal(
                str(
                    realized_pnl_df.filter(pl.col("net_realized_pnl") > 0)[
                        "net_realized_pnl"
                    ].sum()
                )
            )
            gross_loss = abs(
                Decimal(
                    str(
                        realized_pnl_df.filter(pl.col("net_realized_pnl") < 0)[
                            "net_realized_pnl"
                        ].sum()
                    )
                )
            )
            profit_factor = (
                gross_profit / gross_loss if gross_loss > 0 else Decimal("0")
            )

            # Average metrics
            avg_pnl_per_trade = total_pnl / Decimal(str(total_trades)) if total_trades > 0 else Decimal("0")
            avg_pnl_per_day = total_pnl / Decimal(str(total_days)) if total_days > 0 else Decimal("0")

            metrics = {
                "total_pnl": total_pnl,
                "total_fees": total_fees,
                "total_trades": Decimal(str(total_trades)),
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
                "avg_pnl_per_trade": avg_pnl_per_trade,
                "avg_pnl_per_day": avg_pnl_per_day,
                "trading_days": Decimal(str(total_days)),
            }

            logger.info("Performance metrics calculated", total_pnl=str(total_pnl))

            return metrics

        except Exception as e:
            logger.error("Performance metrics calculation failed", error=str(e))
            raise AnalysisError(f"Metrics calculation failed: {e}") from e

    async def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive profit report.

        Returns:
            Profit report dictionary

        Raises:
            AnalysisError: If report generation fails
        """
        try:
            logger.info("Generating profit report")

            # Calculate all components
            realized_pnl = await self.calculate_realized_pnl()
            unrealized_pnl = await self.calculate_unrealized_pnl()
            fees = await self.analyze_fees()
            metrics = await self.calculate_performance_metrics(realized_pnl)

            report = {
                "report_metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "period_start": self.start_date.isoformat(),
                    "period_end": self.end_date.isoformat(),
                    "exchange_filter": self.exchange,
                    "symbol_filter": self.symbol,
                    "base_currency": self.base_currency,
                },
                "performance_summary": {
                    k: str(v) for k, v in metrics.items()
                } if metrics else {},
                "realized_pnl": {
                    "data": realized_pnl.to_dicts() if len(realized_pnl) > 0 else [],
                    "total": str(realized_pnl["net_realized_pnl"].sum())
                    if len(realized_pnl) > 0
                    else "0",
                },
                "unrealized_pnl": {
                    "data": unrealized_pnl.to_dicts()
                    if len(unrealized_pnl) > 0
                    else [],
                    "total": str(unrealized_pnl["total_unrealized_pnl"].sum())
                    if len(unrealized_pnl) > 0
                    else "0",
                },
                "fees": {
                    "data": fees.to_dicts() if len(fees) > 0 else [],
                    "total": str(fees["total_fees"].sum()) if len(fees) > 0 else "0",
                },
            }

            logger.info("Profit report generated successfully")

            return report

        except Exception as e:
            logger.error("Report generation failed", error=str(e))
            raise AnalysisError(f"Report generation failed: {e}") from e

    async def save_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Save report to file.

        Args:
            report: Profit report
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

    parser = argparse.ArgumentParser(description="Quantum Trader AI Profit Report")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--start-date", help="Start date (ISO format)")
    parser.add_argument("--end-date", help="End date (ISO format)")
    parser.add_argument("--exchange", help="Filter by exchange")
    parser.add_argument("--symbol", help="Filter by symbol")
    parser.add_argument(
        "--output", default="reports/profit_report.yaml", help="Output report path"
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

    # Run report
    generator = ProfitReportGenerator(
        args.config, start_date, end_date, args.exchange, args.symbol
    )

    try:
        await generator.initialize()
        report = await generator.generate_report()
        await generator.save_report(report, args.output)

        print(f"✓ Profit report generated. Saved to: {args.output}")

    except Exception as e:
        logger.error("Report generation failed", error=str(e))
        print(f"✗ Report generation failed: {e}")
        sys.exit(1)

    finally:
        await generator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
