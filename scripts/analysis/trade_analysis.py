#!/usr/bin/env python3
"""
Quantum Trader AI - Trade Analysis Script

Analyzes individual trade execution and performance including:
- Trade execution quality
- Entry/exit timing analysis
- Strategy performance breakdown
- Win/loss analysis
- Trade duration statistics
- Market impact analysis

Usage:
    python scripts/analysis/trade_analysis.py --config config/environments/production.yaml
    python scripts/analysis/trade_analysis.py --start-date 2024-01-01 --strategy arbitrage
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


class Trade(BaseModel):
    """Trade data model."""

    trade_id: str
    symbol: str
    exchange: str
    strategy: Optional[str] = None
    side: str
    quantity: Decimal
    entry_price: Decimal
    exit_price: Optional[Decimal] = None
    realized_pnl: Decimal
    fees: Decimal
    opened_at: datetime
    closed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("opened_at", "closed_at")
    def ensure_utc(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure timestamps are UTC."""
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


class TradeAnalyzer:
    """
    Analyzes trade execution and performance.

    Attributes:
        config: Configuration manager
        db: Database handler
        start_date: Analysis start date
        end_date: Analysis end date
        strategy_filter: Optional strategy filter
    """

    def __init__(
        self,
        config_path: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        strategy: Optional[str] = None,
        exchange: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> None:
        """
        Initialize trade analyzer.

        Args:
            config_path: Path to configuration file
            start_date: Analysis start date (defaults to 7 days ago)
            end_date: Analysis end date (defaults to now)
            strategy: Filter by strategy name
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
        self.start_date = start_date or (now - timedelta(days=7))

        # Filters
        self.strategy = strategy
        self.exchange = exchange
        self.symbol = symbol

        # Load analysis parameters from config
        analysis_config = self.config.get("trade_analysis", {})
        self.min_trade_duration_seconds = int(
            analysis_config.get("min_trade_duration_seconds", "60")
        )
        self.max_trade_duration_seconds = int(
            analysis_config.get("max_trade_duration_seconds", "86400")
        )

        logger.info(
            "Trade analyzer initialized",
            start_date=self.start_date.isoformat(),
            end_date=self.end_date.isoformat(),
            strategy=self.strategy,
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

    async def fetch_trades(self) -> pl.DataFrame:
        """
        Fetch trades from database.

        Returns:
            DataFrame with trade data

        Raises:
            AnalysisError: If fetch fails
        """
        try:
            logger.info("Fetching trades")

            # Build query with filters
            filters = [
                "t.closed_at >= $1",
                "t.closed_at <= $2",
                "t.status = 'CLOSED'",
            ]
            params: List[Any] = [self.start_date, self.end_date]

            if self.strategy:
                filters.append(f"t.strategy = ${len(params) + 1}")
                params.append(self.strategy)

            if self.exchange:
                filters.append(f"t.exchange = ${len(params) + 1}")
                params.append(self.exchange)

            if self.symbol:
                filters.append(f"t.symbol = ${len(params) + 1}")
                params.append(self.symbol)

            query = f"""
                SELECT
                    t.trade_id,
                    t.symbol,
                    t.exchange,
                    t.strategy,
                    t.side,
                    t.quantity,
                    t.entry_price,
                    t.exit_price,
                    t.realized_pnl,
                    t.fees,
                    t.opened_at,
                    t.closed_at,
                    EXTRACT(EPOCH FROM (t.closed_at - t.opened_at)) as duration_seconds,
                    t.entry_order_id,
                    t.exit_order_id,
                    t.metadata
                FROM trades t
                WHERE {' AND '.join(filters)}
                ORDER BY t.closed_at DESC
            """

            result = await self.db.query(query, *params)

            if not result:
                logger.warning("No trades found matching criteria")
                return pl.DataFrame()

            df = pl.DataFrame(result)

            logger.info("Trades fetched", count=len(df))

            return df

        except Exception as e:
            logger.error("Failed to fetch trades", error=str(e))
            raise AnalysisError(f"Trade fetch failed: {e}") from e

    async def analyze_win_loss(self, trades_df: pl.DataFrame) -> Dict[str, Any]:
        """
        Analyze win/loss statistics.

        Args:
            trades_df: DataFrame with trade data

        Returns:
            Dictionary with win/loss metrics

        Raises:
            AnalysisError: If analysis fails
        """
        try:
            logger.info("Analyzing win/loss statistics")

            if len(trades_df) == 0:
                return {}

            # Calculate net P&L per trade
            trades_df = trades_df.with_columns(
                [(pl.col("realized_pnl") - pl.col("fees")).alias("net_pnl")]
            )

            # Separate winners and losers
            winners = trades_df.filter(pl.col("net_pnl") > 0)
            losers = trades_df.filter(pl.col("net_pnl") < 0)
            breakeven = trades_df.filter(pl.col("net_pnl") == 0)

            # Calculate statistics
            total_trades = len(trades_df)
            winning_trades = len(winners)
            losing_trades = len(losers)
            breakeven_trades = len(breakeven)

            win_rate = (
                Decimal(str(winning_trades)) / Decimal(str(total_trades))
                if total_trades > 0
                else Decimal("0")
            )

            # Profit metrics
            total_profit = Decimal(str(winners["net_pnl"].sum())) if len(winners) > 0 else Decimal("0")
            total_loss = abs(Decimal(str(losers["net_pnl"].sum()))) if len(losers) > 0 else Decimal("0")
            net_profit = total_profit - total_loss

            avg_win = total_profit / Decimal(str(winning_trades)) if winning_trades > 0 else Decimal("0")
            avg_loss = total_loss / Decimal(str(losing_trades)) if losing_trades > 0 else Decimal("0")

            # Risk/reward ratio
            avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else Decimal("0")

            # Profit factor
            profit_factor = total_profit / total_loss if total_loss > 0 else Decimal("0")

            # Consecutive wins/losses
            trades_sorted = trades_df.sort("closed_at")
            win_streak = 0
            loss_streak = 0
            max_win_streak = 0
            max_loss_streak = 0

            for pnl in trades_sorted["net_pnl"]:
                if pnl > 0:
                    win_streak += 1
                    loss_streak = 0
                    max_win_streak = max(max_win_streak, win_streak)
                elif pnl < 0:
                    loss_streak += 1
                    win_streak = 0
                    max_loss_streak = max(max_loss_streak, loss_streak)

            metrics = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "breakeven_trades": breakeven_trades,
                "win_rate": str(win_rate),
                "total_profit": str(total_profit),
                "total_loss": str(total_loss),
                "net_profit": str(net_profit),
                "avg_win": str(avg_win),
                "avg_loss": str(avg_loss),
                "avg_win_loss_ratio": str(avg_win_loss_ratio),
                "profit_factor": str(profit_factor),
                "max_win_streak": max_win_streak,
                "max_loss_streak": max_loss_streak,
            }

            logger.info(
                "Win/loss analysis complete",
                win_rate=str(win_rate),
                profit_factor=str(profit_factor),
            )

            return metrics

        except Exception as e:
            logger.error("Win/loss analysis failed", error=str(e))
            raise AnalysisError(f"Win/loss analysis failed: {e}") from e

    async def analyze_duration(self, trades_df: pl.DataFrame) -> Dict[str, Any]:
        """
        Analyze trade duration statistics.

        Args:
            trades_df: DataFrame with trade data

        Returns:
            Dictionary with duration metrics

        Raises:
            AnalysisError: If analysis fails
        """
        try:
            logger.info("Analyzing trade duration")

            if len(trades_df) == 0:
                return {}

            # Duration statistics
            avg_duration = Decimal(str(trades_df["duration_seconds"].mean()))
            median_duration = Decimal(str(trades_df["duration_seconds"].median()))
            min_duration = Decimal(str(trades_df["duration_seconds"].min()))
            max_duration = Decimal(str(trades_df["duration_seconds"].max()))
            std_duration = Decimal(str(trades_df["duration_seconds"].std()))

            # Duration distribution
            short_trades = len(
                trades_df.filter(
                    pl.col("duration_seconds") < self.min_trade_duration_seconds
                )
            )
            normal_trades = len(
                trades_df.filter(
                    (pl.col("duration_seconds") >= self.min_trade_duration_seconds)
                    & (pl.col("duration_seconds") <= self.max_trade_duration_seconds)
                )
            )
            long_trades = len(
                trades_df.filter(
                    pl.col("duration_seconds") > self.max_trade_duration_seconds
                )
            )

            # Calculate P&L by duration bucket
            trades_df_with_pnl = trades_df.with_columns(
                [(pl.col("realized_pnl") - pl.col("fees")).alias("net_pnl")]
            )

            short_pnl = (
                Decimal(
                    str(
                        trades_df_with_pnl.filter(
                            pl.col("duration_seconds")
                            < self.min_trade_duration_seconds
                        )["net_pnl"].sum()
                    )
                )
                if short_trades > 0
                else Decimal("0")
            )

            normal_pnl = (
                Decimal(
                    str(
                        trades_df_with_pnl.filter(
                            (
                                pl.col("duration_seconds")
                                >= self.min_trade_duration_seconds
                            )
                            & (
                                pl.col("duration_seconds")
                                <= self.max_trade_duration_seconds
                            )
                        )["net_pnl"].sum()
                    )
                )
                if normal_trades > 0
                else Decimal("0")
            )

            long_pnl = (
                Decimal(
                    str(
                        trades_df_with_pnl.filter(
                            pl.col("duration_seconds")
                            > self.max_trade_duration_seconds
                        )["net_pnl"].sum()
                    )
                )
                if long_trades > 0
                else Decimal("0")
            )

            metrics = {
                "avg_duration_seconds": str(avg_duration),
                "median_duration_seconds": str(median_duration),
                "min_duration_seconds": str(min_duration),
                "max_duration_seconds": str(max_duration),
                "std_duration_seconds": str(std_duration),
                "short_trades": short_trades,
                "normal_trades": normal_trades,
                "long_trades": long_trades,
                "short_trades_pnl": str(short_pnl),
                "normal_trades_pnl": str(normal_pnl),
                "long_trades_pnl": str(long_pnl),
            }

            logger.info(
                "Duration analysis complete",
                avg_duration=str(avg_duration),
            )

            return metrics

        except Exception as e:
            logger.error("Duration analysis failed", error=str(e))
            raise AnalysisError(f"Duration analysis failed: {e}") from e

    async def analyze_by_strategy(self, trades_df: pl.DataFrame) -> pl.DataFrame:
        """
        Analyze performance breakdown by strategy.

        Args:
            trades_df: DataFrame with trade data

        Returns:
            DataFrame with strategy performance

        Raises:
            AnalysisError: If analysis fails
        """
        try:
            logger.info("Analyzing performance by strategy")

            if len(trades_df) == 0:
                return pl.DataFrame()

            # Calculate net P&L
            trades_df = trades_df.with_columns(
                [(pl.col("realized_pnl") - pl.col("fees")).alias("net_pnl")]
            )

            # Group by strategy
            strategy_stats = trades_df.group_by("strategy").agg(
                [
                    pl.col("trade_id").count().alias("trade_count"),
                    pl.col("net_pnl").sum().alias("total_pnl"),
                    pl.col("net_pnl").mean().alias("avg_pnl"),
                    pl.col("fees").sum().alias("total_fees"),
                    (pl.col("net_pnl") > 0).sum().alias("winning_trades"),
                    (pl.col("net_pnl") < 0).sum().alias("losing_trades"),
                    pl.col("duration_seconds").mean().alias("avg_duration"),
                ]
            )

            # Calculate win rate
            strategy_stats = strategy_stats.with_columns(
                [
                    (
                        pl.col("winning_trades") / pl.col("trade_count") * 100
                    ).alias("win_rate_pct")
                ]
            )

            # Sort by total P&L
            strategy_stats = strategy_stats.sort("total_pnl", descending=True)

            logger.info(
                "Strategy analysis complete",
                strategies=len(strategy_stats),
            )

            return strategy_stats

        except Exception as e:
            logger.error("Strategy analysis failed", error=str(e))
            raise AnalysisError(f"Strategy analysis failed: {e}") from e

    async def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive trade analysis report.

        Returns:
            Trade analysis report dictionary

        Raises:
            AnalysisError: If report generation fails
        """
        try:
            logger.info("Generating trade analysis report")

            # Fetch trades
            trades = await self.fetch_trades()

            if len(trades) == 0:
                logger.warning("No trades found for analysis")
                return {
                    "report_metadata": {
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "period_start": self.start_date.isoformat(),
                        "period_end": self.end_date.isoformat(),
                        "filters": {
                            "strategy": self.strategy,
                            "exchange": self.exchange,
                            "symbol": self.symbol,
                        },
                    },
                    "message": "No trades found matching criteria",
                }

            # Run analyses
            win_loss = await self.analyze_win_loss(trades)
            duration = await self.analyze_duration(trades)
            strategy_stats = await self.analyze_by_strategy(trades)

            report = {
                "report_metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "period_start": self.start_date.isoformat(),
                    "period_end": self.end_date.isoformat(),
                    "filters": {
                        "strategy": self.strategy,
                        "exchange": self.exchange,
                        "symbol": self.symbol,
                    },
                },
                "win_loss_metrics": win_loss,
                "duration_metrics": duration,
                "strategy_performance": {
                    "data": strategy_stats.to_dicts()
                    if len(strategy_stats) > 0
                    else []
                },
            }

            logger.info("Trade analysis report generated successfully")

            return report

        except Exception as e:
            logger.error("Report generation failed", error=str(e))
            raise AnalysisError(f"Report generation failed: {e}") from e

    async def save_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Save report to file.

        Args:
            report: Trade analysis report
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

    parser = argparse.ArgumentParser(description="Quantum Trader AI Trade Analysis")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--start-date", help="Start date (ISO format)")
    parser.add_argument("--end-date", help="End date (ISO format)")
    parser.add_argument("--strategy", help="Filter by strategy")
    parser.add_argument("--exchange", help="Filter by exchange")
    parser.add_argument("--symbol", help="Filter by symbol")
    parser.add_argument(
        "--output", default="reports/trade_analysis.yaml", help="Output report path"
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
    analyzer = TradeAnalyzer(
        args.config, start_date, end_date, args.strategy, args.exchange, args.symbol
    )

    try:
        await analyzer.initialize()
        report = await analyzer.generate_report()
        await analyzer.save_report(report, args.output)

        print(f"✓ Trade analysis complete. Report saved to: {args.output}")

    except Exception as e:
        logger.error("Analysis failed", error=str(e))
        print(f"✗ Analysis failed: {e}")
        sys.exit(1)

    finally:
        await analyzer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
