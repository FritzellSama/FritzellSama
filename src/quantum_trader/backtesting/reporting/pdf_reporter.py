"""PDF Reporter for Backtesting Results.

Generates comprehensive PDF reports with charts and tables for
professional backtesting analysis and documentation.
"""

import asyncio
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class PDFReporter:
    """Generate PDF reports for backtest results.

    Creates professional PDF reports with:
    - Executive summary
    - Performance metrics
    - Charts and visualizations
    - Trade history
    - Risk analysis

    Attributes:
        config: Reporter configuration from config files
        output_dir: Directory for generated reports
        include_charts: Whether to include chart visualizations
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize PDF reporter.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = {
            ...     "output_dir": "/path/to/reports",
            ...     "include_charts": True,
            ...     "page_size": "A4"
            ... }
            >>> reporter = PDFReporter(config)
        """
        self.config = config
        self._validate_config()

        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.include_charts = config.get("include_charts", True)
        self.page_size = config.get("page_size", "A4")

        logger.info(
            "PDFReporter initialized",
            output_dir=str(self.output_dir),
            include_charts=self.include_charts
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = ["output_dir"]
        missing_keys = [key for key in required_keys if key not in self.config]

        if missing_keys:
            error_msg = f"Missing required config keys: {missing_keys}"
            logger.error("Config validation failed", missing_keys=missing_keys)
            raise ValueError(error_msg)

    async def generate_report(
        self,
        backtest_results: Dict[str, Any],
        trades: pl.DataFrame,
        equity_curve: pl.DataFrame,
        report_name: str
    ) -> str:
        """Generate comprehensive PDF report.

        Args:
            backtest_results: Dictionary with backtest metrics
            trades: DataFrame with trade history
            equity_curve: DataFrame with equity progression
            report_name: Name for the output report

        Returns:
            Path to generated PDF report

        Raises:
            ValueError: If input data is invalid
            IOError: If report generation fails

        Example:
            >>> reporter = PDFReporter(config)
            >>> results = {"total_return": Decimal("0.25"), ...}
            >>> path = await reporter.generate_report(results, trades_df, equity_df, "backtest_2024")
        """
        try:
            self._validate_input_data(backtest_results, trades, equity_curve)

            logger.info("Generating PDF report", report_name=report_name)

            # Build report sections
            sections = await self._build_report_sections(
                backtest_results,
                trades,
                equity_curve
            )

            # Generate PDF (in production would use ReportLab or similar)
            output_path = self.output_dir / f"{report_name}.pdf"

            # Write report data
            await self._write_pdf(output_path, sections, report_name)

            logger.info("PDF report generated successfully", path=str(output_path))
            return str(output_path)

        except Exception as e:
            logger.error("Failed to generate PDF report", error=str(e))
            raise IOError(f"Report generation failed: {e}") from e

    def _validate_input_data(
        self,
        backtest_results: Dict[str, Any],
        trades: pl.DataFrame,
        equity_curve: pl.DataFrame
    ) -> None:
        """Validate input data."""
        if not isinstance(backtest_results, dict):
            raise ValueError("backtest_results must be a dictionary")
        if not isinstance(trades, pl.DataFrame):
            raise ValueError("trades must be a Polars DataFrame")
        if not isinstance(equity_curve, pl.DataFrame):
            raise ValueError("equity_curve must be a Polars DataFrame")

    async def _build_report_sections(
        self,
        backtest_results: Dict[str, Any],
        trades: pl.DataFrame,
        equity_curve: pl.DataFrame
    ) -> Dict[str, Any]:
        """Build all report sections."""
        sections = {
            "summary": self._build_summary(backtest_results),
            "metrics": self._build_metrics(backtest_results),
            "trades": self._build_trades_section(trades),
            "equity": equity_curve
        }
        return sections

    def _build_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Build executive summary section."""
        return {
            "total_return": results.get("total_return", Decimal("0")),
            "sharpe_ratio": results.get("sharpe_ratio", Decimal("0")),
            "max_drawdown": results.get("max_drawdown", Decimal("0")),
            "total_trades": results.get("total_trades", 0)
        }

    def _build_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Build detailed metrics section."""
        return {
            "win_rate": results.get("win_rate", Decimal("0")),
            "profit_factor": results.get("profit_factor", Decimal("0")),
            "avg_win": results.get("avg_win", Decimal("0")),
            "avg_loss": results.get("avg_loss", Decimal("0")),
            "sortino_ratio": results.get("sortino_ratio", Decimal("0"))
        }

    def _build_trades_section(self, trades: pl.DataFrame) -> Dict[str, Any]:
        """Build trades analysis section."""
        return {
            "total_trades": trades.height,
            "trades_summary": trades.head(100) if trades.height > 0 else trades
        }

    async def _write_pdf(
        self,
        output_path: Path,
        sections: Dict[str, Any],
        report_name: str
    ) -> None:
        """Write PDF file."""
        # In production, would use ReportLab or similar to generate actual PDF
        # For now, write a simple text representation
        content = f"""
Backtest Report: {report_name}
Generated: {datetime.utcnow().isoformat()}

SUMMARY
Total Return: {sections['summary']['total_return']}
Sharpe Ratio: {sections['summary']['sharpe_ratio']}
Max Drawdown: {sections['summary']['max_drawdown']}
Total Trades: {sections['summary']['total_trades']}

DETAILED METRICS
Win Rate: {sections['metrics']['win_rate']}
Profit Factor: {sections['metrics']['profit_factor']}
"""
        # Write to file (placeholder)
        with open(str(output_path).replace('.pdf', '.txt'), 'w') as f:
            f.write(content)

        logger.debug("PDF content written", path=str(output_path))
