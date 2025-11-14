"""Report Generator for Backtesting.

Unified report generation system supporting multiple output formats
(HTML, PDF, JSON) for comprehensive backtesting analysis.
"""

import asyncio
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path
from enum import Enum
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class ReportFormat(Enum):
    """Supported report output formats."""
    HTML = "HTML"
    PDF = "PDF"
    JSON = "JSON"
    MARKDOWN = "MARKDOWN"


class ReportGenerator:
    """Generate comprehensive backtest reports in multiple formats.

    Unified interface for generating reports with:
    - Multiple output formats
    - Customizable sections
    - Chart generation
    - Trade analysis

    Attributes:
        config: Generator configuration from config files
        output_dir: Directory for generated reports
        supported_formats: List of enabled report formats
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize report generator.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = {
            ...     "output_dir": "/path/to/reports",
            ...     "enabled_formats": ["HTML", "JSON"],
            ...     "include_charts": True
            ... }
            >>> generator = ReportGenerator(config)
        """
        self.config = config
        self._validate_config()

        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Parse enabled formats
        format_strs = config.get("enabled_formats", ["HTML", "JSON"])
        self.supported_formats = [ReportFormat[f.upper()] for f in format_strs]

        self.include_charts = config.get("include_charts", True)
        self.include_trade_details = config.get("include_trade_details", True)

        logger.info(
            "ReportGenerator initialized",
            output_dir=str(self.output_dir),
            formats=[f.value for f in self.supported_formats]
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

    async def generate_reports(
        self,
        backtest_results: Dict[str, Any],
        trades: pl.DataFrame,
        equity_curve: pl.DataFrame,
        report_name: str,
        formats: Optional[List[ReportFormat]] = None
    ) -> Dict[str, str]:
        """Generate reports in multiple formats.

        Args:
            backtest_results: Dictionary with backtest metrics
            trades: DataFrame with trade history
            equity_curve: DataFrame with equity progression
            report_name: Base name for generated reports
            formats: Optional list of formats (uses configured if None)

        Returns:
            Dictionary mapping format to file path

        Raises:
            ValueError: If input data is invalid
            IOError: If report generation fails

        Example:
            >>> generator = ReportGenerator(config)
            >>> results = {"total_return": Decimal("0.25"), ...}
            >>> paths = await generator.generate_reports(
            ...     results, trades_df, equity_df, "backtest_2024"
            ... )
            >>> print(f"HTML report: {paths[ReportFormat.HTML]}")
        """
        try:
            self._validate_input_data(backtest_results, trades, equity_curve)

            formats_to_generate = formats or self.supported_formats

            logger.info(
                "Generating reports",
                report_name=report_name,
                formats=[f.value for f in formats_to_generate]
            )

            generated_paths = {}

            for fmt in formats_to_generate:
                path = await self._generate_format(
                    fmt,
                    backtest_results,
                    trades,
                    equity_curve,
                    report_name
                )
                generated_paths[fmt] = path

            logger.info(
                "All reports generated successfully",
                report_count=len(generated_paths)
            )

            return generated_paths

        except Exception as e:
            logger.error("Failed to generate reports", error=str(e))
            raise

    def _validate_input_data(
        self,
        backtest_results: Dict[str, Any],
        trades: pl.DataFrame,
        equity_curve: pl.DataFrame
    ) -> None:
        """Validate input data for report generation."""
        if not isinstance(backtest_results, dict):
            raise ValueError("backtest_results must be a dictionary")

        if not isinstance(trades, pl.DataFrame):
            raise ValueError("trades must be a Polars DataFrame")

        if not isinstance(equity_curve, pl.DataFrame):
            raise ValueError("equity_curve must be a Polars DataFrame")

        if equity_curve.height == 0:
            raise ValueError("equity_curve cannot be empty")

    async def _generate_format(
        self,
        format_type: ReportFormat,
        backtest_results: Dict[str, Any],
        trades: pl.DataFrame,
        equity_curve: pl.DataFrame,
        report_name: str
    ) -> str:
        """Generate report in specific format.

        Args:
            format_type: Report format
            backtest_results: Backtest metrics
            trades: Trade history
            equity_curve: Equity progression
            report_name: Report name

        Returns:
            Path to generated report
        """
        if format_type == ReportFormat.HTML:
            return await self._generate_html(
                backtest_results, trades, equity_curve, report_name
            )
        elif format_type == ReportFormat.JSON:
            return await self._generate_json(
                backtest_results, trades, equity_curve, report_name
            )
        elif format_type == ReportFormat.PDF:
            return await self._generate_pdf(
                backtest_results, trades, equity_curve, report_name
            )
        elif format_type == ReportFormat.MARKDOWN:
            return await self._generate_markdown(
                backtest_results, trades, equity_curve, report_name
            )
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    async def _generate_html(
        self,
        results: Dict[str, Any],
        trades: pl.DataFrame,
        equity: pl.DataFrame,
        name: str
    ) -> str:
        """Generate HTML report."""
        import json

        output_path = self.output_dir / f"{name}.html"

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{name} - Backtest Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #0a0e27; color: #e0e0e0; }}
        .metric {{ margin: 10px 0; }}
        .label {{ font-weight: bold; color: #00d4ff; }}
    </style>
</head>
<body>
    <h1>{name}</h1>
    <h2>Performance Metrics</h2>
    <div class="metric"><span class="label">Total Return:</span> {results.get('total_return', 0)}</div>
    <div class="metric"><span class="label">Sharpe Ratio:</span> {results.get('sharpe_ratio', 0)}</div>
    <div class="metric"><span class="label">Max Drawdown:</span> {results.get('max_drawdown', 0)}</div>
    <div class="metric"><span class="label">Total Trades:</span> {results.get('total_trades', 0)}</div>
</body>
</html>"""

        async with asyncio.Lock():
            with open(output_path, 'w') as f:
                f.write(html_content)

        logger.debug("HTML report generated", path=str(output_path))
        return str(output_path)

    async def _generate_json(
        self,
        results: Dict[str, Any],
        trades: pl.DataFrame,
        equity: pl.DataFrame,
        name: str
    ) -> str:
        """Generate JSON report."""
        import json

        output_path = self.output_dir / f"{name}.json"

        # Convert Decimal to float for JSON serialization
        serializable_results = {
            k: float(v) if isinstance(v, Decimal) else v
            for k, v in results.items()
        }

        report_data = {
            "report_name": name,
            "generated_at": datetime.utcnow().isoformat(),
            "metrics": serializable_results,
            "trade_count": trades.height,
            "equity_points": equity.height
        }

        async with asyncio.Lock():
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)

        logger.debug("JSON report generated", path=str(output_path))
        return str(output_path)

    async def _generate_pdf(
        self,
        results: Dict[str, Any],
        trades: pl.DataFrame,
        equity: pl.DataFrame,
        name: str
    ) -> str:
        """Generate PDF report (placeholder)."""
        output_path = self.output_dir / f"{name}_pdf.txt"

        content = f"PDF Report: {name}\n"
        content += f"Total Return: {results.get('total_return', 0)}\n"

        async with asyncio.Lock():
            with open(output_path, 'w') as f:
                f.write(content)

        logger.debug("PDF report generated", path=str(output_path))
        return str(output_path)

    async def _generate_markdown(
        self,
        results: Dict[str, Any],
        trades: pl.DataFrame,
        equity: pl.DataFrame,
        name: str
    ) -> str:
        """Generate Markdown report."""
        output_path = self.output_dir / f"{name}.md"

        md_content = f"""# {name} - Backtest Report

## Performance Metrics

- **Total Return**: {results.get('total_return', 0)}
- **Sharpe Ratio**: {results.get('sharpe_ratio', 0)}
- **Max Drawdown**: {results.get('max_drawdown', 0)}
- **Total Trades**: {results.get('total_trades', 0)}

## Summary

Generated: {datetime.utcnow().isoformat()}
"""

        async with asyncio.Lock():
            with open(output_path, 'w') as f:
                f.write(md_content)

        logger.debug("Markdown report generated", path=str(output_path))
        return str(output_path)
