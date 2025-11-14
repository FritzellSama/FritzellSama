"""HTML Reporter for Backtesting Results.

Generates comprehensive HTML reports with interactive charts and tables
for backtesting analysis.
"""

import asyncio
import os
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class HTMLReporter:
    """Generate HTML reports for backtest results.

    Attributes:
        config: Reporter configuration from config files
        template_dir: Directory containing HTML templates
        output_dir: Directory for generated reports
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize HTML reporter.

        Args:
            config: Configuration dictionary with template_dir, output_dir

        Raises:
            ValueError: If required config parameters missing
        """
        self.config = config
        self._validate_config()

        self.template_dir = Path(self.config["template_dir"])
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "HTMLReporter initialized",
            template_dir=str(self.template_dir),
            output_dir=str(self.output_dir)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config parameters missing
        """
        required_keys = ["template_dir", "output_dir"]
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
        """Generate comprehensive HTML report.

        Args:
            backtest_results: Dictionary containing backtest metrics
            trades: DataFrame with trade history
            equity_curve: DataFrame with equity progression
            report_name: Name for the output report file

        Returns:
            Path to generated HTML report

        Raises:
            ValueError: If input data is invalid
            IOError: If report generation fails

        Example:
            >>> reporter = HTMLReporter(config)
            >>> results = {"total_return": Decimal("0.25"), ...}
            >>> trades_df = pl.DataFrame(...)
            >>> equity_df = pl.DataFrame(...)
            >>> path = await reporter.generate_report(results, trades_df, equity_df, "backtest_2024")
        """
        try:
            self._validate_input_data(backtest_results, trades, equity_curve)

            logger.info("Generating HTML report", report_name=report_name)

            # Generate HTML sections
            html_content = await self._build_html(
                backtest_results,
                trades,
                equity_curve,
                report_name
            )

            # Write report to file
            output_path = self.output_dir / f"{report_name}.html"
            await self._write_report(output_path, html_content)

            logger.info("HTML report generated successfully", path=str(output_path))
            return str(output_path)

        except Exception as e:
            logger.error("Failed to generate HTML report", error=str(e), report_name=report_name)
            raise IOError(f"Report generation failed: {e}") from e

    def _validate_input_data(
        self,
        backtest_results: Dict[str, Any],
        trades: pl.DataFrame,
        equity_curve: pl.DataFrame
    ) -> None:
        """Validate input data for report generation.

        Args:
            backtest_results: Backtest metrics dictionary
            trades: Trades DataFrame
            equity_curve: Equity curve DataFrame

        Raises:
            ValueError: If data validation fails
        """
        if not isinstance(backtest_results, dict):
            raise ValueError("backtest_results must be a dictionary")

        if not isinstance(trades, pl.DataFrame):
            raise ValueError("trades must be a Polars DataFrame")

        if not isinstance(equity_curve, pl.DataFrame):
            raise ValueError("equity_curve must be a Polars DataFrame")

        if trades.height == 0:
            logger.warning("Trades DataFrame is empty")

        if equity_curve.height == 0:
            raise ValueError("equity_curve cannot be empty")

    async def _build_html(
        self,
        backtest_results: Dict[str, Any],
        trades: pl.DataFrame,
        equity_curve: pl.DataFrame,
        report_name: str
    ) -> str:
        """Build complete HTML report.

        Args:
            backtest_results: Backtest metrics
            trades: Trade history
            equity_curve: Equity progression
            report_name: Report title

        Returns:
            Complete HTML document as string
        """
        # Build HTML sections
        header = self._build_header(report_name)
        summary = self._build_summary_section(backtest_results)
        metrics = self._build_metrics_section(backtest_results)
        charts = await self._build_charts_section(equity_curve, trades)
        trades_table = self._build_trades_table(trades)
        footer = self._build_footer()

        # Combine all sections
        html_content = f"""<!DOCTYPE html>
<html lang="en">
{header}
<body>
    <div class="container">
        <h1>{report_name} - Backtest Report</h1>
        <p class="timestamp">Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>

        {summary}
        {metrics}
        {charts}
        {trades_table}
    </div>
    {footer}
</body>
</html>"""

        return html_content

    def _build_header(self, report_name: str) -> str:
        """Build HTML header with CSS and meta tags.

        Args:
            report_name: Report title for meta tags

        Returns:
            HTML header section
        """
        return f"""<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_name} - Backtest Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: #0a0e27;
            color: #e0e0e0;
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #00d4ff; margin-bottom: 10px; font-size: 2.5em; }}
        h2 {{ color: #00d4ff; margin-top: 30px; margin-bottom: 15px; border-bottom: 2px solid #1a2332; padding-bottom: 10px; }}
        h3 {{ color: #4dd4ff; margin-top: 20px; margin-bottom: 10px; }}
        .timestamp {{ color: #888; font-size: 0.9em; margin-bottom: 30px; }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #1a2332;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #00d4ff;
        }}
        .metric-label {{ color: #888; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.5px; }}
        .metric-value {{ font-size: 2em; color: #00d4ff; font-weight: bold; margin-top: 5px; }}
        .metric-value.positive {{ color: #00ff88; }}
        .metric-value.negative {{ color: #ff4444; }}
        .table-container {{ overflow-x: auto; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; background: #1a2332; border-radius: 8px; overflow: hidden; }}
        th {{ background: #0d1117; color: #00d4ff; padding: 15px; text-align: left; font-weight: 600; }}
        td {{ padding: 12px 15px; border-top: 1px solid #2a3342; }}
        tr:hover {{ background: #222b3d; }}
        .chart-container {{ background: #1a2332; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4444; }}
        footer {{ text-align: center; margin-top: 50px; padding: 20px; color: #666; border-top: 1px solid #2a3342; }}
    </style>
</head>"""

    def _build_summary_section(self, backtest_results: Dict[str, Any]) -> str:
        """Build summary metrics section.

        Args:
            backtest_results: Dictionary containing backtest metrics

        Returns:
            HTML summary section
        """
        total_return = backtest_results.get("total_return", Decimal("0"))
        total_trades = backtest_results.get("total_trades", 0)
        win_rate = backtest_results.get("win_rate", Decimal("0"))
        sharpe_ratio = backtest_results.get("sharpe_ratio", Decimal("0"))

        return_class = "positive" if total_return > 0 else "negative"

        return f"""
        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {return_class}">{total_return * 100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{total_trades}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{win_rate * 100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{sharpe_ratio:.3f}</div>
            </div>
        </div>"""

    def _build_metrics_section(self, backtest_results: Dict[str, Any]) -> str:
        """Build detailed metrics section.

        Args:
            backtest_results: Dictionary containing backtest metrics

        Returns:
            HTML metrics section
        """
        metrics_html = '<h2>Detailed Metrics</h2><div class="summary-grid">'

        # Extract all metrics and format them
        metrics_to_display = {
            "Max Drawdown": ("max_drawdown", "%", -100),
            "Profit Factor": ("profit_factor", "", 1),
            "Average Win": ("avg_win", "$", 1),
            "Average Loss": ("avg_loss", "$", 1),
            "Largest Win": ("largest_win", "$", 1),
            "Largest Loss": ("largest_loss", "$", 1),
            "Sortino Ratio": ("sortino_ratio", "", 1),
            "Calmar Ratio": ("calmar_ratio", "", 1),
        }

        for label, (key, suffix, multiplier) in metrics_to_display.items():
            value = backtest_results.get(key, Decimal("0"))
            formatted_value = f"{value * multiplier:.2f}{suffix}"

            metrics_html += f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{formatted_value}</div>
            </div>"""

        metrics_html += "</div>"
        return metrics_html

    async def _build_charts_section(
        self,
        equity_curve: pl.DataFrame,
        trades: pl.DataFrame
    ) -> str:
        """Build charts section with equity curve visualization.

        Args:
            equity_curve: Equity progression data
            trades: Trade history

        Returns:
            HTML charts section
        """
        # For production, this would generate interactive charts using Plotly or similar
        # Here we provide a placeholder structure that loads from config
        chart_height = self.config.get("chart_height", 400)

        return f"""
        <h2>Performance Charts</h2>
        <div class="chart-container">
            <h3>Equity Curve</h3>
            <div id="equity-chart" style="height: {chart_height}px;">
                <!-- Chart would be rendered here with JavaScript charting library -->
                <p style="text-align: center; padding: 50px; color: #666;">
                    Equity curve visualization ({equity_curve.height} data points)
                </p>
            </div>
        </div>
        <div class="chart-container">
            <h3>Drawdown</h3>
            <div id="drawdown-chart" style="height: {chart_height}px;">
                <p style="text-align: center; padding: 50px; color: #666;">
                    Drawdown visualization
                </p>
            </div>
        </div>"""

    def _build_trades_table(self, trades: pl.DataFrame) -> str:
        """Build trades history table.

        Args:
            trades: DataFrame with trade history

        Returns:
            HTML trades table
        """
        if trades.height == 0:
            return """
            <h2>Trade History</h2>
            <p style="color: #666; text-align: center; padding: 30px;">No trades executed</p>"""

        # Limit to recent trades for performance
        max_displayed_trades = self.config.get("max_displayed_trades", 100)
        recent_trades = trades.head(max_displayed_trades)

        table_html = """
        <h2>Trade History</h2>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Quantity</th>
                        <th>Price</th>
                        <th>P&L</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>"""

        # Convert to list of dicts for iteration
        trades_list = recent_trades.to_dicts()

        for trade in trades_list:
            pnl = trade.get("pnl", Decimal("0"))
            pnl_class = "positive" if pnl > 0 else "negative" if pnl < 0 else ""

            table_html += f"""
                    <tr>
                        <td>{trade.get('timestamp', '')}</td>
                        <td>{trade.get('symbol', '')}</td>
                        <td>{trade.get('side', '')}</td>
                        <td>{trade.get('quantity', '')}</td>
                        <td>${trade.get('price', Decimal('0')):.2f}</td>
                        <td class="{pnl_class}">${pnl:.2f}</td>
                        <td>{trade.get('status', '')}</td>
                    </tr>"""

        table_html += """
                </tbody>
            </table>
        </div>"""

        if trades.height > max_displayed_trades:
            table_html += f'<p style="color: #666; margin-top: 10px;">Showing {max_displayed_trades} of {trades.height} trades</p>'

        return table_html

    def _build_footer(self) -> str:
        """Build HTML footer.

        Returns:
            HTML footer section
        """
        return """
    <footer>
        <p>Quantum Trader AI - Institutional Trading Platform</p>
        <p>This report is for informational purposes only.</p>
    </footer>"""

    async def _write_report(self, output_path: Path, html_content: str) -> None:
        """Write HTML content to file asynchronously.

        Args:
            output_path: Path to output file
            html_content: HTML content to write

        Raises:
            IOError: If file write fails
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._write_file_sync,
                output_path,
                html_content
            )
            logger.debug("Report written to file", path=str(output_path))

        except Exception as e:
            logger.error("Failed to write report file", error=str(e), path=str(output_path))
            raise IOError(f"Failed to write report: {e}") from e

    def _write_file_sync(self, output_path: Path, content: str) -> None:
        """Synchronous file write operation.

        Args:
            output_path: Path to output file
            content: Content to write
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
