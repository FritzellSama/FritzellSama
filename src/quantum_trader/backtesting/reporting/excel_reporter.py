"""
Excel Reporter - Generate Excel reports for backtest results.

This module creates comprehensive Excel reports with multiple sheets containing
performance metrics, trade logs, equity curves, and visualizations.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from pathlib import Path
import os

from structlog import get_logger
import polars as pl

logger = get_logger(__name__)


class ExcelReporter:
    """Generates Excel reports for backtest results.

    Attributes:
        config: Reporter configuration from environment
        output_dir: Output directory for reports
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Excel reporter.

        Args:
            config: Optional configuration override

        Raises:
            ValueError: If configuration invalid
        """
        self.config: Dict[str, Any] = config or self._load_config()
        self.output_dir: Path = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("ExcelReporter initialized", output_dir=str(self.output_dir))

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables.

        Returns:
            Configuration dictionary
        """
        try:
            config = {
                'output_dir': os.getenv('REPORT_OUTPUT_DIR', './reports'),
                'include_charts': os.getenv('REPORT_INCLUDE_CHARTS', 'true').lower() == 'true',
                'include_trade_log': os.getenv('REPORT_INCLUDE_TRADES', 'true').lower() == 'true',
                'decimal_places': int(os.getenv('REPORT_DECIMAL_PLACES', '2')),
                'date_format': os.getenv('REPORT_DATE_FORMAT', '%Y-%m-%d %H:%M:%S'),
            }

            logger.debug("Excel reporter config loaded", config=config)
            return config

        except Exception as e:
            logger.error("Failed to load config", error=str(e))
            raise ValueError(f"Configuration load failed: {e}")

    async def generate_report(
        self,
        backtest_results: Dict[str, Any],
        output_filename: Optional[str] = None
    ) -> str:
        """Generate comprehensive Excel report.

        Args:
            backtest_results: Backtest results dictionary
            output_filename: Optional custom filename

        Returns:
            Path to generated report file

        Example:
            >>> report_path = await reporter.generate_report(results)
            >>> print(f"Report saved to: {report_path}")
        """
        try:
            logger.info("Generating Excel report")

            # Generate filename if not provided
            if not output_filename:
                timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                output_filename = f"backtest_report_{timestamp}.xlsx"

            output_path = self.output_dir / output_filename

            # Prepare report data
            report_data = await self._prepare_report_data(backtest_results)

            # Write to Excel (in production would use openpyxl or xlsxwriter)
            await self._write_excel_file(output_path, report_data)

            logger.info("Excel report generated", path=str(output_path))

            return str(output_path)

        except Exception as e:
            logger.error("Failed to generate report", error=str(e))
            raise

    async def _prepare_report_data(
        self,
        backtest_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare data for Excel report.

        Args:
            backtest_results: Raw backtest results

        Returns:
            Formatted report data
        """
        try:
            logger.debug("Preparing report data")

            report_data = {
                'summary': await self._prepare_summary_sheet(backtest_results),
                'metrics': await self._prepare_metrics_sheet(backtest_results),
                'equity_curve': await self._prepare_equity_curve_sheet(backtest_results),
            }

            if self.config['include_trade_log']:
                report_data['trades'] = await self._prepare_trades_sheet(backtest_results)

            logger.debug("Report data prepared", sheets=len(report_data))

            return report_data

        except Exception as e:
            logger.error("Failed to prepare report data", error=str(e))
            raise

    async def _prepare_summary_sheet(
        self,
        backtest_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare summary sheet data.

        Args:
            backtest_results: Backtest results

        Returns:
            Summary sheet data
        """
        try:
            summary = {
                'Strategy Name': backtest_results.get('strategy_name', 'Unknown'),
                'Start Date': backtest_results.get('start_date', 'N/A'),
                'End Date': backtest_results.get('end_date', 'N/A'),
                'Initial Capital': str(backtest_results.get('initial_capital', Decimal('0'))),
                'Final Capital': str(backtest_results.get('final_capital', Decimal('0'))),
                'Total Return': str(backtest_results.get('total_return', Decimal('0'))),
                'Total Return %': str(backtest_results.get('total_return_pct', Decimal('0'))),
                'Sharpe Ratio': str(backtest_results.get('sharpe_ratio', Decimal('0'))),
                'Sortino Ratio': str(backtest_results.get('sortino_ratio', Decimal('0'))),
                'Max Drawdown': str(backtest_results.get('max_drawdown', Decimal('0'))),
                'Max Drawdown %': str(backtest_results.get('max_drawdown_pct', Decimal('0'))),
                'Win Rate': str(backtest_results.get('win_rate', Decimal('0'))),
                'Total Trades': backtest_results.get('total_trades', 0),
                'Profitable Trades': backtest_results.get('profitable_trades', 0),
                'Losing Trades': backtest_results.get('losing_trades', 0),
            }

            return summary

        except Exception as e:
            logger.error("Failed to prepare summary sheet", error=str(e))
            raise

    async def _prepare_metrics_sheet(
        self,
        backtest_results: Dict[str, Any]
    ) -> pl.DataFrame:
        """Prepare metrics sheet data.

        Args:
            backtest_results: Backtest results

        Returns:
            Metrics DataFrame
        """
        try:
            metrics_data = []

            # Performance metrics
            performance_metrics = backtest_results.get('performance_metrics', {})
            for metric_name, metric_value in performance_metrics.items():
                metrics_data.append({
                    'Category': 'Performance',
                    'Metric': metric_name,
                    'Value': str(metric_value)
                })

            # Risk metrics
            risk_metrics = backtest_results.get('risk_metrics', {})
            for metric_name, metric_value in risk_metrics.items():
                metrics_data.append({
                    'Category': 'Risk',
                    'Metric': metric_name,
                    'Value': str(metric_value)
                })

            # Trade metrics
            trade_metrics = backtest_results.get('trade_metrics', {})
            for metric_name, metric_value in trade_metrics.items():
                metrics_data.append({
                    'Category': 'Trading',
                    'Metric': metric_name,
                    'Value': str(metric_value)
                })

            metrics_df = pl.DataFrame(metrics_data) if metrics_data else pl.DataFrame({
                'Category': [],
                'Metric': [],
                'Value': []
            })

            return metrics_df

        except Exception as e:
            logger.error("Failed to prepare metrics sheet", error=str(e))
            raise

    async def _prepare_equity_curve_sheet(
        self,
        backtest_results: Dict[str, Any]
    ) -> pl.DataFrame:
        """Prepare equity curve sheet data.

        Args:
            backtest_results: Backtest results

        Returns:
            Equity curve DataFrame
        """
        try:
            equity_curve = backtest_results.get('equity_curve')

            if equity_curve is None or not isinstance(equity_curve, pl.DataFrame):
                # Return empty DataFrame with expected schema
                return pl.DataFrame({
                    'timestamp': [],
                    'portfolio_value': [],
                    'cash': [],
                    'return_pct': []
                })

            return equity_curve

        except Exception as e:
            logger.error("Failed to prepare equity curve sheet", error=str(e))
            raise

    async def _prepare_trades_sheet(
        self,
        backtest_results: Dict[str, Any]
    ) -> pl.DataFrame:
        """Prepare trades sheet data.

        Args:
            backtest_results: Backtest results

        Returns:
            Trades DataFrame
        """
        try:
            trades = backtest_results.get('trades')

            if trades is None or not isinstance(trades, pl.DataFrame):
                # Return empty DataFrame with expected schema
                return pl.DataFrame({
                    'timestamp': [],
                    'symbol': [],
                    'side': [],
                    'quantity': [],
                    'price': [],
                    'commission': [],
                    'pnl': []
                })

            return trades

        except Exception as e:
            logger.error("Failed to prepare trades sheet", error=str(e))
            raise

    async def _write_excel_file(
        self,
        output_path: Path,
        report_data: Dict[str, Any]
    ) -> None:
        """Write report data to Excel file.

        Args:
            output_path: Output file path
            report_data: Report data dictionary

        Note:
            In production, this would use openpyxl or xlsxwriter to create
            a formatted Excel file with multiple sheets, charts, and styling.
        """
        try:
            logger.debug("Writing Excel file", path=str(output_path))

            # Simulate Excel file creation
            # In production, would use:
            # import openpyxl
            # wb = openpyxl.Workbook()
            # ws = wb.active
            # ... populate sheets ...
            # wb.save(output_path)

            # For now, create a simple text representation
            with open(output_path, 'w') as f:
                f.write("BACKTEST REPORT\n")
                f.write("=" * 80 + "\n\n")

                # Write summary
                f.write("SUMMARY\n")
                f.write("-" * 80 + "\n")
                if 'summary' in report_data:
                    for key, value in report_data['summary'].items():
                        f.write(f"{key}: {value}\n")
                f.write("\n")

                # Write metrics
                f.write("METRICS\n")
                f.write("-" * 80 + "\n")
                if 'metrics' in report_data:
                    metrics_df = report_data['metrics']
                    if metrics_df.height > 0:
                        f.write(str(metrics_df))
                f.write("\n\n")

                # Write equity curve info
                f.write("EQUITY CURVE\n")
                f.write("-" * 80 + "\n")
                if 'equity_curve' in report_data:
                    equity_df = report_data['equity_curve']
                    f.write(f"Data points: {equity_df.height}\n")
                f.write("\n")

                # Write trades info
                if 'trades' in report_data:
                    f.write("TRADES\n")
                    f.write("-" * 80 + "\n")
                    trades_df = report_data['trades']
                    f.write(f"Total trades: {trades_df.height}\n")
                    f.write("\n")

            logger.debug("Excel file written successfully")

        except Exception as e:
            logger.error("Failed to write Excel file", error=str(e))
            raise

    async def generate_comparison_report(
        self,
        backtest_results_list: List[Dict[str, Any]],
        output_filename: Optional[str] = None
    ) -> str:
        """Generate comparison report for multiple backtest runs.

        Args:
            backtest_results_list: List of backtest results to compare
            output_filename: Optional custom filename

        Returns:
            Path to generated report file

        Example:
            >>> report_path = await reporter.generate_comparison_report([
            ...     results1, results2, results3
            ... ])
        """
        try:
            logger.info(
                "Generating comparison report",
                num_backtests=len(backtest_results_list)
            )

            # Generate filename if not provided
            if not output_filename:
                timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                output_filename = f"backtest_comparison_{timestamp}.xlsx"

            output_path = self.output_dir / output_filename

            # Prepare comparison data
            comparison_data = await self._prepare_comparison_data(backtest_results_list)

            # Write to Excel
            await self._write_excel_file(output_path, comparison_data)

            logger.info("Comparison report generated", path=str(output_path))

            return str(output_path)

        except Exception as e:
            logger.error("Failed to generate comparison report", error=str(e))
            raise

    async def _prepare_comparison_data(
        self,
        backtest_results_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare comparison data.

        Args:
            backtest_results_list: List of backtest results

        Returns:
            Comparison data dictionary
        """
        try:
            logger.debug("Preparing comparison data")

            # Extract key metrics from each backtest
            comparison_rows = []

            for i, results in enumerate(backtest_results_list):
                row = {
                    'Run': f"Run {i + 1}",
                    'Strategy': results.get('strategy_name', 'Unknown'),
                    'Total Return %': str(results.get('total_return_pct', Decimal('0'))),
                    'Sharpe Ratio': str(results.get('sharpe_ratio', Decimal('0'))),
                    'Max DD %': str(results.get('max_drawdown_pct', Decimal('0'))),
                    'Win Rate': str(results.get('win_rate', Decimal('0'))),
                    'Total Trades': results.get('total_trades', 0)
                }
                comparison_rows.append(row)

            comparison_df = pl.DataFrame(comparison_rows) if comparison_rows else pl.DataFrame()

            return {
                'comparison': comparison_df
            }

        except Exception as e:
            logger.error("Failed to prepare comparison data", error=str(e))
            raise
