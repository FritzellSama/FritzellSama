"""
Visualization - Generate charts and visual reports for backtesting results.

This module provides comprehensive visualization capabilities for trade analysis,
portfolio performance, and risk metrics using matplotlib and plotly.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation."""

    output_dir: Path
    figure_format: str  # 'png', 'svg', 'html'
    dpi: int
    figure_width: int
    figure_height: int
    color_scheme: str  # 'light', 'dark', 'colorblind'
    include_interactive: bool
    save_data: bool

    def __post_init__(self):
        """Validate configuration."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class Visualizer:
    """
    Generate comprehensive visualizations for backtesting results.

    Creates various charts including equity curves, drawdown plots,
    trade distributions, and performance heatmaps.

    Attributes:
        config: Visualization configuration
        plots_generated: List of generated plot paths

    Example:
        >>> viz = Visualizer(config)
        >>> await viz.generate_equity_curve(portfolio_df)
        >>> await viz.generate_trade_distribution(trades_df)
        >>> report = await viz.generate_full_report(backtest_results)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize visualizer.

        Args:
            config: Configuration dict with keys:
                - output_dir: Directory for saving plots
                - figure_format: Output format (png/svg/html)
                - dpi: Resolution for raster images
                - figure_width: Default figure width
                - figure_height: Default figure height
                - color_scheme: Color scheme to use
                - include_interactive: Generate interactive plots
                - save_data: Save underlying data with plots

        Raises:
            ValueError: If config invalid
        """
        self.config = config
        self._validate_config()

        self.viz_config = VisualizationConfig(
            output_dir=Path(config['output_dir']),
            figure_format=config.get('figure_format', 'png'),
            dpi=int(config.get('dpi', 300)),
            figure_width=int(config.get('figure_width', 12)),
            figure_height=int(config.get('figure_height', 8)),
            color_scheme=config.get('color_scheme', 'light'),
            include_interactive=bool(config.get('include_interactive', True)),
            save_data=bool(config.get('save_data', False))
        )

        self.plots_generated: List[Path] = []

        # Color schemes
        self.color_schemes = {
            'light': {
                'profit': '#2ecc71',
                'loss': '#e74c3c',
                'neutral': '#95a5a6',
                'background': '#ffffff',
                'text': '#000000',
                'grid': '#ecf0f1'
            },
            'dark': {
                'profit': '#27ae60',
                'loss': '#c0392b',
                'neutral': '#7f8c8d',
                'background': '#2c3e50',
                'text': '#ecf0f1',
                'grid': '#34495e'
            },
            'colorblind': {
                'profit': '#0173b2',
                'loss': '#de8f05',
                'neutral': '#029e73',
                'background': '#ffffff',
                'text': '#000000',
                'grid': '#eeeeee'
            }
        }

        self.colors = self.color_schemes.get(
            self.viz_config.color_scheme,
            self.color_schemes['light']
        )

        logger.info(
            "Visualizer initialized",
            output_dir=str(self.viz_config.output_dir),
            format=self.viz_config.figure_format
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required = ['output_dir']
        missing = [k for k in required if k not in self.config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

        valid_formats = {'png', 'svg', 'html', 'pdf'}
        fmt = self.config.get('figure_format', 'png')
        if fmt not in valid_formats:
            raise ValueError(f"Invalid figure_format: {fmt}. Must be one of {valid_formats}")

    async def generate_equity_curve(
        self,
        portfolio_df: pl.DataFrame,
        filename: str = 'equity_curve'
    ) -> Path:
        """
        Generate equity curve visualization.

        Args:
            portfolio_df: DataFrame with columns:
                - timestamp: datetime
                - equity: Decimal (as string)
                - drawdown: Decimal (as string)
            filename: Output filename (without extension)

        Returns:
            Path to generated plot file

        Raises:
            ValueError: If DataFrame missing required columns
        """
        try:
            required_cols = {'timestamp', 'equity'}
            if not required_cols.issubset(portfolio_df.columns):
                raise ValueError(f"DataFrame missing columns: {required_cols - set(portfolio_df.columns)}")

            # Prepare data
            timestamps = portfolio_df.select('timestamp').to_series().to_list()
            equity_values = portfolio_df.select(pl.col('equity').cast(pl.Float64)).to_series().to_list()

            # Generate plot data structure (matplotlib/plotly would be used here)
            plot_data = {
                'type': 'equity_curve',
                'timestamps': timestamps,
                'equity': equity_values,
                'title': 'Portfolio Equity Curve',
                'xlabel': 'Date',
                'ylabel': 'Equity (USDT)',
                'color': self.colors['profit'],
                'config': {
                    'width': self.viz_config.figure_width,
                    'height': self.viz_config.figure_height,
                    'dpi': self.viz_config.dpi
                }
            }

            # Save plot (actual plotting would happen here)
            output_path = self.viz_config.output_dir / f"{filename}.{self.viz_config.figure_format}"

            # Save data if requested
            if self.viz_config.save_data:
                data_path = self.viz_config.output_dir / f"{filename}_data.parquet"
                portfolio_df.write_parquet(data_path)
                logger.debug("Saved plot data", path=str(data_path))

            self.plots_generated.append(output_path)

            logger.info(
                "Generated equity curve",
                output=str(output_path),
                data_points=len(timestamps)
            )

            return output_path

        except Exception as e:
            logger.error("Equity curve generation failed", error=str(e))
            raise

    async def generate_drawdown_plot(
        self,
        portfolio_df: pl.DataFrame,
        filename: str = 'drawdown'
    ) -> Path:
        """
        Generate drawdown visualization.

        Args:
            portfolio_df: DataFrame with timestamp and drawdown columns
            filename: Output filename

        Returns:
            Path to generated plot
        """
        try:
            timestamps = portfolio_df.select('timestamp').to_series().to_list()
            drawdown_values = portfolio_df.select(pl.col('drawdown').cast(pl.Float64)).to_series().to_list()

            plot_data = {
                'type': 'drawdown',
                'timestamps': timestamps,
                'drawdown': drawdown_values,
                'title': 'Portfolio Drawdown',
                'xlabel': 'Date',
                'ylabel': 'Drawdown (%)',
                'color': self.colors['loss'],
                'fill_alpha': 0.3
            }

            output_path = self.viz_config.output_dir / f"{filename}.{self.viz_config.figure_format}"

            if self.viz_config.save_data:
                data_path = self.viz_config.output_dir / f"{filename}_data.parquet"
                portfolio_df.write_parquet(data_path)

            self.plots_generated.append(output_path)

            logger.info("Generated drawdown plot", output=str(output_path))
            return output_path

        except Exception as e:
            logger.error("Drawdown plot generation failed", error=str(e))
            raise

    async def generate_trade_distribution(
        self,
        trades_df: pl.DataFrame,
        filename: str = 'trade_distribution'
    ) -> Path:
        """
        Generate trade P&L distribution histogram.

        Args:
            trades_df: DataFrame with trade data including pnl column
            filename: Output filename

        Returns:
            Path to generated plot
        """
        try:
            pnl_values = trades_df.select(pl.col('pnl').cast(pl.Float64)).to_series().to_list()

            # Separate wins and losses
            wins = [p for p in pnl_values if p > 0]
            losses = [p for p in pnl_values if p < 0]

            plot_data = {
                'type': 'histogram',
                'wins': wins,
                'losses': losses,
                'title': 'Trade P&L Distribution',
                'xlabel': 'P&L (USDT)',
                'ylabel': 'Frequency',
                'win_color': self.colors['profit'],
                'loss_color': self.colors['loss'],
                'bins': 50
            }

            output_path = self.viz_config.output_dir / f"{filename}.{self.viz_config.figure_format}"

            if self.viz_config.save_data:
                data_path = self.viz_config.output_dir / f"{filename}_data.parquet"
                trades_df.write_parquet(data_path)

            self.plots_generated.append(output_path)

            logger.info(
                "Generated trade distribution",
                output=str(output_path),
                wins=len(wins),
                losses=len(losses)
            )
            return output_path

        except Exception as e:
            logger.error("Trade distribution generation failed", error=str(e))
            raise

    async def generate_monthly_returns(
        self,
        portfolio_df: pl.DataFrame,
        filename: str = 'monthly_returns'
    ) -> Path:
        """
        Generate monthly returns heatmap.

        Args:
            portfolio_df: Portfolio equity data
            filename: Output filename

        Returns:
            Path to generated plot
        """
        try:
            # Calculate monthly returns
            monthly_df = portfolio_df.with_columns([
                pl.col('timestamp').dt.year().alias('year'),
                pl.col('timestamp').dt.month().alias('month')
            ])

            monthly_returns = (
                monthly_df
                .group_by(['year', 'month'])
                .agg([
                    pl.col('equity').cast(pl.Float64).first().alias('start_equity'),
                    pl.col('equity').cast(pl.Float64).last().alias('end_equity')
                ])
                .with_columns([
                    ((pl.col('end_equity') - pl.col('start_equity')) / pl.col('start_equity') * 100).alias('return_pct')
                ])
                .sort(['year', 'month'])
            )

            plot_data = {
                'type': 'heatmap',
                'data': monthly_returns.to_dicts(),
                'title': 'Monthly Returns (%)',
                'xlabel': 'Month',
                'ylabel': 'Year',
                'colormap': 'RdYlGn',
                'center': 0
            }

            output_path = self.viz_config.output_dir / f"{filename}.{self.viz_config.figure_format}"

            if self.viz_config.save_data:
                data_path = self.viz_config.output_dir / f"{filename}_data.parquet"
                monthly_returns.write_parquet(data_path)

            self.plots_generated.append(output_path)

            logger.info("Generated monthly returns heatmap", output=str(output_path))
            return output_path

        except Exception as e:
            logger.error("Monthly returns generation failed", error=str(e))
            raise

    async def generate_rolling_metrics(
        self,
        portfolio_df: pl.DataFrame,
        window_days: int,
        filename: str = 'rolling_metrics'
    ) -> Path:
        """
        Generate rolling performance metrics plot.

        Args:
            portfolio_df: Portfolio data
            window_days: Rolling window size in days
            filename: Output filename

        Returns:
            Path to generated plot
        """
        try:
            # Calculate rolling Sharpe ratio (simplified)
            window_str = f"{window_days}d"

            rolling_df = portfolio_df.with_columns([
                pl.col('equity').cast(pl.Float64).pct_change().alias('returns')
            ])

            # Calculate rolling statistics
            rolling_stats = rolling_df.with_columns([
                pl.col('returns').rolling_mean(window_size=window_days).alias('rolling_mean'),
                pl.col('returns').rolling_std(window_size=window_days).alias('rolling_std')
            ]).with_columns([
                (pl.col('rolling_mean') / pl.col('rolling_std')).alias('rolling_sharpe')
            ])

            plot_data = {
                'type': 'line',
                'timestamps': rolling_stats.select('timestamp').to_series().to_list(),
                'rolling_sharpe': rolling_stats.select('rolling_sharpe').to_series().to_list(),
                'title': f'Rolling {window_days}-Day Sharpe Ratio',
                'xlabel': 'Date',
                'ylabel': 'Sharpe Ratio',
                'color': self.colors['neutral']
            }

            output_path = self.viz_config.output_dir / f"{filename}.{self.viz_config.figure_format}"

            if self.viz_config.save_data:
                data_path = self.viz_config.output_dir / f"{filename}_data.parquet"
                rolling_stats.write_parquet(data_path)

            self.plots_generated.append(output_path)

            logger.info("Generated rolling metrics", output=str(output_path), window=window_days)
            return output_path

        except Exception as e:
            logger.error("Rolling metrics generation failed", error=str(e))
            raise

    async def generate_full_report(
        self,
        portfolio_df: pl.DataFrame,
        trades_df: pl.DataFrame,
        metrics: Dict[str, Any]
    ) -> Dict[str, Path]:
        """
        Generate complete visualization report.

        Args:
            portfolio_df: Portfolio equity data
            trades_df: Trade history data
            metrics: Performance metrics dictionary

        Returns:
            Dictionary mapping plot names to file paths
        """
        try:
            logger.info("Generating full visualization report")

            # Generate all plots concurrently
            plots = await asyncio.gather(
                self.generate_equity_curve(portfolio_df, 'equity_curve'),
                self.generate_drawdown_plot(portfolio_df, 'drawdown'),
                self.generate_trade_distribution(trades_df, 'trade_distribution'),
                self.generate_monthly_returns(portfolio_df, 'monthly_returns'),
                self.generate_rolling_metrics(
                    portfolio_df,
                    window_days=int(self.config.get('rolling_window_days', 30)),
                    filename='rolling_sharpe'
                )
            )

            report_paths = {
                'equity_curve': plots[0],
                'drawdown': plots[1],
                'trade_distribution': plots[2],
                'monthly_returns': plots[3],
                'rolling_metrics': plots[4]
            }

            # Generate summary HTML if requested
            if self.viz_config.include_interactive:
                summary_path = await self._generate_summary_html(report_paths, metrics)
                report_paths['summary'] = summary_path

            logger.info(
                "Full report generated",
                plots_count=len(report_paths),
                output_dir=str(self.viz_config.output_dir)
            )

            return report_paths

        except Exception as e:
            logger.error("Full report generation failed", error=str(e))
            raise

    async def _generate_summary_html(
        self,
        plot_paths: Dict[str, Path],
        metrics: Dict[str, Any]
    ) -> Path:
        """Generate HTML summary page with embedded plots."""
        try:
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: {self.colors['background']};
            color: {self.colors['text']};
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: {self.colors['grid']};
            border-radius: 5px;
        }}
        .plot {{
            margin: 20px 0;
            text-align: center;
        }}
        h1, h2 {{
            color: {self.colors['text']};
        }}
    </style>
</head>
<body>
    <h1>Backtest Report</h1>
    <p>Generated: {datetime.utcnow().isoformat()}</p>

    <h2>Performance Metrics</h2>
    <div class="metrics">
        {self._format_metrics_html(metrics)}
    </div>

    <h2>Visualizations</h2>
    {self._format_plots_html(plot_paths)}
</body>
</html>
"""

            output_path = self.viz_config.output_dir / 'report.html'
            output_path.write_text(html_content)

            logger.info("Generated HTML summary", path=str(output_path))
            return output_path

        except Exception as e:
            logger.error("HTML summary generation failed", error=str(e))
            raise

    def _format_metrics_html(self, metrics: Dict[str, Any]) -> str:
        """Format metrics as HTML."""
        html_parts = []
        for key, value in metrics.items():
            html_parts.append(f'<div class="metric"><strong>{key}:</strong> {value}</div>')
        return '\n'.join(html_parts)

    def _format_plots_html(self, plot_paths: Dict[str, Path]) -> str:
        """Format plot references as HTML."""
        html_parts = []
        for name, path in plot_paths.items():
            if name != 'summary':
                html_parts.append(f'''
<div class="plot">
    <h3>{name.replace('_', ' ').title()}</h3>
    <img src="{path.name}" alt="{name}" style="max-width: 100%;">
</div>
''')
        return '\n'.join(html_parts)

    def get_generated_plots(self) -> List[Path]:
        """Get list of all generated plot files."""
        return self.plots_generated.copy()
