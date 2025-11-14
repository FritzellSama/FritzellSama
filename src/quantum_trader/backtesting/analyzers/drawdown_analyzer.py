"""
Drawdown Analyzer - Analyze drawdown metrics for trading strategies.

This module provides comprehensive drawdown analysis including maximum drawdown,
drawdown duration, recovery time, and drawdown distribution statistics.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
import os

from structlog import get_logger
import polars as pl

logger = get_logger(__name__)


class DrawdownAnalyzer:
    """Analyzes drawdown metrics for trading strategies.

    Attributes:
        config: Analyzer configuration from environment
        equity_curve: Strategy equity curve data
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize drawdown analyzer.

        Args:
            config: Optional configuration override

        Raises:
            ValueError: If configuration invalid
        """
        self.config: Dict[str, Any] = config or self._load_config()
        self.equity_curve: Optional[pl.DataFrame] = None

        logger.info("DrawdownAnalyzer initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables.

        Returns:
            Configuration dictionary
        """
        try:
            config = {
                'drawdown_threshold': Decimal(os.getenv('DRAWDOWN_THRESHOLD', '0.05')),
                'max_acceptable_drawdown': Decimal(os.getenv('MAX_ACCEPTABLE_DRAWDOWN', '0.25')),
                'calculate_underwater_curve': os.getenv('CALC_UNDERWATER_CURVE', 'true').lower() == 'true',
                'percentile_levels': [int(x) for x in os.getenv('DRAWDOWN_PERCENTILES', '25,50,75,90,95,99').split(',')],
            }

            logger.debug("Drawdown analyzer config loaded", config=config)
            return config

        except Exception as e:
            logger.error("Failed to load config", error=str(e))
            raise ValueError(f"Configuration load failed: {e}")

    async def analyze_drawdowns(
        self,
        equity_curve: pl.DataFrame,
        equity_column: str = 'portfolio_value'
    ) -> Dict[str, Any]:
        """Analyze comprehensive drawdown metrics.

        Args:
            equity_curve: DataFrame with equity curve
            equity_column: Name of equity value column

        Returns:
            Dictionary with drawdown metrics

        Example:
            >>> metrics = await analyzer.analyze_drawdowns(equity_df)
            >>> print(f"Max DD: {metrics['max_drawdown']}")
        """
        try:
            logger.info("Analyzing drawdowns", rows=equity_curve.height)

            self.equity_curve = equity_curve

            # Calculate maximum drawdown
            max_dd = await self.calculate_max_drawdown(equity_curve, equity_column)

            # Calculate drawdown duration
            max_dd_duration = await self.calculate_max_drawdown_duration(
                equity_curve,
                equity_column
            )

            # Calculate recovery time
            recovery_time = await self.calculate_recovery_time(equity_curve, equity_column)

            # Get all drawdown periods
            drawdown_periods = await self.identify_drawdown_periods(
                equity_curve,
                equity_column
            )

            # Calculate statistics
            drawdown_stats = await self._calculate_drawdown_statistics(drawdown_periods)

            # Calculate underwater curve
            underwater_curve = None
            if self.config['calculate_underwater_curve']:
                underwater_curve = await self.calculate_underwater_curve(
                    equity_curve,
                    equity_column
                )

            metrics = {
                'max_drawdown': max_dd['max_drawdown'],
                'max_drawdown_percent': max_dd['max_drawdown_percent'],
                'max_drawdown_start': max_dd['start_timestamp'],
                'max_drawdown_end': max_dd['end_timestamp'],
                'max_drawdown_duration_days': max_dd_duration,
                'recovery_time_days': recovery_time,
                'num_drawdown_periods': len(drawdown_periods),
                'avg_drawdown': drawdown_stats['avg_drawdown'],
                'avg_drawdown_duration': drawdown_stats['avg_duration'],
                'drawdown_percentiles': drawdown_stats['percentiles'],
                'underwater_curve': underwater_curve
            }

            logger.info(
                "Drawdown analysis complete",
                max_dd=str(metrics['max_drawdown_percent']),
                num_periods=len(drawdown_periods)
            )

            return metrics

        except Exception as e:
            logger.error("Drawdown analysis failed", error=str(e))
            raise

    async def calculate_max_drawdown(
        self,
        equity_curve: pl.DataFrame,
        equity_column: str = 'portfolio_value'
    ) -> Dict[str, Any]:
        """Calculate maximum drawdown.

        Args:
            equity_curve: Equity curve data
            equity_column: Name of equity column

        Returns:
            Dictionary with max drawdown info

        Example:
            >>> max_dd = await analyzer.calculate_max_drawdown(equity_df)
            >>> max_dd['max_drawdown_percent']
            Decimal('-15.5')
        """
        try:
            logger.debug("Calculating maximum drawdown")

            if equity_curve.height == 0:
                raise ValueError("Empty equity curve")

            # Calculate running maximum
            equity_with_max = equity_curve.with_columns(
                pl.col(equity_column).cum_max().alias('running_max')
            )

            # Calculate drawdown
            equity_with_dd = equity_with_max.with_columns(
                (pl.col(equity_column) - pl.col('running_max')).alias('drawdown'),
                ((pl.col(equity_column) - pl.col('running_max')) / pl.col('running_max') * 100).alias('drawdown_pct')
            )

            # Find maximum drawdown
            max_dd_row = equity_with_dd.filter(
                pl.col('drawdown') == pl.col('drawdown').min()
            ).row(0, named=True)

            max_dd_value = Decimal(str(max_dd_row['drawdown']))
            max_dd_percent = Decimal(str(max_dd_row['drawdown_pct']))

            # Find start of this drawdown period (when running_max was hit)
            max_dd_idx = equity_with_dd.filter(
                pl.col('drawdown') == pl.col('drawdown').min()
            ).select(pl.col('timestamp')).to_series()[0]

            # Find the peak before this drawdown
            peak_value = Decimal(str(max_dd_row['running_max']))
            start_timestamp = equity_with_dd.filter(
                pl.col(equity_column) == float(peak_value)
            ).filter(
                pl.col('timestamp') <= max_dd_idx
            ).select(pl.col('timestamp')).to_series()[-1]

            result = {
                'max_drawdown': max_dd_value,
                'max_drawdown_percent': max_dd_percent,
                'start_timestamp': start_timestamp,
                'end_timestamp': max_dd_row['timestamp'],
                'peak_value': peak_value,
                'trough_value': Decimal(str(max_dd_row[equity_column]))
            }

            logger.debug(
                "Max drawdown calculated",
                dd_percent=str(max_dd_percent)
            )

            return result

        except Exception as e:
            logger.error("Max drawdown calculation failed", error=str(e))
            raise

    async def calculate_max_drawdown_duration(
        self,
        equity_curve: pl.DataFrame,
        equity_column: str = 'portfolio_value'
    ) -> Decimal:
        """Calculate maximum drawdown duration in days.

        Args:
            equity_curve: Equity curve data
            equity_column: Name of equity column

        Returns:
            Maximum drawdown duration in days

        Example:
            >>> duration = await analyzer.calculate_max_drawdown_duration(equity_df)
            >>> duration
            Decimal('45.5')
        """
        try:
            logger.debug("Calculating max drawdown duration")

            max_dd = await self.calculate_max_drawdown(equity_curve, equity_column)

            start = max_dd['start_timestamp']
            end = max_dd['end_timestamp']

            if isinstance(start, datetime) and isinstance(end, datetime):
                duration_seconds = (end - start).total_seconds()
                duration_days = Decimal(str(duration_seconds)) / Decimal('86400')
            else:
                duration_days = Decimal('0')

            logger.debug("Max DD duration calculated", days=str(duration_days))

            return duration_days

        except Exception as e:
            logger.error("Max DD duration calculation failed", error=str(e))
            raise

    async def calculate_recovery_time(
        self,
        equity_curve: pl.DataFrame,
        equity_column: str = 'portfolio_value'
    ) -> Optional[Decimal]:
        """Calculate recovery time from maximum drawdown.

        Args:
            equity_curve: Equity curve data
            equity_column: Name of equity column

        Returns:
            Recovery time in days, or None if not recovered

        Example:
            >>> recovery = await analyzer.calculate_recovery_time(equity_df)
            >>> recovery
            Decimal('30.0')
        """
        try:
            logger.debug("Calculating recovery time")

            max_dd = await self.calculate_max_drawdown(equity_curve, equity_column)

            peak_value = max_dd['peak_value']
            trough_timestamp = max_dd['end_timestamp']

            # Find when equity recovers to peak level
            recovery_data = equity_curve.filter(
                (pl.col('timestamp') > trough_timestamp) &
                (pl.col(equity_column) >= float(peak_value))
            )

            if recovery_data.height == 0:
                logger.debug("No recovery from max drawdown yet")
                return None

            recovery_timestamp = recovery_data.select('timestamp').to_series()[0]

            if isinstance(trough_timestamp, datetime) and isinstance(recovery_timestamp, datetime):
                recovery_seconds = (recovery_timestamp - trough_timestamp).total_seconds()
                recovery_days = Decimal(str(recovery_seconds)) / Decimal('86400')
            else:
                recovery_days = Decimal('0')

            logger.debug("Recovery time calculated", days=str(recovery_days))

            return recovery_days

        except Exception as e:
            logger.error("Recovery time calculation failed", error=str(e))
            raise

    async def identify_drawdown_periods(
        self,
        equity_curve: pl.DataFrame,
        equity_column: str = 'portfolio_value'
    ) -> List[Dict[str, Any]]:
        """Identify all drawdown periods.

        Args:
            equity_curve: Equity curve data
            equity_column: Name of equity column

        Returns:
            List of drawdown period dictionaries

        Example:
            >>> periods = await analyzer.identify_drawdown_periods(equity_df)
            >>> len(periods)
            12
        """
        try:
            logger.debug("Identifying drawdown periods")

            # Calculate running maximum
            equity_with_max = equity_curve.with_columns(
                pl.col(equity_column).cum_max().alias('running_max')
            )

            # Calculate drawdown
            equity_with_dd = equity_with_max.with_columns(
                ((pl.col(equity_column) - pl.col('running_max')) / pl.col('running_max')).alias('dd_pct')
            )

            # Identify when in drawdown
            equity_with_dd = equity_with_dd.with_columns(
                (pl.col('dd_pct') < 0).alias('in_drawdown')
            )

            periods: List[Dict[str, Any]] = []
            in_period = False
            period_start_idx = 0

            for i, row in enumerate(equity_with_dd.iter_rows(named=True)):
                if row['in_drawdown'] and not in_period:
                    # Start of new drawdown period
                    in_period = True
                    period_start_idx = i

                elif not row['in_drawdown'] and in_period:
                    # End of drawdown period
                    in_period = False

                    # Extract period data
                    period_data = equity_with_dd.slice(period_start_idx, i - period_start_idx + 1)

                    # Find max drawdown in this period
                    min_dd = period_data['dd_pct'].min()

                    start_ts = period_data['timestamp'][0]
                    end_ts = period_data['timestamp'][-1]

                    if isinstance(start_ts, datetime) and isinstance(end_ts, datetime):
                        duration = (end_ts - start_ts).total_seconds() / 86400
                    else:
                        duration = 0

                    periods.append({
                        'start_timestamp': start_ts,
                        'end_timestamp': end_ts,
                        'max_drawdown_pct': Decimal(str(min_dd * 100)),
                        'duration_days': Decimal(str(duration))
                    })

            logger.debug("Drawdown periods identified", count=len(periods))

            return periods

        except Exception as e:
            logger.error("Drawdown period identification failed", error=str(e))
            raise

    async def calculate_underwater_curve(
        self,
        equity_curve: pl.DataFrame,
        equity_column: str = 'portfolio_value'
    ) -> pl.DataFrame:
        """Calculate underwater curve (% below peak).

        Args:
            equity_curve: Equity curve data
            equity_column: Name of equity column

        Returns:
            DataFrame with timestamp and underwater percentage

        Example:
            >>> underwater = await analyzer.calculate_underwater_curve(equity_df)
            >>> underwater.columns
            ['timestamp', 'underwater_pct']
        """
        try:
            logger.debug("Calculating underwater curve")

            # Calculate running maximum
            equity_with_max = equity_curve.with_columns(
                pl.col(equity_column).cum_max().alias('running_max')
            )

            # Calculate underwater percentage
            underwater = equity_with_max.with_columns(
                ((pl.col(equity_column) - pl.col('running_max')) / pl.col('running_max') * 100).alias('underwater_pct')
            ).select(['timestamp', 'underwater_pct'])

            logger.debug("Underwater curve calculated")

            return underwater

        except Exception as e:
            logger.error("Underwater curve calculation failed", error=str(e))
            raise

    async def _calculate_drawdown_statistics(
        self,
        drawdown_periods: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate statistics across all drawdown periods.

        Args:
            drawdown_periods: List of drawdown periods

        Returns:
            Statistics dictionary
        """
        try:
            if not drawdown_periods:
                return {
                    'avg_drawdown': Decimal('0'),
                    'avg_duration': Decimal('0'),
                    'percentiles': {}
                }

            # Extract metrics
            drawdowns = [p['max_drawdown_pct'] for p in drawdown_periods]
            durations = [p['duration_days'] for p in drawdown_periods]

            # Calculate averages
            avg_dd = sum(drawdowns) / Decimal(len(drawdowns))
            avg_duration = sum(durations) / Decimal(len(durations))

            # Calculate percentiles
            sorted_dds = sorted([float(dd) for dd in drawdowns])
            percentiles = {}

            for pct in self.config['percentile_levels']:
                idx = int(len(sorted_dds) * pct / 100)
                idx = min(idx, len(sorted_dds) - 1)
                percentiles[f'p{pct}'] = Decimal(str(sorted_dds[idx]))

            return {
                'avg_drawdown': avg_dd,
                'avg_duration': avg_duration,
                'percentiles': percentiles
            }

        except Exception as e:
            logger.error("Failed to calculate drawdown statistics", error=str(e))
            raise

    async def check_drawdown_limits(
        self,
        equity_curve: pl.DataFrame,
        equity_column: str = 'portfolio_value'
    ) -> Tuple[bool, str]:
        """Check if drawdown exceeds acceptable limits.

        Args:
            equity_curve: Equity curve data
            equity_column: Name of equity column

        Returns:
            Tuple of (within_limits, message)

        Example:
            >>> within_limits, msg = await analyzer.check_drawdown_limits(equity_df)
            >>> if not within_limits:
            ...     print(msg)
        """
        try:
            logger.debug("Checking drawdown limits")

            max_dd = await self.calculate_max_drawdown(equity_curve, equity_column)
            max_dd_pct = abs(max_dd['max_drawdown_percent'])

            max_acceptable = self.config['max_acceptable_drawdown'] * Decimal('100')

            within_limits = max_dd_pct <= max_acceptable

            if within_limits:
                message = f"Drawdown {max_dd_pct}% within limit {max_acceptable}%"
            else:
                message = f"Drawdown {max_dd_pct}% exceeds limit {max_acceptable}%"

            logger.debug("Drawdown limit check", within_limits=within_limits)

            return within_limits, message

        except Exception as e:
            logger.error("Drawdown limit check failed", error=str(e))
            raise
