"""
Trade Analyzer - Comprehensive trade analysis and metrics calculation.

This module provides detailed trade-level analysis including win/loss statistics,
profit factors, consecutive wins/losses, and trade distribution metrics.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class TradeMetrics:
    """Comprehensive trade-level metrics."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int

    total_pnl: Decimal
    total_wins: Decimal
    total_losses: Decimal

    average_win: Decimal
    average_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal

    win_rate: Decimal
    profit_factor: Decimal
    expectancy: Decimal

    max_consecutive_wins: int
    max_consecutive_losses: int
    current_streak: int
    current_streak_type: str  # 'win' or 'loss'

    average_trade_duration: timedelta
    average_win_duration: timedelta
    average_loss_duration: timedelta

    risk_reward_ratio: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal

    metadata: Dict[str, Any] = field(default_factory=dict)


class TradeAnalyzer:
    """
    Analyzes individual trades and calculates comprehensive metrics.

    This analyzer processes executed trades and provides detailed statistics
    for performance evaluation and strategy optimization.

    Attributes:
        config: Configuration dictionary with analysis parameters
        trades: DataFrame containing all trades
        risk_free_rate: Risk-free rate for Sharpe/Sortino calculations

    Example:
        >>> analyzer = TradeAnalyzer(config)
        >>> trades_df = pl.DataFrame({...})
        >>> metrics = analyzer.analyze_trades(trades_df)
        >>> print(f"Win rate: {metrics.win_rate}%")
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize trade analyzer.

        Args:
            config: Configuration dict with keys:
                - risk_free_rate: Annual risk-free rate (from config)
                - min_trades: Minimum trades for valid analysis (from config)
                - target_risk_reward: Target risk/reward ratio (from config)

        Raises:
            ValueError: If required config keys missing
        """
        self.config = config
        self._validate_config()

        self.risk_free_rate = Decimal(str(config.get('risk_free_rate', '0.02')))
        self.min_trades = int(config.get('min_trades', 30))
        self.target_risk_reward = Decimal(str(config.get('target_risk_reward', '2.0')))

        self.trades: Optional[pl.DataFrame] = None

        logger.info(
            "TradeAnalyzer initialized",
            risk_free_rate=str(self.risk_free_rate),
            min_trades=self.min_trades
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_keys = ['risk_free_rate', 'min_trades', 'target_risk_reward']
        missing = [k for k in required_keys if k not in self.config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

    def analyze_trades(self, trades_df: pl.DataFrame) -> TradeMetrics:
        """
        Analyze trades and calculate comprehensive metrics.

        Args:
            trades_df: DataFrame with columns:
                - trade_id: str
                - symbol: str
                - entry_time: datetime
                - exit_time: datetime
                - entry_price: Decimal (as string)
                - exit_price: Decimal (as string)
                - quantity: Decimal (as string)
                - side: str ('BUY' or 'SELL')
                - pnl: Decimal (as string)
                - fees: Decimal (as string)
                - strategy: str

        Returns:
            TradeMetrics object with all calculated metrics

        Raises:
            ValueError: If DataFrame has insufficient trades or invalid data
        """
        try:
            if trades_df.height < self.min_trades:
                logger.warning(
                    "Insufficient trades for analysis",
                    trade_count=trades_df.height,
                    min_required=self.min_trades
                )

            self.trades = trades_df

            # Calculate PnL statistics
            total_trades = trades_df.height

            # Separate winning, losing, and breakeven trades
            winning_trades_df = trades_df.filter(pl.col('pnl').cast(pl.Float64) > 0)
            losing_trades_df = trades_df.filter(pl.col('pnl').cast(pl.Float64) < 0)
            breakeven_trades_df = trades_df.filter(pl.col('pnl').cast(pl.Float64) == 0)

            winning_trades = winning_trades_df.height
            losing_trades = losing_trades_df.height
            breakeven_trades = breakeven_trades_df.height

            # Calculate PnL metrics
            total_pnl = self._sum_decimal_column(trades_df, 'pnl')
            total_wins = self._sum_decimal_column(winning_trades_df, 'pnl') if winning_trades > 0 else Decimal('0')
            total_losses = abs(self._sum_decimal_column(losing_trades_df, 'pnl')) if losing_trades > 0 else Decimal('0')

            average_win = total_wins / Decimal(str(winning_trades)) if winning_trades > 0 else Decimal('0')
            average_loss = total_losses / Decimal(str(losing_trades)) if losing_trades > 0 else Decimal('0')

            largest_win = self._max_decimal_column(winning_trades_df, 'pnl') if winning_trades > 0 else Decimal('0')
            largest_loss = abs(self._min_decimal_column(losing_trades_df, 'pnl')) if losing_trades > 0 else Decimal('0')

            # Calculate ratios
            win_rate = (Decimal(str(winning_trades)) / Decimal(str(total_trades)) * Decimal('100')) if total_trades > 0 else Decimal('0')
            profit_factor = total_wins / total_losses if total_losses > Decimal('0') else Decimal('0')
            expectancy = total_pnl / Decimal(str(total_trades)) if total_trades > 0 else Decimal('0')

            # Calculate consecutive streaks
            max_consec_wins, max_consec_losses, current_streak, streak_type = self._calculate_streaks(trades_df)

            # Calculate durations
            avg_duration, avg_win_duration, avg_loss_duration = self._calculate_durations(
                trades_df, winning_trades_df, losing_trades_df
            )

            # Calculate risk/reward ratio
            risk_reward = average_win / average_loss if average_loss > Decimal('0') else Decimal('0')

            # Calculate Sharpe and Sortino ratios
            sharpe = self._calculate_sharpe_ratio(trades_df)
            sortino = self._calculate_sortino_ratio(trades_df)

            metrics = TradeMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                breakeven_trades=breakeven_trades,
                total_pnl=total_pnl,
                total_wins=total_wins,
                total_losses=total_losses,
                average_win=average_win,
                average_loss=average_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                win_rate=win_rate,
                profit_factor=profit_factor,
                expectancy=expectancy,
                max_consecutive_wins=max_consec_wins,
                max_consecutive_losses=max_consec_losses,
                current_streak=current_streak,
                current_streak_type=streak_type,
                average_trade_duration=avg_duration,
                average_win_duration=avg_win_duration,
                average_loss_duration=avg_loss_duration,
                risk_reward_ratio=risk_reward,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                metadata={
                    'analysis_timestamp': datetime.utcnow(),
                    'meets_min_trades': total_trades >= self.min_trades,
                    'meets_target_rr': risk_reward >= self.target_risk_reward
                }
            )

            logger.info(
                "Trade analysis complete",
                total_trades=total_trades,
                win_rate=str(win_rate),
                profit_factor=str(profit_factor)
            )

            return metrics

        except Exception as e:
            logger.error("Trade analysis failed", error=str(e))
            raise

    def _sum_decimal_column(self, df: pl.DataFrame, column: str) -> Decimal:
        """Sum a column and return as Decimal."""
        if df.height == 0:
            return Decimal('0')
        total = df.select(pl.col(column).cast(pl.Float64).sum()).item()
        return Decimal(str(total))

    def _max_decimal_column(self, df: pl.DataFrame, column: str) -> Decimal:
        """Get max value from column as Decimal."""
        if df.height == 0:
            return Decimal('0')
        max_val = df.select(pl.col(column).cast(pl.Float64).max()).item()
        return Decimal(str(max_val))

    def _min_decimal_column(self, df: pl.DataFrame, column: str) -> Decimal:
        """Get min value from column as Decimal."""
        if df.height == 0:
            return Decimal('0')
        min_val = df.select(pl.col(column).cast(pl.Float64).min()).item()
        return Decimal(str(min_val))

    def _calculate_streaks(self, trades_df: pl.DataFrame) -> Tuple[int, int, int, str]:
        """
        Calculate consecutive win/loss streaks.

        Returns:
            Tuple of (max_consecutive_wins, max_consecutive_losses, current_streak, streak_type)
        """
        if trades_df.height == 0:
            return 0, 0, 0, 'none'

        # Convert PnL to win/loss flags
        pnl_values = trades_df.select(pl.col('pnl').cast(pl.Float64)).to_series().to_list()

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for pnl in pnl_values:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:  # breakeven
                current_wins = 0
                current_losses = 0

        # Determine current streak
        current_streak = current_wins if current_wins > 0 else current_losses
        streak_type = 'win' if current_wins > 0 else ('loss' if current_losses > 0 else 'none')

        return max_wins, max_losses, current_streak, streak_type

    def _calculate_durations(
        self,
        all_trades: pl.DataFrame,
        winning_trades: pl.DataFrame,
        losing_trades: pl.DataFrame
    ) -> Tuple[timedelta, timedelta, timedelta]:
        """Calculate average trade durations."""

        def calc_avg_duration(df: pl.DataFrame) -> timedelta:
            if df.height == 0:
                return timedelta(0)

            durations = []
            for row in df.iter_rows(named=True):
                entry = row['entry_time']
                exit_time = row['exit_time']
                if isinstance(entry, datetime) and isinstance(exit_time, datetime):
                    durations.append(exit_time - entry)

            if not durations:
                return timedelta(0)

            avg_seconds = sum(d.total_seconds() for d in durations) / len(durations)
            return timedelta(seconds=avg_seconds)

        avg_all = calc_avg_duration(all_trades)
        avg_wins = calc_avg_duration(winning_trades)
        avg_losses = calc_avg_duration(losing_trades)

        return avg_all, avg_wins, avg_losses

    def _calculate_sharpe_ratio(self, trades_df: pl.DataFrame) -> Decimal:
        """
        Calculate Sharpe ratio for trades.

        Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
        """
        if trades_df.height < 2:
            return Decimal('0')

        try:
            returns = trades_df.select(pl.col('pnl').cast(pl.Float64)).to_series().to_list()

            mean_return = sum(returns) / len(returns)

            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            std_dev = variance ** 0.5

            if std_dev == 0:
                return Decimal('0')

            # Annualized risk-free rate per trade (approximation)
            rf_per_trade = float(self.risk_free_rate) / Decimal('252')  # 252 trading days

            sharpe = (mean_return - rf_per_trade) / std_dev
            return Decimal(str(sharpe))

        except Exception as e:
            logger.warning("Sharpe ratio calculation failed", error=str(e))
            return Decimal('0')

    def _calculate_sortino_ratio(self, trades_df: pl.DataFrame) -> Decimal:
        """
        Calculate Sortino ratio (focuses on downside deviation).

        Sortino = (Mean Return - Risk Free Rate) / Downside Deviation
        """
        if trades_df.height < 2:
            return Decimal('0')

        try:
            returns = trades_df.select(pl.col('pnl').cast(pl.Float64)).to_series().to_list()

            mean_return = sum(returns) / len(returns)

            # Calculate downside deviation (only negative returns)
            downside_returns = [r for r in returns if r < 0]

            if not downside_returns:
                return Decimal('0')

            downside_variance = sum(r ** 2 for r in downside_returns) / len(downside_returns)
            downside_dev = downside_variance ** 0.5

            if downside_dev == 0:
                return Decimal('0')

            rf_per_trade = float(self.risk_free_rate) / Decimal('252')

            sortino = (mean_return - rf_per_trade) / downside_dev
            return Decimal(str(sortino))

        except Exception as e:
            logger.warning("Sortino ratio calculation failed", error=str(e))
            return Decimal('0')

    def get_trade_distribution(self) -> Dict[str, Any]:
        """
        Get trade distribution statistics by various dimensions.

        Returns:
            Dictionary containing:
                - by_symbol: Trade counts per symbol
                - by_strategy: Trade counts per strategy
                - by_hour: Trade counts per hour of day
                - by_day_of_week: Trade counts per day

        Raises:
            ValueError: If analyze_trades() not called yet
        """
        if self.trades is None:
            raise ValueError("Must call analyze_trades() first")

        try:
            # Distribution by symbol
            by_symbol = (
                self.trades
                .group_by('symbol')
                .agg([
                    pl.count('trade_id').alias('count'),
                    pl.col('pnl').cast(pl.Float64).sum().alias('total_pnl')
                ])
                .sort('count', descending=True)
            ).to_dicts()

            # Distribution by strategy
            by_strategy = (
                self.trades
                .group_by('strategy')
                .agg([
                    pl.count('trade_id').alias('count'),
                    pl.col('pnl').cast(pl.Float64).sum().alias('total_pnl')
                ])
                .sort('count', descending=True)
            ).to_dicts()

            # Distribution by hour
            trades_with_hour = self.trades.with_columns(
                pl.col('entry_time').dt.hour().alias('hour')
            )
            by_hour = (
                trades_with_hour
                .group_by('hour')
                .agg([
                    pl.count('trade_id').alias('count')
                ])
                .sort('hour')
            ).to_dicts()

            # Distribution by day of week
            trades_with_dow = self.trades.with_columns(
                pl.col('entry_time').dt.weekday().alias('day_of_week')
            )
            by_day = (
                trades_with_dow
                .group_by('day_of_week')
                .agg([
                    pl.count('trade_id').alias('count')
                ])
                .sort('day_of_week')
            ).to_dicts()

            return {
                'by_symbol': by_symbol,
                'by_strategy': by_strategy,
                'by_hour': by_hour,
                'by_day_of_week': by_day
            }

        except Exception as e:
            logger.error("Trade distribution calculation failed", error=str(e))
            raise

    async def analyze_trades_async(self, trades_df: pl.DataFrame) -> TradeMetrics:
        """
        Async version of analyze_trades for concurrent processing.

        Args:
            trades_df: Trade data DataFrame

        Returns:
            TradeMetrics object
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze_trades, trades_df)
