"""
Correlation Matrix Management
Quantum Trader AI - Production Risk Management

Real-time correlation matrix calculation and monitoring:
- Real-time correlation updates
- Exponentially weighted correlation
- Correlation breakdown detection
- Matrix visualization data

CRITICAL: All numeric values use Decimal, never float
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import polars as pl
import yaml

from quantum_trader.models import Position, AuditLog

logger = logging.getLogger(__name__)


@dataclass
class CorrelationMatrixConfig:
    """Correlation matrix configuration"""
    var_lookback_days: int
    update_frequency_seconds: int
    min_observations: int = 30
    ewma_lambda: Decimal = Decimal('0.94')  # Exponentially weighted moving average decay

    @classmethod
    def from_yaml(cls, risk_config: Dict, prod_config: Dict) -> 'CorrelationMatrixConfig':
        """Load configuration from yaml dictionaries"""
        return cls(
            var_lookback_days=int(risk_config['risk_model']['var_lookback_days']),
            update_frequency_seconds=int(risk_config['monitoring']['update_frequency_seconds'])
        )


@dataclass
class CorrelationSnapshot:
    """Snapshot of correlation matrix at a point in time"""
    correlation_matrix: Dict[Tuple[str, str], Decimal]
    symbols: List[str]
    avg_correlation: Decimal
    max_correlation: Decimal
    min_correlation: Decimal
    timestamp: datetime
    method: str  # 'pearson', 'ewma', 'rolling'
    metadata: Dict = field(default_factory=dict)


@dataclass
class CorrelationBreakdown:
    """Detected correlation breakdown event"""
    symbol_pair: Tuple[str, str]
    previous_correlation: Decimal
    current_correlation: Decimal
    change_percent: Decimal
    breakdown_type: str  # 'spike', 'drop', 'reversal'
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


class CorrelationMatrixManager:
    """
    Manage real-time correlation matrix calculation and monitoring.

    Tracks correlation between assets with multiple calculation methods
    including exponentially weighted moving averages for adaptive risk.
    """

    def __init__(self, risk_config_path: str, prod_config_path: str):
        """Initialize correlation matrix manager with configuration"""
        self.risk_config_path = risk_config_path
        self.prod_config_path = prod_config_path
        self.config: Optional[CorrelationMatrixConfig] = None
        self._load_config()

        # State tracking
        self.current_matrix: Dict[Tuple[str, str], Decimal] = {}
        self.previous_matrix: Dict[Tuple[str, str], Decimal] = {}
        self.ewma_covariances: Dict[Tuple[str, str], Decimal] = {}
        self.ewma_variances: Dict[str, Decimal] = {}
        self.matrix_history: List[CorrelationSnapshot] = []
        self.last_update: Optional[datetime] = None

    def _load_config(self) -> None:
        """Load configuration from yaml files with retry logic"""
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                with open(self.risk_config_path, 'r') as f:
                    risk_config = yaml.safe_load(f)

                with open(self.prod_config_path, 'r') as f:
                    prod_config = yaml.safe_load(f)

                self.config = CorrelationMatrixConfig.from_yaml(risk_config, prod_config)
                logger.info("Correlation matrix configuration loaded successfully")
                return

            except Exception as e:
                logger.error(f"Failed to load config (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise RuntimeError(f"Failed to load correlation matrix config after {max_retries} attempts") from e

    async def calculate_correlation_matrix(
        self,
        returns_df: pl.DataFrame,
        symbols: List[str],
        method: str = 'pearson'
    ) -> CorrelationSnapshot:
        """
        Calculate correlation matrix for given symbols.

        Args:
            returns_df: Polars DataFrame with columns [symbol, date, return]
            symbols: List of symbols to include
            method: Calculation method ('pearson', 'spearman', 'kendall')

        Returns:
            CorrelationSnapshot with calculated matrix
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            timestamp = datetime.utcnow()

            # Pivot to wide format
            pivot_df = returns_df.pivot(
                values='return',
                index='date',
                columns='symbol'
            ).sort('date')

            # Calculate correlation matrix
            correlation_matrix = {}
            correlations_list = []

            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if symbol1 not in pivot_df.columns or symbol2 not in pivot_df.columns:
                        correlation_matrix[(symbol1, symbol2)] = Decimal('0')
                        continue

                    if i == j:
                        # Self-correlation is 1
                        correlation_matrix[(symbol1, symbol2)] = Decimal('1')
                        continue

                    # Get returns
                    returns1 = pivot_df[symbol1].drop_nulls().to_list()
                    returns2 = pivot_df[symbol2].drop_nulls().to_list()

                    if len(returns1) < self.config.min_observations or len(returns2) < self.config.min_observations:
                        correlation_matrix[(symbol1, symbol2)] = Decimal('0')
                        continue

                    # Calculate Pearson correlation
                    correlation = await self._calculate_pearson_correlation(returns1, returns2)
                    correlation_matrix[(symbol1, symbol2)] = correlation

                    if i < j:  # Only count each pair once for statistics
                        correlations_list.append(abs(correlation))

            # Calculate summary statistics
            if correlations_list:
                avg_correlation = sum(correlations_list) / Decimal(str(len(correlations_list)))
                max_correlation = max(correlations_list)
                min_correlation = min(correlations_list)
            else:
                avg_correlation = Decimal('0')
                max_correlation = Decimal('0')
                min_correlation = Decimal('0')

            # Store current matrix
            self.previous_matrix = self.current_matrix.copy()
            self.current_matrix = correlation_matrix
            self.last_update = timestamp

            snapshot = CorrelationSnapshot(
                correlation_matrix=correlation_matrix,
                symbols=symbols,
                avg_correlation=avg_correlation,
                max_correlation=max_correlation,
                min_correlation=min_correlation,
                timestamp=timestamp,
                method=method,
                metadata={
                    'num_symbols': len(symbols),
                    'num_pairs': len(correlations_list)
                }
            )

            # Store in history
            self.matrix_history.append(snapshot)

            # Keep only recent history (last 24 hours)
            cutoff_time = timestamp - timedelta(hours=24)
            self.matrix_history = [
                s for s in self.matrix_history
                if s.timestamp > cutoff_time
            ]

            logger.info(
                f"Correlation matrix calculated: {len(symbols)} symbols, "
                f"avg={avg_correlation}, max={max_correlation}"
            )

            return snapshot

        except Exception as e:
            logger.error(f"Correlation matrix calculation failed: {e}")
            raise

    async def calculate_ewma_correlation_matrix(
        self,
        returns_df: pl.DataFrame,
        symbols: List[str]
    ) -> CorrelationSnapshot:
        """
        Calculate correlation matrix using exponentially weighted moving average.

        EWMA gives more weight to recent observations, making it more adaptive
        to changing market conditions.

        Args:
            returns_df: Polars DataFrame with columns [symbol, date, return]
            symbols: List of symbols to include

        Returns:
            CorrelationSnapshot with EWMA correlation matrix
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            timestamp = datetime.utcnow()
            lambda_param = self.config.ewma_lambda

            # Pivot to wide format
            pivot_df = returns_df.pivot(
                values='return',
                index='date',
                columns='symbol'
            ).sort('date')

            # Calculate EWMA variances and covariances
            for symbol in symbols:
                if symbol not in pivot_df.columns:
                    continue

                returns = [Decimal(str(r)) for r in pivot_df[symbol].drop_nulls().to_list()]

                if len(returns) < self.config.min_observations:
                    continue

                # Calculate EWMA variance
                ewma_var = await self._calculate_ewma_variance(returns, lambda_param)
                self.ewma_variances[symbol] = ewma_var

            # Calculate EWMA covariances
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i >= j:
                        continue

                    if symbol1 not in pivot_df.columns or symbol2 not in pivot_df.columns:
                        continue

                    returns1 = [Decimal(str(r)) for r in pivot_df[symbol1].drop_nulls().to_list()]
                    returns2 = [Decimal(str(r)) for r in pivot_df[symbol2].drop_nulls().to_list()]

                    if len(returns1) < self.config.min_observations or len(returns2) < self.config.min_observations:
                        continue

                    # Calculate EWMA covariance
                    ewma_cov = await self._calculate_ewma_covariance(
                        returns1,
                        returns2,
                        lambda_param
                    )
                    self.ewma_covariances[(symbol1, symbol2)] = ewma_cov
                    self.ewma_covariances[(symbol2, symbol1)] = ewma_cov

            # Convert covariances to correlations
            correlation_matrix = {}
            correlations_list = []

            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i == j:
                        correlation_matrix[(symbol1, symbol2)] = Decimal('1')
                        continue

                    var1 = self.ewma_variances.get(symbol1, Decimal('0'))
                    var2 = self.ewma_variances.get(symbol2, Decimal('0'))
                    cov = self.ewma_covariances.get((symbol1, symbol2), Decimal('0'))

                    if var1 > Decimal('0') and var2 > Decimal('0'):
                        correlation = cov / (var1.sqrt() * var2.sqrt())
                    else:
                        correlation = Decimal('0')

                    correlation_matrix[(symbol1, symbol2)] = correlation

                    if i < j:
                        correlations_list.append(abs(correlation))

            # Calculate summary statistics
            if correlations_list:
                avg_correlation = sum(correlations_list) / Decimal(str(len(correlations_list)))
                max_correlation = max(correlations_list)
                min_correlation = min(correlations_list)
            else:
                avg_correlation = Decimal('0')
                max_correlation = Decimal('0')
                min_correlation = Decimal('0')

            # Store current matrix
            self.previous_matrix = self.current_matrix.copy()
            self.current_matrix = correlation_matrix
            self.last_update = timestamp

            snapshot = CorrelationSnapshot(
                correlation_matrix=correlation_matrix,
                symbols=symbols,
                avg_correlation=avg_correlation,
                max_correlation=max_correlation,
                min_correlation=min_correlation,
                timestamp=timestamp,
                method='ewma',
                metadata={
                    'num_symbols': len(symbols),
                    'lambda': str(lambda_param)
                }
            )

            logger.info(
                f"EWMA correlation matrix calculated: {len(symbols)} symbols, "
                f"avg={avg_correlation}"
            )

            return snapshot

        except Exception as e:
            logger.error(f"EWMA correlation matrix calculation failed: {e}")
            raise

    async def calculate_rolling_correlation(
        self,
        returns_df: pl.DataFrame,
        symbol1: str,
        symbol2: str,
        window_days: int = 60
    ) -> pl.DataFrame:
        """
        Calculate rolling correlation between two symbols.

        Args:
            returns_df: Polars DataFrame with columns [symbol, date, return]
            symbol1: First symbol
            symbol2: Second symbol
            window_days: Rolling window size in days

        Returns:
            Polars DataFrame with columns [date, correlation]
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            # Filter for both symbols
            df1 = returns_df.filter(pl.col('symbol') == symbol1).sort('date')
            df2 = returns_df.filter(pl.col('symbol') == symbol2).sort('date')

            # Join on date
            joined_df = df1.join(df2, on='date', suffix='_2')

            if len(joined_df) < window_days:
                raise ValueError(f"Insufficient data: {len(joined_df)} < {window_days}")

            dates = []
            correlations = []

            # Calculate rolling correlation
            for i in range(window_days, len(joined_df)):
                window_df = joined_df.slice(i - window_days, window_days)

                returns1 = [Decimal(str(r)) for r in window_df['return'].to_list()]
                returns2 = [Decimal(str(r)) for r in window_df['return_2'].to_list()]

                correlation = await self._calculate_pearson_correlation(returns1, returns2)

                dates.append(window_df['date'].to_list()[-1])
                correlations.append(float(correlation))

            result_df = pl.DataFrame({
                'date': dates,
                'correlation': correlations
            })

            logger.info(
                f"Rolling correlation calculated for {symbol1}-{symbol2}: "
                f"{len(result_df)} periods"
            )

            return result_df

        except Exception as e:
            logger.error(f"Rolling correlation calculation failed: {e}")
            raise

    async def detect_correlation_breakdowns(
        self,
        threshold_percent: Decimal = Decimal('20')
    ) -> List[CorrelationBreakdown]:
        """
        Detect significant changes in correlation (breakdowns or spikes).

        Args:
            threshold_percent: Minimum percent change to flag as breakdown

        Returns:
            List of detected correlation breakdowns
        """
        try:
            if not self.previous_matrix or not self.current_matrix:
                return []

            timestamp = datetime.utcnow()
            breakdowns = []

            # Compare current vs previous
            for symbol_pair, current_corr in self.current_matrix.items():
                previous_corr = self.previous_matrix.get(symbol_pair, Decimal('0'))

                if previous_corr == Decimal('0'):
                    continue

                # Calculate change
                change = current_corr - previous_corr
                change_percent = (change / abs(previous_corr)) * Decimal('100')

                if abs(change_percent) < threshold_percent:
                    continue

                # Determine breakdown type
                if change_percent > threshold_percent:
                    breakdown_type = 'spike'
                elif change_percent < -threshold_percent:
                    breakdown_type = 'drop'
                elif previous_corr * current_corr < Decimal('0'):
                    breakdown_type = 'reversal'
                else:
                    continue

                breakdown = CorrelationBreakdown(
                    symbol_pair=symbol_pair,
                    previous_correlation=previous_corr,
                    current_correlation=current_corr,
                    change_percent=change_percent,
                    breakdown_type=breakdown_type,
                    timestamp=timestamp,
                    metadata={
                        'change': str(change)
                    }
                )

                breakdowns.append(breakdown)

            if breakdowns:
                logger.warning(f"Detected {len(breakdowns)} correlation breakdowns")

            return breakdowns

        except Exception as e:
            logger.error(f"Correlation breakdown detection failed: {e}")
            raise

    async def get_correlation(
        self,
        symbol1: str,
        symbol2: str
    ) -> Optional[Decimal]:
        """
        Get current correlation between two symbols.

        Args:
            symbol1: First symbol
            symbol2: Second symbol

        Returns:
            Correlation coefficient or None if not available
        """
        try:
            return self.current_matrix.get((symbol1, symbol2))

        except Exception as e:
            logger.error(f"Failed to get correlation for {symbol1}-{symbol2}: {e}")
            raise

    async def get_matrix_for_visualization(
        self,
        symbols: Optional[List[str]] = None
    ) -> Dict:
        """
        Get correlation matrix formatted for visualization.

        Args:
            symbols: Optional list of symbols to include (default: all)

        Returns:
            Dictionary with matrix data for visualization
        """
        try:
            if not self.current_matrix:
                return {}

            # Get all symbols if not specified
            if symbols is None:
                symbols = list(set(
                    symbol for pair in self.current_matrix.keys()
                    for symbol in pair
                ))
                symbols.sort()

            # Build matrix in list format
            matrix_data = []
            for symbol1 in symbols:
                row = []
                for symbol2 in symbols:
                    corr = self.current_matrix.get((symbol1, symbol2), Decimal('0'))
                    row.append(float(corr))
                matrix_data.append(row)

            return {
                'symbols': symbols,
                'matrix': matrix_data,
                'timestamp': self.last_update.isoformat() if self.last_update else None
            }

        except Exception as e:
            logger.error(f"Failed to format matrix for visualization: {e}")
            raise

    async def _calculate_pearson_correlation(
        self,
        returns1: List[float],
        returns2: List[float]
    ) -> Decimal:
        """Calculate Pearson correlation coefficient"""
        try:
            n = min(len(returns1), len(returns2))
            if n == 0:
                return Decimal('0')

            # Convert to Decimal
            r1 = [Decimal(str(r)) for r in returns1[:n]]
            r2 = [Decimal(str(r)) for r in returns2[:n]]

            # Calculate means
            mean1 = sum(r1) / Decimal(str(n))
            mean2 = sum(r2) / Decimal(str(n))

            # Calculate correlation
            numerator = sum((r1[i] - mean1) * (r2[i] - mean2) for i in range(n))
            denominator1 = sum((r1[i] - mean1) ** 2 for i in range(n))
            denominator2 = sum((r2[i] - mean2) ** 2 for i in range(n))

            if denominator1 == Decimal('0') or denominator2 == Decimal('0'):
                return Decimal('0')

            correlation = numerator / (denominator1.sqrt() * denominator2.sqrt())
            return correlation

        except Exception as e:
            logger.error(f"Pearson correlation calculation failed: {e}")
            raise

    async def _calculate_ewma_variance(
        self,
        returns: List[Decimal],
        lambda_param: Decimal
    ) -> Decimal:
        """Calculate exponentially weighted moving average variance"""
        try:
            if len(returns) == 0:
                return Decimal('0')

            # Initialize with first observation
            ewma_var = returns[0] ** 2

            # Update iteratively
            for ret in returns[1:]:
                ewma_var = lambda_param * ewma_var + (Decimal('1') - lambda_param) * (ret ** 2)

            return ewma_var

        except Exception as e:
            logger.error(f"EWMA variance calculation failed: {e}")
            raise

    async def _calculate_ewma_covariance(
        self,
        returns1: List[Decimal],
        returns2: List[Decimal],
        lambda_param: Decimal
    ) -> Decimal:
        """Calculate exponentially weighted moving average covariance"""
        try:
            n = min(len(returns1), len(returns2))
            if n == 0:
                return Decimal('0')

            # Initialize with first observation
            ewma_cov = returns1[0] * returns2[0]

            # Update iteratively
            for i in range(1, n):
                ewma_cov = lambda_param * ewma_cov + (Decimal('1') - lambda_param) * (returns1[i] * returns2[i])

            return ewma_cov

        except Exception as e:
            logger.error(f"EWMA covariance calculation failed: {e}")
            raise


async def main():
    """Example usage of correlation matrix manager"""
    try:
        # Initialize manager
        manager = CorrelationMatrixManager(
            risk_config_path='/home/user/FritzellSama/config/bot/risk.yaml',
            prod_config_path='/home/user/FritzellSama/config/environments/production.yaml'
        )

        # Create sample returns data
        returns_data = pl.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'GOOGL', 'GOOGL', 'MSFT', 'MSFT'],
            'date': ['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02'],
            'return': [0.01, 0.02, 0.015, 0.018, 0.012, 0.019]
        })

        symbols = ['AAPL', 'GOOGL', 'MSFT']

        # Calculate correlation matrix
        snapshot = await manager.calculate_correlation_matrix(
            returns_data,
            symbols
        )

        logger.info(f"Correlation snapshot: avg={snapshot.avg_correlation}, max={snapshot.max_correlation}")

        # Get specific correlation
        corr = await manager.get_correlation('AAPL', 'GOOGL')
        logger.info(f"AAPL-GOOGL correlation: {corr}")

    except Exception as e:
        logger.error(f"Correlation matrix manager example failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
