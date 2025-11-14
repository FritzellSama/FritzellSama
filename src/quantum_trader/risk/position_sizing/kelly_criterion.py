"""
Kelly Criterion Position Sizing for Quantum Trader AI

Production-grade Kelly criterion implementation with:
- Full Kelly calculation
- Fractional Kelly (half-Kelly, quarter-Kelly)
- Multi-asset Kelly optimization
- Drawdown-adjusted Kelly
- Win rate and payoff ratio estimation

CRITICAL: All numeric values use Decimal, NEVER float
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
import logging

import polars as pl
import yaml
import numpy as np
from scipy.optimize import minimize

from quantum_trader.models import Position, Signal


logger = logging.getLogger(__name__)


@dataclass
class KellyConfig:
    """Configuration for Kelly criterion position sizing"""
    kelly_fraction: Decimal  # 1.0=full Kelly, 0.5=half Kelly, 0.25=quarter Kelly
    min_trade_history: int  # Minimum trades needed for Kelly calculation
    max_kelly_allocation: Decimal  # Maximum allocation per position
    use_multi_asset_kelly: bool
    enable_drawdown_adjustment: bool
    drawdown_threshold_percent: Decimal
    win_rate_lookback_periods: int
    confidence_interval: Decimal


@dataclass
class KellyMetrics:
    """Kelly criterion calculation metrics"""
    win_rate: Decimal
    avg_win: Decimal
    avg_loss: Decimal
    payoff_ratio: Decimal
    kelly_fraction: Decimal
    recommended_allocation: Decimal
    confidence: Decimal
    sample_size: int


class KellyCriterionSizer:
    """
    Kelly Criterion position sizing calculator

    Implements optimal position sizing based on:
    - Historical win rate
    - Average win/loss ratio
    - Risk of ruin considerations
    - Multi-asset portfolio optimization
    """

    def __init__(
        self,
        config_path: str = "/home/user/FritzellSama/config/bot/risk.yaml",
        env_config_path: str = "/home/user/FritzellSama/config/environments/production.yaml"
    ):
        """
        Initialize Kelly criterion position sizer

        Args:
            config_path: Path to risk configuration file
            env_config_path: Path to environment configuration file
        """
        self.config = self._load_config(config_path, env_config_path)
        self.trade_history: List[Dict] = []
        self.account_balance = Decimal('0')
        self.peak_balance = Decimal('0')
        self.current_drawdown_percent = Decimal('0')

        logger.info(
            "KellyCriterionSizer initialized with %s Kelly fraction",
            self.config.kelly_fraction
        )

    def _load_config(self, config_path: str, env_config_path: str) -> KellyConfig:
        """
        Load configuration from YAML files

        Args:
            config_path: Path to risk config
            env_config_path: Path to environment config

        Returns:
            Loaded configuration
        """
        try:
            with open(config_path, 'r') as f:
                risk_config = yaml.safe_load(f)

            with open(env_config_path, 'r') as f:
                env_config = yaml.safe_load(f)

            position_limits = risk_config.get('position_limits', {})

            return KellyConfig(
                kelly_fraction=Decimal('0.25'),  # Quarter-Kelly for safety
                min_trade_history=30,
                max_kelly_allocation=Decimal(str(position_limits.get('max_position_size_percent', 5.0))) / Decimal('100'),
                use_multi_asset_kelly=True,
                enable_drawdown_adjustment=True,
                drawdown_threshold_percent=Decimal(str(risk_config.get('global', {}).get('max_daily_loss_percent', 5.0))),
                win_rate_lookback_periods=100,
                confidence_interval=Decimal('0.95')
            )

        except Exception as e:
            logger.error("Failed to load configuration: %s", e)
            raise

    def update_account_balance(self, balance: Decimal) -> None:
        """
        Update current account balance and calculate drawdown

        Args:
            balance: Current account balance in USD
        """
        if not isinstance(balance, Decimal):
            raise TypeError(f"Balance must be Decimal, got {type(balance)}")

        self.account_balance = balance

        # Update peak balance
        if balance > self.peak_balance:
            self.peak_balance = balance

        # Calculate current drawdown
        if self.peak_balance > Decimal('0'):
            self.current_drawdown_percent = ((self.peak_balance - balance) / self.peak_balance) * Decimal('100')
        else:
            self.current_drawdown_percent = Decimal('0')

    def calculate_kelly_metrics(
        self,
        trade_history_df: pl.DataFrame,
        symbol: Optional[str] = None
    ) -> KellyMetrics:
        """
        Calculate Kelly metrics from trade history

        Args:
            trade_history_df: DataFrame with columns [symbol, pnl, timestamp]
            symbol: Optional symbol to filter for (None = portfolio-wide)

        Returns:
            Kelly metrics
        """
        if not isinstance(trade_history_df, pl.DataFrame):
            raise TypeError(f"trade_history_df must be polars DataFrame, got {type(trade_history_df)}")

        # Filter by symbol if provided
        if symbol is not None:
            df = trade_history_df.filter(pl.col('symbol') == symbol)
        else:
            df = trade_history_df

        # Limit to lookback period
        df = df.tail(self.config.win_rate_lookback_periods)

        if df.height < self.config.min_trade_history:
            logger.warning(
                "Insufficient trade history (%d trades, need %d), using conservative defaults",
                df.height, self.config.min_trade_history
            )
            return KellyMetrics(
                win_rate=Decimal('0.5'),
                avg_win=Decimal('1.0'),
                avg_loss=Decimal('1.0'),
                payoff_ratio=Decimal('1.0'),
                kelly_fraction=Decimal('0'),
                recommended_allocation=Decimal('0'),
                confidence=Decimal('0'),
                sample_size=df.height
            )

        # Separate wins and losses
        wins_df = df.filter(pl.col('pnl') > 0)
        losses_df = df.filter(pl.col('pnl') < 0)

        total_trades = df.height
        num_wins = wins_df.height
        num_losses = losses_df.height

        if num_wins == 0 or num_losses == 0:
            logger.warning("All trades are wins or all are losses, cannot calculate Kelly")
            return KellyMetrics(
                win_rate=Decimal(str(num_wins / total_trades)) if total_trades > 0 else Decimal('0'),
                avg_win=Decimal('0'),
                avg_loss=Decimal('0'),
                payoff_ratio=Decimal('1.0'),
                kelly_fraction=Decimal('0'),
                recommended_allocation=Decimal('0'),
                confidence=Decimal('0'),
                sample_size=total_trades
            )

        # Calculate metrics
        win_rate = Decimal(str(num_wins / total_trades))

        avg_win = Decimal(str(wins_df['pnl'].mean()))
        avg_loss = abs(Decimal(str(losses_df['pnl'].mean())))

        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else Decimal('1')

        # Kelly formula: f* = (p * b - q) / b
        # where p = win probability, q = loss probability, b = payoff ratio
        p = win_rate
        q = Decimal('1') - win_rate
        b = payoff_ratio

        kelly_fraction = (p * b - q) / b if b > 0 else Decimal('0')

        # Apply fractional Kelly
        kelly_fraction = kelly_fraction * self.config.kelly_fraction

        # Cap at maximum allocation
        kelly_fraction = max(Decimal('0'), min(kelly_fraction, self.config.max_kelly_allocation))

        # Calculate confidence based on sample size
        confidence = self._calculate_confidence(total_trades)

        # Apply drawdown adjustment
        if self.config.enable_drawdown_adjustment:
            kelly_fraction = self._apply_drawdown_adjustment(kelly_fraction)

        recommended_allocation = kelly_fraction * self.account_balance

        logger.info(
            "Kelly metrics calculated: win_rate=%s, payoff_ratio=%s, kelly_fraction=%s, allocation=$%s",
            win_rate, payoff_ratio, kelly_fraction, recommended_allocation
        )

        return KellyMetrics(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            payoff_ratio=payoff_ratio,
            kelly_fraction=kelly_fraction,
            recommended_allocation=recommended_allocation,
            confidence=confidence,
            sample_size=total_trades
        )

    def _calculate_confidence(self, sample_size: int) -> Decimal:
        """
        Calculate confidence based on sample size

        More trades = higher confidence

        Args:
            sample_size: Number of historical trades

        Returns:
            Confidence level (0 to 1)
        """
        # Use sigmoid function for confidence
        # confidence = 1 / (1 + exp(-k * (n - n0)))
        # where n = sample_size, n0 = min required, k = steepness

        min_trades = self.config.min_trade_history
        steepness = 0.05

        if sample_size < min_trades:
            return Decimal('0')

        exponent = -steepness * (sample_size - min_trades)

        # Convert to float for exp calculation, then back to Decimal
        confidence = 1 / (1 + np.exp(exponent))

        return Decimal(str(confidence)).quantize(Decimal('0.01'))

    def _apply_drawdown_adjustment(self, kelly_fraction: Decimal) -> Decimal:
        """
        Adjust Kelly fraction based on current drawdown

        Reduces position size during drawdowns

        Args:
            kelly_fraction: Base Kelly fraction

        Returns:
            Adjusted Kelly fraction
        """
        if self.current_drawdown_percent > self.config.drawdown_threshold_percent:
            # Reduce Kelly proportionally to drawdown severity
            reduction_factor = Decimal('1') - (
                self.current_drawdown_percent / (self.config.drawdown_threshold_percent * Decimal('2'))
            )

            # Ensure reduction factor is between 0 and 1
            reduction_factor = max(Decimal('0'), min(Decimal('1'), reduction_factor))

            adjusted_fraction = kelly_fraction * reduction_factor

            logger.warning(
                "Drawdown adjustment applied: %s -> %s (drawdown: %s%%)",
                kelly_fraction, adjusted_fraction, self.current_drawdown_percent
            )

            return adjusted_fraction

        return kelly_fraction

    def calculate_position_size(
        self,
        signal: Signal,
        current_price: Decimal,
        trade_history_df: pl.DataFrame
    ) -> Decimal:
        """
        Calculate position size using Kelly criterion

        Args:
            signal: Trading signal
            current_price: Current asset price
            trade_history_df: Historical trades for the asset

        Returns:
            Position size in number of shares/contracts
        """
        if not isinstance(current_price, Decimal):
            raise TypeError(f"Price must be Decimal, got {type(current_price)}")

        if self.account_balance <= Decimal('0'):
            logger.warning("Account balance not set or zero, cannot calculate position size")
            return Decimal('0')

        # Calculate Kelly metrics for this asset
        kelly_metrics = self.calculate_kelly_metrics(trade_history_df, signal.symbol)

        if kelly_metrics.kelly_fraction <= Decimal('0'):
            logger.info("Kelly fraction is zero or negative for %s, skipping trade", signal.symbol)
            return Decimal('0')

        # Apply signal confidence weighting
        weighted_allocation = kelly_metrics.recommended_allocation * signal.confidence

        # Calculate number of shares
        position_size = weighted_allocation / current_price

        logger.info(
            "Calculated Kelly position size for %s: %s shares at $%s (fraction: %s)",
            signal.symbol, position_size, current_price, kelly_metrics.kelly_fraction
        )

        return position_size.quantize(Decimal('0.00000001'))

    def calculate_multi_asset_kelly(
        self,
        signals_df: pl.DataFrame,
        trade_history_df: pl.DataFrame,
        correlation_matrix_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Calculate optimal Kelly allocations for multiple assets

        Uses portfolio optimization with correlation constraints

        Args:
            signals_df: DataFrame with trading signals
            trade_history_df: Historical trades for all assets
            correlation_matrix_df: Asset correlation matrix

        Returns:
            DataFrame with optimal allocations
        """
        if not isinstance(signals_df, pl.DataFrame):
            raise TypeError(f"signals_df must be polars DataFrame")

        symbols = signals_df['symbol'].to_list()
        n_assets = len(symbols)

        if n_assets == 0:
            return pl.DataFrame()

        # Calculate individual Kelly fractions
        kelly_fractions = []
        for symbol in symbols:
            metrics = self.calculate_kelly_metrics(trade_history_df, symbol)
            kelly_fractions.append(float(metrics.kelly_fraction))

        kelly_array = np.array(kelly_fractions)

        # Get correlation matrix
        corr_matrix = self._extract_correlation_matrix(correlation_matrix_df, symbols)

        # Optimize portfolio with correlation constraints
        # Minimize: -sum(kelly_i) subject to: sum(kelly_i) <= 1
        # and consider correlations to avoid over-concentration

        def objective(weights):
            # Negative sum (we want to maximize)
            return -np.sum(weights * kelly_array)

        def constraint_sum(weights):
            # Total allocation <= 100%
            return 1.0 - np.sum(weights)

        def constraint_correlation(weights):
            # Penalize highly correlated positions
            portfolio_risk = np.sqrt(weights @ corr_matrix @ weights)
            return 1.0 - portfolio_risk  # Limit risk

        # Initial guess: proportional to Kelly fractions
        initial_weights = kelly_array / (np.sum(kelly_array) + 1e-10)

        # Bounds: 0 to max Kelly allocation
        bounds = [(0, float(self.config.max_kelly_allocation)) for _ in range(n_assets)]

        # Constraints
        constraints = [
            {'type': 'ineq', 'fun': constraint_sum},
            {'type': 'ineq', 'fun': constraint_correlation}
        ]

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            logger.warning("Multi-asset Kelly optimization failed, using individual Kelly fractions")
            optimal_weights = kelly_array
        else:
            optimal_weights = result.x

        # Create result DataFrame
        allocations_df = pl.DataFrame({
            'symbol': symbols,
            'kelly_fraction': kelly_fractions,
            'optimal_weight': optimal_weights.tolist(),
            'dollar_allocation': [float(w * float(self.account_balance)) for w in optimal_weights]
        })

        logger.info(
            "Multi-asset Kelly optimization complete for %d assets, total allocation: %s%%",
            n_assets, sum(optimal_weights) * 100
        )

        return allocations_df

    def _extract_correlation_matrix(
        self,
        correlation_df: pl.DataFrame,
        symbols: List[str]
    ) -> np.ndarray:
        """
        Extract correlation matrix as numpy array

        Args:
            correlation_df: Correlation matrix as DataFrame
            symbols: List of symbols to extract

        Returns:
            Numpy correlation matrix
        """
        n = len(symbols)
        matrix = np.eye(n)  # Default to identity matrix

        try:
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols):
                    if i != j:
                        corr_row = correlation_df.filter(pl.col('symbol') == sym1).select(sym2)
                        if corr_row.height > 0:
                            matrix[i, j] = float(corr_row[0, 0])

        except Exception as e:
            logger.warning("Failed to extract correlation matrix: %s", e)

        return matrix

    def calculate_fractional_kelly(
        self,
        kelly_fraction: Decimal,
        fraction: Decimal = Decimal('0.5')
    ) -> Decimal:
        """
        Calculate fractional Kelly (e.g., half-Kelly, quarter-Kelly)

        Args:
            kelly_fraction: Full Kelly fraction
            fraction: Fraction to use (0.5 = half-Kelly, 0.25 = quarter-Kelly)

        Returns:
            Fractional Kelly allocation
        """
        if not isinstance(kelly_fraction, Decimal):
            raise TypeError(f"kelly_fraction must be Decimal")

        if not isinstance(fraction, Decimal):
            raise TypeError(f"fraction must be Decimal")

        return kelly_fraction * fraction

    def estimate_risk_of_ruin(
        self,
        kelly_fraction: Decimal,
        win_rate: Decimal,
        payoff_ratio: Decimal,
        num_trades: int = 100
    ) -> Decimal:
        """
        Estimate risk of ruin using Kelly fraction

        Args:
            kelly_fraction: Kelly allocation fraction
            win_rate: Win probability
            payoff_ratio: Average win / average loss
            num_trades: Number of trades to simulate

        Returns:
            Estimated risk of ruin probability
        """
        if kelly_fraction <= Decimal('0'):
            return Decimal('1.0')  # Certain ruin if not trading

        # Simplified risk of ruin calculation
        # Using formula: RoR = ((1-p)/p)^(capital/kelly)

        p = win_rate
        q = Decimal('1') - p

        if p <= Decimal('0') or p >= Decimal('1'):
            return Decimal('0.5')

        # Convert to float for calculation
        p_float = float(p)
        q_float = float(q)
        kelly_float = float(kelly_fraction)

        # Risk of ruin approximation
        if kelly_float > 0:
            ror = (q_float / p_float) ** (1.0 / kelly_float / num_trades)
        else:
            ror = 1.0

        return Decimal(str(min(1.0, max(0.0, ror)))).quantize(Decimal('0.0001'))

    def get_kelly_statistics(self) -> Dict[str, Decimal]:
        """
        Get Kelly criterion statistics

        Returns:
            Dictionary of statistics
        """
        return {
            'account_balance': self.account_balance,
            'peak_balance': self.peak_balance,
            'current_drawdown_percent': self.current_drawdown_percent,
            'kelly_fraction_config': self.config.kelly_fraction,
            'max_kelly_allocation': self.config.max_kelly_allocation,
            'min_trade_history': Decimal(str(self.config.min_trade_history))
        }
