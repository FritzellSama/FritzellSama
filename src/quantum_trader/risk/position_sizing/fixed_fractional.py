"""
Fixed Fractional Position Sizing for Quantum Trader AI

Production-grade fixed fractional position sizing implementation with:
- Fixed percentage of capital allocation
- Risk normalization across positions
- Dynamic account balance tracking
- Adaptive fraction adjustment based on market conditions

CRITICAL: All numeric values use Decimal, NEVER float
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, Optional
import logging

import polars as pl
import yaml

from quantum_trader.models import Position, Signal


logger = logging.getLogger(__name__)


@dataclass
class FixedFractionalConfig:
    """Configuration for fixed fractional position sizing"""
    default_fraction: Decimal
    min_fraction: Decimal
    max_fraction: Decimal
    min_position_size_usd: Decimal
    max_position_size_percent: Decimal
    enable_volatility_adjustment: bool
    volatility_adjustment_factor: Decimal
    enable_drawdown_adjustment: bool
    drawdown_threshold_percent: Decimal
    drawdown_reduction_factor: Decimal


class FixedFractionalSizer:
    """
    Fixed fractional position sizing calculator

    Allocates a fixed fraction of capital to each position with:
    - Risk normalization
    - Volatility-adjusted sizing
    - Drawdown-based position reduction
    - Real-time account balance tracking
    """

    def __init__(
        self,
        config_path: str = "/home/user/FritzellSama/config/bot/risk.yaml",
        env_config_path: str = "/home/user/FritzellSama/config/environments/production.yaml"
    ):
        """
        Initialize fixed fractional position sizer

        Args:
            config_path: Path to risk configuration file
            env_config_path: Path to environment configuration file
        """
        self.config = self._load_config(config_path, env_config_path)
        self.account_balance = Decimal('0')
        self.peak_balance = Decimal('0')
        self.current_drawdown_percent = Decimal('0')

        logger.info("FixedFractionalSizer initialized with default fraction: %s", self.config.default_fraction)

    def _load_config(self, config_path: str, env_config_path: str) -> FixedFractionalConfig:
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

            return FixedFractionalConfig(
                default_fraction=Decimal(str(position_limits.get('max_position_size_percent', 5.0))) / Decimal('100'),
                min_fraction=Decimal('0.01'),  # 1% minimum
                max_fraction=Decimal(str(position_limits.get('max_position_size_percent', 5.0))) / Decimal('100'),
                min_position_size_usd=Decimal(str(position_limits.get('min_position_size_usd', 500))),
                max_position_size_percent=Decimal(str(position_limits.get('max_position_size_percent', 5.0))),
                enable_volatility_adjustment=risk_config.get('market_conditions', {}).get('reduce_on_high_volatility', True),
                volatility_adjustment_factor=Decimal(str(risk_config.get('market_conditions', {}).get('volatility_position_multiplier', 0.5))),
                enable_drawdown_adjustment=True,
                drawdown_threshold_percent=Decimal(str(risk_config.get('global', {}).get('max_daily_loss_percent', 5.0))),
                drawdown_reduction_factor=Decimal('0.5')
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

        logger.debug(
            "Account balance updated: %s, Peak: %s, Drawdown: %s%%",
            balance, self.peak_balance, self.current_drawdown_percent
        )

    def calculate_position_size(
        self,
        signal: Signal,
        current_price: Decimal,
        volatility: Optional[Decimal] = None,
        stop_loss_percent: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate position size using fixed fractional method

        Args:
            signal: Trading signal
            current_price: Current asset price
            volatility: Asset volatility (optional, for adjustment)
            stop_loss_percent: Stop loss percentage (optional, for risk normalization)

        Returns:
            Position size in number of shares/contracts
        """
        if not isinstance(current_price, Decimal):
            raise TypeError(f"Price must be Decimal, got {type(current_price)}")

        if self.account_balance <= Decimal('0'):
            logger.warning("Account balance not set or zero, cannot calculate position size")
            return Decimal('0')

        # Start with base fraction
        fraction = self.config.default_fraction

        # Apply volatility adjustment
        if self.config.enable_volatility_adjustment and volatility is not None:
            fraction = self._apply_volatility_adjustment(fraction, volatility)

        # Apply drawdown adjustment
        if self.config.enable_drawdown_adjustment:
            fraction = self._apply_drawdown_adjustment(fraction)

        # Apply signal confidence weighting
        fraction = fraction * signal.confidence

        # Ensure fraction is within bounds
        fraction = max(self.config.min_fraction, min(fraction, self.config.max_fraction))

        # Calculate dollar allocation
        dollar_allocation = self.account_balance * fraction

        # Apply minimum position size constraint
        if dollar_allocation < self.config.min_position_size_usd:
            logger.info(
                "Position size %s below minimum %s, returning zero",
                dollar_allocation, self.config.min_position_size_usd
            )
            return Decimal('0')

        # Calculate number of shares
        position_size = dollar_allocation / current_price

        # Apply risk normalization with stop loss
        if stop_loss_percent is not None:
            position_size = self._apply_risk_normalization(
                position_size,
                current_price,
                stop_loss_percent
            )

        logger.info(
            "Calculated position size for %s: %s shares at $%s (fraction: %s)",
            signal.symbol, position_size, current_price, fraction
        )

        return position_size.quantize(Decimal('0.00000001'))

    def _apply_volatility_adjustment(self, fraction: Decimal, volatility: Decimal) -> Decimal:
        """
        Adjust position fraction based on volatility

        High volatility reduces position size

        Args:
            fraction: Base fraction
            volatility: Asset volatility (annualized)

        Returns:
            Adjusted fraction
        """
        if not isinstance(volatility, Decimal):
            raise TypeError(f"Volatility must be Decimal, got {type(volatility)}")

        # Normalize volatility (assume 20% is baseline)
        baseline_volatility = Decimal('0.20')

        if volatility > baseline_volatility:
            # Reduce position size for high volatility
            volatility_ratio = baseline_volatility / volatility
            adjusted_fraction = fraction * volatility_ratio * self.config.volatility_adjustment_factor

            logger.debug(
                "Volatility adjustment applied: %s -> %s (volatility: %s)",
                fraction, adjusted_fraction, volatility
            )
            return adjusted_fraction

        return fraction

    def _apply_drawdown_adjustment(self, fraction: Decimal) -> Decimal:
        """
        Adjust position fraction based on current drawdown

        Reduces position size during drawdowns

        Args:
            fraction: Base fraction

        Returns:
            Adjusted fraction
        """
        if self.current_drawdown_percent > self.config.drawdown_threshold_percent:
            # Reduce position size proportionally to drawdown severity
            reduction_factor = Decimal('1') - (
                (self.current_drawdown_percent / self.config.drawdown_threshold_percent) *
                (Decimal('1') - self.config.drawdown_reduction_factor)
            )

            adjusted_fraction = fraction * reduction_factor

            logger.warning(
                "Drawdown adjustment applied: %s -> %s (drawdown: %s%%)",
                fraction, adjusted_fraction, self.current_drawdown_percent
            )
            return adjusted_fraction

        return fraction

    def _apply_risk_normalization(
        self,
        position_size: Decimal,
        current_price: Decimal,
        stop_loss_percent: Decimal
    ) -> Decimal:
        """
        Normalize position size based on stop loss distance

        Ensures consistent risk across positions

        Args:
            position_size: Raw position size
            current_price: Current price
            stop_loss_percent: Stop loss percentage

        Returns:
            Risk-normalized position size
        """
        if not isinstance(stop_loss_percent, Decimal):
            raise TypeError(f"Stop loss percent must be Decimal, got {type(stop_loss_percent)}")

        # Calculate risk per share
        risk_per_share = current_price * (stop_loss_percent / Decimal('100'))

        # Target risk amount (percentage of account)
        target_risk_amount = self.account_balance * (Decimal('1') / Decimal('100'))  # 1% risk

        # Calculate normalized position size
        normalized_size = target_risk_amount / risk_per_share

        # Use the smaller of raw size and normalized size
        return min(position_size, normalized_size)

    def calculate_batch_sizes(
        self,
        signals_df: pl.DataFrame,
        market_data_df: pl.DataFrame,
        volatility_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """
        Calculate position sizes for batch of signals

        Args:
            signals_df: DataFrame with columns [symbol, action, strength, confidence, timestamp]
            market_data_df: DataFrame with columns [symbol, current_price, timestamp]
            volatility_df: Optional DataFrame with columns [symbol, volatility]

        Returns:
            DataFrame with position sizes added
        """
        if not isinstance(signals_df, pl.DataFrame):
            raise TypeError(f"signals_df must be polars DataFrame, got {type(signals_df)}")

        if not isinstance(market_data_df, pl.DataFrame):
            raise TypeError(f"market_data_df must be polars DataFrame, got {type(market_data_df)}")

        # Join signals with market data
        result_df = signals_df.join(market_data_df, on='symbol', how='left')

        # Join with volatility if provided
        if volatility_df is not None:
            result_df = result_df.join(volatility_df, on='symbol', how='left')

        # Calculate position sizes
        position_sizes = []

        for row in result_df.iter_rows(named=True):
            signal = Signal(
                symbol=row['symbol'],
                action=row['action'],
                strength=Decimal(str(row['strength'])),
                confidence=Decimal(str(row['confidence'])),
                timestamp=row['timestamp'],
                strategy='batch',
                timeframe='1h'
            )

            current_price = Decimal(str(row['current_price']))
            volatility = Decimal(str(row.get('volatility', 0.2))) if 'volatility' in row else None

            position_size = self.calculate_position_size(
                signal=signal,
                current_price=current_price,
                volatility=volatility
            )

            position_sizes.append(float(position_size))

        # Add position sizes to result
        result_df = result_df.with_columns(
            pl.Series('position_size', position_sizes)
        )

        return result_df

    def get_current_fraction(self) -> Decimal:
        """
        Get current effective position fraction after adjustments

        Returns:
            Current position fraction
        """
        fraction = self.config.default_fraction

        if self.config.enable_drawdown_adjustment:
            fraction = self._apply_drawdown_adjustment(fraction)

        return fraction

    def get_metrics(self) -> Dict[str, Decimal]:
        """
        Get current sizer metrics

        Returns:
            Dictionary of current metrics
        """
        return {
            'account_balance': self.account_balance,
            'peak_balance': self.peak_balance,
            'current_drawdown_percent': self.current_drawdown_percent,
            'current_fraction': self.get_current_fraction(),
            'default_fraction': self.config.default_fraction,
            'min_position_size_usd': self.config.min_position_size_usd
        }
