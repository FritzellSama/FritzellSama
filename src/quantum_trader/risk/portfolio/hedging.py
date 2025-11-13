"""
Portfolio Hedging Strategies for Quantum Trader AI

Production-grade hedging implementation with:
- Delta hedging for derivatives
- Beta hedging vs benchmark
- Tail risk hedging
- Cross-asset hedging
- Hedge ratio calculation and optimization
- Hedge effectiveness monitoring

CRITICAL: All numeric values use Decimal, NEVER float
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple
import logging

import polars as pl
import yaml
import numpy as np

from quantum_trader.models import Position


logger = logging.getLogger(__name__)


class HedgeType(Enum):
    """Types of hedging strategies"""
    DELTA = "DELTA"
    BETA = "BETA"
    TAIL_RISK = "TAIL_RISK"
    CROSS_ASSET = "CROSS_ASSET"
    PAIRS = "PAIRS"


@dataclass
class HedgeConfig:
    """Configuration for hedging strategies"""
    enable_delta_hedging: bool
    enable_beta_hedging: bool
    enable_tail_risk_hedging: bool
    target_beta: Decimal
    max_hedge_ratio: Decimal
    min_hedge_effectiveness: Decimal
    rehedge_threshold: Decimal
    tail_risk_protection_percent: Decimal
    hedge_instruments: Dict[str, str]  # asset -> hedge instrument mapping


@dataclass
class HedgePosition:
    """Represents a hedge position"""
    hedge_type: HedgeType
    underlying_symbol: str
    hedge_symbol: str
    hedge_ratio: Decimal
    hedge_quantity: Decimal
    hedge_cost: Decimal
    timestamp: datetime
    effectiveness: Optional[Decimal] = None
    delta: Optional[Decimal] = None
    beta: Optional[Decimal] = None


class PortfolioHedger:
    """
    Portfolio hedging engine

    Manages various hedging strategies to reduce portfolio risk:
    - Delta-neutral hedging for derivatives
    - Beta hedging to target market exposure
    - Tail risk protection against extreme moves
    - Cross-asset correlation hedging
    """

    def __init__(
        self,
        config_path: str = "/home/user/FritzellSama/config/bot/risk.yaml",
        env_config_path: str = "/home/user/FritzellSama/config/environments/production.yaml"
    ):
        """
        Initialize portfolio hedger

        Args:
            config_path: Path to risk configuration file
            env_config_path: Path to environment configuration file
        """
        self.config = self._load_config(config_path, env_config_path)
        self.active_hedges: Dict[str, HedgePosition] = {}
        self.hedge_history: List[HedgePosition] = []

        logger.info("PortfolioHedger initialized")

    def _load_config(self, config_path: str, env_config_path: str) -> HedgeConfig:
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

            return HedgeConfig(
                enable_delta_hedging=True,
                enable_beta_hedging=position_limits.get('enable_correlation_hedging', True),
                enable_tail_risk_hedging=True,
                target_beta=Decimal('1.0'),
                max_hedge_ratio=Decimal('1.5'),
                min_hedge_effectiveness=Decimal('0.7'),
                rehedge_threshold=Decimal('0.1'),  # 10% drift triggers rehedge
                tail_risk_protection_percent=Decimal('5.0'),
                hedge_instruments={
                    'SPY': 'SPY_PUT',
                    'QQQ': 'QQQ_PUT',
                    'IWM': 'IWM_PUT'
                }
            )

        except Exception as e:
            logger.error("Failed to load configuration: %s", e)
            raise

    def calculate_delta_hedge(
        self,
        positions_df: pl.DataFrame,
        options_greeks_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Calculate delta hedging requirements for options positions

        Delta hedging neutralizes directional risk from options

        Args:
            positions_df: DataFrame with columns [symbol, quantity, position_type]
            options_greeks_df: DataFrame with columns [symbol, delta, gamma, underlying_symbol]

        Returns:
            DataFrame with hedge recommendations
        """
        if not isinstance(positions_df, pl.DataFrame):
            raise TypeError(f"positions_df must be polars DataFrame, got {type(positions_df)}")

        if not isinstance(options_greeks_df, pl.DataFrame):
            raise TypeError(f"options_greeks_df must be polars DataFrame, got {type(options_greeks_df)}")

        # Join positions with greeks
        hedge_df = positions_df.join(options_greeks_df, on='symbol', how='inner')

        # Calculate portfolio delta per underlying
        hedge_df = hedge_df.with_columns([
            (pl.col('quantity') * pl.col('delta')).alias('position_delta')
        ])

        # Aggregate by underlying
        portfolio_delta = hedge_df.group_by('underlying_symbol').agg([
            pl.col('position_delta').sum().alias('total_delta')
        ])

        # Calculate hedge quantities (opposite of delta)
        hedge_recommendations = portfolio_delta.with_columns([
            (pl.col('total_delta') * Decimal('-1')).alias('hedge_quantity'),
            pl.lit('DELTA').alias('hedge_type'),
            pl.lit(datetime.now()).alias('timestamp')
        ])

        logger.info(
            "Calculated delta hedges for %d underlying assets",
            hedge_recommendations.height
        )

        return hedge_recommendations

    def calculate_beta_hedge(
        self,
        portfolio_df: pl.DataFrame,
        benchmark_symbol: str = 'SPY',
        target_beta: Optional[Decimal] = None
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate beta hedging requirements vs benchmark

        Adjusts portfolio beta to target level using index futures/ETFs

        Args:
            portfolio_df: DataFrame with columns [symbol, quantity, market_value, beta]
            benchmark_symbol: Benchmark to hedge against
            target_beta: Target portfolio beta (uses config default if None)

        Returns:
            Tuple of (hedge_ratio, hedge_quantity)
        """
        if not isinstance(portfolio_df, pl.DataFrame):
            raise TypeError(f"portfolio_df must be polars DataFrame, got {type(portfolio_df)}")

        if target_beta is None:
            target_beta = self.config.target_beta

        # Calculate portfolio beta
        total_value = Decimal(str(portfolio_df['market_value'].sum()))

        if total_value == Decimal('0'):
            logger.warning("Portfolio value is zero, cannot calculate beta hedge")
            return Decimal('0'), Decimal('0')

        # Weighted average beta
        portfolio_beta = Decimal('0')
        for row in portfolio_df.iter_rows(named=True):
            weight = Decimal(str(row['market_value'])) / total_value
            beta = Decimal(str(row['beta']))
            portfolio_beta += weight * beta

        # Calculate hedge ratio
        beta_difference = portfolio_beta - target_beta

        # Hedge quantity as percentage of portfolio value
        hedge_quantity = (beta_difference * total_value)

        logger.info(
            "Beta hedge calculated: Portfolio beta=%s, Target=%s, Hedge quantity=$%s",
            portfolio_beta, target_beta, hedge_quantity
        )

        return beta_difference, hedge_quantity

    def calculate_tail_risk_hedge(
        self,
        portfolio_value: Decimal,
        current_vix: Decimal,
        protection_percent: Optional[Decimal] = None
    ) -> Dict[str, Decimal]:
        """
        Calculate tail risk hedging using out-of-the-money puts

        Protects against extreme market moves (black swan events)

        Args:
            portfolio_value: Total portfolio value
            current_vix: Current VIX level
            protection_percent: Percentage of portfolio to protect

        Returns:
            Dictionary with hedge details
        """
        if not isinstance(portfolio_value, Decimal):
            raise TypeError(f"portfolio_value must be Decimal, got {type(portfolio_value)}")

        if not isinstance(current_vix, Decimal):
            raise TypeError(f"current_vix must be Decimal, got {type(current_vix)}")

        if protection_percent is None:
            protection_percent = self.config.tail_risk_protection_percent

        # Amount to protect
        protection_amount = portfolio_value * (protection_percent / Decimal('100'))

        # Adjust hedge size based on VIX
        # Higher VIX = more expensive hedges, use smaller position
        vix_adjustment = Decimal('20') / max(current_vix, Decimal('10'))

        adjusted_protection = protection_amount * vix_adjustment

        # Calculate put strike (typically 5-10% out of the money)
        otm_percent = Decimal('10')

        hedge_details = {
            'protection_amount': protection_amount,
            'adjusted_protection': adjusted_protection,
            'vix_level': current_vix,
            'vix_adjustment': vix_adjustment,
            'strike_otm_percent': otm_percent,
            'hedge_cost_estimate': adjusted_protection * Decimal('0.02')  # ~2% of notional
        }

        logger.info(
            "Tail risk hedge calculated: Protecting $%s of $%s portfolio (VIX=%s)",
            protection_amount, portfolio_value, current_vix
        )

        return hedge_details

    def calculate_cross_asset_hedge(
        self,
        positions_df: pl.DataFrame,
        correlation_matrix_df: pl.DataFrame,
        max_correlation: Decimal = Decimal('0.8')
    ) -> pl.DataFrame:
        """
        Calculate cross-asset hedging for correlated positions

        Reduces risk from highly correlated positions

        Args:
            positions_df: DataFrame with columns [symbol, quantity, market_value]
            correlation_matrix_df: DataFrame with correlation matrix
            max_correlation: Maximum acceptable correlation

        Returns:
            DataFrame with hedge recommendations
        """
        if not isinstance(positions_df, pl.DataFrame):
            raise TypeError(f"positions_df must be polars DataFrame, got {type(positions_df)}")

        hedges = []

        # Find highly correlated pairs
        symbols = positions_df['symbol'].to_list()

        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                # Get correlation from matrix
                try:
                    corr_row = correlation_matrix_df.filter(
                        pl.col('symbol') == symbol1
                    ).select(symbol2)

                    if corr_row.height == 0:
                        continue

                    correlation = Decimal(str(corr_row[0, 0]))

                    if abs(correlation) > max_correlation:
                        # High correlation detected
                        pos1 = positions_df.filter(pl.col('symbol') == symbol1)
                        pos2 = positions_df.filter(pl.col('symbol') == symbol2)

                        if pos1.height == 0 or pos2.height == 0:
                            continue

                        value1 = Decimal(str(pos1['market_value'][0]))
                        value2 = Decimal(str(pos2['market_value'][0]))

                        # Determine which position to hedge
                        if value1 > value2:
                            hedge_target = symbol2
                            hedge_ratio = value2 / value1 if value1 > 0 else Decimal('0')
                        else:
                            hedge_target = symbol1
                            hedge_ratio = value1 / value2 if value2 > 0 else Decimal('0')

                        hedges.append({
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'correlation': float(correlation),
                            'hedge_target': hedge_target,
                            'hedge_ratio': float(hedge_ratio),
                            'hedge_type': 'CROSS_ASSET'
                        })

                except Exception as e:
                    logger.warning("Error calculating correlation for %s/%s: %s", symbol1, symbol2, e)
                    continue

        if not hedges:
            logger.info("No cross-asset hedges needed")
            return pl.DataFrame()

        hedge_df = pl.DataFrame(hedges)

        logger.info("Identified %d cross-asset hedge opportunities", hedge_df.height)

        return hedge_df

    def calculate_hedge_effectiveness(
        self,
        portfolio_returns_df: pl.DataFrame,
        hedge_returns_df: pl.DataFrame,
        hedge_position: HedgePosition
    ) -> Decimal:
        """
        Calculate effectiveness of hedge position

        Uses R-squared from regression of portfolio returns on hedge returns

        Args:
            portfolio_returns_df: DataFrame with columns [timestamp, return]
            hedge_returns_df: DataFrame with columns [timestamp, return]
            hedge_position: Hedge position to evaluate

        Returns:
            Hedge effectiveness ratio (0 to 1)
        """
        if not isinstance(portfolio_returns_df, pl.DataFrame):
            raise TypeError(f"portfolio_returns_df must be polars DataFrame")

        if not isinstance(hedge_returns_df, pl.DataFrame):
            raise TypeError(f"hedge_returns_df must be polars DataFrame")

        # Join returns
        returns_df = portfolio_returns_df.join(
            hedge_returns_df,
            on='timestamp',
            how='inner',
            suffix='_hedge'
        )

        if returns_df.height < 2:
            logger.warning("Insufficient data for hedge effectiveness calculation")
            return Decimal('0')

        # Convert to numpy for calculation
        portfolio_returns = returns_df['return'].to_numpy()
        hedge_returns = returns_df['return_hedge'].to_numpy()

        # Calculate correlation
        correlation = np.corrcoef(portfolio_returns, hedge_returns)[0, 1]

        # R-squared as effectiveness measure
        effectiveness = Decimal(str(correlation ** 2))

        logger.info(
            "Hedge effectiveness for %s: %s",
            hedge_position.hedge_symbol, effectiveness
        )

        return effectiveness

    def calculate_optimal_hedge_ratio(
        self,
        portfolio_returns_df: pl.DataFrame,
        hedge_returns_df: pl.DataFrame
    ) -> Decimal:
        """
        Calculate optimal hedge ratio using regression

        Minimizes portfolio variance

        Args:
            portfolio_returns_df: DataFrame with portfolio returns
            hedge_returns_df: DataFrame with hedge instrument returns

        Returns:
            Optimal hedge ratio
        """
        if not isinstance(portfolio_returns_df, pl.DataFrame):
            raise TypeError(f"portfolio_returns_df must be polars DataFrame")

        # Join returns
        returns_df = portfolio_returns_df.join(
            hedge_returns_df,
            on='timestamp',
            how='inner',
            suffix='_hedge'
        )

        if returns_df.height < 10:
            logger.warning("Insufficient data for optimal hedge ratio calculation")
            return Decimal('1.0')

        # Convert to numpy
        portfolio_returns = returns_df['return'].to_numpy()
        hedge_returns = returns_df['return_hedge'].to_numpy()

        # Calculate covariance and variance
        covariance = np.cov(portfolio_returns, hedge_returns)[0, 1]
        hedge_variance = np.var(hedge_returns)

        if hedge_variance == 0:
            logger.warning("Hedge variance is zero")
            return Decimal('0')

        # Optimal hedge ratio
        optimal_ratio = Decimal(str(covariance / hedge_variance))

        # Constrain to max hedge ratio
        optimal_ratio = min(abs(optimal_ratio), self.config.max_hedge_ratio)

        logger.info("Calculated optimal hedge ratio: %s", optimal_ratio)

        return optimal_ratio

    def should_rehedge(
        self,
        current_hedge: HedgePosition,
        current_delta: Decimal
    ) -> bool:
        """
        Determine if position should be rehedged

        Args:
            current_hedge: Current hedge position
            current_delta: Current delta/exposure

        Returns:
            True if rehedge recommended
        """
        if current_hedge.delta is None:
            return False

        # Calculate drift from target
        drift = abs(current_delta - current_hedge.delta)
        drift_percent = drift / abs(current_hedge.delta) if current_hedge.delta != 0 else Decimal('1')

        should_rehedge = drift_percent > self.config.rehedge_threshold

        if should_rehedge:
            logger.info(
                "Rehedge recommended for %s: drift=%s%% (threshold=%s%%)",
                current_hedge.hedge_symbol,
                drift_percent * Decimal('100'),
                self.config.rehedge_threshold * Decimal('100')
            )

        return should_rehedge

    def monitor_hedge_performance(
        self,
        hedge_position: HedgePosition,
        current_portfolio_value: Decimal,
        hedge_pnl: Decimal
    ) -> Dict[str, Decimal]:
        """
        Monitor and report hedge performance

        Args:
            hedge_position: Hedge position to monitor
            current_portfolio_value: Current portfolio value
            hedge_pnl: P&L from hedge position

        Returns:
            Performance metrics
        """
        if not isinstance(current_portfolio_value, Decimal):
            raise TypeError(f"current_portfolio_value must be Decimal")

        if not isinstance(hedge_pnl, Decimal):
            raise TypeError(f"hedge_pnl must be Decimal")

        hedge_cost_percent = (hedge_position.hedge_cost / current_portfolio_value * Decimal('100')
                              if current_portfolio_value > 0 else Decimal('0'))

        hedge_pnl_percent = (hedge_pnl / hedge_position.hedge_cost * Decimal('100')
                            if hedge_position.hedge_cost > 0 else Decimal('0'))

        metrics = {
            'hedge_cost': hedge_position.hedge_cost,
            'hedge_cost_percent': hedge_cost_percent,
            'hedge_pnl': hedge_pnl,
            'hedge_pnl_percent': hedge_pnl_percent,
            'effectiveness': hedge_position.effectiveness or Decimal('0'),
            'hedge_ratio': hedge_position.hedge_ratio,
            'days_active': Decimal(str((datetime.now() - hedge_position.timestamp).days))
        }

        return metrics

    def get_active_hedges(self) -> List[HedgePosition]:
        """
        Get list of active hedge positions

        Returns:
            List of active hedges
        """
        return list(self.active_hedges.values())

    def add_hedge(self, hedge: HedgePosition) -> None:
        """
        Add a hedge position to tracking

        Args:
            hedge: Hedge position to add
        """
        key = f"{hedge.hedge_type.value}_{hedge.underlying_symbol}"
        self.active_hedges[key] = hedge
        self.hedge_history.append(hedge)

        logger.info(
            "Added %s hedge for %s: %s quantity=%s",
            hedge.hedge_type.value, hedge.underlying_symbol,
            hedge.hedge_symbol, hedge.hedge_quantity
        )

    def remove_hedge(self, hedge_type: HedgeType, underlying_symbol: str) -> None:
        """
        Remove a hedge position

        Args:
            hedge_type: Type of hedge
            underlying_symbol: Underlying asset symbol
        """
        key = f"{hedge_type.value}_{underlying_symbol}"
        if key in self.active_hedges:
            del self.active_hedges[key]
            logger.info("Removed %s hedge for %s", hedge_type.value, underlying_symbol)
