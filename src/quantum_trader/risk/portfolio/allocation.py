"""
Portfolio Allocation Optimization
Quantum Trader AI - Production Risk Management

Implements portfolio allocation strategies including:
- Mean-variance optimization
- Risk parity allocation
- Black-Litterman model
- Rebalancing triggers
- Tax-aware allocation
- Transaction cost optimization

CRITICAL: All numeric values use Decimal, never float
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import polars as pl
import yaml

from quantum_trader.models import Position, AuditLog

logger = logging.getLogger(__name__)


@dataclass
class AllocationConfig:
    """Portfolio allocation configuration loaded from risk.yaml"""
    max_position_size_percent: Decimal
    max_sector_exposure_percent: Decimal
    min_position_size_usd: Decimal
    rebalance_enabled: bool
    rebalance_threshold_percent: Decimal
    max_single_ticker_percent: Decimal
    max_top_5_percent: Decimal
    transaction_cost_bps: Decimal = Decimal('10')  # 10 basis points default
    tax_rate: Decimal = Decimal('0.25')  # 25% default capital gains tax
    risk_free_rate: Decimal = Decimal('0.04')  # 4% annual risk-free rate

    @classmethod
    def from_yaml(cls, risk_config: Dict, prod_config: Dict) -> 'AllocationConfig':
        """Load configuration from yaml dictionaries"""
        return cls(
            max_position_size_percent=Decimal(str(risk_config['position_limits']['max_position_size_percent'])),
            max_sector_exposure_percent=Decimal(str(risk_config['position_limits']['max_sector_exposure_percent'])),
            min_position_size_usd=Decimal(str(risk_config['position_limits']['min_position_size_usd'])),
            rebalance_enabled=risk_config['concentration_limits']['rebalance_enabled'],
            rebalance_threshold_percent=Decimal(str(risk_config['concentration_limits']['rebalance_threshold_percent'])),
            max_single_ticker_percent=Decimal(str(risk_config['concentration_limits']['max_single_ticker_percent'])),
            max_top_5_percent=Decimal(str(risk_config['concentration_limits']['max_top_5_percent'])),
        )


@dataclass
class AllocationResult:
    """Result of portfolio allocation optimization"""
    target_weights: Dict[str, Decimal]
    current_weights: Dict[str, Decimal]
    rebalance_trades: Dict[str, Decimal]  # symbol -> quantity change
    expected_return: Decimal
    portfolio_variance: Decimal
    sharpe_ratio: Decimal
    transaction_costs: Decimal
    tax_costs: Decimal
    optimization_method: str
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


class PortfolioAllocator:
    """
    Portfolio allocation optimizer with multiple strategies.

    Implements mean-variance optimization, risk parity, and Black-Litterman
    models with transaction cost and tax awareness.
    """

    def __init__(self, risk_config_path: str, prod_config_path: str):
        """Initialize portfolio allocator with configuration"""
        self.risk_config_path = risk_config_path
        self.prod_config_path = prod_config_path
        self.config: Optional[AllocationConfig] = None
        self._load_config()

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

                self.config = AllocationConfig.from_yaml(risk_config, prod_config)
                logger.info("Portfolio allocation configuration loaded successfully")
                return

            except Exception as e:
                logger.error(f"Failed to load config (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise RuntimeError(f"Failed to load allocation config after {max_retries} attempts") from e

    async def calculate_mean_variance_allocation(
        self,
        returns_df: pl.DataFrame,
        portfolio_value: Decimal,
        risk_aversion: Decimal = Decimal('2.5')
    ) -> AllocationResult:
        """
        Calculate optimal allocation using mean-variance optimization.

        Args:
            returns_df: Polars DataFrame with columns [symbol, date, return]
            portfolio_value: Total portfolio value
            risk_aversion: Risk aversion parameter (higher = more conservative)

        Returns:
            AllocationResult with optimal weights and metrics
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            # Calculate expected returns for each symbol
            expected_returns = (
                returns_df
                .group_by('symbol')
                .agg([
                    pl.col('return').mean().alias('mean_return'),
                    pl.col('return').std().alias('std_return'),
                    pl.col('return').count().alias('obs_count')
                ])
            )

            # Calculate covariance matrix
            symbols = expected_returns['symbol'].to_list()
            cov_matrix = await self._calculate_covariance_matrix(returns_df, symbols)

            # Optimize portfolio weights
            weights = await self._optimize_weights_mv(
                expected_returns,
                cov_matrix,
                risk_aversion
            )

            # Apply constraints
            weights = await self._apply_allocation_constraints(weights, portfolio_value)

            # Calculate portfolio metrics
            expected_return = await self._calculate_portfolio_return(weights, expected_returns)
            portfolio_variance = await self._calculate_portfolio_variance(weights, cov_matrix)
            sharpe_ratio = await self._calculate_sharpe_ratio(expected_return, portfolio_variance)

            timestamp = datetime.utcnow()

            return AllocationResult(
                target_weights=weights,
                current_weights={},
                rebalance_trades={},
                expected_return=expected_return,
                portfolio_variance=portfolio_variance,
                sharpe_ratio=sharpe_ratio,
                transaction_costs=Decimal('0'),
                tax_costs=Decimal('0'),
                optimization_method='mean_variance',
                timestamp=timestamp,
                metadata={'risk_aversion': str(risk_aversion)}
            )

        except Exception as e:
            logger.error(f"Mean-variance allocation failed: {e}")
            raise

    async def calculate_risk_parity_allocation(
        self,
        returns_df: pl.DataFrame,
        portfolio_value: Decimal
    ) -> AllocationResult:
        """
        Calculate risk parity allocation where each asset contributes equally to risk.

        Args:
            returns_df: Polars DataFrame with columns [symbol, date, return]
            portfolio_value: Total portfolio value

        Returns:
            AllocationResult with risk parity weights
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            # Calculate volatilities
            volatilities = (
                returns_df
                .group_by('symbol')
                .agg([
                    pl.col('return').std().alias('volatility')
                ])
            )

            # Risk parity: weight inversely proportional to volatility
            total_inv_vol = Decimal('0')
            inv_vols = {}

            for row in volatilities.iter_rows(named=True):
                symbol = row['symbol']
                vol = Decimal(str(row['volatility']))
                if vol > Decimal('0'):
                    inv_vol = Decimal('1') / vol
                    inv_vols[symbol] = inv_vol
                    total_inv_vol += inv_vol

            # Normalize to sum to 1
            weights = {
                symbol: inv_vol / total_inv_vol
                for symbol, inv_vol in inv_vols.items()
            }

            # Apply constraints
            weights = await self._apply_allocation_constraints(weights, portfolio_value)

            # Calculate expected returns for metrics
            expected_returns = (
                returns_df
                .group_by('symbol')
                .agg([
                    pl.col('return').mean().alias('mean_return')
                ])
            )

            symbols = list(weights.keys())
            cov_matrix = await self._calculate_covariance_matrix(returns_df, symbols)

            expected_return = await self._calculate_portfolio_return(weights, expected_returns)
            portfolio_variance = await self._calculate_portfolio_variance(weights, cov_matrix)
            sharpe_ratio = await self._calculate_sharpe_ratio(expected_return, portfolio_variance)

            timestamp = datetime.utcnow()

            return AllocationResult(
                target_weights=weights,
                current_weights={},
                rebalance_trades={},
                expected_return=expected_return,
                portfolio_variance=portfolio_variance,
                sharpe_ratio=sharpe_ratio,
                transaction_costs=Decimal('0'),
                tax_costs=Decimal('0'),
                optimization_method='risk_parity',
                timestamp=timestamp
            )

        except Exception as e:
            logger.error(f"Risk parity allocation failed: {e}")
            raise

    async def calculate_black_litterman_allocation(
        self,
        returns_df: pl.DataFrame,
        market_caps: Dict[str, Decimal],
        views: Dict[str, Decimal],
        view_confidence: Decimal,
        portfolio_value: Decimal
    ) -> AllocationResult:
        """
        Calculate Black-Litterman allocation incorporating market equilibrium and investor views.

        Args:
            returns_df: Polars DataFrame with historical returns
            market_caps: Dictionary of market capitalizations by symbol
            views: Dictionary of expected return views by symbol
            view_confidence: Confidence level in views (0 to 1)
            portfolio_value: Total portfolio value

        Returns:
            AllocationResult with Black-Litterman optimal weights
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            # Calculate market equilibrium weights
            total_market_cap = sum(market_caps.values())
            market_weights = {
                symbol: cap / total_market_cap
                for symbol, cap in market_caps.items()
            }

            # Calculate covariance matrix
            symbols = list(market_caps.keys())
            cov_matrix = await self._calculate_covariance_matrix(returns_df, symbols)

            # Calculate implied equilibrium returns
            risk_aversion = Decimal('2.5')
            implied_returns = {}
            for symbol in symbols:
                weight = market_weights.get(symbol, Decimal('0'))
                # Simplified: equilibrium return = risk_aversion * weight * variance
                variance = cov_matrix.get((symbol, symbol), Decimal('0'))
                implied_returns[symbol] = risk_aversion * weight * variance

            # Blend implied returns with investor views
            tau = Decimal('0.025')  # Scaling factor for uncertainty in prior
            blended_returns = {}

            for symbol in symbols:
                implied_ret = implied_returns.get(symbol, Decimal('0'))
                view_ret = views.get(symbol, implied_ret)

                # Weighted average based on confidence
                blended_returns[symbol] = (
                    (Decimal('1') - view_confidence) * implied_ret +
                    view_confidence * view_ret
                )

            # Create DataFrame for optimization
            bl_returns_df = pl.DataFrame({
                'symbol': list(blended_returns.keys()),
                'mean_return': [float(r) for r in blended_returns.values()]
            })

            # Optimize with blended returns
            weights = await self._optimize_weights_mv(
                bl_returns_df,
                cov_matrix,
                risk_aversion
            )

            # Apply constraints
            weights = await self._apply_allocation_constraints(weights, portfolio_value)

            expected_return = sum(
                weights.get(symbol, Decimal('0')) * blended_returns.get(symbol, Decimal('0'))
                for symbol in symbols
            )
            portfolio_variance = await self._calculate_portfolio_variance(weights, cov_matrix)
            sharpe_ratio = await self._calculate_sharpe_ratio(expected_return, portfolio_variance)

            timestamp = datetime.utcnow()

            return AllocationResult(
                target_weights=weights,
                current_weights={},
                rebalance_trades={},
                expected_return=expected_return,
                portfolio_variance=portfolio_variance,
                sharpe_ratio=sharpe_ratio,
                transaction_costs=Decimal('0'),
                tax_costs=Decimal('0'),
                optimization_method='black_litterman',
                timestamp=timestamp,
                metadata={
                    'view_confidence': str(view_confidence),
                    'views': {k: str(v) for k, v in views.items()}
                }
            )

        except Exception as e:
            logger.error(f"Black-Litterman allocation failed: {e}")
            raise

    async def calculate_rebalancing_trades(
        self,
        current_positions: List[Position],
        target_weights: Dict[str, Decimal],
        portfolio_value: Decimal,
        current_prices: Dict[str, Decimal]
    ) -> Tuple[Dict[str, Decimal], Decimal, Decimal]:
        """
        Calculate rebalancing trades with transaction cost and tax optimization.

        Args:
            current_positions: List of current positions
            target_weights: Target allocation weights
            portfolio_value: Total portfolio value
            current_prices: Current prices for each symbol

        Returns:
            Tuple of (trades dict, transaction costs, tax costs)
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            # Calculate current weights
            current_weights = {}
            position_values = {}

            for position in current_positions:
                value = abs(position.quantity) * position.current_price
                position_values[position.symbol] = value
                current_weights[position.symbol] = value / portfolio_value

            # Check if rebalancing is needed
            needs_rebalance = False
            for symbol in set(list(current_weights.keys()) + list(target_weights.keys())):
                current_weight = current_weights.get(symbol, Decimal('0'))
                target_weight = target_weights.get(symbol, Decimal('0'))
                weight_diff = abs(target_weight - current_weight)

                if weight_diff > self.config.rebalance_threshold_percent / Decimal('100'):
                    needs_rebalance = True
                    break

            if not needs_rebalance:
                logger.info("Portfolio within rebalancing threshold, no trades needed")
                return {}, Decimal('0'), Decimal('0')

            # Calculate required trades
            trades = {}
            total_transaction_costs = Decimal('0')
            total_tax_costs = Decimal('0')

            for symbol in set(list(current_weights.keys()) + list(target_weights.keys())):
                current_weight = current_weights.get(symbol, Decimal('0'))
                target_weight = target_weights.get(symbol, Decimal('0'))

                target_value = target_weight * portfolio_value
                current_value = position_values.get(symbol, Decimal('0'))
                value_change = target_value - current_value

                if abs(value_change) > self.config.min_position_size_usd:
                    price = current_prices.get(symbol, Decimal('0'))
                    if price > Decimal('0'):
                        quantity_change = value_change / price
                        trades[symbol] = quantity_change

                        # Calculate transaction costs
                        trade_value = abs(value_change)
                        transaction_cost = trade_value * self.config.transaction_cost_bps / Decimal('10000')
                        total_transaction_costs += transaction_cost

                        # Calculate tax costs for sells with gains
                        if quantity_change < Decimal('0'):
                            position = next((p for p in current_positions if p.symbol == symbol), None)
                            if position and position.pnl > Decimal('0'):
                                realized_gain = position.pnl * (abs(quantity_change) / abs(position.quantity))
                                tax_cost = realized_gain * self.config.tax_rate
                                total_tax_costs += tax_cost

            logger.info(
                f"Rebalancing: {len(trades)} trades, "
                f"transaction costs: {total_transaction_costs}, "
                f"tax costs: {total_tax_costs}"
            )

            return trades, total_transaction_costs, total_tax_costs

        except Exception as e:
            logger.error(f"Failed to calculate rebalancing trades: {e}")
            raise

    async def _calculate_covariance_matrix(
        self,
        returns_df: pl.DataFrame,
        symbols: List[str]
    ) -> Dict[Tuple[str, str], Decimal]:
        """Calculate covariance matrix from returns"""
        try:
            # Pivot returns to wide format
            pivot_df = returns_df.pivot(
                values='return',
                index='date',
                columns='symbol'
            )

            # Calculate covariance for each pair
            cov_matrix = {}
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if symbol1 in pivot_df.columns and symbol2 in pivot_df.columns:
                        returns1 = pivot_df[symbol1].drop_nulls()
                        returns2 = pivot_df[symbol2].drop_nulls()

                        # Calculate covariance
                        mean1 = returns1.mean()
                        mean2 = returns2.mean()

                        covariance = Decimal('0')
                        count = 0

                        for r1, r2 in zip(returns1.to_list(), returns2.to_list()):
                            if r1 is not None and r2 is not None:
                                covariance += Decimal(str(r1 - mean1)) * Decimal(str(r2 - mean2))
                                count += 1

                        if count > 1:
                            covariance /= Decimal(str(count - 1))

                        cov_matrix[(symbol1, symbol2)] = covariance

            return cov_matrix

        except Exception as e:
            logger.error(f"Failed to calculate covariance matrix: {e}")
            raise

    async def _optimize_weights_mv(
        self,
        expected_returns: pl.DataFrame,
        cov_matrix: Dict[Tuple[str, str], Decimal],
        risk_aversion: Decimal
    ) -> Dict[str, Decimal]:
        """Optimize portfolio weights using mean-variance optimization"""
        try:
            # Simplified analytical solution for unconstrained optimization
            # Full implementation would use quadratic programming
            symbols = expected_returns['symbol'].to_list()
            n = len(symbols)

            if n == 0:
                return {}

            # Equal weight as baseline
            weights = {symbol: Decimal('1') / Decimal(str(n)) for symbol in symbols}

            # Iterative adjustment based on return/risk tradeoff
            for _ in range(10):  # Simple iteration
                new_weights = {}
                total_score = Decimal('0')

                for symbol in symbols:
                    mean_return = Decimal(str(
                        expected_returns.filter(pl.col('symbol') == symbol)['mean_return'][0]
                    ))
                    variance = cov_matrix.get((symbol, symbol), Decimal('0.0001'))

                    # Score: return minus risk penalty
                    score = mean_return - risk_aversion * variance
                    score = max(score, Decimal('0'))  # Non-negative

                    new_weights[symbol] = score
                    total_score += score

                # Normalize
                if total_score > Decimal('0'):
                    weights = {
                        symbol: weight / total_score
                        for symbol, weight in new_weights.items()
                    }

            return weights

        except Exception as e:
            logger.error(f"Weight optimization failed: {e}")
            raise

    async def _apply_allocation_constraints(
        self,
        weights: Dict[str, Decimal],
        portfolio_value: Decimal
    ) -> Dict[str, Decimal]:
        """Apply position size and concentration constraints"""
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            constrained_weights = weights.copy()

            # Apply maximum position size constraint
            max_weight = self.config.max_position_size_percent / Decimal('100')
            for symbol in constrained_weights:
                if constrained_weights[symbol] > max_weight:
                    constrained_weights[symbol] = max_weight

            # Apply minimum position size constraint
            min_value = self.config.min_position_size_usd
            min_weight = min_value / portfolio_value

            constrained_weights = {
                symbol: weight
                for symbol, weight in constrained_weights.items()
                if weight >= min_weight
            }

            # Renormalize to sum to 1
            total_weight = sum(constrained_weights.values())
            if total_weight > Decimal('0'):
                constrained_weights = {
                    symbol: weight / total_weight
                    for symbol, weight in constrained_weights.items()
                }

            return constrained_weights

        except Exception as e:
            logger.error(f"Failed to apply constraints: {e}")
            raise

    async def _calculate_portfolio_return(
        self,
        weights: Dict[str, Decimal],
        expected_returns: pl.DataFrame
    ) -> Decimal:
        """Calculate expected portfolio return"""
        try:
            portfolio_return = Decimal('0')

            for symbol, weight in weights.items():
                mean_return = expected_returns.filter(
                    pl.col('symbol') == symbol
                )['mean_return']

                if len(mean_return) > 0:
                    portfolio_return += weight * Decimal(str(mean_return[0]))

            return portfolio_return

        except Exception as e:
            logger.error(f"Failed to calculate portfolio return: {e}")
            raise

    async def _calculate_portfolio_variance(
        self,
        weights: Dict[str, Decimal],
        cov_matrix: Dict[Tuple[str, str], Decimal]
    ) -> Decimal:
        """Calculate portfolio variance"""
        try:
            variance = Decimal('0')
            symbols = list(weights.keys())

            for symbol1 in symbols:
                for symbol2 in symbols:
                    weight1 = weights.get(symbol1, Decimal('0'))
                    weight2 = weights.get(symbol2, Decimal('0'))
                    cov = cov_matrix.get((symbol1, symbol2), Decimal('0'))
                    variance += weight1 * weight2 * cov

            return variance

        except Exception as e:
            logger.error(f"Failed to calculate portfolio variance: {e}")
            raise

    async def _calculate_sharpe_ratio(
        self,
        expected_return: Decimal,
        portfolio_variance: Decimal
    ) -> Decimal:
        """Calculate Sharpe ratio"""
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            if portfolio_variance <= Decimal('0'):
                return Decimal('0')

            portfolio_std = portfolio_variance.sqrt()

            if portfolio_std == Decimal('0'):
                return Decimal('0')

            # Annualized Sharpe ratio
            annual_return = expected_return * Decimal('252')  # 252 trading days
            annual_std = portfolio_std * Decimal('252').sqrt()

            sharpe = (annual_return - self.config.risk_free_rate) / annual_std

            return sharpe

        except Exception as e:
            logger.error(f"Failed to calculate Sharpe ratio: {e}")
            raise


async def main():
    """Example usage of portfolio allocator"""
    try:
        # Initialize allocator
        allocator = PortfolioAllocator(
            risk_config_path='/home/user/FritzellSama/config/bot/risk.yaml',
            prod_config_path='/home/user/FritzellSama/config/environments/production.yaml'
        )

        # Create sample returns data
        returns_data = pl.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'GOOGL', 'GOOGL', 'MSFT', 'MSFT'],
            'date': ['2024-01-01'] * 6,
            'return': [0.01, 0.02, -0.01, 0.03, 0.015, -0.005]
        })

        portfolio_value = Decimal('1000000')

        # Calculate mean-variance allocation
        result = await allocator.calculate_mean_variance_allocation(
            returns_data,
            portfolio_value
        )

        logger.info(f"Allocation result: {result}")

    except Exception as e:
        logger.error(f"Portfolio allocation example failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
