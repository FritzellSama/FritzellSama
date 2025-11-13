"""
Portfolio Rebalancing Engine for Quantum Trader AI

Production-grade rebalancing implementation with:
- Threshold-based rebalancing
- Calendar-based rebalancing (daily, weekly, monthly)
- Tax-loss harvesting opportunities
- Cost-aware rebalancing (transaction costs)
- Drift monitoring and alerts
- Optimal rebalancing trade generation

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

from quantum_trader.models import Position, Order, OrderSide, OrderType


logger = logging.getLogger(__name__)


class RebalanceFrequency(Enum):
    """Rebalancing frequency options"""
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    THRESHOLD = "THRESHOLD"


class RebalanceReason(Enum):
    """Reason for rebalancing"""
    DRIFT_THRESHOLD = "DRIFT_THRESHOLD"
    CALENDAR = "CALENDAR"
    TAX_LOSS_HARVEST = "TAX_LOSS_HARVEST"
    RISK_LIMIT = "RISK_LIMIT"
    MANUAL = "MANUAL"


@dataclass
class RebalancingConfig:
    """Configuration for portfolio rebalancing"""
    enable_rebalancing: bool
    frequency: RebalanceFrequency
    drift_threshold_percent: Decimal
    min_trade_value_usd: Decimal
    max_trades_per_rebalance: int
    transaction_cost_percent: Decimal
    tax_loss_harvest_threshold: Decimal
    enable_tax_optimization: bool
    short_term_cap_gains_rate: Decimal
    long_term_cap_gains_rate: Decimal


@dataclass
class RebalanceAction:
    """Represents a rebalancing trade action"""
    symbol: str
    current_weight: Decimal
    target_weight: Decimal
    drift_percent: Decimal
    action: OrderSide
    quantity: Decimal
    estimated_value: Decimal
    reason: RebalanceReason
    priority: int
    estimated_cost: Decimal
    tax_impact: Optional[Decimal] = None


@dataclass
class RebalanceResult:
    """Result of rebalancing operation"""
    timestamp: datetime
    reason: RebalanceReason
    actions: List[RebalanceAction]
    total_drift: Decimal
    estimated_total_cost: Decimal
    estimated_tax_impact: Decimal
    orders_generated: int


class PortfolioRebalancer:
    """
    Portfolio rebalancing engine

    Manages portfolio rebalancing to maintain target allocations:
    - Threshold-based drift detection
    - Calendar-based scheduled rebalancing
    - Tax-aware trading
    - Transaction cost optimization
    """

    def __init__(
        self,
        config_path: str = "/home/user/FritzellSama/config/bot/risk.yaml",
        env_config_path: str = "/home/user/FritzellSama/config/environments/production.yaml"
    ):
        """
        Initialize portfolio rebalancer

        Args:
            config_path: Path to risk configuration file
            env_config_path: Path to environment configuration file
        """
        self.config = self._load_config(config_path, env_config_path)
        self.last_rebalance_time: Optional[datetime] = None
        self.rebalance_history: List[RebalanceResult] = []

        logger.info("PortfolioRebalancer initialized with frequency: %s", self.config.frequency.value)

    def _load_config(self, config_path: str, env_config_path: str) -> RebalancingConfig:
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

            concentration = risk_config.get('concentration_limits', {})

            return RebalancingConfig(
                enable_rebalancing=concentration.get('rebalance_enabled', True),
                frequency=RebalanceFrequency.THRESHOLD,
                drift_threshold_percent=Decimal(str(concentration.get('rebalance_threshold_percent', 15.0))),
                min_trade_value_usd=Decimal('100'),
                max_trades_per_rebalance=20,
                transaction_cost_percent=Decimal('0.1'),  # 0.1% transaction cost
                tax_loss_harvest_threshold=Decimal('-3.0'),  # -3% loss
                enable_tax_optimization=True,
                short_term_cap_gains_rate=Decimal('0.37'),  # 37% short-term rate
                long_term_cap_gains_rate=Decimal('0.20')    # 20% long-term rate
            )

        except Exception as e:
            logger.error("Failed to load configuration: %s", e)
            raise

    def should_rebalance(
        self,
        current_time: datetime,
        portfolio_df: pl.DataFrame,
        target_weights: Dict[str, Decimal]
    ) -> Tuple[bool, Optional[RebalanceReason]]:
        """
        Determine if portfolio should be rebalanced

        Args:
            current_time: Current timestamp
            portfolio_df: DataFrame with current portfolio positions
            target_weights: Target allocation weights by symbol

        Returns:
            Tuple of (should_rebalance, reason)
        """
        if not self.config.enable_rebalancing:
            return False, None

        # Check calendar-based rebalancing
        if self.config.frequency != RebalanceFrequency.THRESHOLD:
            if self._is_calendar_rebalance_due(current_time):
                return True, RebalanceReason.CALENDAR

        # Check drift threshold
        max_drift = self.calculate_portfolio_drift(portfolio_df, target_weights)

        if max_drift > self.config.drift_threshold_percent:
            logger.info(
                "Rebalancing needed: max drift %s%% exceeds threshold %s%%",
                max_drift, self.config.drift_threshold_percent
            )
            return True, RebalanceReason.DRIFT_THRESHOLD

        return False, None

    def _is_calendar_rebalance_due(self, current_time: datetime) -> bool:
        """
        Check if calendar-based rebalancing is due

        Args:
            current_time: Current timestamp

        Returns:
            True if rebalancing is due
        """
        if self.last_rebalance_time is None:
            return True

        time_since_last = current_time - self.last_rebalance_time

        if self.config.frequency == RebalanceFrequency.DAILY:
            return time_since_last >= timedelta(days=1)
        elif self.config.frequency == RebalanceFrequency.WEEKLY:
            return time_since_last >= timedelta(days=7)
        elif self.config.frequency == RebalanceFrequency.MONTHLY:
            return time_since_last >= timedelta(days=30)
        elif self.config.frequency == RebalanceFrequency.QUARTERLY:
            return time_since_last >= timedelta(days=90)

        return False

    def calculate_portfolio_drift(
        self,
        portfolio_df: pl.DataFrame,
        target_weights: Dict[str, Decimal]
    ) -> Decimal:
        """
        Calculate maximum drift from target allocation

        Args:
            portfolio_df: DataFrame with columns [symbol, market_value]
            target_weights: Target weights by symbol

        Returns:
            Maximum drift percentage
        """
        if not isinstance(portfolio_df, pl.DataFrame):
            raise TypeError(f"portfolio_df must be polars DataFrame")

        # Calculate total portfolio value
        total_value = Decimal(str(portfolio_df['market_value'].sum()))

        if total_value <= Decimal('0'):
            return Decimal('0')

        # Calculate current weights
        max_drift = Decimal('0')

        for row in portfolio_df.iter_rows(named=True):
            symbol = row['symbol']
            current_value = Decimal(str(row['market_value']))
            current_weight = (current_value / total_value) * Decimal('100')

            target_weight = target_weights.get(symbol, Decimal('0'))

            drift = abs(current_weight - target_weight)
            max_drift = max(max_drift, drift)

        return max_drift

    def generate_rebalance_actions(
        self,
        portfolio_df: pl.DataFrame,
        target_weights: Dict[str, Decimal],
        current_prices: Dict[str, Decimal],
        reason: RebalanceReason = RebalanceReason.DRIFT_THRESHOLD
    ) -> List[RebalanceAction]:
        """
        Generate optimal rebalancing trades

        Args:
            portfolio_df: Current portfolio DataFrame
            target_weights: Target allocation weights
            current_prices: Current prices by symbol
            reason: Reason for rebalancing

        Returns:
            List of rebalancing actions
        """
        if not isinstance(portfolio_df, pl.DataFrame):
            raise TypeError(f"portfolio_df must be polars DataFrame")

        # Calculate total portfolio value
        total_value = Decimal(str(portfolio_df['market_value'].sum()))

        if total_value <= Decimal('0'):
            logger.warning("Portfolio value is zero, cannot generate rebalance actions")
            return []

        actions = []

        # Calculate current weights and required changes
        for row in portfolio_df.iter_rows(named=True):
            symbol = row['symbol']
            current_value = Decimal(str(row['market_value']))
            current_quantity = Decimal(str(row['quantity']))
            current_weight = (current_value / total_value) * Decimal('100')

            target_weight = target_weights.get(symbol, Decimal('0'))
            drift = current_weight - target_weight

            # Skip if drift is below threshold
            if abs(drift) < Decimal('1.0'):  # 1% minimum drift
                continue

            # Calculate target value and quantity change
            target_value = (target_weight / Decimal('100')) * total_value
            value_change = target_value - current_value

            current_price = current_prices.get(symbol, Decimal('0'))

            if current_price <= Decimal('0'):
                logger.warning("Price not available for %s, skipping", symbol)
                continue

            quantity_change = value_change / current_price

            # Determine action
            if quantity_change > Decimal('0'):
                action_side = OrderSide.BUY
                quantity = abs(quantity_change)
            else:
                action_side = OrderSide.SELL
                quantity = abs(quantity_change)

            # Skip if trade value too small
            trade_value = abs(value_change)
            if trade_value < self.config.min_trade_value_usd:
                continue

            # Estimate transaction cost
            estimated_cost = trade_value * (self.config.transaction_cost_percent / Decimal('100'))

            # Calculate priority (higher drift = higher priority)
            priority = int(abs(drift))

            action = RebalanceAction(
                symbol=symbol,
                current_weight=current_weight,
                target_weight=target_weight,
                drift_percent=drift,
                action=action_side,
                quantity=quantity,
                estimated_value=trade_value,
                reason=reason,
                priority=priority,
                estimated_cost=estimated_cost
            )

            actions.append(action)

        # Sort by priority (highest drift first)
        actions.sort(key=lambda a: a.priority, reverse=True)

        # Limit number of trades
        actions = actions[:self.config.max_trades_per_rebalance]

        logger.info("Generated %d rebalance actions", len(actions))

        return actions

    def identify_tax_loss_harvesting_opportunities(
        self,
        portfolio_df: pl.DataFrame,
        current_prices: Dict[str, Decimal]
    ) -> List[RebalanceAction]:
        """
        Identify tax-loss harvesting opportunities

        Finds positions with losses that can be sold to offset gains

        Args:
            portfolio_df: Portfolio with columns [symbol, quantity, entry_price, market_value, opened_at]
            current_prices: Current prices by symbol

        Returns:
            List of tax-loss harvesting actions
        """
        if not isinstance(portfolio_df, pl.DataFrame):
            raise TypeError(f"portfolio_df must be polars DataFrame")

        if not self.config.enable_tax_optimization:
            return []

        actions = []

        for row in portfolio_df.iter_rows(named=True):
            symbol = row['symbol']
            quantity = Decimal(str(row['quantity']))
            entry_price = Decimal(str(row['entry_price']))
            current_price = current_prices.get(symbol, Decimal('0'))

            if current_price <= Decimal('0'):
                continue

            # Calculate unrealized loss
            loss_percent = ((current_price - entry_price) / entry_price) * Decimal('100')

            # Check if loss exceeds threshold
            if loss_percent < self.config.tax_loss_harvest_threshold:
                # This is a tax-loss harvesting opportunity
                market_value = current_price * quantity

                # Estimate tax benefit
                capital_loss = entry_price * quantity - current_price * quantity

                # Determine if short-term or long-term (assume short-term for conservative estimate)
                tax_benefit = capital_loss * self.config.short_term_cap_gains_rate

                action = RebalanceAction(
                    symbol=symbol,
                    current_weight=Decimal('0'),
                    target_weight=Decimal('0'),
                    drift_percent=loss_percent,
                    action=OrderSide.SELL,
                    quantity=quantity,
                    estimated_value=market_value,
                    reason=RebalanceReason.TAX_LOSS_HARVEST,
                    priority=int(abs(loss_percent)),
                    estimated_cost=market_value * (self.config.transaction_cost_percent / Decimal('100')),
                    tax_impact=tax_benefit
                )

                actions.append(action)

                logger.info(
                    "Tax-loss harvest opportunity: %s (loss: %s%%, tax benefit: $%s)",
                    symbol, loss_percent, tax_benefit
                )

        # Sort by tax benefit
        actions.sort(key=lambda a: a.tax_impact or Decimal('0'), reverse=True)

        return actions

    def calculate_optimal_rebalance_with_costs(
        self,
        actions: List[RebalanceAction],
        max_cost_percent: Decimal = Decimal('0.5')
    ) -> List[RebalanceAction]:
        """
        Optimize rebalancing actions considering transaction costs

        Filters out actions where cost exceeds expected benefit

        Args:
            actions: List of potential rebalance actions
            max_cost_percent: Maximum acceptable cost as % of trade value

        Returns:
            Filtered list of cost-effective actions
        """
        optimal_actions = []

        for action in actions:
            # Calculate cost as percentage of trade
            cost_percent = (action.estimated_cost / action.estimated_value * Decimal('100')
                          if action.estimated_value > 0 else Decimal('100'))

            # Skip if cost too high relative to trade value
            if cost_percent > max_cost_percent:
                logger.debug(
                    "Skipping %s rebalance: cost %s%% exceeds threshold %s%%",
                    action.symbol, cost_percent, max_cost_percent
                )
                continue

            # For tax-loss harvesting, always include if tax benefit > cost
            if action.reason == RebalanceReason.TAX_LOSS_HARVEST:
                if action.tax_impact and action.tax_impact > action.estimated_cost:
                    optimal_actions.append(action)
                    continue

            # For drift-based rebalancing, include if drift significant
            if abs(action.drift_percent) > Decimal('2.0'):  # 2% minimum for cost efficiency
                optimal_actions.append(action)

        logger.info(
            "Optimized rebalance: %d of %d actions are cost-effective",
            len(optimal_actions), len(actions)
        )

        return optimal_actions

    def execute_rebalance(
        self,
        portfolio_df: pl.DataFrame,
        target_weights: Dict[str, Decimal],
        current_prices: Dict[str, Decimal],
        reason: RebalanceReason = RebalanceReason.DRIFT_THRESHOLD
    ) -> RebalanceResult:
        """
        Execute portfolio rebalancing

        Args:
            portfolio_df: Current portfolio
            target_weights: Target allocation weights
            current_prices: Current prices
            reason: Rebalancing reason

        Returns:
            Rebalancing result
        """
        timestamp = datetime.now()

        # Generate actions
        actions = self.generate_rebalance_actions(
            portfolio_df,
            target_weights,
            current_prices,
            reason
        )

        # Optimize for costs
        actions = self.calculate_optimal_rebalance_with_costs(actions)

        # Calculate total drift
        total_drift = self.calculate_portfolio_drift(portfolio_df, target_weights)

        # Calculate totals
        total_cost = sum(a.estimated_cost for a in actions)
        total_tax_impact = sum(a.tax_impact or Decimal('0') for a in actions)

        result = RebalanceResult(
            timestamp=timestamp,
            reason=reason,
            actions=actions,
            total_drift=total_drift,
            estimated_total_cost=total_cost,
            estimated_tax_impact=total_tax_impact,
            orders_generated=len(actions)
        )

        # Update last rebalance time
        self.last_rebalance_time = timestamp

        # Store in history
        self.rebalance_history.append(result)

        logger.info(
            "Rebalancing executed: %d orders, total cost=$%s, drift=%s%%",
            len(actions), total_cost, total_drift
        )

        return result

    def generate_rebalance_orders(
        self,
        actions: List[RebalanceAction],
        exchange: str = "default",
        strategy: str = "rebalance"
    ) -> List[Order]:
        """
        Convert rebalance actions to executable orders

        Args:
            actions: List of rebalance actions
            exchange: Target exchange
            strategy: Strategy name

        Returns:
            List of orders ready for execution
        """
        orders = []

        for action in actions:
            order = Order(
                symbol=action.symbol,
                side=action.action,
                quantity=action.quantity,
                order_type=OrderType.MARKET,
                exchange=exchange,
                strategy=strategy,
                timestamp=datetime.now(),
                metadata={
                    'rebalance_reason': action.reason.value,
                    'target_weight': float(action.target_weight),
                    'current_weight': float(action.current_weight),
                    'drift_percent': float(action.drift_percent),
                    'priority': action.priority
                }
            )

            orders.append(order)

        logger.info("Generated %d rebalance orders", len(orders))

        return orders

    def get_drift_report(
        self,
        portfolio_df: pl.DataFrame,
        target_weights: Dict[str, Decimal]
    ) -> pl.DataFrame:
        """
        Generate drift report showing current vs target allocations

        Args:
            portfolio_df: Current portfolio
            target_weights: Target weights

        Returns:
            DataFrame with drift analysis
        """
        if not isinstance(portfolio_df, pl.DataFrame):
            raise TypeError(f"portfolio_df must be polars DataFrame")

        total_value = Decimal(str(portfolio_df['market_value'].sum()))

        if total_value <= Decimal('0'):
            return pl.DataFrame()

        drift_data = []

        for row in portfolio_df.iter_rows(named=True):
            symbol = row['symbol']
            current_value = Decimal(str(row['market_value']))
            current_weight = (current_value / total_value) * Decimal('100')
            target_weight = target_weights.get(symbol, Decimal('0'))
            drift = current_weight - target_weight

            drift_data.append({
                'symbol': symbol,
                'current_weight': float(current_weight),
                'target_weight': float(target_weight),
                'drift_percent': float(drift),
                'market_value': float(current_value),
                'needs_rebalance': abs(drift) > float(self.config.drift_threshold_percent)
            })

        return pl.DataFrame(drift_data)

    def get_rebalance_history(self) -> pl.DataFrame:
        """
        Get rebalancing history as DataFrame

        Returns:
            DataFrame with rebalancing history
        """
        if not self.rebalance_history:
            return pl.DataFrame()

        history_data = []

        for result in self.rebalance_history:
            history_data.append({
                'timestamp': result.timestamp,
                'reason': result.reason.value,
                'num_actions': result.orders_generated,
                'total_drift': float(result.total_drift),
                'total_cost': float(result.estimated_total_cost),
                'tax_impact': float(result.estimated_tax_impact)
            })

        return pl.DataFrame(history_data)

    def get_metrics(self) -> Dict[str, Decimal]:
        """
        Get rebalancing metrics

        Returns:
            Dictionary of metrics
        """
        return {
            'drift_threshold_percent': self.config.drift_threshold_percent,
            'min_trade_value_usd': self.config.min_trade_value_usd,
            'transaction_cost_percent': self.config.transaction_cost_percent,
            'rebalances_executed': Decimal(str(len(self.rebalance_history))),
            'last_rebalance_days_ago': (
                Decimal(str((datetime.now() - self.last_rebalance_time).days))
                if self.last_rebalance_time else Decimal('-1')
            )
        }
