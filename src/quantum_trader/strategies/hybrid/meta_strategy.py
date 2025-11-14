"""
Meta Strategy - Strategy of Strategies.

Dynamically allocates capital across multiple sub-strategies based on
their recent performance, market conditions, and correlation.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import deque
import polars as pl
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LIMIT = "STOP_LIMIT"


class SignalAction(Enum):
    """Signal action enumeration."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


@dataclass
class Order:
    """Trading order - immutable after creation."""
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Optional[Decimal] = None
    order_type: OrderType = OrderType.MARKET
    exchange: str = ""
    strategy: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    """Trading signal from strategy."""
    symbol: str
    action: SignalAction
    strength: Decimal
    confidence: Decimal
    timestamp: datetime
    strategy: str
    timeframe: str
    indicators: Dict[str, Decimal] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyAllocation:
    """Capital allocation for a sub-strategy."""
    strategy_name: str
    weight: Decimal  # Allocation weight (0.0 to 1.0)
    sharpe_ratio: Decimal
    recent_returns: Decimal
    max_drawdown: Decimal
    correlation_score: Decimal
    is_active: bool


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""
    strategy_name: str
    total_trades: int
    winning_trades: int
    total_pnl: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    avg_return_per_trade: Decimal
    last_update: datetime


class MetaStrategy:
    """
    Meta Strategy - Dynamic Multi-Strategy Allocation.

    Manages a portfolio of trading strategies, dynamically allocating
    capital based on performance, correlation, and market regime.

    Attributes:
        config: Strategy configuration
        risk_manager: Risk management instance
        sub_strategies: Dictionary of registered sub-strategies
        allocations: Current capital allocations
        performance_history: Historical performance tracking
    """

    def __init__(self, config: Dict[str, Any], risk_manager: Any) -> None:
        """
        Initialize Meta Strategy.

        Args:
            config: Configuration dictionary with strategy parameters
            risk_manager: RiskManager instance for position validation

        Raises:
            ValueError: If required config parameters missing
        """
        self.config = config
        self.risk_manager = risk_manager
        self._validate_config()

        # Load parameters from config
        self.rebalance_interval_seconds = int(config["rebalance_interval_seconds"])
        self.min_strategy_weight = Decimal(str(config["min_strategy_weight"]))
        self.max_strategy_weight = Decimal(str(config["max_strategy_weight"]))
        self.lookback_periods = int(config["lookback_periods"])
        self.min_sharpe_threshold = Decimal(str(config["min_sharpe_threshold"]))
        self.max_correlation = Decimal(str(config["max_correlation"]))
        self.performance_decay_factor = Decimal(str(config["performance_decay_factor"]))

        # State tracking
        self.sub_strategies: Dict[str, Any] = {}
        self.allocations: Dict[str, StrategyAllocation] = {}
        self.performance_history: Dict[str, deque] = {}
        self.signal_history: Dict[str, deque] = {}
        self.last_rebalance_time: Optional[datetime] = None
        self.total_capital = Decimal(str(config.get("total_capital", "100000")))

        logger.info(
            "meta_strategy_initialized",
            rebalance_interval=self.rebalance_interval_seconds,
            min_sharpe=float(self.min_sharpe_threshold)
        )

    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required_params = [
            "rebalance_interval_seconds",
            "min_strategy_weight",
            "max_strategy_weight",
            "lookback_periods",
            "min_sharpe_threshold",
            "max_correlation",
            "performance_decay_factor",
            "symbol",
            "exchange",
            "timeframe"
        ]

        missing = [p for p in required_params if p not in self.config]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")

    async def register_strategy(
        self,
        strategy_name: str,
        strategy_instance: Any,
        initial_weight: Optional[Decimal] = None
    ) -> None:
        """
        Register a sub-strategy with the meta strategy.

        Args:
            strategy_name: Unique strategy identifier
            strategy_instance: Strategy instance
            initial_weight: Initial allocation weight (default: equal weight)

        Raises:
            ValueError: If strategy_name already registered
        """
        if strategy_name in self.sub_strategies:
            raise ValueError(f"Strategy {strategy_name} already registered")

        self.sub_strategies[strategy_name] = strategy_instance

        # Initialize performance tracking
        self.performance_history[strategy_name] = deque(maxlen=self.lookback_periods)
        self.signal_history[strategy_name] = deque(maxlen=self.lookback_periods)

        # Set initial allocation
        if initial_weight is None:
            # Equal weight across all strategies
            n_strategies = len(self.sub_strategies)
            initial_weight = Decimal("1") / Decimal(str(n_strategies))

        self.allocations[strategy_name] = StrategyAllocation(
            strategy_name=strategy_name,
            weight=initial_weight,
            sharpe_ratio=Decimal("0"),
            recent_returns=Decimal("0"),
            max_drawdown=Decimal("0"),
            correlation_score=Decimal("0"),
            is_active=True
        )

        logger.info(
            "strategy_registered",
            strategy_name=strategy_name,
            initial_weight=float(initial_weight)
        )

    async def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """
        Generate aggregated signals from all sub-strategies.

        Args:
            market_data: Polars DataFrame with market data

        Returns:
            List of aggregated Signal objects

        Raises:
            ValueError: If market_data format invalid
        """
        try:
            all_signals = []

            if market_data.is_empty():
                logger.warning("empty_market_data")
                return all_signals

            # Get latest timestamp
            latest_row = market_data.sort("timestamp", descending=True).row(0, named=True)
            current_time = latest_row["timestamp"]

            # Check if rebalancing is needed
            if await self._should_rebalance(current_time):
                await self._rebalance_allocations(current_time)

            # Collect signals from each active sub-strategy
            strategy_signals: Dict[str, List[Signal]] = {}

            for strategy_name, strategy_instance in self.sub_strategies.items():
                allocation = self.allocations.get(strategy_name)

                if not allocation or not allocation.is_active:
                    continue

                if allocation.weight < self.min_strategy_weight:
                    logger.debug(
                        "skipping_low_weight_strategy",
                        strategy=strategy_name,
                        weight=float(allocation.weight)
                    )
                    continue

                try:
                    # Generate signals from sub-strategy
                    signals = await strategy_instance.generate_signals(market_data)

                    if signals:
                        strategy_signals[strategy_name] = signals
                        self.signal_history[strategy_name].extend(signals)

                        logger.debug(
                            "strategy_signals_generated",
                            strategy=strategy_name,
                            signal_count=len(signals)
                        )

                except Exception as e:
                    logger.error(
                        "strategy_signal_generation_failed",
                        strategy=strategy_name,
                        error=str(e)
                    )
                    continue

            # Aggregate signals
            aggregated_signals = await self._aggregate_signals(
                strategy_signals=strategy_signals,
                current_time=current_time
            )

            return aggregated_signals

        except Exception as e:
            logger.error("meta_signal_generation_failed", error=str(e), exc_info=True)
            raise

    async def _should_rebalance(self, current_time: datetime) -> bool:
        """
        Check if portfolio should be rebalanced.

        Args:
            current_time: Current timestamp

        Returns:
            True if rebalancing needed
        """
        if self.last_rebalance_time is None:
            return True

        time_delta = (current_time - self.last_rebalance_time).total_seconds()

        return time_delta >= self.rebalance_interval_seconds

    async def _rebalance_allocations(self, current_time: datetime) -> None:
        """
        Rebalance capital allocations across strategies.

        Args:
            current_time: Current timestamp
        """
        try:
            logger.info("rebalancing_allocations")

            # Calculate performance metrics for each strategy
            performances: Dict[str, StrategyPerformance] = {}

            for strategy_name in self.sub_strategies.keys():
                perf = await self._calculate_strategy_performance(strategy_name)
                performances[strategy_name] = perf

            # Calculate correlation matrix
            correlation_matrix = await self._calculate_correlation_matrix()

            # Optimize allocations
            new_weights = await self._optimize_allocations(
                performances=performances,
                correlation_matrix=correlation_matrix
            )

            # Update allocations
            for strategy_name, new_weight in new_weights.items():
                if strategy_name in self.allocations:
                    allocation = self.allocations[strategy_name]
                    old_weight = allocation.weight

                    allocation.weight = new_weight
                    allocation.sharpe_ratio = performances[strategy_name].sharpe_ratio
                    allocation.recent_returns = performances[strategy_name].avg_return_per_trade
                    allocation.max_drawdown = performances[strategy_name].max_drawdown

                    # Deactivate strategies with poor performance
                    if allocation.sharpe_ratio < self.min_sharpe_threshold:
                        allocation.is_active = False
                        allocation.weight = Decimal("0")

                    logger.info(
                        "allocation_updated",
                        strategy=strategy_name,
                        old_weight=float(old_weight),
                        new_weight=float(new_weight),
                        sharpe=float(allocation.sharpe_ratio)
                    )

            self.last_rebalance_time = current_time

        except Exception as e:
            logger.error("rebalancing_failed", error=str(e), exc_info=True)

    async def _calculate_strategy_performance(
        self,
        strategy_name: str
    ) -> StrategyPerformance:
        """
        Calculate performance metrics for a strategy.

        Args:
            strategy_name: Strategy identifier

        Returns:
            StrategyPerformance object
        """
        history = self.performance_history.get(strategy_name, deque())

        if not history:
            return StrategyPerformance(
                strategy_name=strategy_name,
                total_trades=0,
                winning_trades=0,
                total_pnl=Decimal("0"),
                sharpe_ratio=Decimal("0"),
                max_drawdown=Decimal("0"),
                avg_return_per_trade=Decimal("0"),
                last_update=datetime.now(timezone.utc)
            )

        # Calculate metrics from history
        returns = [Decimal(str(h.get("return", 0))) for h in history]

        total_trades = len(returns)
        winning_trades = sum(1 for r in returns if r > Decimal("0"))
        total_pnl = sum(returns)

        # Calculate Sharpe ratio
        if len(returns) >= 2:
            mean_return = sum(returns) / Decimal(str(len(returns)))
            variance = sum((r - mean_return) ** 2 for r in returns) / Decimal(str(len(returns)))
            std_dev = Decimal(str(np.sqrt(float(variance))))

            sharpe_ratio = mean_return / std_dev if std_dev > Decimal("0") else Decimal("0")
        else:
            sharpe_ratio = Decimal("0")

        # Calculate max drawdown
        cumulative = Decimal("0")
        peak = Decimal("0")
        max_dd = Decimal("0")

        for ret in returns:
            cumulative += ret
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_dd:
                max_dd = drawdown

        avg_return = total_pnl / Decimal(str(total_trades)) if total_trades > 0 else Decimal("0")

        return StrategyPerformance(
            strategy_name=strategy_name,
            total_trades=total_trades,
            winning_trades=winning_trades,
            total_pnl=total_pnl,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_dd,
            avg_return_per_trade=avg_return,
            last_update=datetime.now(timezone.utc)
        )

    async def _calculate_correlation_matrix(self) -> Dict[Tuple[str, str], Decimal]:
        """
        Calculate correlation matrix between strategies.

        Returns:
            Dictionary mapping (strategy1, strategy2) -> correlation
        """
        correlation_matrix = {}

        strategy_names = list(self.sub_strategies.keys())

        for i, strategy1 in enumerate(strategy_names):
            for strategy2 in strategy_names[i:]:
                if strategy1 == strategy2:
                    correlation_matrix[(strategy1, strategy2)] = Decimal("1")
                    continue

                # Calculate correlation based on signal directions
                corr = await self._calculate_signal_correlation(strategy1, strategy2)
                correlation_matrix[(strategy1, strategy2)] = corr
                correlation_matrix[(strategy2, strategy1)] = corr

        return correlation_matrix

    async def _calculate_signal_correlation(
        self,
        strategy1: str,
        strategy2: str
    ) -> Decimal:
        """
        Calculate correlation between two strategies' signals.

        Args:
            strategy1: First strategy name
            strategy2: Second strategy name

        Returns:
            Correlation coefficient (-1 to +1)
        """
        signals1 = list(self.signal_history.get(strategy1, deque()))
        signals2 = list(self.signal_history.get(strategy2, deque()))

        if not signals1 or not signals2:
            return Decimal("0")

        # Convert signals to directional values
        def signal_to_value(signal: Signal) -> Decimal:
            direction = Decimal("1") if signal.action == SignalAction.BUY else Decimal("-1")
            return direction * signal.strength

        # Align signals by timestamp (simplified - use overlapping period)
        min_length = min(len(signals1), len(signals2))

        if min_length < 2:
            return Decimal("0")

        values1 = [signal_to_value(s) for s in signals1[-min_length:]]
        values2 = [signal_to_value(s) for s in signals2[-min_length:]]

        # Calculate correlation
        mean1 = sum(values1) / Decimal(str(len(values1)))
        mean2 = sum(values2) / Decimal(str(len(values2)))

        covariance = sum(
            (v1 - mean1) * (v2 - mean2)
            for v1, v2 in zip(values1, values2)
        ) / Decimal(str(len(values1)))

        var1 = sum((v - mean1) ** 2 for v in values1) / Decimal(str(len(values1)))
        var2 = sum((v - mean2) ** 2 for v in values2) / Decimal(str(len(values2)))

        std1 = Decimal(str(np.sqrt(float(var1))))
        std2 = Decimal(str(np.sqrt(float(var2))))

        if std1 > Decimal("0") and std2 > Decimal("0"):
            correlation = covariance / (std1 * std2)
        else:
            correlation = Decimal("0")

        return max(Decimal("-1"), min(Decimal("1"), correlation))

    async def _optimize_allocations(
        self,
        performances: Dict[str, StrategyPerformance],
        correlation_matrix: Dict[Tuple[str, str], Decimal]
    ) -> Dict[str, Decimal]:
        """
        Optimize capital allocations using performance and correlation.

        Args:
            performances: Strategy performance metrics
            correlation_matrix: Strategy correlations

        Returns:
            Dictionary mapping strategy_name -> optimal_weight
        """
        # Simple optimization: weight by Sharpe ratio, penalize high correlation

        strategy_scores = {}

        for strategy_name, perf in performances.items():
            # Base score from Sharpe ratio
            score = max(Decimal("0"), perf.sharpe_ratio)

            # Penalize strategies with high correlation to others
            avg_correlation = Decimal("0")
            correlation_count = 0

            for other_strategy in performances.keys():
                if other_strategy != strategy_name:
                    corr = correlation_matrix.get((strategy_name, other_strategy), Decimal("0"))
                    avg_correlation += abs(corr)
                    correlation_count += 1

            if correlation_count > 0:
                avg_correlation /= Decimal(str(correlation_count))

            # Reduce score for highly correlated strategies
            correlation_penalty = Decimal("1") - (avg_correlation * Decimal("0.5"))
            score *= correlation_penalty

            strategy_scores[strategy_name] = max(Decimal("0"), score)

        # Normalize scores to weights
        total_score = sum(strategy_scores.values())

        if total_score > Decimal("0"):
            weights = {
                name: score / total_score
                for name, score in strategy_scores.items()
            }
        else:
            # Equal weight if no scores
            n = len(strategy_scores)
            weights = {name: Decimal("1") / Decimal(str(n)) for name in strategy_scores.keys()}

        # Apply weight constraints
        for name in weights:
            weights[name] = max(
                self.min_strategy_weight,
                min(self.max_strategy_weight, weights[name])
            )

        # Renormalize to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > Decimal("0"):
            weights = {name: w / total_weight for name, w in weights.items()}

        return weights

    async def _aggregate_signals(
        self,
        strategy_signals: Dict[str, List[Signal]],
        current_time: datetime
    ) -> List[Signal]:
        """
        Aggregate signals from multiple strategies.

        Args:
            strategy_signals: Dictionary mapping strategy_name -> signals
            current_time: Current timestamp

        Returns:
            List of aggregated signals
        """
        if not strategy_signals:
            return []

        # Group signals by action
        buy_signals = []
        sell_signals = []

        for strategy_name, signals in strategy_signals.items():
            allocation = self.allocations.get(strategy_name)
            if not allocation:
                continue

            for signal in signals:
                # Weight signal by strategy allocation
                weighted_signal = Signal(
                    symbol=signal.symbol,
                    action=signal.action,
                    strength=signal.strength * allocation.weight,
                    confidence=signal.confidence,
                    timestamp=signal.timestamp,
                    strategy=f"meta_{strategy_name}",
                    timeframe=signal.timeframe,
                    indicators=signal.indicators,
                    metadata={
                        **signal.metadata,
                        "sub_strategy": strategy_name,
                        "allocation_weight": str(allocation.weight)
                    }
                )

                if signal.action == SignalAction.BUY:
                    buy_signals.append(weighted_signal)
                elif signal.action == SignalAction.SELL:
                    sell_signals.append(weighted_signal)

        # Aggregate weighted signals
        aggregated = []

        if buy_signals:
            total_buy_strength = sum(s.strength for s in buy_signals)
            avg_buy_confidence = sum(s.confidence for s in buy_signals) / Decimal(str(len(buy_signals)))

            # Create aggregated buy signal
            buy_signal = Signal(
                symbol=buy_signals[0].symbol,
                action=SignalAction.BUY,
                strength=min(Decimal("1.0"), total_buy_strength),
                confidence=avg_buy_confidence,
                timestamp=current_time,
                strategy="meta_strategy",
                timeframe=self.config["timeframe"],
                indicators=await self.calculate_indicators(pl.DataFrame()),
                metadata={
                    "component_strategies": [s.metadata.get("sub_strategy") for s in buy_signals],
                    "component_count": len(buy_signals)
                }
            )
            aggregated.append(buy_signal)

        if sell_signals:
            total_sell_strength = sum(s.strength for s in sell_signals)
            avg_sell_confidence = sum(s.confidence for s in sell_signals) / Decimal(str(len(sell_signals)))

            sell_signal = Signal(
                symbol=sell_signals[0].symbol,
                action=SignalAction.SELL,
                strength=min(Decimal("1.0"), total_sell_strength),
                confidence=avg_sell_confidence,
                timestamp=current_time,
                strategy="meta_strategy",
                timeframe=self.config["timeframe"],
                indicators=await self.calculate_indicators(pl.DataFrame()),
                metadata={
                    "component_strategies": [s.metadata.get("sub_strategy") for s in sell_signals],
                    "component_count": len(sell_signals)
                }
            )
            aggregated.append(sell_signal)

        return aggregated

    async def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """
        Calculate meta-strategy indicators.

        Args:
            data: Market data (unused, kept for interface compatibility)

        Returns:
            Dictionary of indicator values
        """
        indicators = {}

        # Portfolio-level metrics
        active_strategies = sum(1 for a in self.allocations.values() if a.is_active)
        indicators["active_strategies"] = Decimal(str(active_strategies))

        # Average Sharpe across strategies
        if self.allocations:
            active_sharpes = [
                a.sharpe_ratio for a in self.allocations.values()
                if a.is_active and a.sharpe_ratio > Decimal("0")
            ]
            if active_sharpes:
                indicators["avg_sharpe"] = sum(active_sharpes) / Decimal(str(len(active_sharpes)))

        # Concentration (Herfindahl index)
        if self.allocations:
            concentration = sum(a.weight ** 2 for a in self.allocations.values() if a.is_active)
            indicators["concentration_index"] = concentration

        return indicators

    def validate_signal(self, signal: Signal) -> bool:
        """
        Validate aggregated signal.

        Args:
            signal: Signal to validate

        Returns:
            True if signal is valid
        """
        try:
            # Check strength
            if signal.strength <= Decimal("0") or signal.strength > Decimal("1"):
                return False

            # Check confidence
            min_confidence = Decimal(str(self.config.get("min_confidence", "0.3")))
            if signal.confidence < min_confidence:
                return False

            return True

        except Exception as e:
            logger.error("signal_validation_error", error=str(e))
            return False

    async def record_trade_result(
        self,
        strategy_name: str,
        pnl: Decimal,
        timestamp: datetime
    ) -> None:
        """
        Record trade result for performance tracking.

        Args:
            strategy_name: Strategy that generated the trade
            pnl: Profit/loss from trade
            timestamp: Trade timestamp
        """
        if strategy_name not in self.performance_history:
            return

        self.performance_history[strategy_name].append({
            "return": pnl,
            "timestamp": timestamp
        })

        logger.debug(
            "trade_result_recorded",
            strategy=strategy_name,
            pnl=float(pnl)
        )
