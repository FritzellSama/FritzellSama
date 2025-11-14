"""
Multi-Strategy Portfolio Approach.

Runs multiple strategy instances simultaneously with independent
risk allocations and combines their signals for diversified trading.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import deque
import polars as pl
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


class MultiStrategy:
    """
    Multi-Strategy Portfolio Approach.

    Manages multiple trading strategies in parallel, each with independent
    risk budgets, and aggregates their signals with conflict resolution.

    Attributes:
        config: Strategy configuration
        risk_manager: Risk management instance
        strategies: Dictionary of active sub-strategies
        strategy_allocations: Capital allocation per strategy
        signal_aggregator: Signal combination logic
    """

    def __init__(self, config: Dict[str, Any], risk_manager: Any) -> None:
        """
        Initialize Multi-Strategy.

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
        self.max_concurrent_strategies = int(config["max_concurrent_strategies"])
        self.signal_conflict_mode = config.get("signal_conflict_mode", "weighted_vote")
        self.min_agreement_threshold = Decimal(str(config["min_agreement_threshold"]))
        self.capital_allocation_mode = config.get("capital_allocation_mode", "equal_weight")
        self.rebalance_interval_seconds = int(config.get("rebalance_interval_seconds", 3600))

        # State tracking
        self.strategies: Dict[str, Any] = {}
        self.strategy_allocations: Dict[str, Decimal] = {}
        self.strategy_performance: Dict[str, Dict[str, Decimal]] = {}
        self.aggregated_signals: deque = deque(maxlen=100)
        self.last_rebalance: Optional[datetime] = None

        logger.info(
            "multi_strategy_initialized",
            max_strategies=self.max_concurrent_strategies,
            conflict_mode=self.signal_conflict_mode,
            allocation_mode=self.capital_allocation_mode
        )

    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required_params = [
            "max_concurrent_strategies",
            "min_agreement_threshold",
            "symbol",
            "exchange",
            "timeframe"
        ]

        missing = [p for p in required_params if p not in self.config]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")

    async def add_strategy(
        self,
        strategy_name: str,
        strategy_instance: Any,
        allocation: Optional[Decimal] = None
    ) -> None:
        """
        Add a sub-strategy to the portfolio.

        Args:
            strategy_name: Unique strategy identifier
            strategy_instance: Strategy object instance
            allocation: Capital allocation (0-1), None for automatic

        Raises:
            ValueError: If max strategies exceeded or duplicate name
        """
        if len(self.strategies) >= self.max_concurrent_strategies:
            raise ValueError(
                f"Maximum {self.max_concurrent_strategies} strategies allowed"
            )

        if strategy_name in self.strategies:
            raise ValueError(f"Strategy {strategy_name} already exists")

        self.strategies[strategy_name] = strategy_instance

        # Set allocation
        if allocation is None:
            # Equal weight allocation
            n_strategies = len(self.strategies)
            allocation = Decimal("1") / Decimal(str(n_strategies))

            # Rebalance existing allocations
            for name in self.strategy_allocations:
                self.strategy_allocations[name] = Decimal("1") / Decimal(str(n_strategies))

        self.strategy_allocations[strategy_name] = allocation

        # Initialize performance tracking
        self.strategy_performance[strategy_name] = {
            "total_signals": Decimal("0"),
            "winning_signals": Decimal("0"),
            "total_pnl": Decimal("0"),
            "sharpe": Decimal("0")
        }

        logger.info(
            "strategy_added",
            name=strategy_name,
            allocation=float(allocation),
            total_strategies=len(self.strategies)
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
            if market_data.is_empty():
                logger.warning("empty_market_data")
                return []

            # Validate required columns
            required_cols = ["timestamp", "symbol"]
            missing_cols = [c for c in required_cols if c not in market_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Get latest timestamp
            latest_row = market_data.sort("timestamp", descending=True).row(0, named=True)
            current_time = latest_row["timestamp"]
            symbol = latest_row["symbol"]

            # Check if rebalancing needed
            if await self._should_rebalance(current_time):
                await self._rebalance_allocations()

            # Collect signals from all strategies
            strategy_signals: Dict[str, List[Signal]] = {}

            for strategy_name, strategy_instance in self.strategies.items():
                try:
                    signals = await strategy_instance.generate_signals(market_data)

                    if signals:
                        strategy_signals[strategy_name] = signals

                        logger.debug(
                            "strategy_signals_collected",
                            strategy=strategy_name,
                            signal_count=len(signals)
                        )

                except Exception as e:
                    logger.error(
                        "strategy_signal_error",
                        strategy=strategy_name,
                        error=str(e)
                    )
                    continue

            # Aggregate signals using configured mode
            aggregated = await self._aggregate_signals(
                strategy_signals=strategy_signals,
                current_time=current_time,
                symbol=symbol
            )

            # Store aggregated signals
            self.aggregated_signals.extend(aggregated)

            return aggregated

        except Exception as e:
            logger.error("signal_generation_failed", error=str(e), exc_info=True)
            raise

    async def _aggregate_signals(
        self,
        strategy_signals: Dict[str, List[Signal]],
        current_time: datetime,
        symbol: str
    ) -> List[Signal]:
        """
        Aggregate signals from multiple strategies.

        Args:
            strategy_signals: Dictionary mapping strategy_name -> signals
            current_time: Current timestamp
            symbol: Trading symbol

        Returns:
            List of aggregated signals
        """
        if not strategy_signals:
            return []

        if self.signal_conflict_mode == "weighted_vote":
            return await self._weighted_vote_aggregation(
                strategy_signals, current_time, symbol
            )
        elif self.signal_conflict_mode == "unanimous":
            return await self._unanimous_aggregation(
                strategy_signals, current_time, symbol
            )
        elif self.signal_conflict_mode == "majority":
            return await self._majority_aggregation(
                strategy_signals, current_time, symbol
            )
        else:
            # Default: weighted vote
            return await self._weighted_vote_aggregation(
                strategy_signals, current_time, symbol
            )

    async def _weighted_vote_aggregation(
        self,
        strategy_signals: Dict[str, List[Signal]],
        current_time: datetime,
        symbol: str
    ) -> List[Signal]:
        """
        Aggregate using weighted voting based on allocations.

        Args:
            strategy_signals: Strategy signals dictionary
            current_time: Current timestamp
            symbol: Trading symbol

        Returns:
            List of aggregated signals
        """
        buy_weight = Decimal("0")
        sell_weight = Decimal("0")
        buy_signals = []
        sell_signals = []

        for strategy_name, signals in strategy_signals.items():
            allocation = self.strategy_allocations.get(strategy_name, Decimal("0"))

            for signal in signals:
                weighted_strength = signal.strength * allocation

                if signal.action == SignalAction.BUY:
                    buy_weight += weighted_strength
                    buy_signals.append(signal)
                elif signal.action == SignalAction.SELL:
                    sell_weight += weighted_strength
                    sell_signals.append(signal)

        aggregated = []

        # Generate buy signal if weight threshold met
        if buy_weight > sell_weight and buy_weight >= self.min_agreement_threshold:
            avg_confidence = (
                sum(s.confidence for s in buy_signals) / Decimal(str(len(buy_signals)))
                if buy_signals else Decimal("0.5")
            )

            indicators = await self.calculate_indicators(pl.DataFrame())

            buy_signal = Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                strength=min(Decimal("1.0"), buy_weight),
                confidence=avg_confidence,
                timestamp=current_time,
                strategy="multi_strategy",
                timeframe=self.config["timeframe"],
                indicators=indicators,
                metadata={
                    "total_weight": str(buy_weight),
                    "component_count": len(buy_signals),
                    "aggregation_mode": "weighted_vote"
                }
            )

            if self.validate_signal(buy_signal):
                aggregated.append(buy_signal)

        # Generate sell signal
        elif sell_weight > buy_weight and sell_weight >= self.min_agreement_threshold:
            avg_confidence = (
                sum(s.confidence for s in sell_signals) / Decimal(str(len(sell_signals)))
                if sell_signals else Decimal("0.5")
            )

            indicators = await self.calculate_indicators(pl.DataFrame())

            sell_signal = Signal(
                symbol=symbol,
                action=SignalAction.SELL,
                strength=min(Decimal("1.0"), sell_weight),
                confidence=avg_confidence,
                timestamp=current_time,
                strategy="multi_strategy",
                timeframe=self.config["timeframe"],
                indicators=indicators,
                metadata={
                    "total_weight": str(sell_weight),
                    "component_count": len(sell_signals),
                    "aggregation_mode": "weighted_vote"
                }
            )

            if self.validate_signal(sell_signal):
                aggregated.append(sell_signal)

        return aggregated

    async def _unanimous_aggregation(
        self,
        strategy_signals: Dict[str, List[Signal]],
        current_time: datetime,
        symbol: str
    ) -> List[Signal]:
        """
        Require all strategies to agree on direction.

        Args:
            strategy_signals: Strategy signals dictionary
            current_time: Current timestamp
            symbol: Trading symbol

        Returns:
            List of aggregated signals (empty if no unanimity)
        """
        if not strategy_signals:
            return []

        # Count signals by action
        action_counts = {
            SignalAction.BUY: 0,
            SignalAction.SELL: 0,
            SignalAction.HOLD: 0
        }

        all_signals = []

        for signals in strategy_signals.values():
            for signal in signals:
                if signal.action in action_counts:
                    action_counts[signal.action] += 1
                    all_signals.append(signal)

        # Check for unanimity (all active strategies agree)
        total_strategies = len(strategy_signals)

        if action_counts[SignalAction.BUY] == total_strategies:
            # Unanimous buy
            avg_strength = sum(s.strength for s in all_signals) / Decimal(str(len(all_signals)))
            avg_confidence = sum(s.confidence for s in all_signals) / Decimal(str(len(all_signals)))

            indicators = await self.calculate_indicators(pl.DataFrame())

            return [Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                strength=avg_strength,
                confidence=avg_confidence,
                timestamp=current_time,
                strategy="multi_strategy_unanimous",
                timeframe=self.config["timeframe"],
                indicators=indicators,
                metadata={"aggregation_mode": "unanimous"}
            )]

        elif action_counts[SignalAction.SELL] == total_strategies:
            # Unanimous sell
            avg_strength = sum(s.strength for s in all_signals) / Decimal(str(len(all_signals)))
            avg_confidence = sum(s.confidence for s in all_signals) / Decimal(str(len(all_signals)))

            indicators = await self.calculate_indicators(pl.DataFrame())

            return [Signal(
                symbol=symbol,
                action=SignalAction.SELL,
                strength=avg_strength,
                confidence=avg_confidence,
                timestamp=current_time,
                strategy="multi_strategy_unanimous",
                timeframe=self.config["timeframe"],
                indicators=indicators,
                metadata={"aggregation_mode": "unanimous"}
            )]

        return []

    async def _majority_aggregation(
        self,
        strategy_signals: Dict[str, List[Signal]],
        current_time: datetime,
        symbol: str
    ) -> List[Signal]:
        """
        Use majority voting for signal aggregation.

        Args:
            strategy_signals: Strategy signals dictionary
            current_time: Current timestamp
            symbol: Trading symbol

        Returns:
            List of aggregated signals
        """
        buy_votes = 0
        sell_votes = 0
        buy_signals = []
        sell_signals = []

        for signals in strategy_signals.values():
            # Each strategy gets one vote
            has_buy = any(s.action == SignalAction.BUY for s in signals)
            has_sell = any(s.action == SignalAction.SELL for s in signals)

            if has_buy:
                buy_votes += 1
                buy_signals.extend([s for s in signals if s.action == SignalAction.BUY])

            if has_sell:
                sell_votes += 1
                sell_signals.extend([s for s in signals if s.action == SignalAction.SELL])

        total_strategies = len(strategy_signals)
        majority_threshold = (total_strategies / 2) + 1

        aggregated = []

        if buy_votes >= majority_threshold and buy_votes > sell_votes:
            avg_strength = sum(s.strength for s in buy_signals) / Decimal(str(len(buy_signals)))
            avg_confidence = sum(s.confidence for s in buy_signals) / Decimal(str(len(buy_signals)))

            indicators = await self.calculate_indicators(pl.DataFrame())

            buy_signal = Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                strength=avg_strength,
                confidence=avg_confidence,
                timestamp=current_time,
                strategy="multi_strategy_majority",
                timeframe=self.config["timeframe"],
                indicators=indicators,
                metadata={
                    "votes": buy_votes,
                    "total_strategies": total_strategies,
                    "aggregation_mode": "majority"
                }
            )

            if self.validate_signal(buy_signal):
                aggregated.append(buy_signal)

        elif sell_votes >= majority_threshold and sell_votes > buy_votes:
            avg_strength = sum(s.strength for s in sell_signals) / Decimal(str(len(sell_signals)))
            avg_confidence = sum(s.confidence for s in sell_signals) / Decimal(str(len(sell_signals)))

            indicators = await self.calculate_indicators(pl.DataFrame())

            sell_signal = Signal(
                symbol=symbol,
                action=SignalAction.SELL,
                strength=avg_strength,
                confidence=avg_confidence,
                timestamp=current_time,
                strategy="multi_strategy_majority",
                timeframe=self.config["timeframe"],
                indicators=indicators,
                metadata={
                    "votes": sell_votes,
                    "total_strategies": total_strategies,
                    "aggregation_mode": "majority"
                }
            )

            if self.validate_signal(sell_signal):
                aggregated.append(sell_signal)

        return aggregated

    async def _should_rebalance(self, current_time: datetime) -> bool:
        """
        Check if strategy allocations should be rebalanced.

        Args:
            current_time: Current timestamp

        Returns:
            True if rebalancing needed
        """
        if self.last_rebalance is None:
            return True

        if self.capital_allocation_mode == "static":
            return False

        time_delta = (current_time - self.last_rebalance).total_seconds()

        return time_delta >= self.rebalance_interval_seconds

    async def _rebalance_allocations(self) -> None:
        """Rebalance capital allocations based on performance."""
        try:
            if self.capital_allocation_mode == "equal_weight":
                # Equal allocation
                n_strategies = len(self.strategies)
                if n_strategies > 0:
                    allocation = Decimal("1") / Decimal(str(n_strategies))
                    for name in self.strategies:
                        self.strategy_allocations[name] = allocation

            elif self.capital_allocation_mode == "performance_weighted":
                # Weight by Sharpe ratio
                total_sharpe = sum(
                    max(Decimal("0"), perf.get("sharpe", Decimal("0")))
                    for perf in self.strategy_performance.values()
                )

                if total_sharpe > Decimal("0"):
                    for name, perf in self.strategy_performance.items():
                        sharpe = max(Decimal("0"), perf.get("sharpe", Decimal("0")))
                        self.strategy_allocations[name] = sharpe / total_sharpe
                else:
                    # Fall back to equal weight
                    n = len(self.strategies)
                    for name in self.strategies:
                        self.strategy_allocations[name] = Decimal("1") / Decimal(str(n))

            self.last_rebalance = datetime.now(timezone.utc)

            logger.info(
                "allocations_rebalanced",
                allocations={k: float(v) for k, v in self.strategy_allocations.items()}
            )

        except Exception as e:
            logger.error("rebalancing_failed", error=str(e))

    async def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """
        Calculate multi-strategy indicators.

        Args:
            data: Market data (unused, kept for interface compatibility)

        Returns:
            Dictionary of indicator values
        """
        indicators = {}

        # Number of active strategies
        indicators["active_strategies"] = Decimal(str(len(self.strategies)))

        # Average allocation
        if self.strategy_allocations:
            avg_allocation = sum(self.strategy_allocations.values()) / Decimal(
                str(len(self.strategy_allocations))
            )
            indicators["avg_allocation"] = avg_allocation

        # Portfolio Sharpe (weighted average)
        total_sharpe = Decimal("0")
        for name, allocation in self.strategy_allocations.items():
            sharpe = self.strategy_performance.get(name, {}).get("sharpe", Decimal("0"))
            total_sharpe += sharpe * allocation

        indicators["portfolio_sharpe"] = total_sharpe

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
            if signal.confidence < Decimal("0.3"):
                return False

            # For weighted vote, check threshold
            if "total_weight" in signal.metadata:
                weight = Decimal(str(signal.metadata["total_weight"]))
                if weight < self.min_agreement_threshold:
                    return False

            return True

        except Exception as e:
            logger.error("signal_validation_error", error=str(e))
            return False

    async def update_strategy_performance(
        self,
        strategy_name: str,
        pnl: Decimal,
        success: bool
    ) -> None:
        """
        Update strategy performance metrics.

        Args:
            strategy_name: Strategy identifier
            pnl: Profit/loss from trade
            success: Whether trade was profitable
        """
        if strategy_name not in self.strategy_performance:
            return

        perf = self.strategy_performance[strategy_name]

        perf["total_signals"] += Decimal("1")
        if success:
            perf["winning_signals"] += Decimal("1")

        perf["total_pnl"] += pnl

        # Update Sharpe (simplified)
        if perf["total_signals"] > Decimal("0"):
            avg_pnl = perf["total_pnl"] / perf["total_signals"]
            # Simplified Sharpe = avg / std (std approximated)
            perf["sharpe"] = avg_pnl * Decimal("10")  # Simplified

        logger.debug(
            "strategy_performance_updated",
            strategy=strategy_name,
            total_signals=int(perf["total_signals"]),
            total_pnl=float(perf["total_pnl"])
        )
