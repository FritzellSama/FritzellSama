"""Strategy Manager for Lifecycle and State Management.

Manages the lifecycle, state, and coordination of all active trading strategies.
Handles initialization, execution, monitoring, and graceful shutdown.

Performance Target: <50ms per strategy coordination cycle
Capital Allocation: N/A (management module)
Risk: State corruption, resource leaks
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timezone, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class StrategyState(Enum):
    """Strategy lifecycle states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StrategyStatus:
    """Status of a managed strategy."""
    strategy_id: str
    strategy_type: str
    state: StrategyState
    signals_generated: int
    last_signal_time: Optional[datetime]
    performance_metrics: Dict[str, Any]
    health_status: str  # 'healthy', 'degraded', 'unhealthy'
    error_count: int
    last_error: Optional[str]
    uptime_seconds: float
    timestamp: datetime


@dataclass
class StrategyAllocation:
    """Capital allocation for a strategy."""
    strategy_id: str
    allocated_capital: Decimal
    used_capital: Decimal
    available_capital: Decimal
    max_capital: Decimal
    allocation_percent: Decimal
    last_updated: datetime


class StrategyManager:
    """Strategy lifecycle and state manager.

    Manages all trading strategies throughout their lifecycle:
    - Initialization and configuration
    - State management and transitions
    - Signal coordination
    - Health monitoring
    - Resource allocation
    - Graceful shutdown

    Key Features:
    - Thread-safe state management
    - Automatic health monitoring
    - Resource lifecycle management
    - Error handling and recovery
    - Performance tracking
    - Graceful degradation

    Attributes:
        strategies: Managed strategy instances
        states: Current states of strategies
        allocations: Capital allocations
        health_checks: Health check tasks

    Example:
        >>> manager = StrategyManager(config, factory, risk_manager)
        >>> await manager.initialize()
        >>> await manager.start_all()
        >>> signals = await manager.collect_signals(market_data)
        >>> await manager.shutdown()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        strategy_factory: Any,
        risk_manager: Any
    ) -> None:
        """Initialize strategy manager.

        Args:
            config: Global configuration
            strategy_factory: Strategy factory instance
            risk_manager: Risk manager instance

        Raises:
            ValueError: If configuration invalid
        """
        self.config = config
        self.strategy_factory = strategy_factory
        self.risk_manager = risk_manager

        # Strategy instances and metadata
        self.strategies: Dict[str, Any] = {}
        self.states: Dict[str, StrategyState] = {}
        self.status: Dict[str, StrategyStatus] = {}
        self.allocations: Dict[str, StrategyAllocation] = {}

        # Performance tracking
        self.signal_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.start_times: Dict[str, datetime] = {}

        # Health monitoring
        self.health_check_interval: int = config.get('health_check_interval_seconds', 60)
        self.health_check_tasks: Dict[str, asyncio.Task] = {}

        # Lifecycle control
        self.is_running: bool = False
        self.shutdown_requested: bool = False

        # Thread safety
        self._state_lock = asyncio.Lock()
        self._signal_lock = asyncio.Lock()

        # Manager statistics
        self.manager_start_time: Optional[datetime] = None
        self.total_signals_processed: int = 0

        logger.info(
            "strategy_manager_initialized",
            health_check_interval=self.health_check_interval
        )

    async def initialize(self) -> None:
        """Initialize all configured strategies.

        Raises:
            RuntimeError: If initialization fails
        """
        try:
            logger.info("strategy_manager_initializing")

            # Get enabled strategies from config
            enabled_strategies = self._get_enabled_strategies()

            # Initialize each strategy
            for strategy_type in enabled_strategies:
                await self._initialize_strategy(strategy_type)

            # Setup capital allocations
            await self._setup_allocations()

            self.manager_start_time = datetime.now(timezone.utc)
            logger.info(
                "strategy_manager_initialized_successfully",
                strategies_count=len(self.strategies)
            )

        except Exception as e:
            logger.error(
                "strategy_manager_initialization_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def _get_enabled_strategies(self) -> List[str]:
        """Get list of enabled strategies from configuration.

        Returns:
            List of enabled strategy types
        """
        enabled = []

        # Check global config
        strategies_config = self.config.get('strategies', {})

        for strategy_type, strategy_cfg in strategies_config.items():
            if isinstance(strategy_cfg, dict) and strategy_cfg.get('enabled', False):
                enabled.append(strategy_type)

        logger.debug("enabled_strategies_identified", count=len(enabled), strategies=enabled)
        return enabled

    async def _initialize_strategy(self, strategy_type: str) -> None:
        """Initialize a single strategy.

        Args:
            strategy_type: Type of strategy to initialize

        Raises:
            ValueError: If strategy initialization fails
        """
        async with self._state_lock:
            try:
                # Set state to initializing
                strategy_id = f"{strategy_type}_1"
                self.states[strategy_id] = StrategyState.INITIALIZING

                # Create strategy instance
                strategy_config = self.config.get('strategies', {}).get(strategy_type, {})
                strategy = self.strategy_factory.create_strategy(strategy_type, strategy_config)

                # Store strategy
                self.strategies[strategy_id] = strategy
                self.start_times[strategy_id] = datetime.now(timezone.utc)

                # Initialize status
                self.status[strategy_id] = StrategyStatus(
                    strategy_id=strategy_id,
                    strategy_type=strategy_type,
                    state=StrategyState.ACTIVE,
                    signals_generated=0,
                    last_signal_time=None,
                    performance_metrics={},
                    health_status='healthy',
                    error_count=0,
                    last_error=None,
                    uptime_seconds=0.0,
                    timestamp=datetime.now(timezone.utc)
                )

                # Set state to active
                self.states[strategy_id] = StrategyState.ACTIVE

                logger.info(
                    "strategy_initialized",
                    strategy_id=strategy_id,
                    strategy_type=strategy_type
                )

            except Exception as e:
                self.states[strategy_id] = StrategyState.ERROR
                self.error_counts[strategy_id] += 1

                logger.error(
                    "strategy_initialization_failed",
                    strategy_id=strategy_id,
                    strategy_type=strategy_type,
                    error=str(e)
                )
                raise

    async def _setup_allocations(self) -> None:
        """Setup capital allocations for strategies."""
        total_capital = Decimal(str(self.config.get('total_capital', 1000000)))

        for strategy_id, strategy in self.strategies.items():
            # Get allocation percentage from config
            strategy_type = self.status[strategy_id].strategy_type
            strategy_config = self.config.get('strategies', {}).get(strategy_type, {})
            allocation_pct = Decimal(str(strategy_config.get('allocation_percent', 10)))

            # Calculate allocated capital
            allocated = total_capital * allocation_pct / Decimal('100')

            self.allocations[strategy_id] = StrategyAllocation(
                strategy_id=strategy_id,
                allocated_capital=allocated,
                used_capital=Decimal('0'),
                available_capital=allocated,
                max_capital=allocated,
                allocation_percent=allocation_pct,
                last_updated=datetime.now(timezone.utc)
            )

            logger.debug(
                "strategy_allocation_setup",
                strategy_id=strategy_id,
                allocated_capital=float(allocated),
                allocation_percent=float(allocation_pct)
            )

    async def start_all(self) -> None:
        """Start all initialized strategies."""
        try:
            self.is_running = True

            # Start health monitoring for each strategy
            for strategy_id in self.strategies.keys():
                await self._start_health_monitoring(strategy_id)

            logger.info("all_strategies_started", count=len(self.strategies))

        except Exception as e:
            logger.error("strategy_start_failed", error=str(e))
            raise

    async def _start_health_monitoring(self, strategy_id: str) -> None:
        """Start health monitoring task for strategy.

        Args:
            strategy_id: Strategy identifier
        """
        async def health_check_loop():
            while not self.shutdown_requested:
                try:
                    await asyncio.sleep(self.health_check_interval)
                    await self._check_strategy_health(strategy_id)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(
                        "health_check_error",
                        strategy_id=strategy_id,
                        error=str(e)
                    )

        task = asyncio.create_task(health_check_loop())
        self.health_check_tasks[strategy_id] = task

        logger.debug("health_monitoring_started", strategy_id=strategy_id)

    async def _check_strategy_health(self, strategy_id: str) -> None:
        """Check health of a strategy.

        Args:
            strategy_id: Strategy to check
        """
        if strategy_id not in self.status:
            return

        status = self.status[strategy_id]

        # Check error rate
        if status.error_count > 10:
            status.health_status = 'unhealthy'
        elif status.error_count > 5:
            status.health_status = 'degraded'
        else:
            status.health_status = 'healthy'

        # Update uptime
        if strategy_id in self.start_times:
            uptime = (datetime.now(timezone.utc) - self.start_times[strategy_id]).total_seconds()
            status.uptime_seconds = uptime

        status.timestamp = datetime.now(timezone.utc)

        logger.debug(
            "health_check_completed",
            strategy_id=strategy_id,
            health=status.health_status,
            uptime=status.uptime_seconds
        )

    async def collect_signals(self, market_data: pl.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Collect signals from all active strategies.

        Args:
            market_data: Market data for signal generation

        Returns:
            Dictionary mapping strategy_id to list of signals

        Raises:
            RuntimeError: If signal collection fails critically
        """
        async with self._signal_lock:
            all_signals = {}

            for strategy_id, strategy in self.strategies.items():
                # Check if strategy is active
                if self.states.get(strategy_id) != StrategyState.ACTIVE:
                    continue

                try:
                    # Generate signals
                    signals = await asyncio.wait_for(
                        self._collect_strategy_signals(strategy_id, strategy, market_data),
                        timeout=5.0  # 5 second timeout
                    )

                    all_signals[strategy_id] = signals

                    # Update statistics
                    if signals:
                        self.signal_counts[strategy_id] += len(signals)
                        self.total_signals_processed += len(signals)

                        # Update status
                        if strategy_id in self.status:
                            self.status[strategy_id].signals_generated = self.signal_counts[strategy_id]
                            self.status[strategy_id].last_signal_time = datetime.now(timezone.utc)

                except asyncio.TimeoutError:
                    logger.warning("strategy_signal_timeout", strategy_id=strategy_id)
                    self.error_counts[strategy_id] += 1

                except Exception as e:
                    logger.error(
                        "strategy_signal_collection_failed",
                        strategy_id=strategy_id,
                        error=str(e)
                    )
                    self.error_counts[strategy_id] += 1

                    if strategy_id in self.status:
                        self.status[strategy_id].error_count = self.error_counts[strategy_id]
                        self.status[strategy_id].last_error = str(e)

            return all_signals

    async def _collect_strategy_signals(
        self,
        strategy_id: str,
        strategy: Any,
        market_data: pl.DataFrame
    ) -> List[Dict[str, Any]]:
        """Collect signals from a single strategy.

        Args:
            strategy_id: Strategy identifier
            strategy: Strategy instance
            market_data: Market data

        Returns:
            List of signals
        """
        # Check if strategy has generate_signals method
        if not hasattr(strategy, 'generate_signals'):
            logger.warning("strategy_missing_generate_signals", strategy_id=strategy_id)
            return []

        # Call strategy's signal generation
        signals = await strategy.generate_signals(market_data)

        return signals if signals else []

    async def pause_strategy(self, strategy_id: str) -> None:
        """Pause a strategy.

        Args:
            strategy_id: Strategy to pause
        """
        async with self._state_lock:
            if strategy_id in self.states:
                self.states[strategy_id] = StrategyState.PAUSED

                logger.info("strategy_paused", strategy_id=strategy_id)

    async def resume_strategy(self, strategy_id: str) -> None:
        """Resume a paused strategy.

        Args:
            strategy_id: Strategy to resume
        """
        async with self._state_lock:
            if strategy_id in self.states and self.states[strategy_id] == StrategyState.PAUSED:
                self.states[strategy_id] = StrategyState.ACTIVE

                logger.info("strategy_resumed", strategy_id=strategy_id)

    async def stop_strategy(self, strategy_id: str) -> None:
        """Stop a strategy.

        Args:
            strategy_id: Strategy to stop
        """
        async with self._state_lock:
            if strategy_id not in self.strategies:
                return

            self.states[strategy_id] = StrategyState.STOPPING

            # Cancel health check task
            if strategy_id in self.health_check_tasks:
                self.health_check_tasks[strategy_id].cancel()
                del self.health_check_tasks[strategy_id]

            # Remove strategy
            del self.strategies[strategy_id]
            self.states[strategy_id] = StrategyState.STOPPED

            logger.info("strategy_stopped", strategy_id=strategy_id)

    async def shutdown(self) -> None:
        """Gracefully shutdown all strategies."""
        try:
            logger.info("strategy_manager_shutdown_initiated")

            self.shutdown_requested = True
            self.is_running = False

            # Cancel all health check tasks
            for task in self.health_check_tasks.values():
                task.cancel()

            await asyncio.gather(*self.health_check_tasks.values(), return_exceptions=True)

            # Stop all strategies
            strategy_ids = list(self.strategies.keys())
            for strategy_id in strategy_ids:
                await self.stop_strategy(strategy_id)

            logger.info("strategy_manager_shutdown_complete")

        except Exception as e:
            logger.error("shutdown_error", error=str(e))
            raise

    def get_strategy_status(self, strategy_id: str) -> Optional[StrategyStatus]:
        """Get status of a strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            StrategyStatus or None
        """
        return self.status.get(strategy_id)

    def get_all_statuses(self) -> Dict[str, StrategyStatus]:
        """Get statuses of all strategies.

        Returns:
            Dictionary of strategy statuses
        """
        return self.status.copy()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get manager performance metrics.

        Returns:
            Performance metrics dictionary
        """
        uptime = 0.0
        if self.manager_start_time:
            uptime = (datetime.now(timezone.utc) - self.manager_start_time).total_seconds()

        return {
            'active_strategies': sum(1 for s in self.states.values() if s == StrategyState.ACTIVE),
            'total_strategies': len(self.strategies),
            'total_signals_processed': self.total_signals_processed,
            'manager_uptime_seconds': uptime,
            'health_checks_running': len(self.health_check_tasks),
            'is_running': self.is_running
        }
