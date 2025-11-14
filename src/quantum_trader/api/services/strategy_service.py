"""
Strategy Service - Business logic for strategy management.

This module handles strategy lifecycle management, performance tracking,
and configuration management with proper async operations.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import uuid
import os

from structlog import get_logger
import polars as pl

from quantum_trader.api.schemas.strategy_schemas import (
    StrategyResponse,
    StrategyCreateRequest,
    StrategyUpdateRequest,
    StrategyPerformanceResponse,
    StrategyConfigResponse,
    StrategyType,
    StrategyStatus
)
from quantum_trader.core.exceptions import (
    StrategyNotFoundError,
    ValidationError,
    ConfigurationError
)

logger = get_logger(__name__)


class StrategyService:
    """Service layer for strategy management operations.

    Attributes:
        config: Service configuration loaded from environment
        strategies: In-memory strategy registry
        performance_cache: Performance metrics cache
    """

    def __init__(self) -> None:
        """Initialize strategy service.

        Raises:
            ConfigurationError: If required configuration is missing
        """
        self.config: Dict[str, Any] = self._load_config()
        self.strategies: Dict[str, Dict[str, Any]] = {}
        self.performance_cache: Dict[str, Dict[str, Any]] = {}
        self._initialized: bool = False

        logger.info("StrategyService initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load service configuration from environment variables.

        Returns:
            Dict containing service configuration

        Raises:
            ConfigurationError: If required config missing
        """
        try:
            config = {
                'max_strategies': int(os.getenv('STRATEGY_MAX_COUNT', '100')),
                'performance_cache_ttl': int(os.getenv('STRATEGY_PERF_CACHE_TTL', '300')),
                'default_timeframe': os.getenv('STRATEGY_DEFAULT_TIMEFRAME', '1h'),
                'metrics_enabled': os.getenv('STRATEGY_METRICS_ENABLED', 'true').lower() == 'true',
            }

            logger.debug("Strategy service config loaded", config=config)
            return config

        except Exception as e:
            logger.error("Failed to load strategy service config", error=str(e))
            raise ConfigurationError(f"Configuration load failed: {e}")

    async def initialize(self) -> None:
        """Initialize service resources.

        Raises:
            ConfigurationError: If initialization fails
        """
        if self._initialized:
            return

        try:
            # Load existing strategies from storage (would be DB in production)
            await self._load_existing_strategies()

            self._initialized = True
            logger.info("StrategyService initialized successfully")

        except Exception as e:
            logger.error("StrategyService initialization failed", error=str(e))
            raise ConfigurationError(f"Service initialization failed: {e}")

    async def _load_existing_strategies(self) -> None:
        """Load existing strategies from persistent storage.

        In production, this would load from database.
        """
        try:
            # Simulated async DB load
            await asyncio.sleep(0.001)
            logger.debug("Loaded existing strategies from storage")

        except Exception as e:
            logger.error("Failed to load strategies", error=str(e))
            raise

    async def list_strategies(
        self,
        active_only: bool = False,
        exchange: Optional[str] = None
    ) -> List[StrategyResponse]:
        """List all strategies with optional filtering.

        Args:
            active_only: Return only active strategies
            exchange: Filter by exchange

        Returns:
            List of strategy responses

        Example:
            >>> strategies = await service.list_strategies(active_only=True)
            >>> len(strategies)
            5
        """
        try:
            logger.debug(
                "Listing strategies",
                active_only=active_only,
                exchange=exchange,
                total_count=len(self.strategies)
            )

            result: List[StrategyResponse] = []

            for strategy_id, strategy_data in self.strategies.items():
                # Apply filters
                if active_only and not strategy_data.get('is_active', False):
                    continue

                if exchange and strategy_data.get('exchange') != exchange:
                    continue

                result.append(self._to_strategy_response(strategy_id, strategy_data))

            logger.info("Strategies listed", count=len(result))
            return result

        except Exception as e:
            logger.error("Failed to list strategies", error=str(e))
            raise

    async def get_strategy(self, strategy_id: str) -> Optional[StrategyResponse]:
        """Get detailed strategy information.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Strategy response or None if not found

        Raises:
            StrategyNotFoundError: If strategy doesn't exist
        """
        try:
            logger.debug("Retrieving strategy", strategy_id=strategy_id)

            if strategy_id not in self.strategies:
                raise StrategyNotFoundError(f"Strategy {strategy_id} not found")

            strategy_data = self.strategies[strategy_id]
            return self._to_strategy_response(strategy_id, strategy_data)

        except StrategyNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to get strategy", error=str(e), strategy_id=strategy_id)
            raise

    async def create_strategy(self, request: StrategyCreateRequest) -> StrategyResponse:
        """Create a new trading strategy.

        Args:
            request: Strategy creation parameters

        Returns:
            Created strategy response

        Raises:
            ValidationError: If strategy validation fails
            ConfigurationError: If max strategies exceeded

        Example:
            >>> request = StrategyCreateRequest(name="Test", ...)
            >>> strategy = await service.create_strategy(request)
            >>> strategy.strategy_id
            'strat_abc123'
        """
        try:
            # Check max strategies limit
            if len(self.strategies) >= self.config['max_strategies']:
                raise ConfigurationError(
                    f"Maximum strategies limit ({self.config['max_strategies']}) reached"
                )

            # Validate request
            self._validate_strategy_request(request)

            # Generate unique ID
            strategy_id = f"strat_{uuid.uuid4().hex[:12]}"

            now = datetime.now(timezone.utc)

            strategy_data = {
                'name': request.name,
                'strategy_type': request.strategy_type.value,
                'description': request.description,
                'config': request.config,
                'risk_limits': request.risk_limits,
                'exchange': request.exchange,
                'symbols': request.symbols,
                'status': StrategyStatus.INACTIVE.value if not request.enabled else StrategyStatus.INITIALIZING.value,
                'is_active': False,
                'created_at': now,
                'updated_at': now,
                'started_at': None,
                'stopped_at': None
            }

            self.strategies[strategy_id] = strategy_data

            logger.info(
                "Strategy created",
                strategy_id=strategy_id,
                name=request.name,
                type=request.strategy_type.value
            )

            # If enabled, start it
            if request.enabled:
                await self._async_start_strategy(strategy_id)

            return self._to_strategy_response(strategy_id, strategy_data)

        except (ValidationError, ConfigurationError):
            raise
        except Exception as e:
            logger.error("Failed to create strategy", error=str(e))
            raise ValidationError(f"Strategy creation failed: {e}")

    async def update_strategy(
        self,
        strategy_id: str,
        request: StrategyUpdateRequest
    ) -> StrategyResponse:
        """Update an existing strategy.

        Args:
            strategy_id: Strategy to update
            request: Update parameters

        Returns:
            Updated strategy response

        Raises:
            StrategyNotFoundError: If strategy doesn't exist
            ValidationError: If update validation fails
        """
        try:
            if strategy_id not in self.strategies:
                raise StrategyNotFoundError(f"Strategy {strategy_id} not found")

            strategy_data = self.strategies[strategy_id]

            # Update fields if provided
            if request.name is not None:
                strategy_data['name'] = request.name

            if request.description is not None:
                strategy_data['description'] = request.description

            if request.config is not None:
                strategy_data['config'].update(request.config)

            if request.risk_limits is not None:
                strategy_data['risk_limits'].update(request.risk_limits)

            if request.enabled is not None:
                if request.enabled and not strategy_data['is_active']:
                    await self._async_start_strategy(strategy_id)
                elif not request.enabled and strategy_data['is_active']:
                    await self._async_stop_strategy(strategy_id)

            strategy_data['updated_at'] = datetime.now(timezone.utc)

            logger.info("Strategy updated", strategy_id=strategy_id)

            return self._to_strategy_response(strategy_id, strategy_data)

        except StrategyNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to update strategy", error=str(e), strategy_id=strategy_id)
            raise ValidationError(f"Strategy update failed: {e}")

    async def delete_strategy(self, strategy_id: str) -> None:
        """Delete a strategy.

        Args:
            strategy_id: Strategy to delete

        Raises:
            StrategyNotFoundError: If strategy doesn't exist
        """
        try:
            if strategy_id not in self.strategies:
                raise StrategyNotFoundError(f"Strategy {strategy_id} not found")

            # Stop if active
            if self.strategies[strategy_id]['is_active']:
                await self._async_stop_strategy(strategy_id)

            del self.strategies[strategy_id]

            # Clear performance cache
            if strategy_id in self.performance_cache:
                del self.performance_cache[strategy_id]

            logger.info("Strategy deleted", strategy_id=strategy_id)

        except StrategyNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to delete strategy", error=str(e), strategy_id=strategy_id)
            raise

    async def start_strategy(self, strategy_id: str) -> StrategyResponse:
        """Start a strategy.

        Args:
            strategy_id: Strategy to start

        Returns:
            Updated strategy response

        Raises:
            StrategyNotFoundError: If strategy doesn't exist
        """
        try:
            if strategy_id not in self.strategies:
                raise StrategyNotFoundError(f"Strategy {strategy_id} not found")

            await self._async_start_strategy(strategy_id)

            strategy_data = self.strategies[strategy_id]
            return self._to_strategy_response(strategy_id, strategy_data)

        except StrategyNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to start strategy", error=str(e), strategy_id=strategy_id)
            raise

    async def stop_strategy(self, strategy_id: str) -> StrategyResponse:
        """Stop a strategy.

        Args:
            strategy_id: Strategy to stop

        Returns:
            Updated strategy response

        Raises:
            StrategyNotFoundError: If strategy doesn't exist
        """
        try:
            if strategy_id not in self.strategies:
                raise StrategyNotFoundError(f"Strategy {strategy_id} not found")

            await self._async_stop_strategy(strategy_id)

            strategy_data = self.strategies[strategy_id]
            return self._to_strategy_response(strategy_id, strategy_data)

        except StrategyNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to stop strategy", error=str(e), strategy_id=strategy_id)
            raise

    async def get_strategy_performance(
        self,
        strategy_id: str
    ) -> StrategyPerformanceResponse:
        """Get strategy performance metrics.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Performance metrics

        Raises:
            StrategyNotFoundError: If strategy doesn't exist
        """
        try:
            if strategy_id not in self.strategies:
                raise StrategyNotFoundError(f"Strategy {strategy_id} not found")

            # Check cache
            if strategy_id in self.performance_cache:
                cached = self.performance_cache[strategy_id]
                return StrategyPerformanceResponse(**cached)

            # Calculate performance metrics
            performance = await self._calculate_performance(strategy_id)

            # Cache results
            self.performance_cache[strategy_id] = performance

            return StrategyPerformanceResponse(**performance)

        except StrategyNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to get performance", error=str(e), strategy_id=strategy_id)
            raise

    async def get_strategy_config(self, strategy_id: str) -> StrategyConfigResponse:
        """Get strategy configuration.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Strategy configuration

        Raises:
            StrategyNotFoundError: If strategy doesn't exist
        """
        try:
            if strategy_id not in self.strategies:
                raise StrategyNotFoundError(f"Strategy {strategy_id} not found")

            strategy_data = self.strategies[strategy_id]

            return StrategyConfigResponse(
                strategy_id=strategy_id,
                config=strategy_data['config'],
                risk_limits=strategy_data['risk_limits'],
                updated_at=strategy_data['updated_at']
            )

        except StrategyNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to get config", error=str(e), strategy_id=strategy_id)
            raise

    async def _async_start_strategy(self, strategy_id: str) -> None:
        """Internal method to start a strategy."""
        strategy_data = self.strategies[strategy_id]

        strategy_data['status'] = StrategyStatus.ACTIVE.value
        strategy_data['is_active'] = True
        strategy_data['started_at'] = datetime.now(timezone.utc)

        logger.info("Strategy started", strategy_id=strategy_id)

    async def _async_stop_strategy(self, strategy_id: str) -> None:
        """Internal method to stop a strategy."""
        strategy_data = self.strategies[strategy_id]

        strategy_data['status'] = StrategyStatus.INACTIVE.value
        strategy_data['is_active'] = False
        strategy_data['stopped_at'] = datetime.now(timezone.utc)

        logger.info("Strategy stopped", strategy_id=strategy_id)

    async def _calculate_performance(self, strategy_id: str) -> Dict[str, Any]:
        """Calculate performance metrics for a strategy.

        In production, this would query trade history from database.
        """
        # Simulated performance calculation
        await asyncio.sleep(0.001)

        now = datetime.now(timezone.utc)

        return {
            'strategy_id': strategy_id,
            'total_pnl': str(Decimal('0.00')),
            'total_pnl_percent': str(Decimal('0.00')),
            'sharpe_ratio': str(Decimal('0.00')),
            'sortino_ratio': str(Decimal('0.00')),
            'max_drawdown': str(Decimal('0.00')),
            'win_rate': str(Decimal('0.00')),
            'total_trades': 0,
            'profitable_trades': 0,
            'losing_trades': 0,
            'avg_win': str(Decimal('0.00')),
            'avg_loss': str(Decimal('0.00')),
            'avg_trade_duration': str(Decimal('0')),
            'period_start': now,
            'period_end': now
        }

    def _validate_strategy_request(self, request: StrategyCreateRequest) -> None:
        """Validate strategy creation request.

        Raises:
            ValidationError: If validation fails
        """
        # Validate symbols
        if not request.symbols:
            raise ValidationError("At least one symbol required")

        # Validate risk limits
        for limit_name, limit_value in request.risk_limits.items():
            try:
                value = Decimal(limit_value)
                if value <= Decimal('0'):
                    raise ValidationError(f"Risk limit {limit_name} must be positive")
            except Exception as e:
                raise ValidationError(f"Invalid risk limit {limit_name}: {e}")

    def _to_strategy_response(
        self,
        strategy_id: str,
        strategy_data: Dict[str, Any]
    ) -> StrategyResponse:
        """Convert internal strategy data to response model."""
        return StrategyResponse(
            strategy_id=strategy_id,
            name=strategy_data['name'],
            strategy_type=StrategyType(strategy_data['strategy_type']),
            description=strategy_data.get('description'),
            status=StrategyStatus(strategy_data['status']),
            is_active=strategy_data['is_active'],
            exchange=strategy_data['exchange'],
            symbols=strategy_data['symbols'],
            config=strategy_data['config'],
            risk_limits=strategy_data['risk_limits'],
            created_at=strategy_data['created_at'],
            updated_at=strategy_data['updated_at'],
            started_at=strategy_data.get('started_at'),
            stopped_at=strategy_data.get('stopped_at')
        )
