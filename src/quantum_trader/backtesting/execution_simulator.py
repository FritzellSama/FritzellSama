"""
Execution Simulator - Simulate realistic order execution for backtesting.

This module simulates order execution with realistic fill prices, slippage,
partial fills, and market impact to create accurate backtest results.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum
import os

from structlog import get_logger
import polars as pl

logger = get_logger(__name__)


class ExecutionModel(str, Enum):
    """Order execution models."""
    IMMEDIATE = "immediate"
    REALISTIC = "realistic"
    CONSERVATIVE = "conservative"
    MARKET_IMPACT = "market_impact"


class ExecutionSimulator:
    """Simulates realistic order execution for backtesting.

    Attributes:
        config: Simulator configuration from environment
        execution_model: Execution model to use
        market_data: Current market data for execution
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize execution simulator.

        Args:
            config: Optional configuration override

        Raises:
            ValueError: If configuration invalid
        """
        self.config: Dict[str, Any] = config or self._load_config()
        self.execution_model: ExecutionModel = ExecutionModel(self.config['execution_model'])
        self.market_data: Optional[pl.DataFrame] = None

        logger.info("ExecutionSimulator initialized", model=self.execution_model.value)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables.

        Returns:
            Configuration dictionary
        """
        try:
            config = {
                'execution_model': os.getenv('EXECUTION_MODEL', 'realistic'),
                'slippage_bps': Decimal(os.getenv('SLIPPAGE_BPS', '5')),
                'market_impact_factor': Decimal(os.getenv('MARKET_IMPACT_FACTOR', '0.1')),
                'partial_fill_threshold': Decimal(os.getenv('PARTIAL_FILL_THRESHOLD', '0.3')),
                'fill_delay_ms': int(os.getenv('FILL_DELAY_MS', '10')),
                'use_bid_ask_spread': os.getenv('USE_BID_ASK_SPREAD', 'false').lower() == 'true',
                'bid_ask_spread_bps': Decimal(os.getenv('BID_ASK_SPREAD_BPS', '10')),
            }

            logger.debug("Execution simulator config loaded", config=config)
            return config

        except Exception as e:
            logger.error("Failed to load config", error=str(e))
            raise ValueError(f"Configuration load failed: {e}")

    async def execute_market_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: Decimal,
        current_price: Decimal,
        volume: Decimal,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Simulate market order execution.

        Args:
            order_id: Order identifier
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            current_price: Current market price
            volume: Current market volume
            timestamp: Execution timestamp

        Returns:
            Execution result dictionary

        Example:
            >>> result = await simulator.execute_market_order(
            ...     'ord_123',
            ...     'BTC/USDT',
            ...     'BUY',
            ...     Decimal('1.0'),
            ...     Decimal('50000'),
            ...     Decimal('100'),
            ...     datetime.now(timezone.utc)
            ... )
            >>> result['fill_price']
            Decimal('50025.00')
        """
        try:
            logger.debug(
                "Executing market order",
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=str(quantity)
            )

            # Add execution delay
            await asyncio.sleep(self.config['fill_delay_ms'] / 1000)

            # Calculate fill price based on execution model
            fill_price = await self._calculate_fill_price(
                side,
                quantity,
                current_price,
                volume
            )

            # Determine if partial fill
            filled_quantity, is_partial = await self._determine_fill_quantity(
                quantity,
                volume
            )

            # Calculate commission
            commission = await self._calculate_commission(
                filled_quantity,
                fill_price
            )

            result = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'order_type': 'MARKET',
                'requested_quantity': str(quantity),
                'filled_quantity': str(filled_quantity),
                'fill_price': str(fill_price),
                'commission': str(commission),
                'is_partial_fill': is_partial,
                'timestamp': timestamp,
                'execution_model': self.execution_model.value
            }

            logger.debug(
                "Market order executed",
                order_id=order_id,
                filled_qty=str(filled_quantity),
                fill_price=str(fill_price)
            )

            return result

        except Exception as e:
            logger.error("Market order execution failed", error=str(e))
            raise

    async def execute_limit_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: Decimal,
        limit_price: Decimal,
        current_price: Decimal,
        high_price: Decimal,
        low_price: Decimal,
        volume: Decimal,
        timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """Simulate limit order execution.

        Args:
            order_id: Order identifier
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            limit_price: Limit price
            current_price: Current market price
            high_price: Period high price
            low_price: Period low price
            volume: Current market volume
            timestamp: Execution timestamp

        Returns:
            Execution result dictionary, or None if order not filled

        Example:
            >>> result = await simulator.execute_limit_order(
            ...     'ord_124',
            ...     'BTC/USDT',
            ...     'BUY',
            ...     Decimal('1.0'),
            ...     Decimal('49000'),
            ...     Decimal('50000'),
            ...     Decimal('50500'),
            ...     Decimal('49500'),
            ...     Decimal('100'),
            ...     datetime.now(timezone.utc)
            ... )
        """
        try:
            logger.debug(
                "Executing limit order",
                order_id=order_id,
                side=side,
                limit_price=str(limit_price)
            )

            # Check if limit order would be filled
            would_fill = await self._check_limit_order_fill(
                side,
                limit_price,
                high_price,
                low_price
            )

            if not would_fill:
                logger.debug("Limit order not filled", order_id=order_id)
                return None

            # Add execution delay
            await asyncio.sleep(self.config['fill_delay_ms'] / 1000)

            # For limit orders, fill price is limit price (best case)
            # In realistic mode, might get slight improvement
            fill_price = limit_price

            if self.execution_model == ExecutionModel.REALISTIC:
                # Slight price improvement possible
                improvement = limit_price * Decimal('0.0001')
                if side == 'BUY':
                    fill_price = max(limit_price - improvement, low_price)
                else:
                    fill_price = min(limit_price + improvement, high_price)

            # Determine fill quantity
            filled_quantity, is_partial = await self._determine_fill_quantity(
                quantity,
                volume
            )

            # Calculate commission
            commission = await self._calculate_commission(
                filled_quantity,
                fill_price
            )

            result = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'order_type': 'LIMIT',
                'requested_quantity': str(quantity),
                'filled_quantity': str(filled_quantity),
                'fill_price': str(fill_price),
                'limit_price': str(limit_price),
                'commission': str(commission),
                'is_partial_fill': is_partial,
                'timestamp': timestamp,
                'execution_model': self.execution_model.value
            }

            logger.debug(
                "Limit order executed",
                order_id=order_id,
                fill_price=str(fill_price)
            )

            return result

        except Exception as e:
            logger.error("Limit order execution failed", error=str(e))
            raise

    async def _calculate_fill_price(
        self,
        side: str,
        quantity: Decimal,
        current_price: Decimal,
        volume: Decimal
    ) -> Decimal:
        """Calculate realistic fill price.

        Args:
            side: Order side
            quantity: Order quantity
            current_price: Current market price
            volume: Market volume

        Returns:
            Fill price
        """
        try:
            fill_price = current_price

            if self.execution_model == ExecutionModel.IMMEDIATE:
                # No slippage or impact
                return fill_price

            # Add bid-ask spread
            if self.config['use_bid_ask_spread']:
                spread_bps = self.config['bid_ask_spread_bps']
                spread = current_price * (spread_bps / Decimal('10000'))

                if side == 'BUY':
                    fill_price = current_price + (spread / Decimal('2'))
                else:
                    fill_price = current_price - (spread / Decimal('2'))

            # Add slippage
            slippage_bps = self.config['slippage_bps']
            slippage = current_price * (slippage_bps / Decimal('10000'))

            if side == 'BUY':
                fill_price = fill_price + slippage
            else:
                fill_price = fill_price - slippage

            # Add market impact (for large orders relative to volume)
            if self.execution_model == ExecutionModel.MARKET_IMPACT:
                if volume > Decimal('0'):
                    order_volume_ratio = quantity / volume
                    impact_factor = self.config['market_impact_factor']
                    market_impact = current_price * order_volume_ratio * impact_factor

                    if side == 'BUY':
                        fill_price = fill_price + market_impact
                    else:
                        fill_price = fill_price - market_impact

            return fill_price

        except Exception as e:
            logger.error("Failed to calculate fill price", error=str(e))
            return current_price

    async def _determine_fill_quantity(
        self,
        quantity: Decimal,
        volume: Decimal
    ) -> Tuple[Decimal, bool]:
        """Determine if order fully or partially fills.

        Args:
            quantity: Requested quantity
            volume: Market volume

        Returns:
            Tuple of (filled_quantity, is_partial)
        """
        try:
            if self.execution_model == ExecutionModel.IMMEDIATE:
                return quantity, False

            # Check if order size is too large relative to volume
            if volume > Decimal('0'):
                volume_ratio = quantity / volume

                if volume_ratio > self.config['partial_fill_threshold']:
                    # Partial fill
                    filled_quantity = quantity * Decimal('0.5')
                    return filled_quantity, True

            return quantity, False

        except Exception as e:
            logger.error("Failed to determine fill quantity", error=str(e))
            return quantity, False

    async def _check_limit_order_fill(
        self,
        side: str,
        limit_price: Decimal,
        high_price: Decimal,
        low_price: Decimal
    ) -> bool:
        """Check if limit order would be filled.

        Args:
            side: Order side
            limit_price: Limit price
            high_price: Period high price
            low_price: Period low price

        Returns:
            True if order would fill
        """
        try:
            if side == 'BUY':
                # Buy limit fills if market went at or below limit price
                return low_price <= limit_price
            else:
                # Sell limit fills if market went at or above limit price
                return high_price >= limit_price

        except Exception as e:
            logger.error("Failed to check limit fill", error=str(e))
            return False

    async def _calculate_commission(
        self,
        quantity: Decimal,
        price: Decimal
    ) -> Decimal:
        """Calculate trading commission.

        Args:
            quantity: Trade quantity
            price: Trade price

        Returns:
            Commission amount
        """
        try:
            # Use default commission rate from config
            commission_rate = Decimal(os.getenv('DEFAULT_FEE_RATE', '0.001'))

            trade_value = quantity * price
            commission = trade_value * commission_rate

            return commission

        except Exception as e:
            logger.error("Failed to calculate commission", error=str(e))
            return Decimal('0')

    def set_execution_model(self, model: ExecutionModel) -> None:
        """Set execution model.

        Args:
            model: Execution model to use

        Example:
            >>> simulator.set_execution_model(ExecutionModel.CONSERVATIVE)
        """
        try:
            self.execution_model = model
            logger.info("Execution model set", model=model.value)

        except Exception as e:
            logger.error("Failed to set execution model", error=str(e))
            raise

    async def estimate_execution_cost(
        self,
        side: str,
        quantity: Decimal,
        current_price: Decimal,
        volume: Decimal
    ) -> Decimal:
        """Estimate total execution cost including slippage and commission.

        Args:
            side: Order side
            quantity: Order quantity
            current_price: Current price
            volume: Market volume

        Returns:
            Estimated execution cost

        Example:
            >>> cost = await simulator.estimate_execution_cost(
            ...     'BUY', Decimal('10'), Decimal('50000'), Decimal('100')
            ... )
        """
        try:
            # Calculate expected fill price
            fill_price = await self._calculate_fill_price(
                side,
                quantity,
                current_price,
                volume
            )

            # Calculate slippage cost
            slippage_cost = abs(fill_price - current_price) * quantity

            # Calculate commission
            commission = await self._calculate_commission(quantity, fill_price)

            # Total execution cost
            total_cost = slippage_cost + commission

            logger.debug(
                "Execution cost estimated",
                slippage=str(slippage_cost),
                commission=str(commission),
                total=str(total_cost)
            )

            return total_cost

        except Exception as e:
            logger.error("Failed to estimate execution cost", error=str(e))
            raise
