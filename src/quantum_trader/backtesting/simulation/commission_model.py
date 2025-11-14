"""
Commission Model - Trading fee and commission calculation for backtesting.

This module provides realistic commission and fee models for various exchanges
and order types to ensure accurate backtest results.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from enum import Enum
import os

from structlog import get_logger
import polars as pl

logger = get_logger(__name__)


class FeeStructure(str, Enum):
    """Commission fee structure types."""
    PERCENTAGE = "percentage"
    FIXED = "fixed"
    TIERED = "tiered"
    MAKER_TAKER = "maker_taker"


class CommissionModel:
    """Models trading commissions and fees for backtesting.

    Attributes:
        config: Commission configuration from environment
        exchange_fees: Fee schedules by exchange
        volume_tiers: Volume-based fee tiers
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize commission model.

        Args:
            config: Optional configuration override

        Raises:
            ValueError: If configuration invalid
        """
        self.config: Dict[str, Any] = config or self._load_config()
        self.exchange_fees: Dict[str, Dict[str, Any]] = self._load_exchange_fees()
        self.volume_tiers: Dict[str, List[Dict[str, Any]]] = self._load_volume_tiers()
        self._cumulative_volume: Dict[str, Decimal] = {}

        logger.info("CommissionModel initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables.

        Returns:
            Configuration dictionary
        """
        try:
            config = {
                'default_fee_rate': Decimal(os.getenv('DEFAULT_FEE_RATE', '0.001')),
                'default_exchange': os.getenv('DEFAULT_EXCHANGE', 'BINANCE'),
                'include_slippage': os.getenv('INCLUDE_SLIPPAGE', 'true').lower() == 'true',
                'slippage_rate': Decimal(os.getenv('SLIPPAGE_RATE', '0.0001')),
                'min_commission': Decimal(os.getenv('MIN_COMMISSION', '0.01')),
            }

            logger.debug("Commission model config loaded", config=config)
            return config

        except Exception as e:
            logger.error("Failed to load config", error=str(e))
            raise ValueError(f"Configuration load failed: {e}")

    def _load_exchange_fees(self) -> Dict[str, Dict[str, Any]]:
        """Load exchange-specific fee schedules.

        Returns:
            Dictionary mapping exchange to fee schedule
        """
        try:
            # Production would load from config files
            exchange_fees = {
                'BINANCE': {
                    'structure': FeeStructure.MAKER_TAKER.value,
                    'maker_fee': Decimal(os.getenv('BINANCE_MAKER_FEE', '0.001')),
                    'taker_fee': Decimal(os.getenv('BINANCE_TAKER_FEE', '0.001')),
                },
                'BYBIT': {
                    'structure': FeeStructure.MAKER_TAKER.value,
                    'maker_fee': Decimal(os.getenv('BYBIT_MAKER_FEE', '0.001')),
                    'taker_fee': Decimal(os.getenv('BYBIT_TAKER_FEE', '0.0006')),
                },
                'OKX': {
                    'structure': FeeStructure.TIERED.value,
                    'base_maker_fee': Decimal(os.getenv('OKX_MAKER_FEE', '0.0008')),
                    'base_taker_fee': Decimal(os.getenv('OKX_TAKER_FEE', '0.001')),
                },
                'KUCOIN': {
                    'structure': FeeStructure.MAKER_TAKER.value,
                    'maker_fee': Decimal(os.getenv('KUCOIN_MAKER_FEE', '0.001')),
                    'taker_fee': Decimal(os.getenv('KUCOIN_TAKER_FEE', '0.001')),
                },
                'BITGET': {
                    'structure': FeeStructure.MAKER_TAKER.value,
                    'maker_fee': Decimal(os.getenv('BITGET_MAKER_FEE', '0.001')),
                    'taker_fee': Decimal(os.getenv('BITGET_TAKER_FEE', '0.0006')),
                }
            }

            logger.debug("Exchange fees loaded", exchanges=list(exchange_fees.keys()))
            return exchange_fees

        except Exception as e:
            logger.error("Failed to load exchange fees", error=str(e))
            raise

    def _load_volume_tiers(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load volume-based fee tiers for tiered exchanges.

        Returns:
            Dictionary mapping exchange to tier schedules
        """
        try:
            # Production would load from config files
            volume_tiers = {
                'OKX': [
                    {
                        'min_volume': Decimal('0'),
                        'max_volume': Decimal('50000'),
                        'maker_fee': Decimal('0.0008'),
                        'taker_fee': Decimal('0.001')
                    },
                    {
                        'min_volume': Decimal('50000'),
                        'max_volume': Decimal('500000'),
                        'maker_fee': Decimal('0.0007'),
                        'taker_fee': Decimal('0.0009')
                    },
                    {
                        'min_volume': Decimal('500000'),
                        'max_volume': Decimal('999999999'),
                        'maker_fee': Decimal('0.0005'),
                        'taker_fee': Decimal('0.0008')
                    }
                ]
            }

            logger.debug("Volume tiers loaded")
            return volume_tiers

        except Exception as e:
            logger.error("Failed to load volume tiers", error=str(e))
            raise

    async def calculate_commission(
        self,
        exchange: str,
        order_type: str,
        quantity: Decimal,
        price: Decimal,
        is_maker: bool = False
    ) -> Decimal:
        """Calculate commission for a trade.

        Args:
            exchange: Exchange name
            order_type: Order type (MARKET, LIMIT, etc.)
            quantity: Order quantity
            price: Execution price
            is_maker: Whether order is maker or taker

        Returns:
            Commission amount in quote currency

        Example:
            >>> commission = await model.calculate_commission(
            ...     'BINANCE', 'LIMIT', Decimal('1.0'), Decimal('50000'), is_maker=True
            ... )
            >>> commission
            Decimal('50.00')
        """
        try:
            logger.debug(
                "Calculating commission",
                exchange=exchange,
                order_type=order_type,
                quantity=str(quantity),
                price=str(price),
                is_maker=is_maker
            )

            # Get exchange fee schedule
            if exchange not in self.exchange_fees:
                logger.warning(f"Unknown exchange {exchange}, using default fees")
                fee_rate = self.config['default_fee_rate']
            else:
                fee_rate = await self._get_fee_rate(exchange, is_maker, quantity, price)

            # Calculate base commission
            trade_value = quantity * price
            commission = trade_value * fee_rate

            # Apply minimum commission
            commission = max(commission, self.config['min_commission'])

            # Add slippage if configured
            if self.config['include_slippage'] and not is_maker:
                slippage = trade_value * self.config['slippage_rate']
                commission += slippage

            # Update cumulative volume
            if exchange not in self._cumulative_volume:
                self._cumulative_volume[exchange] = Decimal('0')
            self._cumulative_volume[exchange] += trade_value

            logger.debug("Commission calculated", commission=str(commission))

            return commission

        except Exception as e:
            logger.error("Commission calculation failed", error=str(e))
            raise

    async def calculate_batch_commissions(
        self,
        trades: pl.DataFrame
    ) -> pl.DataFrame:
        """Calculate commissions for a batch of trades.

        Args:
            trades: DataFrame with columns: exchange, order_type, quantity, price, is_maker

        Returns:
            DataFrame with added commission column

        Example:
            >>> trades_with_fees = await model.calculate_batch_commissions(trades_df)
            >>> trades_with_fees.select('commission').sum()
            Decimal('1234.56')
        """
        try:
            logger.info("Calculating batch commissions", trade_count=trades.height)

            commissions = []

            for row in trades.iter_rows(named=True):
                commission = await self.calculate_commission(
                    exchange=row['exchange'],
                    order_type=row.get('order_type', 'MARKET'),
                    quantity=Decimal(str(row['quantity'])),
                    price=Decimal(str(row['price'])),
                    is_maker=row.get('is_maker', False)
                )
                commissions.append(float(commission))

            # Add commission column
            result = trades.with_columns(
                pl.Series('commission', commissions)
            )

            total_commission = sum(Decimal(str(c)) for c in commissions)
            logger.info(
                "Batch commissions calculated",
                total_commission=str(total_commission)
            )

            return result

        except Exception as e:
            logger.error("Batch commission calculation failed", error=str(e))
            raise

    async def get_effective_price(
        self,
        exchange: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        is_maker: bool = False
    ) -> Decimal:
        """Calculate effective execution price including fees.

        Args:
            exchange: Exchange name
            side: Order side (BUY or SELL)
            quantity: Order quantity
            price: Nominal price
            is_maker: Whether order is maker

        Returns:
            Effective price including commissions

        Example:
            >>> effective = await model.get_effective_price(
            ...     'BINANCE', 'BUY', Decimal('1.0'), Decimal('50000')
            ... )
            >>> effective
            Decimal('50050.00')
        """
        try:
            logger.debug(
                "Calculating effective price",
                exchange=exchange,
                side=side,
                price=str(price)
            )

            # Calculate commission
            commission = await self.calculate_commission(
                exchange=exchange,
                order_type='MARKET',
                quantity=quantity,
                price=price,
                is_maker=is_maker
            )

            # Adjust price based on side
            if side.upper() == 'BUY':
                # Commission increases effective buy price
                effective_price = price + (commission / quantity)
            else:  # SELL
                # Commission decreases effective sell price
                effective_price = price - (commission / quantity)

            logger.debug("Effective price calculated", effective_price=str(effective_price))

            return effective_price

        except Exception as e:
            logger.error("Effective price calculation failed", error=str(e))
            raise

    async def estimate_total_fees(
        self,
        exchange: str,
        total_trades: int,
        avg_trade_value: Decimal,
        maker_ratio: Decimal = Decimal('0.5')
    ) -> Decimal:
        """Estimate total fees for a trading strategy.

        Args:
            exchange: Exchange name
            total_trades: Expected number of trades
            avg_trade_value: Average trade value
            maker_ratio: Ratio of maker orders (0.0 to 1.0)

        Returns:
            Estimated total fees

        Example:
            >>> total_fees = await model.estimate_total_fees(
            ...     'BINANCE', 1000, Decimal('10000'), Decimal('0.6')
            ... )
            >>> total_fees
            Decimal('9000.00')
        """
        try:
            logger.info(
                "Estimating total fees",
                exchange=exchange,
                total_trades=total_trades,
                avg_trade_value=str(avg_trade_value)
            )

            if maker_ratio < Decimal('0') or maker_ratio > Decimal('1'):
                raise ValueError("maker_ratio must be between 0 and 1")

            # Get fee rates
            maker_fee = await self._get_fee_rate(exchange, True, Decimal('1'), avg_trade_value)
            taker_fee = await self._get_fee_rate(exchange, False, Decimal('1'), avg_trade_value)

            # Calculate weighted average fee
            avg_fee = (maker_fee * maker_ratio) + (taker_fee * (Decimal('1') - maker_ratio))

            # Estimate total fees
            total_value = avg_trade_value * Decimal(total_trades)
            total_fees = total_value * avg_fee

            # Add slippage for taker orders if configured
            if self.config['include_slippage']:
                taker_trades = Decimal(total_trades) * (Decimal('1') - maker_ratio)
                slippage_cost = avg_trade_value * taker_trades * self.config['slippage_rate']
                total_fees += slippage_cost

            logger.info("Total fees estimated", total_fees=str(total_fees))

            return total_fees

        except Exception as e:
            logger.error("Fee estimation failed", error=str(e))
            raise

    async def _get_fee_rate(
        self,
        exchange: str,
        is_maker: bool,
        quantity: Decimal,
        price: Decimal
    ) -> Decimal:
        """Get fee rate for exchange considering structure and volume.

        Args:
            exchange: Exchange name
            is_maker: Whether order is maker
            quantity: Order quantity
            price: Order price

        Returns:
            Fee rate as decimal
        """
        try:
            if exchange not in self.exchange_fees:
                return self.config['default_fee_rate']

            fee_schedule = self.exchange_fees[exchange]
            structure = fee_schedule['structure']

            if structure == FeeStructure.MAKER_TAKER.value:
                return fee_schedule['maker_fee'] if is_maker else fee_schedule['taker_fee']

            elif structure == FeeStructure.TIERED.value:
                # Get current volume
                current_volume = self._cumulative_volume.get(exchange, Decimal('0'))

                # Find appropriate tier
                if exchange in self.volume_tiers:
                    for tier in self.volume_tiers[exchange]:
                        if tier['min_volume'] <= current_volume < tier['max_volume']:
                            return tier['maker_fee'] if is_maker else tier['taker_fee']

                # Fallback to base fee
                return fee_schedule['base_maker_fee'] if is_maker else fee_schedule['base_taker_fee']

            elif structure == FeeStructure.PERCENTAGE.value:
                return fee_schedule.get('fee_rate', self.config['default_fee_rate'])

            else:
                return self.config['default_fee_rate']

        except Exception as e:
            logger.error("Failed to get fee rate", error=str(e))
            return self.config['default_fee_rate']

    def reset_volume(self, exchange: Optional[str] = None) -> None:
        """Reset cumulative volume tracking.

        Args:
            exchange: Specific exchange to reset, or None for all
        """
        try:
            if exchange:
                self._cumulative_volume[exchange] = Decimal('0')
                logger.debug("Volume reset", exchange=exchange)
            else:
                self._cumulative_volume.clear()
                logger.debug("All volumes reset")

        except Exception as e:
            logger.error("Failed to reset volume", error=str(e))

    def get_cumulative_volume(self, exchange: str) -> Decimal:
        """Get cumulative trading volume for an exchange.

        Args:
            exchange: Exchange name

        Returns:
            Cumulative volume
        """
        return self._cumulative_volume.get(exchange, Decimal('0'))
