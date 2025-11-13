"""
Portfolio Rebalancing Manager
CRITICAL: Automated portfolio rebalancing logic and execution
"""

import logging
import numpy as np
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class PortfolioRebalancer:
    """Manage portfolio rebalancing operations"""

    def __init__(self):
        """Initialize portfolio rebalancer with config"""
        self.config = get_config()

        # Load rebalancing parameters
        self.rebalance_threshold_pct = self.config.get_decimal("risk", "portfolio.rebalance_threshold_pct")
        self.rebalance_frequency_hours = self.config.get_int("risk", "portfolio.rebalance_frequency_hours")
        self.min_trade_size_usd = self.config.get_decimal("risk", "position_sizing.min_position_size_usd")
        self.max_position_size_pct = self.config.get_decimal("risk", "position_sizing.max_position_size_pct")

        # Track last rebalance time
        self.last_rebalance_time = None

        logger.info(
            f"PortfolioRebalancer initialized: threshold={self.rebalance_threshold_pct:.2%}, "
            f"frequency={self.rebalance_frequency_hours}h"
        )

    async def check_rebalance_needed(
        self,
        current_weights: Dict[str, Decimal],
        target_weights: Dict[str, Decimal],
        force_check: bool = False
    ) -> Tuple[bool, Dict[str, Decimal]]:
        """
        Check if portfolio rebalancing is needed

        Args:
            current_weights: Current portfolio weights by symbol
            target_weights: Target portfolio weights by symbol
            force_check: Force rebalance check regardless of time

        Returns:
            Tuple of (needs_rebalance, weight_deviations)
            weight_deviations: Dict mapping symbol to deviation from target
        """
        try:
            # Check time-based rebalancing
            current_time = datetime.now()
            if not force_check and self.last_rebalance_time is not None:
                time_since_rebalance = current_time - self.last_rebalance_time
                hours_elapsed = time_since_rebalance.total_seconds() / 3600

                if hours_elapsed < self.rebalance_frequency_hours:
                    logger.debug(
                        f"Rebalance check skipped: only {hours_elapsed:.1f}h elapsed "
                        f"(need {self.rebalance_frequency_hours}h)"
                    )
                    return False, {}

            # Calculate deviations from target weights
            weight_deviations = {}
            all_symbols = set(list(current_weights.keys()) + list(target_weights.keys()))

            for symbol in all_symbols:
                current = current_weights.get(symbol, Decimal("0"))
                target = target_weights.get(symbol, Decimal("0"))
                deviation = abs(current - target)
                weight_deviations[symbol] = deviation

            # Check if any asset exceeds threshold
            max_deviation = max(weight_deviations.values()) if weight_deviations else Decimal("0")
            needs_rebalance = max_deviation >= self.rebalance_threshold_pct

            if needs_rebalance:
                logger.info(
                    f"Rebalance needed: max deviation {max_deviation:.4%} "
                    f"exceeds threshold {self.rebalance_threshold_pct:.4%}"
                )
                # Log top deviations
                sorted_deviations = sorted(
                    weight_deviations.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                for symbol, dev in sorted_deviations:
                    logger.info(
                        f"  {symbol}: current={current_weights.get(symbol, Decimal('0')):.4%}, "
                        f"target={target_weights.get(symbol, Decimal('0')):.4%}, "
                        f"deviation={dev:.4%}"
                    )
            else:
                logger.debug(
                    f"Rebalance not needed: max deviation {max_deviation:.4%} "
                    f"below threshold {self.rebalance_threshold_pct:.4%}"
                )

            return needs_rebalance, weight_deviations

        except Exception as e:
            logger.error(f"Error checking rebalance need: {e}", exc_info=True)
            raise

    async def calculate_rebalance_trades(
        self,
        current_positions: Dict[str, Decimal],
        target_weights: Dict[str, Decimal],
        portfolio_value: Decimal,
        current_prices: Dict[str, Decimal]
    ) -> Dict[str, Dict[str, Decimal]]:
        """
        Calculate trades needed to rebalance portfolio

        Args:
            current_positions: Current positions (quantity held) by symbol
            target_weights: Target portfolio weights by symbol
            portfolio_value: Total portfolio value
            current_prices: Current prices by symbol

        Returns:
            Dictionary of trades by symbol:
            {
                'symbol': {
                    'action': 'buy' or 'sell',
                    'quantity': Decimal,
                    'current_quantity': Decimal,
                    'target_quantity': Decimal,
                    'current_value': Decimal,
                    'target_value': Decimal,
                    'trade_value': Decimal
                }
            }
        """
        try:
            if portfolio_value <= Decimal("0"):
                logger.error("Cannot calculate rebalance trades: invalid portfolio value")
                raise ValueError("Portfolio value must be positive")

            trades = {}
            all_symbols = set(list(current_positions.keys()) + list(target_weights.keys()))

            for symbol in all_symbols:
                # Get current position
                current_qty = current_positions.get(symbol, Decimal("0"))
                current_price = current_prices.get(symbol, Decimal("0"))

                if current_price <= Decimal("0"):
                    logger.warning(f"Invalid price for {symbol}, skipping")
                    continue

                current_value = current_qty * current_price
                current_weight = current_value / portfolio_value if portfolio_value > 0 else Decimal("0")

                # Get target allocation
                target_weight = target_weights.get(symbol, Decimal("0"))
                target_value = target_weight * portfolio_value
                target_qty = target_value / current_price if current_price > 0 else Decimal("0")

                # Calculate trade needed
                trade_qty = target_qty - current_qty
                trade_value = abs(trade_qty * current_price)

                # Skip if trade is too small
                if trade_value < self.min_trade_size_usd:
                    logger.debug(
                        f"Skipping {symbol}: trade value ${trade_value} below minimum "
                        f"${self.min_trade_size_usd}"
                    )
                    continue

                # Determine action
                if trade_qty > Decimal("0"):
                    action = "buy"
                elif trade_qty < Decimal("0"):
                    action = "sell"
                    trade_qty = abs(trade_qty)
                else:
                    continue  # No trade needed

                trades[symbol] = {
                    "action": action,
                    "quantity": trade_qty.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP),
                    "current_quantity": current_qty.quantize(Decimal("0.00000001")),
                    "target_quantity": target_qty.quantize(Decimal("0.00000001")),
                    "current_value": current_value.quantize(Decimal("0.01")),
                    "target_value": target_value.quantize(Decimal("0.01")),
                    "trade_value": trade_value.quantize(Decimal("0.01")),
                    "current_weight": current_weight.quantize(Decimal("0.000001")),
                    "target_weight": target_weight.quantize(Decimal("0.000001")),
                    "price": current_price.quantize(Decimal("0.00000001"))
                }

            logger.info(
                f"Calculated {len(trades)} rebalance trades: "
                f"{sum(1 for t in trades.values() if t['action'] == 'buy')} buys, "
                f"{sum(1 for t in trades.values() if t['action'] == 'sell')} sells"
            )

            # Log trade details
            total_buy_value = sum(
                t['trade_value'] for t in trades.values() if t['action'] == 'buy'
            )
            total_sell_value = sum(
                t['trade_value'] for t in trades.values() if t['action'] == 'sell'
            )
            logger.info(
                f"Total rebalance volume: buy=${total_buy_value:,.2f}, "
                f"sell=${total_sell_value:,.2f}"
            )

            return trades

        except Exception as e:
            logger.error(f"Error calculating rebalance trades: {e}", exc_info=True)
            raise

    async def execute_rebalance(
        self,
        trades: Dict[str, Dict[str, Decimal]],
        execution_callback: Optional[callable] = None,
        dry_run: bool = False
    ) -> Dict[str, any]:
        """
        Execute rebalancing trades

        Args:
            trades: Dictionary of trades from calculate_rebalance_trades()
            execution_callback: Optional async callback function to execute each trade
                                Should accept (symbol, action, quantity, price)
            dry_run: If True, simulate execution without placing real orders

        Returns:
            Dictionary with execution results:
            {
                'executed_trades': List[Dict],
                'failed_trades': List[Dict],
                'total_executed': int,
                'total_failed': int,
                'total_buy_value': Decimal,
                'total_sell_value': Decimal,
                'execution_time': float
            }
        """
        try:
            if not trades:
                logger.info("No trades to execute")
                return {
                    'executed_trades': [],
                    'failed_trades': [],
                    'total_executed': 0,
                    'total_failed': 0,
                    'total_buy_value': Decimal("0"),
                    'total_sell_value': Decimal("0"),
                    'execution_time': 0.0
                }

            start_time = datetime.now()
            executed_trades = []
            failed_trades = []

            logger.info(
                f"{'DRY RUN: ' if dry_run else ''}Executing {len(trades)} rebalance trades"
            )

            # Sort trades: sells first, then buys (to free up capital)
            sorted_trades = sorted(
                trades.items(),
                key=lambda x: (0 if x[1]['action'] == 'sell' else 1, x[0])
            )

            for symbol, trade in sorted_trades:
                try:
                    action = trade['action']
                    quantity = trade['quantity']
                    price = trade['price']
                    trade_value = trade['trade_value']

                    logger.info(
                        f"{'[DRY RUN] ' if dry_run else ''}{action.upper()} {quantity} "
                        f"{symbol} @ ${price} (value: ${trade_value:,.2f})"
                    )

                    if not dry_run and execution_callback:
                        # Execute trade via callback
                        result = await execution_callback(symbol, action, quantity, price)
                        if result.get('success', False):
                            executed_trades.append({
                                'symbol': symbol,
                                'action': action,
                                'quantity': quantity,
                                'price': price,
                                'value': trade_value,
                                'order_id': result.get('order_id'),
                                'timestamp': datetime.now().timestamp()
                            })
                        else:
                            raise Exception(result.get('error', 'Unknown error'))
                    else:
                        # Dry run - just log
                        executed_trades.append({
                            'symbol': symbol,
                            'action': action,
                            'quantity': quantity,
                            'price': price,
                            'value': trade_value,
                            'order_id': 'DRY_RUN',
                            'timestamp': datetime.now().timestamp()
                        })

                except Exception as e:
                    logger.error(f"Failed to execute trade for {symbol}: {e}")
                    failed_trades.append({
                        'symbol': symbol,
                        'action': action,
                        'quantity': quantity,
                        'error': str(e),
                        'timestamp': datetime.now().timestamp()
                    })

            # Calculate totals
            total_buy_value = sum(
                t['value'] for t in executed_trades if t['action'] == 'buy'
            )
            total_sell_value = sum(
                t['value'] for t in executed_trades if t['action'] == 'sell'
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # Update last rebalance time on success
            if executed_trades and not dry_run:
                self.last_rebalance_time = datetime.now()

            result = {
                'executed_trades': executed_trades,
                'failed_trades': failed_trades,
                'total_executed': len(executed_trades),
                'total_failed': len(failed_trades),
                'total_buy_value': total_buy_value,
                'total_sell_value': total_sell_value,
                'execution_time': execution_time,
                'dry_run': dry_run
            }

            logger.info(
                f"Rebalance {'DRY RUN ' if dry_run else ''}completed: "
                f"{len(executed_trades)} succeeded, {len(failed_trades)} failed, "
                f"time={execution_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Error executing rebalance: {e}", exc_info=True)
            raise

    async def optimize_rebalance_timing(
        self,
        market_data: pl.DataFrame,
        current_hour: int
    ) -> Tuple[bool, str]:
        """
        Determine optimal timing for rebalancing based on market conditions

        Args:
            market_data: Recent market data with columns ['timestamp', 'volatility', 'volume']
            current_hour: Current hour of day (0-23)

        Returns:
            Tuple of (should_rebalance_now, reason)
        """
        try:
            # Avoid rebalancing during high volatility
            if not market_data.is_empty():
                recent_data = market_data.tail(20)  # Last 20 periods
                current_vol = recent_data.select('volatility').tail(1).to_numpy()[0, 0]
                avg_vol = recent_data.select('volatility').mean().to_numpy()[0, 0]

                if current_vol > avg_vol * 1.5:
                    logger.info(
                        f"Delaying rebalance: high volatility "
                        f"(current={current_vol:.6f}, avg={avg_vol:.6f})"
                    )
                    return False, "high_volatility"

            # Avoid rebalancing during market open/close (high volatility hours)
            # Assuming US market hours: 9:30 AM - 4:00 PM ET
            if current_hour in [9, 10, 15, 16]:
                logger.info(
                    f"Delaying rebalance: market open/close hour (hour={current_hour})"
                )
                return False, "market_open_close"

            # Prefer mid-day rebalancing
            if 11 <= current_hour <= 14:
                logger.info("Optimal rebalance timing: mid-day trading hours")
                return True, "optimal_timing"

            # Acceptable but not optimal
            logger.info(f"Acceptable rebalance timing (hour={current_hour})")
            return True, "acceptable_timing"

        except Exception as e:
            logger.error(f"Error optimizing rebalance timing: {e}", exc_info=True)
            # Default to allowing rebalance on error
            return True, "error_default"

    async def calculate_rebalance_cost(
        self,
        trades: Dict[str, Dict[str, Decimal]],
        trading_fee_pct: Decimal = Decimal("0.001"),
        slippage_pct: Decimal = Decimal("0.001")
    ) -> Dict[str, Decimal]:
        """
        Estimate total cost of rebalancing

        Args:
            trades: Dictionary of trades from calculate_rebalance_trades()
            trading_fee_pct: Trading fee as percentage (default: 0.1%)
            slippage_pct: Expected slippage as percentage (default: 0.1%)

        Returns:
            Dictionary with cost breakdown:
            {
                'total_cost': Decimal,
                'trading_fees': Decimal,
                'slippage_cost': Decimal,
                'total_trade_value': Decimal,
                'cost_pct': Decimal
            }
        """
        try:
            total_trade_value = Decimal("0")
            trading_fees = Decimal("0")
            slippage_cost = Decimal("0")

            for symbol, trade in trades.items():
                trade_value = trade['trade_value']
                total_trade_value += trade_value

                # Calculate fees and slippage
                fee = trade_value * trading_fee_pct
                slippage = trade_value * slippage_pct

                trading_fees += fee
                slippage_cost += slippage

            total_cost = trading_fees + slippage_cost
            cost_pct = (total_cost / total_trade_value * Decimal("100")) if total_trade_value > 0 else Decimal("0")

            result = {
                'total_cost': total_cost.quantize(Decimal("0.01")),
                'trading_fees': trading_fees.quantize(Decimal("0.01")),
                'slippage_cost': slippage_cost.quantize(Decimal("0.01")),
                'total_trade_value': total_trade_value.quantize(Decimal("0.01")),
                'cost_pct': cost_pct.quantize(Decimal("0.0001"))
            }

            logger.info(
                f"Rebalance cost estimate: ${total_cost:,.2f} "
                f"({cost_pct:.4%} of trade value)"
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating rebalance cost: {e}", exc_info=True)
            raise
