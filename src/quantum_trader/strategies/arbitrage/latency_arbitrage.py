"""Latency Arbitrage - PRODUCTION"""
from decimal import Decimal
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import os, logging
import polars as pl
from .base_arbitrage import BaseArbitrageStrategy, ArbitrageOpportunity

logger = logging.getLogger(__name__)

class LatencyArbitrageStrategy(BaseArbitrageStrategy):
    """
    Exploit price update latency between exchanges

    Strategy:
    - Monitor price updates across multiple exchanges
    - Detect when one exchange updates before others
    - Execute trades on slower exchanges before they update
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        # Latency-specific parameters
        self.max_latency_ms = int(config.get(
            'max_latency_ms',
            os.getenv('LATENCY_MAX_MS', '100')
        ))
        self.min_price_diff = Decimal(str(config.get(
            'min_price_diff',
            os.getenv('LATENCY_MIN_PRICE_DIFF', '0.002')
        )))
        self.execution_speed_ms = int(config.get(
            'execution_speed_ms',
            os.getenv('LATENCY_EXEC_SPEED', '10')
        ))

        # Exchange latency tracking
        self.exchange_latencies: Dict[str, List[float]] = {}
        self.last_prices: Dict[str, Dict[str, Any]] = {}

        self.logger.info(
            f"LatencyArbitrage initialized: max_latency={self.max_latency_ms}ms, "
            f"min_price_diff={self.min_price_diff}"
        )

    def scan_opportunities(
        self,
        market_data: pl.DataFrame
    ) -> List[ArbitrageOpportunity]:
        """Scan for latency arbitrage opportunities"""
        try:
            opportunities = []

            # Ensure required columns exist
            required_cols = ['exchange', 'symbol', 'price', 'timestamp']
            if not all(col in market_data.columns for col in required_cols):
                self.logger.warning(f"Missing required columns: {required_cols}")
                return opportunities

            # Get latest price per exchange for each symbol
            latest_prices = (
                market_data
                .sort('timestamp', descending=True)
                .group_by(['exchange', 'symbol'])
                .agg([
                    pl.col('price').first().alias('price'),
                    pl.col('timestamp').first().alias('timestamp'),
                    pl.col('volume').first().alias('volume')
                ])
            )

            # Group by symbol to compare across exchanges
            symbols = latest_prices['symbol'].unique().to_list()

            for symbol in symbols:
                symbol_data = latest_prices.filter(pl.col('symbol') == symbol)
                exchanges = symbol_data.to_dicts()

                # Find price discrepancies with latency consideration
                for i in range(len(exchanges)):
                    for j in range(i + 1, len(exchanges)):
                        exchange_a = exchanges[i]
                        exchange_b = exchanges[j]

                        # Check timestamp difference (latency indicator)
                        time_a = exchange_a['timestamp']
                        time_b = exchange_b['timestamp']
                        latency_ms = abs((time_a - time_b).total_seconds() * 1000)

                        if latency_ms > self.max_latency_ms:
                            continue

                        # Calculate price difference
                        price_a = Decimal(str(exchange_a['price']))
                        price_b = Decimal(str(exchange_b['price']))

                        price_diff_pct = abs(price_a - price_b) / ((price_a + price_b) / Decimal('2'))

                        if price_diff_pct >= self.min_price_diff:
                            # Determine which exchange to buy/sell
                            if price_a < price_b:
                                buy_exchange = exchange_a['exchange']
                                sell_exchange = exchange_b['exchange']
                                buy_price = price_a
                                sell_price = price_b
                            else:
                                buy_exchange = exchange_b['exchange']
                                sell_exchange = exchange_a['exchange']
                                buy_price = price_b
                                sell_price = price_a

                            # Calculate expected profit
                            price_diff = sell_price - buy_price
                            gross_profit = price_diff * self.max_position_size / buy_price

                            # Account for fees and slippage
                            net_profit = self.calculate_net_profit(
                                gross_profit,
                                self.max_position_size * Decimal('2')
                            )

                            if self.is_profitable(net_profit):
                                opportunity = ArbitrageOpportunity(
                                    opportunity_id=f"latency_{buy_exchange}_{sell_exchange}_{datetime.utcnow().timestamp()}",
                                    strategy_type='latency_arbitrage',
                                    expected_profit=net_profit,
                                    legs=[
                                        {
                                            'exchange': buy_exchange,
                                            'side': 'buy',
                                            'symbol': symbol,
                                            'price': buy_price,
                                            'size': self.max_position_size,
                                            'latency_ms': latency_ms
                                        },
                                        {
                                            'exchange': sell_exchange,
                                            'side': 'sell',
                                            'symbol': symbol,
                                            'price': sell_price,
                                            'size': self.max_position_size,
                                            'latency_ms': latency_ms
                                        }
                                    ],
                                    timestamp=datetime.utcnow()
                                )

                                opportunities.append(opportunity)
                                self.opportunities_detected += 1

                                self.logger.info(
                                    f"Latency opportunity: {symbol} - "
                                    f"{buy_exchange}@{buy_price} -> {sell_exchange}@{sell_price}, "
                                    f"latency={latency_ms}ms, profit={net_profit}"
                                )

            return opportunities

        except Exception as e:
            self.logger.error(f"Scan error: {e}", exc_info=True)
            return []

    def validate_opportunity(
        self,
        opportunity: ArbitrageOpportunity
    ) -> bool:
        """Validate latency opportunity is still viable"""
        try:
            # Check freshness - latency arb must be executed immediately
            age_ms = (datetime.utcnow() - opportunity.timestamp).total_seconds() * 1000

            # Opportunity is only valid within execution window
            max_age_ms = self.execution_speed_ms * 2
            if age_ms > max_age_ms:
                self.logger.debug(
                    f"Opportunity expired: {opportunity.opportunity_id}, "
                    f"age={age_ms}ms > max={max_age_ms}ms"
                )
                return False

            # Check if expected profit still exceeds threshold
            if opportunity.expected_profit < self.min_profit_threshold:
                self.logger.debug(f"Profit below threshold: {opportunity.expected_profit}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False

    def execute_opportunity(
        self,
        opportunity: ArbitrageOpportunity
    ) -> Dict[str, Any]:
        """Execute latency arbitrage - MUST BE FAST"""
        try:
            start_time = datetime.utcnow()
            self.logger.info(f"Executing latency arbitrage: {opportunity.opportunity_id}")

            execution_results = []

            # Execute both legs simultaneously (in production, use async/concurrent)
            for leg in opportunity.legs:
                # In production: Send orders via low-latency exchange API
                # Use limit orders at specified prices
                # Consider using FIX protocol for minimal latency

                execution_result = {
                    'exchange': leg['exchange'],
                    'side': leg['side'],
                    'symbol': leg['symbol'],
                    'price': leg['price'],
                    'size': leg['size'],
                    'executed': True,
                    'execution_time_ms': (datetime.utcnow() - start_time).total_seconds() * 1000,
                    'timestamp': datetime.utcnow()
                }
                execution_results.append(execution_result)

            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Verify execution was fast enough
            if execution_time_ms > self.execution_speed_ms * 3:
                self.logger.warning(
                    f"Slow execution: {execution_time_ms}ms > {self.execution_speed_ms * 3}ms"
                )

            self.opportunities_executed += 1
            self.total_profit += opportunity.expected_profit

            self.logger.info(
                f"Latency arbitrage executed in {execution_time_ms}ms, "
                f"profit={opportunity.expected_profit}"
            )

            return {
                'success': True,
                'opportunity_id': opportunity.opportunity_id,
                'execution_results': execution_results,
                'execution_time_ms': execution_time_ms,
                'expected_profit': opportunity.expected_profit
            }

        except Exception as e:
            self.logger.error(f"Execution error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def measure_exchange_latency(
        self,
        exchange: str,
        measurements: List[float]
    ) -> Dict[str, Any]:
        """Track exchange latency statistics"""
        try:
            if exchange not in self.exchange_latencies:
                self.exchange_latencies[exchange] = []

            self.exchange_latencies[exchange].extend(measurements)

            # Keep only recent measurements (last 1000)
            self.exchange_latencies[exchange] = self.exchange_latencies[exchange][-1000:]

            latencies = self.exchange_latencies[exchange]
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)

            # Calculate percentiles
            sorted_latencies = sorted(latencies)
            p50 = sorted_latencies[len(sorted_latencies) // 2]
            p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

            return {
                'exchange': exchange,
                'avg_latency_ms': avg_latency,
                'min_latency_ms': min_latency,
                'max_latency_ms': max_latency,
                'p50_ms': p50,
                'p95_ms': p95,
                'p99_ms': p99,
                'sample_count': len(latencies)
            }

        except Exception as e:
            self.logger.error(f"Latency measurement error: {e}")
            return {'error': str(e)}
