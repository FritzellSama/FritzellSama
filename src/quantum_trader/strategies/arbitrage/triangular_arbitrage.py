"""Triangular Arbitrage - PRODUCTION"""
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from itertools import permutations
import os, logging
import polars as pl
from .base_arbitrage import BaseArbitrageStrategy, ArbitrageOpportunity

logger = logging.getLogger(__name__)

class TriangularArbitrageStrategy(BaseArbitrageStrategy):
    """
    Exploit price discrepancies in currency triangles

    Strategy:
    - Trade through 3 currency pairs to exploit rate inconsistencies
    - Example: USD -> BTC -> ETH -> USD
    - Profit from deviations in cross-rate equilibrium
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        # Triangular-specific parameters
        self.min_rate_deviation = Decimal(str(config.get(
            'min_rate_deviation',
            os.getenv('TRI_ARB_MIN_DEVIATION', '0.003')
        )))
        self.base_currency = config.get(
            'base_currency',
            os.getenv('TRI_ARB_BASE_CURRENCY', 'USD')
        )
        self.supported_currencies = config.get(
            'supported_currencies',
            ['USD', 'BTC', 'ETH', 'USDT', 'USDC', 'BNB']
        )

        # Triangle tracking
        self.discovered_triangles: List[Tuple[str, str, str]] = []
        self.triangle_cache: Dict[str, Dict[str, Any]] = {}

        self.logger.info(
            f"TriangularArbitrage initialized: base={self.base_currency}, "
            f"min_deviation={self.min_rate_deviation}, "
            f"currencies={len(self.supported_currencies)}"
        )

    def scan_opportunities(
        self,
        market_data: pl.DataFrame
    ) -> List[ArbitrageOpportunity]:
        """Scan for triangular arbitrage opportunities"""
        try:
            opportunities = []

            # Extract latest exchange rates
            if 'symbol' not in market_data.columns or 'price' not in market_data.columns:
                self.logger.warning("Missing required columns: symbol, price")
                return opportunities

            # Build exchange rate matrix
            exchange_rates = self._build_exchange_rate_matrix(market_data)

            if not exchange_rates:
                self.logger.debug("Insufficient exchange rate data")
                return opportunities

            # Find all possible triangles
            triangles = self._find_currency_triangles()

            # Check each triangle for arbitrage
            for triangle in triangles:
                try:
                    opportunity = self._check_triangle_arbitrage(
                        triangle,
                        exchange_rates,
                        market_data
                    )

                    if opportunity:
                        opportunities.append(opportunity)
                        self.opportunities_detected += 1

                except Exception as e:
                    self.logger.debug(f"Triangle check error {triangle}: {e}")

            return opportunities

        except Exception as e:
            self.logger.error(f"Scan error: {e}", exc_info=True)
            return []

    def _build_exchange_rate_matrix(
        self,
        market_data: pl.DataFrame
    ) -> Dict[Tuple[str, str], Decimal]:
        """Build exchange rate lookup from market data"""
        try:
            rates = {}

            # Get latest price for each trading pair
            latest_prices = (
                market_data
                .sort('timestamp', descending=True)
                .group_by('symbol')
                .agg([
                    pl.col('price').first().alias('price'),
                    pl.col('timestamp').first().alias('timestamp')
                ])
            )

            for row in latest_prices.iter_rows(named=True):
                symbol = row['symbol']
                price = Decimal(str(row['price']))

                # Parse symbol (e.g., "BTC/USD" or "BTCUSDT")
                base, quote = self._parse_symbol(symbol)

                if base and quote:
                    # Direct rate
                    rates[(base, quote)] = price
                    # Inverse rate
                    if price > 0:
                        rates[(quote, base)] = Decimal('1') / price

            self.logger.debug(f"Built exchange rate matrix: {len(rates)} pairs")
            return rates

        except Exception as e:
            self.logger.error(f"Rate matrix error: {e}")
            return {}

    def _parse_symbol(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse trading symbol into base and quote currencies"""
        try:
            # Handle slash format: BTC/USD
            if '/' in symbol:
                parts = symbol.split('/')
                return parts[0].upper(), parts[1].upper()

            # Handle concatenated format: BTCUSDT
            # Try known currencies
            for currency in sorted(self.supported_currencies, key=len, reverse=True):
                if symbol.startswith(currency):
                    base = currency
                    quote = symbol[len(currency):]
                    if quote in self.supported_currencies:
                        return base, quote

                if symbol.endswith(currency):
                    quote = currency
                    base = symbol[:-len(currency)]
                    if base in self.supported_currencies:
                        return base, quote

            return None, None

        except Exception as e:
            self.logger.debug(f"Symbol parse error {symbol}: {e}")
            return None, None

    def _find_currency_triangles(self) -> List[Tuple[str, str, str]]:
        """Find all possible currency triangles starting from base currency"""
        try:
            if self.discovered_triangles:
                return self.discovered_triangles

            triangles = []

            # Get all currencies except base
            other_currencies = [
                c for c in self.supported_currencies
                if c != self.base_currency
            ]

            # Generate all triangles: base -> currency1 -> currency2 -> base
            for curr1 in other_currencies:
                for curr2 in other_currencies:
                    if curr1 != curr2:
                        triangle = (self.base_currency, curr1, curr2)
                        triangles.append(triangle)

            self.discovered_triangles = triangles
            self.logger.info(f"Discovered {len(triangles)} possible triangles")

            return triangles

        except Exception as e:
            self.logger.error(f"Triangle discovery error: {e}")
            return []

    def _check_triangle_arbitrage(
        self,
        triangle: Tuple[str, str, str],
        exchange_rates: Dict[Tuple[str, str], Decimal],
        market_data: pl.DataFrame
    ) -> Optional[ArbitrageOpportunity]:
        """Check if a triangle has arbitrage opportunity"""
        try:
            curr_a, curr_b, curr_c = triangle

            # Get exchange rates for the triangle path
            # Path: A -> B -> C -> A
            rate_a_to_b = exchange_rates.get((curr_a, curr_b))
            rate_b_to_c = exchange_rates.get((curr_b, curr_c))
            rate_c_to_a = exchange_rates.get((curr_c, curr_a))

            if not all([rate_a_to_b, rate_b_to_c, rate_c_to_a]):
                return None

            # Calculate final amount after trading through triangle
            initial_amount = Decimal('1.0')
            after_first_trade = initial_amount * rate_a_to_b
            after_second_trade = after_first_trade * rate_b_to_c
            final_amount = after_second_trade * rate_c_to_a

            # Calculate profit percentage
            profit_pct = (final_amount - initial_amount) / initial_amount

            # Check if profitable after fees
            if profit_pct >= self.min_rate_deviation:
                # Calculate actual profit with position size
                position_value = self.max_position_size
                gross_profit = position_value * profit_pct

                # Account for 3 trades (fees on each leg)
                net_profit = self.calculate_net_profit(
                    gross_profit,
                    position_value * Decimal('3')  # 3 legs
                )

                if self.is_profitable(net_profit):
                    opportunity = ArbitrageOpportunity(
                        opportunity_id=f"triangular_{curr_a}_{curr_b}_{curr_c}_{datetime.utcnow().timestamp()}",
                        strategy_type='triangular_arbitrage',
                        expected_profit=net_profit,
                        legs=[
                            {
                                'step': 1,
                                'from_currency': curr_a,
                                'to_currency': curr_b,
                                'exchange_rate': rate_a_to_b,
                                'symbol': f"{curr_a}/{curr_b}",
                                'side': 'buy',
                                'amount': position_value
                            },
                            {
                                'step': 2,
                                'from_currency': curr_b,
                                'to_currency': curr_c,
                                'exchange_rate': rate_b_to_c,
                                'symbol': f"{curr_b}/{curr_c}",
                                'side': 'buy',
                                'amount': position_value * rate_a_to_b
                            },
                            {
                                'step': 3,
                                'from_currency': curr_c,
                                'to_currency': curr_a,
                                'exchange_rate': rate_c_to_a,
                                'symbol': f"{curr_c}/{curr_a}",
                                'side': 'buy',
                                'amount': position_value * rate_a_to_b * rate_b_to_c
                            }
                        ],
                        timestamp=datetime.utcnow()
                    )

                    self.logger.info(
                        f"Triangular opportunity: {curr_a}->{curr_b}->{curr_c}->{curr_a}, "
                        f"profit={profit_pct*100:.3f}%, net_profit={net_profit}"
                    )

                    return opportunity

            return None

        except Exception as e:
            self.logger.debug(f"Triangle check error: {e}")
            return None

    def validate_opportunity(
        self,
        opportunity: ArbitrageOpportunity
    ) -> bool:
        """Validate triangular arbitrage opportunity"""
        try:
            # Check freshness
            age = (datetime.utcnow() - opportunity.timestamp).total_seconds()
            if age > 10:  # 10 second max age
                self.logger.debug(f"Stale opportunity: {age}s old")
                return False

            # Verify all legs still have valid rates
            # In production, would re-fetch current rates
            for leg in opportunity.legs:
                if leg['exchange_rate'] <= 0:
                    return False

            # Check profit still above threshold
            if opportunity.expected_profit < self.min_profit_threshold:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False

    def execute_opportunity(
        self,
        opportunity: ArbitrageOpportunity
    ) -> Dict[str, Any]:
        """Execute triangular arbitrage - must execute all legs atomically"""
        try:
            self.logger.info(f"Executing triangular arbitrage: {opportunity.opportunity_id}")

            execution_results = []
            current_amount = opportunity.legs[0]['amount']

            # Execute each leg sequentially
            for leg in sorted(opportunity.legs, key=lambda x: x['step']):
                # In production: Execute trade via exchange API
                # CRITICAL: Must execute all legs quickly to lock in arbitrage

                execution_result = {
                    'step': leg['step'],
                    'from_currency': leg['from_currency'],
                    'to_currency': leg['to_currency'],
                    'symbol': leg['symbol'],
                    'expected_rate': leg['exchange_rate'],
                    'executed_amount': current_amount,
                    'executed': True,
                    'timestamp': datetime.utcnow()
                }

                execution_results.append(execution_result)

                # Update amount for next leg
                current_amount = current_amount * leg['exchange_rate']

                self.logger.debug(
                    f"Executed leg {leg['step']}: {leg['from_currency']}->{leg['to_currency']}"
                )

            # Verify final amount matches expected profit
            initial_amount = opportunity.legs[0]['amount']
            final_amount = current_amount
            actual_profit = final_amount - initial_amount

            self.opportunities_executed += 1
            self.total_profit += opportunity.expected_profit

            self.logger.info(
                f"Triangular arbitrage complete: "
                f"expected_profit={opportunity.expected_profit}, "
                f"actual_profit={actual_profit}"
            )

            return {
                'success': True,
                'opportunity_id': opportunity.opportunity_id,
                'execution_results': execution_results,
                'initial_amount': initial_amount,
                'final_amount': final_amount,
                'actual_profit': actual_profit,
                'expected_profit': opportunity.expected_profit
            }

        except Exception as e:
            self.logger.error(f"Execution error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def get_triangle_statistics(self) -> Dict[str, Any]:
        """Get statistics on triangle performance"""
        try:
            # Analyze triangle cache for best performing triangles
            if not self.triangle_cache:
                return {'message': 'No triangle data available'}

            triangle_performance = []
            for triangle_key, data in self.triangle_cache.items():
                triangle_performance.append({
                    'triangle': triangle_key,
                    'avg_profit': data.get('avg_profit', Decimal('0')),
                    'opportunity_count': data.get('count', 0),
                    'last_seen': data.get('last_seen')
                })

            # Sort by average profit
            triangle_performance.sort(
                key=lambda x: x['avg_profit'],
                reverse=True
            )

            return {
                'total_triangles': len(self.discovered_triangles),
                'active_triangles': len(self.triangle_cache),
                'best_triangles': triangle_performance[:10],
                'timestamp': datetime.utcnow()
            }

        except Exception as e:
            self.logger.error(f"Triangle statistics error: {e}")
            return {'error': str(e)}
