"""Arbitrage Opportunity Scanner - PRODUCTION"""
from decimal import Decimal
from typing import Dict, Any, List, Optional, Type
from datetime import datetime
import os, logging
import asyncio
import polars as pl

from .base_arbitrage import BaseArbitrageStrategy, ArbitrageOpportunity
from .funding_arbitrage import FundingArbitrageStrategy
from .latency_arbitrage import LatencyArbitrageStrategy

logger = logging.getLogger(__name__)

class OpportunityScanner:
    """
    Coordinates multiple arbitrage strategies and scans for opportunities

    Manages:
    - Multiple arbitrage strategy instances
    - Concurrent opportunity scanning
    - Opportunity prioritization and filtering
    - Execution coordination
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Scanner parameters
        self.scan_interval_ms = int(config.get(
            'scan_interval_ms',
            os.getenv('SCANNER_INTERVAL', '100')
        ))
        self.max_concurrent_opportunities = int(config.get(
            'max_concurrent_opportunities',
            os.getenv('SCANNER_MAX_CONCURRENT', '5')
        ))
        self.enable_auto_execution = config.get(
            'enable_auto_execution',
            os.getenv('SCANNER_AUTO_EXEC', 'false').lower() == 'true'
        )

        # Initialize strategies
        self.strategies: Dict[str, BaseArbitrageStrategy] = {}
        self._initialize_strategies(config)

        # Tracking
        self.active_opportunities: Dict[str, ArbitrageOpportunity] = {}
        self.scan_count = 0
        self.last_scan_time: Optional[datetime] = None

        self.logger.info(
            f"OpportunityScanner initialized: {len(self.strategies)} strategies, "
            f"scan_interval={self.scan_interval_ms}ms, "
            f"auto_exec={self.enable_auto_execution}"
        )

    def _initialize_strategies(self, config: Dict[str, Any]) -> None:
        """Initialize all enabled arbitrage strategies"""
        try:
            strategy_config = config.get('strategies', {})

            # Funding arbitrage
            if strategy_config.get('funding_enabled', True):
                self.strategies['funding'] = FundingArbitrageStrategy(
                    config.get('funding_config', {})
                )
                self.logger.info("Funding arbitrage strategy enabled")

            # Latency arbitrage
            if strategy_config.get('latency_enabled', True):
                self.strategies['latency'] = LatencyArbitrageStrategy(
                    config.get('latency_config', {})
                )
                self.logger.info("Latency arbitrage strategy enabled")

            # Triangular arbitrage (if implemented)
            # if strategy_config.get('triangular_enabled', False):
            #     from .triangular_arbitrage import TriangularArbitrageStrategy
            #     self.strategies['triangular'] = TriangularArbitrageStrategy(
            #         config.get('triangular_config', {})
            #     )

        except Exception as e:
            self.logger.error(f"Strategy initialization error: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize strategies: {e}")

    def scan_all_strategies(
        self,
        market_data: pl.DataFrame
    ) -> List[ArbitrageOpportunity]:
        """Scan all strategies for opportunities"""
        try:
            scan_start = datetime.utcnow()
            all_opportunities = []

            for strategy_name, strategy in self.strategies.items():
                try:
                    opportunities = strategy.scan_opportunities(market_data)
                    all_opportunities.extend(opportunities)

                    self.logger.debug(
                        f"{strategy_name}: Found {len(opportunities)} opportunities"
                    )

                except Exception as e:
                    self.logger.error(
                        f"Error scanning {strategy_name}: {e}",
                        exc_info=True
                    )

            # Sort by expected profit (descending)
            all_opportunities.sort(
                key=lambda x: x.expected_profit,
                reverse=True
            )

            # Update tracking
            self.scan_count += 1
            self.last_scan_time = datetime.utcnow()
            scan_duration = (self.last_scan_time - scan_start).total_seconds() * 1000

            self.logger.info(
                f"Scan #{self.scan_count} complete: {len(all_opportunities)} opportunities "
                f"found in {scan_duration:.2f}ms"
            )

            return all_opportunities

        except Exception as e:
            self.logger.error(f"Scan error: {e}", exc_info=True)
            return []

    async def scan_all_strategies_async(
        self,
        market_data: pl.DataFrame
    ) -> List[ArbitrageOpportunity]:
        """Scan all strategies concurrently (async)"""
        try:
            scan_start = datetime.utcnow()

            # Create tasks for each strategy
            tasks = []
            for strategy_name, strategy in self.strategies.items():
                task = asyncio.create_task(
                    self._scan_strategy_async(strategy_name, strategy, market_data)
                )
                tasks.append(task)

            # Wait for all scans to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect all opportunities
            all_opportunities = []
            for result in results:
                if isinstance(result, list):
                    all_opportunities.extend(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Async scan error: {result}")

            # Sort by profit
            all_opportunities.sort(
                key=lambda x: x.expected_profit,
                reverse=True
            )

            # Update tracking
            self.scan_count += 1
            self.last_scan_time = datetime.utcnow()
            scan_duration = (self.last_scan_time - scan_start).total_seconds() * 1000

            self.logger.info(
                f"Async scan #{self.scan_count}: {len(all_opportunities)} opportunities "
                f"in {scan_duration:.2f}ms"
            )

            return all_opportunities

        except Exception as e:
            self.logger.error(f"Async scan error: {e}", exc_info=True)
            return []

    async def _scan_strategy_async(
        self,
        strategy_name: str,
        strategy: BaseArbitrageStrategy,
        market_data: pl.DataFrame
    ) -> List[ArbitrageOpportunity]:
        """Async wrapper for strategy scanning"""
        try:
            # Run blocking scan in executor
            loop = asyncio.get_event_loop()
            opportunities = await loop.run_in_executor(
                None,
                strategy.scan_opportunities,
                market_data
            )
            return opportunities

        except Exception as e:
            self.logger.error(f"Strategy {strategy_name} scan error: {e}")
            return []

    def filter_opportunities(
        self,
        opportunities: List[ArbitrageOpportunity],
        min_profit: Optional[Decimal] = None,
        max_count: Optional[int] = None
    ) -> List[ArbitrageOpportunity]:
        """Filter and prioritize opportunities"""
        try:
            filtered = opportunities

            # Filter by minimum profit
            if min_profit is not None:
                filtered = [
                    opp for opp in filtered
                    if opp.expected_profit >= min_profit
                ]

            # Limit count
            if max_count is not None:
                filtered = filtered[:max_count]

            self.logger.debug(
                f"Filtered {len(opportunities)} -> {len(filtered)} opportunities"
            )

            return filtered

        except Exception as e:
            self.logger.error(f"Filter error: {e}")
            return opportunities

    def execute_opportunities(
        self,
        opportunities: List[ArbitrageOpportunity]
    ) -> List[Dict[str, Any]]:
        """Execute multiple opportunities"""
        try:
            results = []

            for opportunity in opportunities:
                # Check if we're at max concurrent
                if len(self.active_opportunities) >= self.max_concurrent_opportunities:
                    self.logger.warning(
                        f"Max concurrent opportunities reached: {self.max_concurrent_opportunities}"
                    )
                    break

                # Get the appropriate strategy
                strategy = self._get_strategy_for_opportunity(opportunity)
                if strategy is None:
                    continue

                # Validate before execution
                if not strategy.validate_opportunity(opportunity):
                    self.logger.debug(
                        f"Opportunity failed validation: {opportunity.opportunity_id}"
                    )
                    continue

                # Execute
                result = strategy.execute_opportunity(opportunity)
                results.append(result)

                if result.get('success'):
                    self.active_opportunities[opportunity.opportunity_id] = opportunity
                    self.logger.info(
                        f"Executed: {opportunity.opportunity_id}, "
                        f"profit={opportunity.expected_profit}"
                    )

            return results

        except Exception as e:
            self.logger.error(f"Execution error: {e}", exc_info=True)
            return []

    def _get_strategy_for_opportunity(
        self,
        opportunity: ArbitrageOpportunity
    ) -> Optional[BaseArbitrageStrategy]:
        """Get the strategy instance for an opportunity"""
        try:
            strategy_type = opportunity.strategy_type

            if 'funding' in strategy_type:
                return self.strategies.get('funding')
            elif 'latency' in strategy_type:
                return self.strategies.get('latency')
            elif 'triangular' in strategy_type:
                return self.strategies.get('triangular')
            else:
                self.logger.warning(f"Unknown strategy type: {strategy_type}")
                return None

        except Exception as e:
            self.logger.error(f"Strategy lookup error: {e}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get scanner statistics"""
        try:
            strategy_stats = {}
            for name, strategy in self.strategies.items():
                strategy_stats[name] = strategy.get_statistics()

            total_detected = sum(
                s['opportunities_detected'] for s in strategy_stats.values()
            )
            total_executed = sum(
                s['opportunities_executed'] for s in strategy_stats.values()
            )
            total_profit = sum(
                s['total_profit'] for s in strategy_stats.values()
            )

            return {
                'scan_count': self.scan_count,
                'last_scan_time': self.last_scan_time,
                'active_opportunities': len(self.active_opportunities),
                'total_detected': total_detected,
                'total_executed': total_executed,
                'total_profit': total_profit,
                'strategies': strategy_stats,
                'timestamp': datetime.utcnow()
            }

        except Exception as e:
            self.logger.error(f"Statistics error: {e}")
            return {'error': str(e)}
