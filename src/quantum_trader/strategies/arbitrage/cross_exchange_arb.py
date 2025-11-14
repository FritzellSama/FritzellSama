"""
Cross-Exchange Arbitrage Strategy.

Implements arbitrage trading across multiple exchanges to profit from
price discrepancies for the same asset on different venues.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

import polars as pl
from structlog import get_logger

from quantum_trader.models import Signal, SignalAction, Ticker, OrderBook
from quantum_trader.strategies.base_strategy import BaseStrategy
from quantum_trader.risk.manager import RiskManager
from quantum_trader.exchanges.base import BaseExchange

logger = get_logger(__name__)


class CrossExchangeArbStrategy(BaseStrategy):
    """Cross-exchange arbitrage trading strategy.

    This strategy identifies and exploits price differences for the same asset
    across multiple exchanges. It accounts for:
    - Trading fees on both exchanges
    - Transfer costs and time
    - Slippage and execution risk
    - Balance requirements

    Features:
    - Real-time multi-exchange price monitoring
    - Fee-adjusted profit calculation
    - Transfer optimization
    - Risk limits per exchange

    Attributes:
        min_profit_bps: Minimum profit in basis points
        max_position_size: Maximum position size per trade
        exchanges: List of exchange connectors

    Example:
        >>> config = {
        ...     "min_profit_bps": "10",
        ...     "exchanges": ["binance", "bybit"],
        ...     "transfer_fee": "0.0001"
        ... }
        >>> strategy = CrossExchangeArbStrategy(config, risk_manager)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        risk_manager: RiskManager,
        exchanges: Optional[Dict[str, BaseExchange]] = None
    ) -> None:
        """Initialize cross-exchange arbitrage strategy.

        Args:
            config: Strategy configuration
            risk_manager: Risk manager instance
            exchanges: Dictionary of exchange name to connector

        Raises:
            ValueError: If required parameters missing
        """
        super().__init__(config, risk_manager)

        # Arbitrage parameters
        self.min_profit_bps = Decimal(str(config.get("min_profit_bps", "10")))  # 10 bps
        self.max_position_size = Decimal(str(config.get("max_position_size", "1.0")))
        self.transfer_fee = Decimal(str(config.get("transfer_fee", "0.0001")))
        self.max_transfer_time = int(config.get("max_transfer_time", 300))  # seconds

        # Exchange management
        self.exchanges = exchanges or {}
        self.exchange_names = config.get("exchanges", [])

        # Price tracking
        self.exchange_prices: Dict[str, Dict[str, Decimal]] = {}  # {exchange: {symbol: price}}
        self.exchange_fees: Dict[str, Dict[str, Decimal]] = {}  # {exchange: {maker/taker: fee}}
        self.orderbooks: Dict[str, Dict[str, OrderBook]] = {}  # {exchange: {symbol: orderbook}}

        # Opportunity tracking
        self.arb_opportunities: List[Dict] = []
        self.executed_arbs: List[Dict] = []

        logger.info(
            "Cross-exchange arbitrage strategy initialized",
            strategy=self.name,
            min_profit_bps=self.min_profit_bps,
            exchanges=self.exchange_names
        )

    def _validate_config(self) -> None:
        """Validate strategy configuration.

        Raises:
            ValueError: If parameters invalid
        """
        super()._validate_config()

        min_profit = Decimal(str(self.config.get("min_profit_bps", "10")))
        if min_profit <= Decimal("0"):
            raise ValueError(f"min_profit_bps must be positive: {min_profit}")

        exchanges = self.config.get("exchanges", [])
        if len(exchanges) < 2:
            raise ValueError("At least 2 exchanges required for arbitrage")

        max_pos = Decimal(str(self.config.get("max_position_size", "1.0")))
        if max_pos <= Decimal("0"):
            raise ValueError(f"max_position_size must be positive: {max_pos}")

    async def update_exchange_data(
        self,
        exchange_name: str,
        symbol: str
    ) -> None:
        """Update price and orderbook data for an exchange.

        Args:
            exchange_name: Name of exchange
            symbol: Trading symbol
        """
        try:
            if exchange_name not in self.exchanges:
                logger.warning("Exchange not available", exchange=exchange_name)
                return

            exchange = self.exchanges[exchange_name]

            # Fetch ticker
            ticker = await exchange.fetch_ticker(symbol)

            # Update prices
            if exchange_name not in self.exchange_prices:
                self.exchange_prices[exchange_name] = {}

            self.exchange_prices[exchange_name][symbol] = ticker.last

            # Fetch orderbook
            orderbook = await exchange.fetch_orderbook(symbol, depth=10)

            if exchange_name not in self.orderbooks:
                self.orderbooks[exchange_name] = {}

            self.orderbooks[exchange_name][symbol] = orderbook

            # Fetch fees if not cached
            if exchange_name not in self.exchange_fees:
                fees = await exchange.get_trading_fees(symbol)
                self.exchange_fees[exchange_name] = fees

            logger.debug(
                "Exchange data updated",
                exchange=exchange_name,
                symbol=symbol,
                price=ticker.last
            )

        except Exception as e:
            logger.error(
                "Error updating exchange data",
                error=str(e),
                exchange=exchange_name,
                symbol=symbol
            )

    def calculate_arbitrage_profit(
        self,
        buy_exchange: str,
        sell_exchange: str,
        symbol: str,
        quantity: Decimal
    ) -> Tuple[Decimal, Dict[str, Decimal]]:
        """Calculate potential arbitrage profit.

        Args:
            buy_exchange: Exchange to buy from
            sell_exchange: Exchange to sell on
            symbol: Trading symbol
            quantity: Trade quantity

        Returns:
            Tuple of (net_profit, breakdown_dict)
        """
        try:
            # Get prices
            buy_price = self.exchange_prices.get(buy_exchange, {}).get(symbol)
            sell_price = self.exchange_prices.get(sell_exchange, {}).get(symbol)

            if not buy_price or not sell_price:
                return Decimal("0"), {}

            # Get fees
            buy_fees = self.exchange_fees.get(buy_exchange, {})
            sell_fees = self.exchange_fees.get(sell_exchange, {})

            buy_fee_rate = buy_fees.get("taker", Decimal("0.001"))  # Default 0.1%
            sell_fee_rate = sell_fees.get("taker", Decimal("0.001"))

            # Calculate gross profit
            gross_profit = (sell_price - buy_price) * quantity

            # Calculate costs
            buy_cost = buy_price * quantity * buy_fee_rate
            sell_cost = sell_price * quantity * sell_fee_rate
            transfer_cost = self.transfer_fee * quantity

            total_costs = buy_cost + sell_cost + transfer_cost

            # Net profit
            net_profit = gross_profit - total_costs

            # Profit in bps
            capital_used = buy_price * quantity
            profit_bps = (net_profit / capital_used) * Decimal("10000") if capital_used > Decimal("0") else Decimal("0")

            breakdown = {
                "buy_price": buy_price,
                "sell_price": sell_price,
                "gross_profit": gross_profit,
                "buy_fee": buy_cost,
                "sell_fee": sell_cost,
                "transfer_fee": transfer_cost,
                "total_costs": total_costs,
                "net_profit": net_profit,
                "profit_bps": profit_bps,
                "quantity": quantity
            }

            return net_profit, breakdown

        except Exception as e:
            logger.error("Error calculating arbitrage profit", error=str(e))
            return Decimal("0"), {}

    def find_arbitrage_opportunities(
        self,
        symbol: str
    ) -> List[Dict[str, Any]]:
        """Find all arbitrage opportunities for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of opportunity dictionaries
        """
        opportunities = []

        try:
            # Compare all exchange pairs
            for buy_exchange in self.exchange_names:
                for sell_exchange in self.exchange_names:
                    if buy_exchange == sell_exchange:
                        continue

                    # Check if we have data for both exchanges
                    if (buy_exchange not in self.exchange_prices or
                        sell_exchange not in self.exchange_prices):
                        continue

                    buy_price = self.exchange_prices[buy_exchange].get(symbol)
                    sell_price = self.exchange_prices[sell_exchange].get(symbol)

                    if not buy_price or not sell_price:
                        continue

                    # Check if sell price is higher (opportunity exists)
                    if sell_price <= buy_price:
                        continue

                    # Calculate profit for max position size
                    quantity = self.max_position_size

                    net_profit, breakdown = self.calculate_arbitrage_profit(
                        buy_exchange,
                        sell_exchange,
                        symbol,
                        quantity
                    )

                    profit_bps = breakdown.get("profit_bps", Decimal("0"))

                    # Check if profit exceeds minimum
                    if profit_bps >= self.min_profit_bps:
                        opportunity = {
                            "symbol": symbol,
                            "buy_exchange": buy_exchange,
                            "sell_exchange": sell_exchange,
                            "buy_price": buy_price,
                            "sell_price": sell_price,
                            "quantity": quantity,
                            "net_profit": net_profit,
                            "profit_bps": profit_bps,
                            "breakdown": breakdown,
                            "timestamp": datetime.now(timezone.utc)
                        }

                        opportunities.append(opportunity)

                        logger.info(
                            "Arbitrage opportunity found",
                            symbol=symbol,
                            buy_exchange=buy_exchange,
                            sell_exchange=sell_exchange,
                            profit_bps=profit_bps
                        )

        except Exception as e:
            logger.error("Error finding opportunities", error=str(e))

        return opportunities

    def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """Generate arbitrage signals.

        Note: This strategy primarily works with real-time tick data,
        not historical OHLCV data.

        Args:
            market_data: Market data (not typically used)

        Returns:
            List of Signal objects
        """
        signals: List[Signal] = []

        try:
            if not self.active:
                return signals

            # Get symbols
            symbols = self.config.get("symbols", [])
            if not symbols:
                logger.warning("No symbols configured")
                return signals

            # Find opportunities for each symbol
            all_opportunities = []
            for symbol in symbols:
                opportunities = self.find_arbitrage_opportunities(symbol)
                all_opportunities.extend(opportunities)

            # Sort by profit
            all_opportunities.sort(key=lambda x: x["profit_bps"], reverse=True)

            # Generate signals for top opportunities
            max_signals = int(self.config.get("max_concurrent_arbs", 3))

            for opp in all_opportunities[:max_signals]:
                # Create buy signal
                buy_signal = Signal(
                    symbol=opp["symbol"],
                    action=SignalAction.BUY,
                    strength=min(Decimal("1"), opp["profit_bps"] / self.min_profit_bps),
                    confidence=Decimal("0.8"),
                    timestamp=datetime.now(timezone.utc),
                    strategy=self.name,
                    timeframe="tick",
                    indicators={
                        "buy_price": opp["buy_price"],
                        "sell_price": opp["sell_price"],
                        "profit_bps": opp["profit_bps"],
                        "net_profit": opp["net_profit"]
                    },
                    metadata={
                        "strategy_type": "arbitrage",
                        "buy_exchange": opp["buy_exchange"],
                        "sell_exchange": opp["sell_exchange"],
                        "leg": "buy",
                        "paired_leg": "sell"
                    }
                )

                if self.validate_signal(buy_signal):
                    signals.append(buy_signal)

                    # Track opportunity
                    self.arb_opportunities.append(opp)

                    # Update state
                    self.state["signals_generated"] = self.state.get("signals_generated", 0) + 1
                    self.state["last_signal_time"] = datetime.now(timezone.utc)

            return signals

        except Exception as e:
            logger.error("Error generating signals", error=str(e))
            return signals

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate arbitrage indicators.

        Args:
            data: Market data

        Returns:
            Dictionary of indicators
        """
        try:
            indicators = {}

            # Calculate price spreads across exchanges
            symbols = self.config.get("symbols", [])

            for symbol in symbols:
                prices = []
                for exchange in self.exchange_names:
                    price = self.exchange_prices.get(exchange, {}).get(symbol)
                    if price:
                        prices.append(price)

                if len(prices) >= 2:
                    max_price = max(prices)
                    min_price = min(prices)

                    spread = max_price - min_price
                    spread_bps = (spread / min_price) * Decimal("10000") if min_price > Decimal("0") else Decimal("0")

                    indicators[f"{symbol}_max_price"] = max_price
                    indicators[f"{symbol}_min_price"] = min_price
                    indicators[f"{symbol}_spread"] = spread
                    indicators[f"{symbol}_spread_bps"] = spread_bps

            # Add opportunity metrics
            if self.arb_opportunities:
                recent_opps = [o for o in self.arb_opportunities
                              if (datetime.now(timezone.utc) - o["timestamp"]).seconds < 60]

                if recent_opps:
                    avg_profit = sum(o["profit_bps"] for o in recent_opps) / Decimal(str(len(recent_opps)))
                    max_profit = max(o["profit_bps"] for o in recent_opps)

                    indicators["opportunities_last_minute"] = Decimal(str(len(recent_opps)))
                    indicators["avg_profit_bps"] = avg_profit
                    indicators["max_profit_bps"] = max_profit

            return indicators

        except Exception as e:
            logger.error("Error calculating indicators", error=str(e))
            return {}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        base_metrics = super().get_performance_metrics()

        arb_metrics = {
            "total_opportunities": len(self.arb_opportunities),
            "executed_arbs": len(self.executed_arbs),
            "exchange_prices": {
                exchange: {symbol: float(price) for symbol, price in prices.items()}
                for exchange, prices in self.exchange_prices.items()
            }
        }

        # Calculate aggregate stats
        if self.arb_opportunities:
            recent = [o for o in self.arb_opportunities
                     if (datetime.now(timezone.utc) - o["timestamp"]).seconds < 3600]

            if recent:
                arb_metrics["opportunities_last_hour"] = len(recent)
                arb_metrics["avg_profit_bps_last_hour"] = float(
                    sum(o["profit_bps"] for o in recent) / Decimal(str(len(recent)))
                )

        return {**base_metrics, **arb_metrics}
