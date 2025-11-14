"""Spread analysis processor for arbitrage and market making.

This module analyzes bid-ask spreads across multiple exchanges and symbols
to identify arbitrage opportunities and optimal execution venues.
Processes spreads in real-time with batch optimization.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from structlog import get_logger

from quantum_trader.models import Ticker

logger = get_logger(__name__)


class SpreadProcessor:
    """Processes spread data for arbitrage detection.

    Analyzes bid-ask spreads within and across exchanges to identify:
    - Cross-exchange arbitrage opportunities
    - Abnormal spread conditions
    - Optimal routing for execution
    - Market quality metrics

    Attributes:
        config: Configuration dictionary
        min_arbitrage_bps: Minimum arbitrage opportunity in basis points
        max_spread_bps: Maximum acceptable spread
        checkpoint_interval: Interval for checkpointing state
        _processed_count: Count of processed spreads
        _last_tickers: Cache of last tickers per exchange/symbol

    Example:
        >>> processor = SpreadProcessor(config)
        >>> opportunities = await processor.detect_arbitrage(tickers)
        >>> for opp in opportunities:
        ...     print(f"{opp['profit_bps']}bps opportunity")
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize spread processor.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If required configuration is missing
        """
        self.config = config
        self._validate_config()

        processor_config = config.get("spread_processor", {})
        self.min_arbitrage_bps = Decimal(
            str(processor_config.get("min_arbitrage_bps", 10))
        )
        self.max_spread_bps = Decimal(
            str(processor_config.get("max_spread_bps", 100))
        )
        self.min_liquidity = Decimal(
            str(processor_config.get("min_liquidity_usdt", 1000))
        )
        self.checkpoint_interval = processor_config.get("checkpoint_interval", 1000)

        # Processing state
        self._processed_count = 0
        self._last_tickers: Dict[str, Ticker] = {}
        self._checkpoint_data: Dict[str, Any] = {}

        # Metrics
        self._metrics: Dict[str, int] = {
            "total_processed": 0,
            "total_errors": 0,
            "arbitrage_opportunities": 0,
            "abnormal_spreads": 0,
            "checkpoints_saved": 0
        }

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required configuration is missing
        """
        if "spread_processor" not in self.config:
            raise ValueError("Missing spread_processor configuration")

    async def process_ticker(
        self,
        ticker: Ticker,
        exchange: str
    ) -> Dict[str, Any]:
        """Process single ticker for spread analysis.

        Args:
            ticker: Ticker instance
            exchange: Exchange name

        Returns:
            Dictionary of spread metrics

        Example:
            >>> metrics = await processor.process_ticker(ticker, "binance")
            >>> print(f"Spread: {metrics['spread_bps']}bps")
        """
        try:
            metrics = {}

            # Basic spread metrics
            spread = ticker.ask - ticker.bid
            mid_price = (ticker.bid + ticker.ask) / Decimal("2")
            spread_bps = (spread / mid_price) * Decimal("10000")

            metrics["symbol"] = ticker.symbol
            metrics["exchange"] = exchange
            metrics["bid"] = ticker.bid
            metrics["ask"] = ticker.ask
            metrics["mid_price"] = mid_price
            metrics["spread"] = spread
            metrics["spread_bps"] = spread_bps
            metrics["timestamp"] = ticker.timestamp

            # Check for abnormal spread
            if spread_bps > self.max_spread_bps:
                metrics["abnormal_spread"] = True
                self._metrics["abnormal_spreads"] += 1
                logger.warning(
                    "abnormal_spread_detected",
                    symbol=ticker.symbol,
                    exchange=exchange,
                    spread_bps=str(spread_bps)
                )
            else:
                metrics["abnormal_spread"] = False

            # Cache ticker
            cache_key = f"{exchange}:{ticker.symbol}"
            self._last_tickers[cache_key] = ticker

            # Update metrics
            self._metrics["total_processed"] += 1
            self._processed_count += 1

            # Checkpoint if needed
            if self._processed_count % self.checkpoint_interval == 0:
                await self._save_checkpoint()

            return metrics

        except Exception as e:
            logger.error(
                "ticker_spread_processing_failed",
                symbol=ticker.symbol,
                exchange=exchange,
                error=str(e)
            )
            self._metrics["total_errors"] += 1
            raise

    async def detect_arbitrage(
        self,
        tickers: Dict[str, Dict[str, Ticker]]
    ) -> List[Dict[str, Any]]:
        """Detect cross-exchange arbitrage opportunities.

        Args:
            tickers: Dict of {exchange: {symbol: Ticker}}

        Returns:
            List of arbitrage opportunities

        Example:
            >>> tickers = {
            ...     "binance": {"BTC/USDT": ticker1},
            ...     "bybit": {"BTC/USDT": ticker2}
            ... }
            >>> opportunities = await processor.detect_arbitrage(tickers)
        """
        opportunities = []

        # Group tickers by symbol
        symbols_by_exchange: Dict[str, Dict[str, Ticker]] = {}

        for exchange, exchange_tickers in tickers.items():
            for symbol, ticker in exchange_tickers.items():
                if symbol not in symbols_by_exchange:
                    symbols_by_exchange[symbol] = {}
                symbols_by_exchange[symbol][exchange] = ticker

        # Check each symbol across exchanges
        for symbol, exchange_tickers in symbols_by_exchange.items():
            if len(exchange_tickers) < 2:
                continue

            # Find all pairwise arbitrage opportunities
            exchanges = list(exchange_tickers.keys())
            for i in range(len(exchanges)):
                for j in range(i + 1, len(exchanges)):
                    exchange_a = exchanges[i]
                    exchange_b = exchanges[j]

                    ticker_a = exchange_tickers[exchange_a]
                    ticker_b = exchange_tickers[exchange_b]

                    # Check both directions
                    # Direction 1: Buy on A, Sell on B
                    profit_a_b = await self._calculate_arbitrage_profit(
                        buy_ticker=ticker_a,
                        sell_ticker=ticker_b,
                        buy_exchange=exchange_a,
                        sell_exchange=exchange_b
                    )

                    if profit_a_b and profit_a_b["profit_bps"] >= self.min_arbitrage_bps:
                        opportunities.append(profit_a_b)
                        self._metrics["arbitrage_opportunities"] += 1

                    # Direction 2: Buy on B, Sell on A
                    profit_b_a = await self._calculate_arbitrage_profit(
                        buy_ticker=ticker_b,
                        sell_ticker=ticker_a,
                        buy_exchange=exchange_b,
                        sell_exchange=exchange_a
                    )

                    if profit_b_a and profit_b_a["profit_bps"] >= self.min_arbitrage_bps:
                        opportunities.append(profit_b_a)
                        self._metrics["arbitrage_opportunities"] += 1

        return opportunities

    async def _calculate_arbitrage_profit(
        self,
        buy_ticker: Ticker,
        sell_ticker: Ticker,
        buy_exchange: str,
        sell_exchange: str
    ) -> Optional[Dict[str, Any]]:
        """Calculate arbitrage profit for a specific direction.

        Args:
            buy_ticker: Ticker to buy on
            sell_ticker: Ticker to sell on
            buy_exchange: Exchange to buy from
            sell_exchange: Exchange to sell on

        Returns:
            Arbitrage opportunity details or None if not profitable
        """
        # Buy at ask on buy exchange
        buy_price = buy_ticker.ask

        # Sell at bid on sell exchange
        sell_price = sell_ticker.bid

        # Calculate profit
        if sell_price <= buy_price:
            return None

        profit = sell_price - buy_price
        profit_bps = (profit / buy_price) * Decimal("10000")

        # Estimate available volume (use minimum of buy/sell)
        # Use volume from tickers as proxy for available liquidity
        estimated_volume = min(buy_ticker.volume, sell_ticker.volume) / Decimal("100")

        opportunity = {
            "symbol": buy_ticker.symbol,
            "buy_exchange": buy_exchange,
            "sell_exchange": sell_exchange,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "profit": profit,
            "profit_bps": profit_bps,
            "estimated_volume": estimated_volume,
            "estimated_profit_usdt": profit * estimated_volume,
            "timestamp": datetime.now(timezone.utc)
        }

        logger.info(
            "arbitrage_opportunity_detected",
            symbol=buy_ticker.symbol,
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            profit_bps=str(profit_bps)
        )

        return opportunity

    async def process_batch(
        self,
        tickers_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Process batch of tickers for spread analysis.

        Args:
            tickers_df: Polars DataFrame with ticker data

        Returns:
            DataFrame with spread metrics

        Example:
            >>> df = pl.DataFrame([...])
            >>> spreads_df = await processor.process_batch(df)
        """
        try:
            if tickers_df.is_empty():
                return pl.DataFrame()

            results = []

            for row in tickers_df.to_dicts():
                # Reconstruct Ticker from row
                ticker = Ticker(
                    symbol=row["symbol"],
                    bid=Decimal(str(row["bid"])),
                    ask=Decimal(str(row["ask"])),
                    last=Decimal(str(row["last"])),
                    volume=Decimal(str(row["volume"])),
                    timestamp=row["timestamp"]
                )

                metrics = await self.process_ticker(
                    ticker,
                    row.get("exchange", "unknown")
                )
                results.append(metrics)

            # Convert to Polars DataFrame
            result_df = pl.DataFrame(results)

            logger.debug(
                "spread_batch_processed",
                count=len(results)
            )

            return result_df

        except Exception as e:
            logger.error(
                "spread_batch_processing_failed",
                rows=len(tickers_df),
                error=str(e)
            )
            self._metrics["total_errors"] += 1
            raise

    async def calculate_best_execution_venue(
        self,
        symbol: str,
        side: str,
        tickers: Dict[str, Ticker]
    ) -> Optional[Dict[str, Any]]:
        """Calculate best exchange for execution.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            tickers: Dict of {exchange: Ticker}

        Returns:
            Best venue details or None

        Example:
            >>> best = await processor.calculate_best_execution_venue(
            ...     "BTC/USDT", "buy", tickers
            ... )
            >>> print(f"Best exchange: {best['exchange']}")
        """
        if not tickers:
            return None

        best_exchange = None
        best_price = None

        for exchange, ticker in tickers.items():
            if side.lower() == "buy":
                price = ticker.ask
                if best_price is None or price < best_price:
                    best_price = price
                    best_exchange = exchange
            else:  # sell
                price = ticker.bid
                if best_price is None or price > best_price:
                    best_price = price
                    best_exchange = exchange

        if best_exchange and best_price:
            return {
                "symbol": symbol,
                "side": side,
                "best_exchange": best_exchange,
                "best_price": best_price,
                "timestamp": datetime.now(timezone.utc)
            }

        return None

    async def _save_checkpoint(self) -> None:
        """Save processing checkpoint for recovery."""
        try:
            self._checkpoint_data = {
                "processed_count": self._processed_count,
                "timestamp": datetime.now(timezone.utc),
                "metrics": self._metrics.copy()
            }

            self._metrics["checkpoints_saved"] += 1

            logger.debug(
                "spread_checkpoint_saved",
                processed=self._processed_count
            )

        except Exception as e:
            logger.error("checkpoint_save_failed", error=str(e))

    def get_checkpoint(self) -> Dict[str, Any]:
        """Get current checkpoint data.

        Returns:
            Checkpoint dictionary
        """
        return self._checkpoint_data.copy()

    def restore_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore from checkpoint.

        Args:
            checkpoint: Checkpoint data to restore
        """
        self._processed_count = checkpoint.get("processed_count", 0)
        self._metrics = checkpoint.get("metrics", {}).copy()

        logger.info(
            "spread_checkpoint_restored",
            processed=self._processed_count
        )

    def get_metrics(self) -> Dict[str, int]:
        """Get processor metrics.

        Returns:
            Dictionary of metrics
        """
        return self._metrics.copy()
