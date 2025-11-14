"""Order book data processor with advanced analytics.

This module processes order book data to extract market microstructure features,
imbalance metrics, and liquidity indicators. Optimized for stream processing
with Polars DataFrames.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from structlog import get_logger

from quantum_trader.models import OrderBook

logger = get_logger(__name__)


class OrderbookProcessor:
    """Processes order book data for analytics and features.

    Computes market microstructure metrics including:
    - Order book imbalance
    - Weighted mid-price
    - Liquidity depth
    - Spread metrics
    - Volume-weighted levels

    Attributes:
        config: Configuration dictionary
        depth_levels: Number of levels to analyze
        checkpoint_interval: Interval for checkpointing state
        _processed_count: Count of processed orderbooks
        _checkpoint_data: Checkpoint state for recovery

    Example:
        >>> processor = OrderbookProcessor(config)
        >>> features = await processor.process_orderbook(orderbook)
        >>> print(features["imbalance"])
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize orderbook processor.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If required configuration is missing
        """
        self.config = config
        self._validate_config()

        processor_config = config.get("orderbook_processor", {})
        self.depth_levels = processor_config.get("depth_levels", 10)
        self.imbalance_depth = processor_config.get("imbalance_depth", 5)
        self.checkpoint_interval = processor_config.get("checkpoint_interval", 1000)
        self.enable_advanced_metrics = processor_config.get(
            "enable_advanced_metrics",
            True
        )

        # Processing state
        self._processed_count = 0
        self._checkpoint_data: Dict[str, Any] = {}
        self._last_orderbooks: Dict[str, OrderBook] = {}

        # Metrics
        self._metrics: Dict[str, int] = {
            "total_processed": 0,
            "total_errors": 0,
            "checkpoints_saved": 0
        }

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required configuration is missing
        """
        if "orderbook_processor" not in self.config:
            raise ValueError("Missing orderbook_processor configuration")

    async def process_orderbook(
        self,
        orderbook: OrderBook
    ) -> Dict[str, Decimal]:
        """Process single orderbook and extract features.

        Args:
            orderbook: OrderBook instance to process

        Returns:
            Dictionary of computed features

        Example:
            >>> features = await processor.process_orderbook(orderbook)
            >>> print(f"Spread: {features['spread_bps']}")
        """
        try:
            features = {}

            # Basic spread metrics
            if orderbook.bids and orderbook.asks:
                best_bid = orderbook.bids[0][0]
                best_ask = orderbook.asks[0][0]

                features["best_bid"] = best_bid
                features["best_ask"] = best_ask
                features["mid_price"] = (best_bid + best_ask) / Decimal("2")
                features["spread"] = best_ask - best_bid
                features["spread_bps"] = (
                    (best_ask - best_bid) / features["mid_price"] * Decimal("10000")
                )

                # Weighted mid price
                features["weighted_mid_price"] = self._calculate_weighted_mid(
                    orderbook
                )

                # Order book imbalance
                features["imbalance"] = self._calculate_imbalance(orderbook)

                # Liquidity metrics
                liquidity = self._calculate_liquidity_metrics(orderbook)
                features.update(liquidity)

                # Advanced metrics if enabled
                if self.enable_advanced_metrics:
                    advanced = self._calculate_advanced_metrics(orderbook)
                    features.update(advanced)

            # Cache for comparison
            self._last_orderbooks[orderbook.symbol] = orderbook

            # Update metrics
            self._metrics["total_processed"] += 1
            self._processed_count += 1

            # Checkpoint if needed
            if self._processed_count % self.checkpoint_interval == 0:
                await self._save_checkpoint()

            return features

        except Exception as e:
            logger.error(
                "orderbook_processing_failed",
                symbol=orderbook.symbol,
                error=str(e)
            )
            self._metrics["total_errors"] += 1
            raise

    async def process_batch(
        self,
        orderbooks: pl.DataFrame
    ) -> pl.DataFrame:
        """Process batch of orderbooks efficiently.

        Args:
            orderbooks: Polars DataFrame with orderbook data

        Returns:
            DataFrame with computed features

        Example:
            >>> df = pl.DataFrame([...])
            >>> features_df = await processor.process_batch(df)
        """
        try:
            if orderbooks.is_empty():
                return pl.DataFrame()

            # Process each orderbook
            results = []

            for row in orderbooks.to_dicts():
                # Reconstruct OrderBook from row
                orderbook = OrderBook(
                    symbol=row["symbol"],
                    bids=row["bids"],
                    asks=row["asks"],
                    timestamp=row["timestamp"]
                )

                features = await self.process_orderbook(orderbook)
                features["symbol"] = orderbook.symbol
                features["timestamp"] = orderbook.timestamp
                results.append(features)

            # Convert to Polars DataFrame
            result_df = pl.DataFrame(results)

            logger.debug(
                "batch_processed",
                count=len(results)
            )

            return result_df

        except Exception as e:
            logger.error(
                "batch_processing_failed",
                rows=len(orderbooks),
                error=str(e)
            )
            self._metrics["total_errors"] += 1
            raise

    def _calculate_weighted_mid(
        self,
        orderbook: OrderBook
    ) -> Decimal:
        """Calculate volume-weighted mid price.

        Args:
            orderbook: OrderBook instance

        Returns:
            Weighted mid price
        """
        if not orderbook.bids or not orderbook.asks:
            return Decimal("0")

        # Use top level volumes
        bid_price, bid_size = orderbook.bids[0]
        ask_price, ask_size = orderbook.asks[0]

        total_size = bid_size + ask_size
        if total_size == 0:
            return (bid_price + ask_price) / Decimal("2")

        weighted_mid = (
            (bid_price * ask_size + ask_price * bid_size) / total_size
        )

        return weighted_mid

    def _calculate_imbalance(
        self,
        orderbook: OrderBook
    ) -> Decimal:
        """Calculate order book imbalance ratio.

        Imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

        Args:
            orderbook: OrderBook instance

        Returns:
            Imbalance ratio between -1 and 1
        """
        # Sum volume for configured depth
        bid_volume = sum(
            size for _, size in orderbook.bids[:self.imbalance_depth]
        )
        ask_volume = sum(
            size for _, size in orderbook.asks[:self.imbalance_depth]
        )

        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return Decimal("0")

        imbalance = (bid_volume - ask_volume) / total_volume

        return imbalance

    def _calculate_liquidity_metrics(
        self,
        orderbook: OrderBook
    ) -> Dict[str, Decimal]:
        """Calculate liquidity depth metrics.

        Args:
            orderbook: OrderBook instance

        Returns:
            Dictionary of liquidity metrics
        """
        metrics = {}

        # Total bid/ask volume at depth
        bid_volume = sum(
            size for _, size in orderbook.bids[:self.depth_levels]
        )
        ask_volume = sum(
            size for _, size in orderbook.asks[:self.depth_levels]
        )

        metrics["bid_liquidity"] = bid_volume
        metrics["ask_liquidity"] = ask_volume
        metrics["total_liquidity"] = bid_volume + ask_volume

        # Notional values
        if orderbook.bids and orderbook.asks:
            mid_price = (orderbook.bids[0][0] + orderbook.asks[0][0]) / Decimal("2")

            metrics["bid_liquidity_notional"] = bid_volume * mid_price
            metrics["ask_liquidity_notional"] = ask_volume * mid_price

            # Average price levels
            if bid_volume > 0:
                weighted_bid = sum(
                    price * size for price, size in orderbook.bids[:self.depth_levels]
                )
                metrics["avg_bid_price"] = weighted_bid / bid_volume
            else:
                metrics["avg_bid_price"] = Decimal("0")

            if ask_volume > 0:
                weighted_ask = sum(
                    price * size for price, size in orderbook.asks[:self.depth_levels]
                )
                metrics["avg_ask_price"] = weighted_ask / ask_volume
            else:
                metrics["avg_ask_price"] = Decimal("0")

        return metrics

    def _calculate_advanced_metrics(
        self,
        orderbook: OrderBook
    ) -> Dict[str, Decimal]:
        """Calculate advanced market microstructure metrics.

        Args:
            orderbook: OrderBook instance

        Returns:
            Dictionary of advanced metrics
        """
        metrics = {}

        if not orderbook.bids or not orderbook.asks:
            return metrics

        # Price impact estimation (cost of eating N levels)
        impact_levels = min(5, len(orderbook.asks))
        if impact_levels > 0:
            base_price = orderbook.asks[0][0]
            total_cost = Decimal("0")
            total_volume = Decimal("0")

            for price, size in orderbook.asks[:impact_levels]:
                total_cost += price * size
                total_volume += size

            if total_volume > 0:
                avg_fill_price = total_cost / total_volume
                metrics["ask_slippage_bps"] = (
                    (avg_fill_price - base_price) / base_price * Decimal("10000")
                )

        # Bid side impact
        impact_levels = min(5, len(orderbook.bids))
        if impact_levels > 0:
            base_price = orderbook.bids[0][0]
            total_proceeds = Decimal("0")
            total_volume = Decimal("0")

            for price, size in orderbook.bids[:impact_levels]:
                total_proceeds += price * size
                total_volume += size

            if total_volume > 0:
                avg_fill_price = total_proceeds / total_volume
                metrics["bid_slippage_bps"] = (
                    (base_price - avg_fill_price) / base_price * Decimal("10000")
                )

        # Depth at percentage levels
        mid_price = (orderbook.bids[0][0] + orderbook.asks[0][0]) / Decimal("2")

        for pct in [Decimal("0.001"), Decimal("0.005"), Decimal("0.01")]:
            # Bid depth at price level
            threshold_price = mid_price * (Decimal("1") - pct)
            bid_depth = sum(
                size for price, size in orderbook.bids
                if price >= threshold_price
            )
            metrics[f"bid_depth_{int(pct * 10000)}bps"] = bid_depth

            # Ask depth at price level
            threshold_price = mid_price * (Decimal("1") + pct)
            ask_depth = sum(
                size for price, size in orderbook.asks
                if price <= threshold_price
            )
            metrics[f"ask_depth_{int(pct * 10000)}bps"] = ask_depth

        # Order book pressure (derivative metrics)
        if orderbook.symbol in self._last_orderbooks:
            last_ob = self._last_orderbooks[orderbook.symbol]

            # Changes in top level volumes
            curr_bid_vol = orderbook.bids[0][1] if orderbook.bids else Decimal("0")
            last_bid_vol = last_ob.bids[0][1] if last_ob.bids else Decimal("0")
            metrics["bid_volume_change"] = curr_bid_vol - last_bid_vol

            curr_ask_vol = orderbook.asks[0][1] if orderbook.asks else Decimal("0")
            last_ask_vol = last_ob.asks[0][1] if last_ob.asks else Decimal("0")
            metrics["ask_volume_change"] = curr_ask_vol - last_ask_vol

        return metrics

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
                "checkpoint_saved",
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
            "checkpoint_restored",
            processed=self._processed_count
        )

    def get_metrics(self) -> Dict[str, int]:
        """Get processor metrics.

        Returns:
            Dictionary of metrics
        """
        return self._metrics.copy()

    async def calculate_vwap(
        self,
        orderbook: OrderBook,
        target_volume: Decimal,
        side: str
    ) -> Decimal:
        """Calculate VWAP for executing target volume.

        Args:
            orderbook: OrderBook instance
            target_volume: Volume to execute
            side: "buy" or "sell"

        Returns:
            Volume-weighted average price

        Example:
            >>> vwap = await processor.calculate_vwap(ob, Decimal("10"), "buy")
        """
        levels = orderbook.asks if side.lower() == "buy" else orderbook.bids

        total_cost = Decimal("0")
        remaining_volume = target_volume

        for price, size in levels:
            if remaining_volume <= 0:
                break

            fill_volume = min(remaining_volume, size)
            total_cost += price * fill_volume
            remaining_volume -= fill_volume

        if remaining_volume > 0:
            # Not enough liquidity
            logger.warning(
                "insufficient_liquidity",
                symbol=orderbook.symbol,
                target=str(target_volume),
                remaining=str(remaining_volume)
            )

        filled_volume = target_volume - remaining_volume
        if filled_volume > 0:
            return total_cost / filled_volume
        else:
            return Decimal("0")
