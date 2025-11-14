"""
Order Book Microstructure Features for Trading.

This module extracts microstructure features from order book data
including depth imbalance, spread, and liquidity metrics.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class OrderBookSnapshot:
    """Order book snapshot data structure.

    Attributes:
        symbol: Trading symbol
        timestamp: Snapshot timestamp
        bids: List of (price, quantity) tuples for bids
        asks: List of (price, quantity) tuples for asks
        mid_price: Mid price
        spread: Bid-ask spread
    """
    symbol: str
    timestamp: datetime
    bids: List[Tuple[Decimal, Decimal]]
    asks: List[Tuple[Decimal, Decimal]]
    mid_price: Decimal
    spread: Decimal


class OrderBookFeatures:
    """Production-ready order book feature extractor.

    Extracts microstructure features from order book data for
    high-frequency trading and market making strategies.

    Attributes:
        config: Configuration dictionary
        depth_levels: Number of order book levels to analyze
        imbalance_window: Window for imbalance calculation
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize order book feature extractor.

        Args:
            config: Configuration dictionary containing:
                - features.microstructure.depth_levels
                - features.microstructure.imbalance_window
                - features.microstructure.volume_window
                - features.microstructure.spread_normalization
        """
        self.config = config
        self._validate_config()

        microstructure_config = self.config["features"]["microstructure"]
        self.depth_levels = microstructure_config["depth_levels"]
        self.imbalance_window = microstructure_config["imbalance_window"]
        self.volume_window = microstructure_config["volume_window"]
        self.spread_normalization = microstructure_config["spread_normalization"]

        logger.info("orderbook_features_initialized",
                   depth_levels=self.depth_levels,
                   imbalance_window=self.imbalance_window)

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config parameters are missing
        """
        required_keys = [
            "features.microstructure.depth_levels",
            "features.microstructure.imbalance_window",
            "features.microstructure.volume_window",
            "features.microstructure.spread_normalization"
        ]

        for key in required_keys:
            parts = key.split(".")
            current = self.config
            for part in parts:
                if part not in current:
                    raise ValueError(f"Missing required config key: {key}")
                current = current[part]

    def calculate_bid_ask_spread(
        self,
        snapshot: OrderBookSnapshot
    ) -> Dict[str, Decimal]:
        """Calculate bid-ask spread metrics.

        Args:
            snapshot: Order book snapshot

        Returns:
            Dictionary of spread metrics

        Raises:
            ValueError: If order book is empty
        """
        try:
            if not snapshot.bids or not snapshot.asks:
                raise ValueError("Empty order book")

            best_bid = snapshot.bids[0][0]
            best_ask = snapshot.asks[0][0]

            # Absolute spread
            absolute_spread = best_ask - best_bid

            # Relative spread (percentage)
            mid_price = (best_bid + best_ask) / Decimal("2")
            relative_spread = (absolute_spread / mid_price) * Decimal("100") if mid_price > 0 else Decimal("0")

            # Basis points
            spread_bps = relative_spread * Decimal("100")

            return {
                "absolute_spread": absolute_spread,
                "relative_spread": relative_spread,
                "spread_bps": spread_bps,
                "mid_price": mid_price,
                "best_bid": best_bid,
                "best_ask": best_ask
            }

        except Exception as e:
            logger.error("spread_calculation_failed", error=str(e))
            raise

    def calculate_depth_imbalance(
        self,
        snapshot: OrderBookSnapshot,
        levels: Optional[int] = None
    ) -> Dict[str, Decimal]:
        """Calculate order book depth imbalance.

        Args:
            snapshot: Order book snapshot
            levels: Number of levels to analyze (uses config default if None)

        Returns:
            Dictionary of imbalance metrics

        Raises:
            ValueError: If order book is invalid
        """
        try:
            num_levels = levels if levels is not None else self.depth_levels

            if not snapshot.bids or not snapshot.asks:
                raise ValueError("Empty order book")

            # Calculate total volume at each side
            bid_volume = Decimal("0")
            ask_volume = Decimal("0")

            for i in range(min(num_levels, len(snapshot.bids))):
                bid_volume += snapshot.bids[i][1]

            for i in range(min(num_levels, len(snapshot.asks))):
                ask_volume += snapshot.asks[i][1]

            # Total volume
            total_volume = bid_volume + ask_volume

            # Volume imbalance (-1 to 1, positive = more bids)
            if total_volume > 0:
                volume_imbalance = (bid_volume - ask_volume) / total_volume
            else:
                volume_imbalance = Decimal("0")

            # Bid/ask ratio
            bid_ask_ratio = bid_volume / ask_volume if ask_volume > 0 else Decimal("0")

            # Weighted mid price
            if bid_volume + ask_volume > 0:
                weighted_mid = (
                    snapshot.bids[0][0] * ask_volume +
                    snapshot.asks[0][0] * bid_volume
                ) / (bid_volume + ask_volume)
            else:
                weighted_mid = snapshot.mid_price

            return {
                "volume_imbalance": volume_imbalance,
                "bid_volume": bid_volume,
                "ask_volume": ask_volume,
                "total_volume": total_volume,
                "bid_ask_ratio": bid_ask_ratio,
                "weighted_mid_price": weighted_mid
            }

        except Exception as e:
            logger.error("imbalance_calculation_failed", error=str(e))
            raise

    def calculate_liquidity_metrics(
        self,
        snapshot: OrderBookSnapshot,
        levels: Optional[int] = None
    ) -> Dict[str, Decimal]:
        """Calculate liquidity metrics.

        Args:
            snapshot: Order book snapshot
            levels: Number of levels to analyze

        Returns:
            Dictionary of liquidity metrics
        """
        try:
            num_levels = levels if levels is not None else self.depth_levels

            if not snapshot.bids or not snapshot.asks:
                return {
                    "total_liquidity": Decimal("0"),
                    "bid_liquidity": Decimal("0"),
                    "ask_liquidity": Decimal("0"),
                    "liquidity_imbalance": Decimal("0"),
                    "vwap_bid": Decimal("0"),
                    "vwap_ask": Decimal("0")
                }

            # Calculate liquidity (volume * price)
            bid_liquidity = Decimal("0")
            ask_liquidity = Decimal("0")
            bid_vwap_numerator = Decimal("0")
            ask_vwap_numerator = Decimal("0")
            bid_volume_total = Decimal("0")
            ask_volume_total = Decimal("0")

            for i in range(min(num_levels, len(snapshot.bids))):
                price, volume = snapshot.bids[i]
                liquidity = price * volume
                bid_liquidity += liquidity
                bid_vwap_numerator += price * volume
                bid_volume_total += volume

            for i in range(min(num_levels, len(snapshot.asks))):
                price, volume = snapshot.asks[i]
                liquidity = price * volume
                ask_liquidity += liquidity
                ask_vwap_numerator += price * volume
                ask_volume_total += volume

            total_liquidity = bid_liquidity + ask_liquidity

            # Liquidity imbalance
            if total_liquidity > 0:
                liquidity_imbalance = (bid_liquidity - ask_liquidity) / total_liquidity
            else:
                liquidity_imbalance = Decimal("0")

            # VWAP calculations
            vwap_bid = bid_vwap_numerator / bid_volume_total if bid_volume_total > 0 else Decimal("0")
            vwap_ask = ask_vwap_numerator / ask_volume_total if ask_volume_total > 0 else Decimal("0")

            return {
                "total_liquidity": total_liquidity,
                "bid_liquidity": bid_liquidity,
                "ask_liquidity": ask_liquidity,
                "liquidity_imbalance": liquidity_imbalance,
                "vwap_bid": vwap_bid,
                "vwap_ask": vwap_ask
            }

        except Exception as e:
            logger.error("liquidity_calculation_failed", error=str(e))
            raise

    def calculate_price_impact(
        self,
        snapshot: OrderBookSnapshot,
        order_size: Decimal
    ) -> Dict[str, Decimal]:
        """Calculate estimated price impact for given order size.

        Args:
            snapshot: Order book snapshot
            order_size: Order size to estimate impact for

        Returns:
            Dictionary of price impact metrics
        """
        try:
            if not snapshot.bids or not snapshot.asks:
                return {
                    "buy_impact": Decimal("0"),
                    "sell_impact": Decimal("0"),
                    "avg_buy_price": Decimal("0"),
                    "avg_sell_price": Decimal("0")
                }

            # Calculate buy impact (walking up the ask book)
            remaining_size = order_size
            total_cost = Decimal("0")
            volume_filled = Decimal("0")

            for price, volume in snapshot.asks:
                if remaining_size <= 0:
                    break

                fill_volume = min(remaining_size, volume)
                total_cost += fill_volume * price
                volume_filled += fill_volume
                remaining_size -= fill_volume

            avg_buy_price = total_cost / volume_filled if volume_filled > 0 else snapshot.asks[0][0]
            buy_impact = (avg_buy_price - snapshot.mid_price) / snapshot.mid_price * Decimal("100")

            # Calculate sell impact (walking down the bid book)
            remaining_size = order_size
            total_proceeds = Decimal("0")
            volume_filled = Decimal("0")

            for price, volume in snapshot.bids:
                if remaining_size <= 0:
                    break

                fill_volume = min(remaining_size, volume)
                total_proceeds += fill_volume * price
                volume_filled += fill_volume
                remaining_size -= fill_volume

            avg_sell_price = total_proceeds / volume_filled if volume_filled > 0 else snapshot.bids[0][0]
            sell_impact = (snapshot.mid_price - avg_sell_price) / snapshot.mid_price * Decimal("100")

            return {
                "buy_impact": buy_impact,
                "sell_impact": sell_impact,
                "avg_buy_price": avg_buy_price,
                "avg_sell_price": avg_sell_price
            }

        except Exception as e:
            logger.error("price_impact_calculation_failed", error=str(e))
            raise

    def calculate_market_depth(
        self,
        snapshot: OrderBookSnapshot,
        price_range_pct: Decimal = Decimal("0.1")
    ) -> Dict[str, Decimal]:
        """Calculate market depth within price range.

        Args:
            snapshot: Order book snapshot
            price_range_pct: Price range percentage from mid price

        Returns:
            Dictionary of depth metrics
        """
        try:
            if not snapshot.bids or not snapshot.asks:
                return {
                    "bid_depth": Decimal("0"),
                    "ask_depth": Decimal("0"),
                    "total_depth": Decimal("0"),
                    "depth_ratio": Decimal("0")
                }

            # Calculate price thresholds
            lower_bound = snapshot.mid_price * (Decimal("1") - price_range_pct)
            upper_bound = snapshot.mid_price * (Decimal("1") + price_range_pct)

            # Sum volumes within range
            bid_depth = Decimal("0")
            for price, volume in snapshot.bids:
                if price >= lower_bound:
                    bid_depth += volume
                else:
                    break

            ask_depth = Decimal("0")
            for price, volume in snapshot.asks:
                if price <= upper_bound:
                    ask_depth += volume
                else:
                    break

            total_depth = bid_depth + ask_depth
            depth_ratio = bid_depth / ask_depth if ask_depth > 0 else Decimal("0")

            return {
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
                "total_depth": total_depth,
                "depth_ratio": depth_ratio,
                "price_range_pct": price_range_pct
            }

        except Exception as e:
            logger.error("market_depth_calculation_failed", error=str(e))
            raise

    async def extract_all_features(
        self,
        snapshot: OrderBookSnapshot,
        order_size: Optional[Decimal] = None
    ) -> pl.DataFrame:
        """Extract all order book features.

        Args:
            snapshot: Order book snapshot
            order_size: Optional order size for impact calculation

        Returns:
            Polars DataFrame with all features

        Raises:
            ValueError: If snapshot is invalid
        """
        try:
            logger.debug("extracting_orderbook_features",
                        symbol=snapshot.symbol,
                        timestamp=snapshot.timestamp.isoformat())

            # Calculate all feature sets
            spread_metrics = self.calculate_bid_ask_spread(snapshot)
            imbalance_metrics = self.calculate_depth_imbalance(snapshot)
            liquidity_metrics = self.calculate_liquidity_metrics(snapshot)
            depth_metrics = self.calculate_market_depth(snapshot)

            # Combine all metrics
            features = {
                "symbol": snapshot.symbol,
                "timestamp": snapshot.timestamp,
                **{f"spread_{k}": str(v) for k, v in spread_metrics.items()},
                **{f"imbalance_{k}": str(v) for k, v in imbalance_metrics.items()},
                **{f"liquidity_{k}": str(v) for k, v in liquidity_metrics.items()},
                **{f"depth_{k}": str(v) for k, v in depth_metrics.items()}
            }

            # Add price impact if order size provided
            if order_size is not None:
                impact_metrics = self.calculate_price_impact(snapshot, order_size)
                features.update({f"impact_{k}": str(v) for k, v in impact_metrics.items()})

            # Create DataFrame
            df = pl.DataFrame([features])

            logger.debug("orderbook_features_extracted",
                        feature_count=len(features))

            return df

        except Exception as e:
            logger.error("feature_extraction_failed", error=str(e))
            raise

    async def process_orderbook_stream(
        self,
        snapshots: List[OrderBookSnapshot]
    ) -> pl.DataFrame:
        """Process stream of order book snapshots.

        Args:
            snapshots: List of order book snapshots

        Returns:
            Polars DataFrame with features for all snapshots

        Raises:
            ValueError: If snapshots list is empty
        """
        if not snapshots:
            raise ValueError("Snapshots list cannot be empty")

        try:
            logger.info("processing_orderbook_stream",
                       snapshot_count=len(snapshots))

            # Extract features for each snapshot
            feature_dfs = []

            for snapshot in snapshots:
                try:
                    df = await self.extract_all_features(snapshot)
                    feature_dfs.append(df)

                except Exception as e:
                    logger.error("snapshot_processing_failed",
                               symbol=snapshot.symbol,
                               timestamp=snapshot.timestamp.isoformat(),
                               error=str(e))
                    continue

            if not feature_dfs:
                raise ValueError("No features extracted from snapshots")

            # Concatenate all DataFrames
            result_df = pl.concat(feature_dfs)

            logger.info("orderbook_stream_processed",
                       snapshots_processed=len(feature_dfs),
                       feature_count=len(result_df.columns))

            return result_df

        except Exception as e:
            logger.error("orderbook_stream_processing_failed", error=str(e))
            raise
