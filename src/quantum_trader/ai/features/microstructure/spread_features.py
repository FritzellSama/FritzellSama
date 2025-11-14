"""
Bid-Ask Spread Microstructure Features

Production-ready feature extraction from order book spread dynamics.
Analyzes bid-ask spreads, depth imbalances, and microstructure signals.
"""

import asyncio
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from collections import deque
import os

import polars as pl
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)

# Set high precision
getcontext().prec = 28


class SpreadFeatures:
    """
    Extract microstructure features from bid-ask spread dynamics.

    Analyzes order book spreads, depth, and imbalances to identify
    liquidity conditions and potential price movements.

    Attributes:
        config: Configuration dictionary
        lookback_window: Number of observations for rolling calculations
        spread_percentiles: Percentile levels for spread analysis
        depth_levels: Number of order book levels to analyze

    Example:
        >>> config = {"features": {"microstructure": {"lookback_window": 100}}}
        >>> spread = SpreadFeatures(config)
        >>> features = await spread.extract_features(orderbook_df, symbol="BTC/USDT")
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize spread features extractor.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        microstructure_config = self.config.get("features", {}).get("microstructure", {})

        # Load configuration
        self.lookback_window: int = microstructure_config.get(
            "lookback_window",
            int(os.getenv("SPREAD_LOOKBACK_WINDOW", "100"))
        )
        self.depth_levels: int = microstructure_config.get(
            "depth_levels",
            int(os.getenv("SPREAD_DEPTH_LEVELS", "10"))
        )
        self.spread_percentiles: List[int] = microstructure_config.get(
            "spread_percentiles",
            [int(x) for x in os.getenv("SPREAD_PERCENTILES", "25,50,75,90,95").split(",")]
        )

        # Thresholds for classification
        self.tight_spread_threshold: Decimal = Decimal(
            str(microstructure_config.get("tight_spread_threshold", os.getenv("TIGHT_SPREAD_THRESHOLD", "0.001")))
        )
        self.wide_spread_threshold: Decimal = Decimal(
            str(microstructure_config.get("wide_spread_threshold", os.getenv("WIDE_SPREAD_THRESHOLD", "0.01")))
        )

        # Rolling buffers for historical analysis
        self.spread_buffer: deque = deque(maxlen=self.lookback_window)
        self.imbalance_buffer: deque = deque(maxlen=self.lookback_window)

        logger.info(
            "spread_features_initialized",
            lookback_window=self.lookback_window,
            depth_levels=self.depth_levels,
            percentiles=self.spread_percentiles
        )

    def _validate_config(self) -> None:
        """
        Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        microstructure_config = self.config.get("features", {}).get("microstructure", {})

        if microstructure_config:
            lookback = microstructure_config.get("lookback_window", 100)
            if lookback <= 0:
                raise ValueError("lookback_window must be positive")

            depth = microstructure_config.get("depth_levels", 10)
            if depth <= 0 or depth > 100:
                raise ValueError("depth_levels must be between 1 and 100")

    async def extract_features(
        self,
        orderbook_data: pl.DataFrame,
        symbol: str,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Decimal]:
        """
        Extract spread features from orderbook data.

        Args:
            orderbook_data: DataFrame with orderbook snapshots
                Required columns: timestamp, bid_price, ask_price, bid_volume, ask_volume
                Optional: bid_depth_[1-N], ask_depth_[1-N]
            symbol: Trading symbol
            timestamp: Reference timestamp (defaults to latest)

        Returns:
            Dictionary of spread features

        Raises:
            ValueError: If data schema is invalid

        Example:
            >>> features = await spread.extract_features(
            ...     orderbook_df,
            ...     symbol="BTC/USDT",
            ...     timestamp=datetime.now(timezone.utc)
            ... )
        """
        try:
            # Validate schema
            required_cols = ["timestamp", "bid_price", "ask_price", "bid_volume", "ask_volume"]
            if not all(col in orderbook_data.columns for col in required_cols):
                raise ValueError(f"DataFrame missing required columns: {required_cols}")

            if len(orderbook_data) == 0:
                logger.warning("empty_orderbook_data", symbol=symbol)
                return self._get_default_features()

            # Sort by timestamp
            data = orderbook_data.sort("timestamp")

            # Get latest snapshot if timestamp not specified
            if timestamp is None:
                latest_data = data[-1:]
            else:
                latest_data = data.filter(pl.col("timestamp") <= timestamp).tail(1)

            if len(latest_data) == 0:
                logger.warning("no_data_for_timestamp", symbol=symbol, timestamp=timestamp)
                return self._get_default_features()

            # Extract features
            features = {}

            # Current spread metrics
            features.update(self._calculate_current_spread(latest_data))

            # Historical spread statistics
            if len(data) >= 2:
                features.update(self._calculate_spread_statistics(data))

            # Order book imbalance
            features.update(self._calculate_imbalance(latest_data))

            # Depth analysis
            if self._has_depth_data(orderbook_data):
                features.update(self._calculate_depth_features(latest_data))

            # Spread dynamics
            if len(data) >= 2:
                features.update(self._calculate_spread_dynamics(data))

            # Liquidity metrics
            features.update(self._calculate_liquidity_metrics(latest_data))

            logger.debug(
                "spread_features_extracted",
                symbol=symbol,
                spread_bps=str(features.get("spread_bps", 0)),
                imbalance=str(features.get("order_imbalance", 0))
            )

            return features

        except Exception as e:
            logger.error("spread_feature_extraction_failed", symbol=symbol, error=str(e))
            raise

    def _calculate_current_spread(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate current spread metrics."""
        row = data.to_dicts()[0]

        bid_price = Decimal(str(row["bid_price"]))
        ask_price = Decimal(str(row["ask_price"]))

        # Absolute spread
        spread = ask_price - bid_price

        # Relative spread (basis points)
        mid_price = (bid_price + ask_price) / Decimal("2")
        spread_bps = (spread / mid_price) * Decimal("10000") if mid_price > 0 else Decimal("0")

        # Classify spread
        is_tight = Decimal("1") if spread_bps < self.tight_spread_threshold * Decimal("10000") else Decimal("0")
        is_wide = Decimal("1") if spread_bps > self.wide_spread_threshold * Decimal("10000") else Decimal("0")

        # Add to buffer
        self.spread_buffer.append(spread_bps)

        return {
            "spread_absolute": spread,
            "spread_bps": spread_bps,
            "mid_price": mid_price,
            "is_tight_spread": is_tight,
            "is_wide_spread": is_wide
        }

    def _calculate_spread_statistics(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate historical spread statistics."""
        # Calculate spreads for all rows
        spreads = []

        for row in data.iter_rows(named=True):
            bid = Decimal(str(row["bid_price"]))
            ask = Decimal(str(row["ask_price"]))
            mid = (bid + ask) / Decimal("2")

            spread_bps = ((ask - bid) / mid) * Decimal("10000") if mid > 0 else Decimal("0")
            spreads.append(float(spread_bps))

        spreads_array = np.array(spreads)

        # Calculate statistics
        mean_spread = Decimal(str(np.mean(spreads_array)))
        std_spread = Decimal(str(np.std(spreads_array)))
        min_spread = Decimal(str(np.min(spreads_array)))
        max_spread = Decimal(str(np.max(spreads_array)))

        # Percentiles
        percentile_features = {}
        for pct in self.spread_percentiles:
            pct_value = Decimal(str(np.percentile(spreads_array, pct)))
            percentile_features[f"spread_p{pct}"] = pct_value

        return {
            "spread_mean": mean_spread,
            "spread_std": std_spread,
            "spread_min": min_spread,
            "spread_max": max_spread,
            **percentile_features
        }

    def _calculate_imbalance(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate order book imbalance."""
        row = data.to_dicts()[0]

        bid_volume = Decimal(str(row["bid_volume"]))
        ask_volume = Decimal(str(row["ask_volume"]))

        total_volume = bid_volume + ask_volume

        # Order imbalance: positive = more bids, negative = more asks
        if total_volume > 0:
            imbalance = (bid_volume - ask_volume) / total_volume
        else:
            imbalance = Decimal("0")

        # Imbalance ratio
        if ask_volume > 0:
            imbalance_ratio = bid_volume / ask_volume
        else:
            imbalance_ratio = Decimal("10")  # Cap at 10x

        # Add to buffer
        self.imbalance_buffer.append(imbalance)

        return {
            "order_imbalance": imbalance,
            "imbalance_ratio": imbalance_ratio,
            "total_top_volume": total_volume
        }

    def _calculate_depth_features(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate order book depth features."""
        row = data.to_dicts()[0]

        bid_depth_total = Decimal("0")
        ask_depth_total = Decimal("0")

        # Sum depth across levels
        for level in range(1, self.depth_levels + 1):
            bid_col = f"bid_depth_{level}"
            ask_col = f"ask_depth_{level}"

            if bid_col in row:
                bid_depth_total += Decimal(str(row[bid_col]))
            if ask_col in row:
                ask_depth_total += Decimal(str(row[ask_col]))

        total_depth = bid_depth_total + ask_depth_total

        # Depth imbalance
        if total_depth > 0:
            depth_imbalance = (bid_depth_total - ask_depth_total) / total_depth
        else:
            depth_imbalance = Decimal("0")

        # Average depth per level
        avg_bid_depth = bid_depth_total / Decimal(str(self.depth_levels))
        avg_ask_depth = ask_depth_total / Decimal(str(self.depth_levels))

        return {
            "bid_depth_total": bid_depth_total,
            "ask_depth_total": ask_depth_total,
            "total_depth": total_depth,
            "depth_imbalance": depth_imbalance,
            "avg_bid_depth": avg_bid_depth,
            "avg_ask_depth": avg_ask_depth
        }

    def _calculate_spread_dynamics(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate spread dynamics over time."""
        if len(self.spread_buffer) < 2:
            return {
                "spread_change": Decimal("0"),
                "spread_volatility": Decimal("0"),
                "spread_trend": Decimal("0")
            }

        spreads = list(self.spread_buffer)
        spreads_array = np.array(spreads)

        # Spread change (latest - previous)
        spread_change = Decimal(str(spreads[-1] - spreads[-2]))

        # Spread volatility
        spread_volatility = Decimal(str(np.std(spreads_array)))

        # Spread trend (simple linear regression slope)
        if len(spreads) >= 3:
            x = np.arange(len(spreads))
            coeffs = np.polyfit(x, spreads_array, 1)
            spread_trend = Decimal(str(coeffs[0]))
        else:
            spread_trend = Decimal("0")

        return {
            "spread_change": spread_change,
            "spread_volatility": spread_volatility,
            "spread_trend": spread_trend
        }

    def _calculate_liquidity_metrics(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate liquidity-related metrics."""
        row = data.to_dicts()[0]

        bid_price = Decimal(str(row["bid_price"]))
        ask_price = Decimal(str(row["ask_price"]))
        bid_volume = Decimal(str(row["bid_volume"]))
        ask_volume = Decimal(str(row["ask_volume"]))

        # Volume-weighted spread
        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            vw_spread = (
                (ask_price * ask_volume + bid_price * bid_volume) / total_volume
            ) - ((bid_price + ask_price) / Decimal("2"))
        else:
            vw_spread = Decimal("0")

        # Effective spread (half-spread)
        mid_price = (bid_price + ask_price) / Decimal("2")
        effective_spread = (ask_price - bid_price) / Decimal("2")
        effective_spread_bps = (effective_spread / mid_price) * Decimal("10000") if mid_price > 0 else Decimal("0")

        # Price impact (simplified - would need trade data for real calculation)
        # Using volume at top of book as proxy
        min_volume = min(bid_volume, ask_volume)
        liquidity_score = min_volume / (effective_spread + Decimal("0.0001"))  # Avoid div by zero

        return {
            "volume_weighted_spread": vw_spread,
            "effective_spread_bps": effective_spread_bps,
            "liquidity_score": liquidity_score
        }

    def _has_depth_data(self, data: pl.DataFrame) -> bool:
        """Check if orderbook has depth data."""
        return "bid_depth_1" in data.columns and "ask_depth_1" in data.columns

    def _get_default_features(self) -> Dict[str, Decimal]:
        """Return default features when insufficient data."""
        features = {
            "spread_absolute": Decimal("0"),
            "spread_bps": Decimal("0"),
            "mid_price": Decimal("0"),
            "is_tight_spread": Decimal("0"),
            "is_wide_spread": Decimal("0"),
            "order_imbalance": Decimal("0"),
            "imbalance_ratio": Decimal("1"),
            "total_top_volume": Decimal("0"),
            "spread_change": Decimal("0"),
            "spread_volatility": Decimal("0"),
            "spread_trend": Decimal("0"),
            "volume_weighted_spread": Decimal("0"),
            "effective_spread_bps": Decimal("0"),
            "liquidity_score": Decimal("0")
        }

        # Add percentile defaults
        for pct in self.spread_percentiles:
            features[f"spread_p{pct}"] = Decimal("0")

        return features

    def reset(self) -> None:
        """Reset internal buffers."""
        self.spread_buffer.clear()
        self.imbalance_buffer.clear()
        logger.info("spread_features_reset")
