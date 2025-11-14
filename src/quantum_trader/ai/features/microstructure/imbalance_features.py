"""Market microstructure imbalance features.

This module extracts order flow imbalance features from order book and
trade data for high-frequency trading and market making strategies.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import polars as pl
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)


class ImbalanceFeatureExtractor:
    """Extract order flow imbalance features.

    Computes various imbalance metrics from order book and trade data,
    including volume imbalance, order imbalance, and trade imbalance.

    Attributes:
        config: Configuration dictionary
        levels: Number of order book levels to use
        time_windows: Time windows for aggregation

    Example:
        >>> config = {
        ...     "levels": 5,
        ...     "time_windows": ["1s", "5s", "30s"],
        ...     "volume_threshold": "1000.0"
        ... }
        >>> extractor = ImbalanceFeatureExtractor(config)
        >>> features = await extractor.extract_features(orderbook_data, trade_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize imbalance feature extractor.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        self.levels = config.get("levels", 5)
        self.time_windows = config.get("time_windows", ["1s", "5s", "30s"])
        self.volume_threshold = Decimal(str(config.get("volume_threshold", "1000.0")))

        logger.info(
            "Imbalance feature extractor initialized",
            levels=self.levels,
            time_windows=self.time_windows
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If required config missing
        """
        required_keys = ["levels", "time_windows"]

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    async def extract_features(
        self,
        orderbook_data: pl.DataFrame,
        trade_data: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """Extract imbalance features.

        Args:
            orderbook_data: Order book snapshots with columns
                [timestamp, symbol, bid_prices, bid_volumes, ask_prices, ask_volumes]
            trade_data: Optional trade data with columns
                [timestamp, symbol, price, volume, side]

        Returns:
            DataFrame with imbalance features

        Raises:
            ValueError: If input data is invalid
        """
        try:
            self._validate_orderbook_data(orderbook_data)

            logger.info(
                "Extracting imbalance features",
                orderbook_rows=len(orderbook_data),
                trade_rows=len(trade_data) if trade_data is not None else 0
            )

            # Calculate order book imbalance features
            features = await self._calculate_orderbook_imbalance(orderbook_data)

            # Add volume-based features
            features = await self._calculate_volume_imbalance(features, orderbook_data)

            # Add price-based features
            features = await self._calculate_price_imbalance(features, orderbook_data)

            # Add trade imbalance if trade data available
            if trade_data is not None:
                self._validate_trade_data(trade_data)
                features = await self._calculate_trade_imbalance(
                    features,
                    orderbook_data,
                    trade_data
                )

            # Add temporal features
            features = await self._calculate_temporal_imbalance(features, orderbook_data)

            logger.info(
                "Imbalance features extracted",
                feature_count=len(features.columns),
                rows=len(features)
            )

            return features

        except Exception as e:
            logger.error("Failed to extract imbalance features", error=str(e))
            raise

    def _validate_orderbook_data(self, data: pl.DataFrame) -> None:
        """Validate order book data structure.

        Args:
            data: Order book DataFrame

        Raises:
            ValueError: If data structure is invalid
        """
        required_columns = [
            "timestamp",
            "symbol",
            "bid_prices",
            "bid_volumes",
            "ask_prices",
            "ask_volumes"
        ]

        missing = set(required_columns) - set(data.columns)

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if len(data) == 0:
            raise ValueError("Empty orderbook data provided")

    def _validate_trade_data(self, data: pl.DataFrame) -> None:
        """Validate trade data structure.

        Args:
            data: Trade DataFrame

        Raises:
            ValueError: If data structure is invalid
        """
        required_columns = ["timestamp", "symbol", "price", "volume", "side"]

        missing = set(required_columns) - set(data.columns)

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    async def _calculate_orderbook_imbalance(
        self,
        data: pl.DataFrame
    ) -> pl.DataFrame:
        """Calculate basic order book imbalance metrics.

        Args:
            data: Order book data

        Returns:
            DataFrame with imbalance features
        """
        try:
            features = data.select([
                "timestamp",
                "symbol"
            ])

            # Calculate for each level
            for level in range(self.levels):
                # Volume imbalance at level
                bid_vol_col = f"bid_volume_l{level}"
                ask_vol_col = f"ask_volume_l{level}"

                # Extract volumes at level
                features = features.with_columns([
                    pl.lit(data["bid_volumes"].list.get(level).fill_null(0))
                    .alias(bid_vol_col),

                    pl.lit(data["ask_volumes"].list.get(level).fill_null(0))
                    .alias(ask_vol_col)
                ])

                # Imbalance ratio
                imbalance_col = f"imbalance_ratio_l{level}"
                features = features.with_columns(
                    ((pl.col(bid_vol_col) - pl.col(ask_vol_col)) /
                     (pl.col(bid_vol_col) + pl.col(ask_vol_col) + Decimal("0.001")))
                    .alias(imbalance_col)
                )

            # Aggregate imbalance across all levels
            bid_sum = sum(
                pl.col(f"bid_volume_l{i}")
                for i in range(self.levels)
            )

            ask_sum = sum(
                pl.col(f"ask_volume_l{i}")
                for i in range(self.levels)
            )

            features = features.with_columns([
                bid_sum.alias("total_bid_volume"),
                ask_sum.alias("total_ask_volume"),

                ((bid_sum - ask_sum) / (bid_sum + ask_sum + Decimal("0.001")))
                .alias("total_imbalance_ratio")
            ])

            return features

        except Exception as e:
            logger.error("Failed to calculate orderbook imbalance", error=str(e))
            raise

    async def _calculate_volume_imbalance(
        self,
        features: pl.DataFrame,
        orderbook_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Calculate volume-weighted imbalance metrics.

        Args:
            features: Existing features
            orderbook_data: Order book data

        Returns:
            DataFrame with added volume imbalance features
        """
        try:
            # Weighted imbalance (closer levels weighted more)
            weights = np.exp(-np.arange(self.levels) * 0.5)  # Exponential decay

            weighted_bid = Decimal("0")
            weighted_ask = Decimal("0")

            for level in range(self.levels):
                weight = Decimal(str(weights[level]))

                weighted_bid += pl.col(f"bid_volume_l{level}") * weight
                weighted_ask += pl.col(f"ask_volume_l{level}") * weight

            features = features.with_columns([
                weighted_bid.alias("weighted_bid_volume"),
                weighted_ask.alias("weighted_ask_volume"),

                ((weighted_bid - weighted_ask) /
                 (weighted_bid + weighted_ask + Decimal("0.001")))
                .alias("weighted_imbalance_ratio")
            ])

            # Cumulative volume at each level
            for level in range(1, self.levels):
                cum_bid = sum(
                    pl.col(f"bid_volume_l{i}")
                    for i in range(level + 1)
                )

                cum_ask = sum(
                    pl.col(f"ask_volume_l{i}")
                    for i in range(level + 1)
                )

                features = features.with_columns(
                    ((cum_bid - cum_ask) / (cum_bid + cum_ask + Decimal("0.001")))
                    .alias(f"cumulative_imbalance_l{level}")
                )

            return features

        except Exception as e:
            logger.error("Failed to calculate volume imbalance", error=str(e))
            raise

    async def _calculate_price_imbalance(
        self,
        features: pl.DataFrame,
        orderbook_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Calculate price-based imbalance metrics.

        Args:
            features: Existing features
            orderbook_data: Order book data

        Returns:
            DataFrame with added price imbalance features
        """
        try:
            # Mid price
            best_bid = orderbook_data["bid_prices"].list.get(0)
            best_ask = orderbook_data["ask_prices"].list.get(0)

            features = features.with_columns([
                pl.lit(best_bid).alias("best_bid"),
                pl.lit(best_ask).alias("best_ask"),

                ((best_bid + best_ask) / Decimal("2.0"))
                .alias("mid_price"),

                (best_ask - best_bid).alias("spread"),

                ((best_ask - best_bid) / ((best_bid + best_ask) / Decimal("2.0")))
                .alias("relative_spread")
            ])

            # VWAP imbalance (volume-weighted average price)
            bid_vwap = Decimal("0")
            ask_vwap = Decimal("0")
            total_bid_vol = Decimal("0.001")
            total_ask_vol = Decimal("0.001")

            for level in range(self.levels):
                bid_price = orderbook_data["bid_prices"].list.get(level)
                ask_price = orderbook_data["ask_prices"].list.get(level)

                bid_vwap += bid_price * pl.col(f"bid_volume_l{level}")
                ask_vwap += ask_price * pl.col(f"ask_volume_l{level}")

                total_bid_vol += pl.col(f"bid_volume_l{level}")
                total_ask_vol += pl.col(f"ask_volume_l{level}")

            features = features.with_columns([
                (bid_vwap / total_bid_vol).alias("bid_vwap"),
                (ask_vwap / total_ask_vol).alias("ask_vwap"),

                ((bid_vwap / total_bid_vol - ask_vwap / total_ask_vol) /
                 pl.col("mid_price"))
                .alias("vwap_imbalance")
            ])

            return features

        except Exception as e:
            logger.error("Failed to calculate price imbalance", error=str(e))
            raise

    async def _calculate_trade_imbalance(
        self,
        features: pl.DataFrame,
        orderbook_data: pl.DataFrame,
        trade_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Calculate trade flow imbalance metrics.

        Args:
            features: Existing features
            orderbook_data: Order book data
            trade_data: Trade data

        Returns:
            DataFrame with added trade imbalance features
        """
        try:
            # Aggregate trades by time window
            for window in self.time_windows:
                window_ms = self._parse_time_window(window)

                # Calculate trade imbalance per symbol and time window
                trade_agg = (
                    trade_data
                    .sort("timestamp")
                    .group_by_dynamic("timestamp", every=window, by="symbol")
                    .agg([
                        # Buy volume
                        pl.when(pl.col("side") == "buy")
                        .then(pl.col("volume"))
                        .otherwise(0)
                        .sum()
                        .alias(f"buy_volume_{window}"),

                        # Sell volume
                        pl.when(pl.col("side") == "sell")
                        .then(pl.col("volume"))
                        .otherwise(0)
                        .sum()
                        .alias(f"sell_volume_{window}"),

                        # Trade count
                        pl.count().alias(f"trade_count_{window}")
                    ])
                )

                # Calculate imbalance
                trade_agg = trade_agg.with_columns([
                    ((pl.col(f"buy_volume_{window}") - pl.col(f"sell_volume_{window}")) /
                     (pl.col(f"buy_volume_{window}") + pl.col(f"sell_volume_{window}") + Decimal("0.001")))
                    .alias(f"trade_imbalance_{window}")
                ])

                # Join with features (asof join for nearest timestamp)
                features = features.join_asof(
                    trade_agg,
                    on="timestamp",
                    by="symbol",
                    strategy="backward"
                )

            return features

        except Exception as e:
            logger.error("Failed to calculate trade imbalance", error=str(e))
            raise

    async def _calculate_temporal_imbalance(
        self,
        features: pl.DataFrame,
        orderbook_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Calculate temporal changes in imbalance.

        Args:
            features: Existing features
            orderbook_data: Order book data

        Returns:
            DataFrame with temporal imbalance features
        """
        try:
            # Sort by symbol and timestamp
            features = features.sort(["symbol", "timestamp"])

            # Calculate changes in imbalance
            features = features.with_columns([
                # Change in total imbalance
                (pl.col("total_imbalance_ratio") -
                 pl.col("total_imbalance_ratio").shift(1).over("symbol"))
                .alias("imbalance_change"),

                # Imbalance momentum (rate of change)
                ((pl.col("total_imbalance_ratio") -
                  pl.col("total_imbalance_ratio").shift(5).over("symbol")) /
                 Decimal("5.0"))
                .alias("imbalance_momentum"),

                # Imbalance volatility (rolling std)
                pl.col("total_imbalance_ratio")
                .rolling_std(window_size=20, min_periods=5)
                .over("symbol")
                .alias("imbalance_volatility")
            ])

            return features

        except Exception as e:
            logger.error("Failed to calculate temporal imbalance", error=str(e))
            raise

    def _parse_time_window(self, window: str) -> int:
        """Parse time window string to milliseconds.

        Args:
            window: Time window string (e.g., '1s', '5m', '1h')

        Returns:
            Duration in milliseconds
        """
        unit = window[-1]
        value = int(window[:-1])

        if unit == 's':
            return value * 1000
        elif unit == 'm':
            return value * 60 * 1000
        elif unit == 'h':
            return value * 60 * 60 * 1000
        else:
            raise ValueError(f"Invalid time unit: {unit}")

    def calculate_imbalance_score(
        self,
        features: pl.DataFrame,
        symbol: str,
        timestamp: datetime
    ) -> Decimal:
        """Calculate aggregate imbalance score.

        Args:
            features: Feature DataFrame
            symbol: Symbol to calculate score for
            timestamp: Timestamp to evaluate at

        Returns:
            Imbalance score between -1 (sell pressure) and 1 (buy pressure)

        Raises:
            ValueError: If data not found
        """
        try:
            # Filter to symbol and timestamp
            row = features.filter(
                (pl.col("symbol") == symbol) &
                (pl.col("timestamp") == timestamp)
            )

            if len(row) == 0:
                raise ValueError(f"No data found for {symbol} at {timestamp}")

            # Weighted combination of imbalance metrics
            weights = {
                "total": Decimal(str(self.config.get("weight_total", "0.3"))),
                "weighted": Decimal(str(self.config.get("weight_weighted", "0.4"))),
                "vwap": Decimal(str(self.config.get("weight_vwap", "0.3")))
            }

            total_imb = Decimal(str(row["total_imbalance_ratio"][0]))
            weighted_imb = Decimal(str(row["weighted_imbalance_ratio"][0]))
            vwap_imb = Decimal(str(row["vwap_imbalance"][0]))

            # Aggregate score
            score = (
                weights["total"] * total_imb +
                weights["weighted"] * weighted_imb +
                weights["vwap"] * vwap_imb
            )

            # Clamp to [-1, 1]
            score = max(Decimal("-1"), min(Decimal("1"), score))

            logger.info(
                "Calculated imbalance score",
                symbol=symbol,
                score=float(score)
            )

            return score

        except Exception as e:
            logger.error(
                "Failed to calculate imbalance score",
                symbol=symbol,
                error=str(e)
            )
            raise
