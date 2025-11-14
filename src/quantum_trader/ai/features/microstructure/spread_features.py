"""Spread-based features for market microstructure analysis.

This module extracts features from bid-ask spreads and order book data
to capture market microstructure dynamics and liquidity conditions.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import os

import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class SpreadFeatures:
    """Extract spread-based microstructure features.

    Analyzes bid-ask spreads, effective spreads, and realized spreads
    to quantify liquidity, transaction costs, and market impact.

    Attributes:
        config: Configuration dictionary
        spread_window: Window for spread calculations
        percentiles: Percentiles to calculate for spread distribution

    Examples:
        >>> config = {"spread_window": 100, "percentiles": [25, 50, 75]}
        >>> extractor = SpreadFeatures(config)
        >>> features = await extractor.extract(
        ...     data=pl.DataFrame({"bid": [...], "ask": [...], "price": [...]})
        ... )
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize spread feature extractor.

        Args:
            config: Configuration with window sizes, percentiles, etc.

        Raises:
            ValueError: If required config parameters missing
        """
        self.config = config
        self._validate_config()

        self.spread_window: int = int(config.get("spread_window", os.getenv("SPREAD_WINDOW", "100")))
        self.percentiles: List[int] = config.get("percentiles", [int(x) for x in os.getenv("SPREAD_PERCENTILES", "25,50,75").split(",")])
        self.min_observations: int = int(config.get("min_observations", os.getenv("SPREAD_MIN_OBS", "10")))
        self.tick_size: Decimal = Decimal(str(config.get("tick_size", os.getenv("SPREAD_TICK_SIZE", "0.01"))))

        logger.info(
            "Spread features extractor initialized",
            window=self.spread_window,
            percentiles=self.percentiles,
            min_observations=self.min_observations
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config validation fails
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        if "spread_window" in self.config:
            window = int(self.config["spread_window"])
            if window <= 0:
                raise ValueError(f"spread_window must be positive, got {window}")

        if "min_observations" in self.config:
            min_obs = int(self.config["min_observations"])
            if min_obs <= 0:
                raise ValueError(f"min_observations must be positive, got {min_obs}")

    async def extract(
        self,
        data: pl.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ) -> pl.DataFrame:
        """Extract spread features from order book data.

        Args:
            data: DataFrame with 'bid', 'ask', 'price', 'volume' columns
            metadata: Optional metadata

        Returns:
            DataFrame with spread features

        Raises:
            ValueError: If data DataFrame invalid
        """
        try:
            if data.is_empty():
                logger.warning("Empty data DataFrame provided")
                return self._empty_features()

            required_cols = ["bid", "ask"]
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Limit to window size
            if len(data) > self.spread_window:
                data = data.tail(self.spread_window)

            if len(data) < self.min_observations:
                logger.warning(
                    "Insufficient observations for spread features",
                    observations=len(data),
                    min_required=self.min_observations
                )
                return self._empty_features()

            # Calculate quoted spread
            quoted_spread_features = await self._calculate_quoted_spread(data)

            # Calculate effective spread (if trade price available)
            effective_spread_features = {}
            if "price" in data.columns:
                effective_spread_features = await self._calculate_effective_spread(data)

            # Calculate realized spread (if price available)
            realized_spread_features = {}
            if "price" in data.columns:
                realized_spread_features = await self._calculate_realized_spread(data)

            # Calculate spread volatility
            volatility_features = await self._calculate_spread_volatility(data)

            # Calculate relative spread
            relative_spread_features = await self._calculate_relative_spread(data)

            # Combine all features
            features = self._combine_features(
                quoted_spread_features,
                effective_spread_features,
                realized_spread_features,
                volatility_features,
                relative_spread_features
            )

            logger.debug(
                "Spread features extracted",
                num_observations=len(data),
                num_features=len(features)
            )

            return features

        except Exception as e:
            logger.error(
                "Spread feature extraction failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def _calculate_quoted_spread(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate quoted spread features.

        Args:
            data: DataFrame with bid and ask prices

        Returns:
            Dictionary of quoted spread metrics
        """
        # Calculate spreads
        spreads = []
        for bid, ask in zip(data["bid"].to_list(), data["ask"].to_list()):
            if bid is not None and ask is not None:
                bid_dec = Decimal(str(bid))
                ask_dec = Decimal(str(ask))
                spread = ask_dec - bid_dec
                spreads.append(spread)

        if not spreads:
            return {
                "quoted_spread_mean": Decimal("0"),
                "quoted_spread_std": Decimal("0"),
                "quoted_spread_min": Decimal("0"),
                "quoted_spread_max": Decimal("0")
            }

        spreads_array = np.array([float(s) for s in spreads], dtype=np.float64)

        features = {
            "quoted_spread_mean": Decimal(str(np.mean(spreads_array))),
            "quoted_spread_std": Decimal(str(np.std(spreads_array))),
            "quoted_spread_min": Decimal(str(np.min(spreads_array))),
            "quoted_spread_max": Decimal(str(np.max(spreads_array)))
        }

        # Add percentiles
        for pct in self.percentiles:
            features[f"quoted_spread_p{pct}"] = Decimal(str(np.percentile(spreads_array, pct)))

        return features

    async def _calculate_effective_spread(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate effective spread features.

        Effective spread = 2 * |price - midpoint|

        Args:
            data: DataFrame with bid, ask, and price

        Returns:
            Dictionary of effective spread metrics
        """
        effective_spreads = []

        for bid, ask, price in zip(data["bid"].to_list(), data["ask"].to_list(), data["price"].to_list()):
            if bid is not None and ask is not None and price is not None:
                bid_dec = Decimal(str(bid))
                ask_dec = Decimal(str(ask))
                price_dec = Decimal(str(price))

                midpoint = (bid_dec + ask_dec) / Decimal("2")
                effective_spread = Decimal("2") * abs(price_dec - midpoint)
                effective_spreads.append(effective_spread)

        if not effective_spreads:
            return {}

        spreads_array = np.array([float(s) for s in effective_spreads], dtype=np.float64)

        return {
            "effective_spread_mean": Decimal(str(np.mean(spreads_array))),
            "effective_spread_std": Decimal(str(np.std(spreads_array))),
            "effective_spread_median": Decimal(str(np.median(spreads_array)))
        }

    async def _calculate_realized_spread(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate realized spread features.

        Realized spread measures the temporary price impact of a trade.

        Args:
            data: DataFrame with bid, ask, and price

        Returns:
            Dictionary of realized spread metrics
        """
        if len(data) < 2:
            return {}

        realized_spreads = []

        for i in range(len(data) - 1):
            bid_curr = data["bid"][i]
            ask_curr = data["ask"][i]
            price_curr = data["price"][i]
            bid_next = data["bid"][i + 1]
            ask_next = data["ask"][i + 1]

            if all(x is not None for x in [bid_curr, ask_curr, price_curr, bid_next, ask_next]):
                bid_curr_dec = Decimal(str(bid_curr))
                ask_curr_dec = Decimal(str(ask_curr))
                price_curr_dec = Decimal(str(price_curr))
                bid_next_dec = Decimal(str(bid_next))
                ask_next_dec = Decimal(str(ask_next))

                midpoint_curr = (bid_curr_dec + ask_curr_dec) / Decimal("2")
                midpoint_next = (bid_next_dec + ask_next_dec) / Decimal("2")

                realized_spread = Decimal("2") * (price_curr_dec - midpoint_curr) * (midpoint_next - midpoint_curr)
                realized_spreads.append(realized_spread)

        if not realized_spreads:
            return {}

        spreads_array = np.array([float(s) for s in realized_spreads], dtype=np.float64)

        return {
            "realized_spread_mean": Decimal(str(np.mean(spreads_array))),
            "realized_spread_std": Decimal(str(np.std(spreads_array)))
        }

    async def _calculate_spread_volatility(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate spread volatility features.

        Args:
            data: DataFrame with bid and ask

        Returns:
            Dictionary of volatility metrics
        """
        spreads = []
        for bid, ask in zip(data["bid"].to_list(), data["ask"].to_list()):
            if bid is not None and ask is not None:
                spread = Decimal(str(ask)) - Decimal(str(bid))
                spreads.append(spread)

        if len(spreads) < 2:
            return {"spread_volatility": Decimal("0")}

        spreads_array = np.array([float(s) for s in spreads], dtype=np.float64)

        # Calculate rolling volatility
        volatility = Decimal(str(np.std(spreads_array)))

        return {
            "spread_volatility": volatility,
            "spread_cv": volatility / Decimal(str(np.mean(spreads_array) + 1e-10))  # Coefficient of variation
        }

    async def _calculate_relative_spread(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate relative spread (spread / midpoint).

        Args:
            data: DataFrame with bid and ask

        Returns:
            Dictionary of relative spread metrics
        """
        relative_spreads = []

        for bid, ask in zip(data["bid"].to_list(), data["ask"].to_list()):
            if bid is not None and ask is not None:
                bid_dec = Decimal(str(bid))
                ask_dec = Decimal(str(ask))
                midpoint = (bid_dec + ask_dec) / Decimal("2")

                if midpoint > Decimal("0"):
                    relative_spread = (ask_dec - bid_dec) / midpoint
                    relative_spreads.append(relative_spread)

        if not relative_spreads:
            return {"relative_spread_mean": Decimal("0")}

        spreads_array = np.array([float(s) for s in relative_spreads], dtype=np.float64)

        return {
            "relative_spread_mean": Decimal(str(np.mean(spreads_array))),
            "relative_spread_std": Decimal(str(np.std(spreads_array)))
        }

    def _combine_features(self, *feature_dicts: Dict[str, Decimal]) -> pl.DataFrame:
        """Combine multiple feature dictionaries.

        Args:
            *feature_dicts: Variable number of feature dictionaries

        Returns:
            Combined features DataFrame
        """
        all_features = {}
        for features in feature_dicts:
            all_features.update(features)

        if not all_features:
            return self._empty_features()

        return pl.DataFrame({
            "feature": list(all_features.keys()),
            "value": [str(v) for v in all_features.values()]
        })

    def _empty_features(self) -> pl.DataFrame:
        """Return empty features DataFrame.

        Returns:
            DataFrame with zero-valued features
        """
        zero_features = {
            "quoted_spread_mean": "0",
            "quoted_spread_std": "0",
            "quoted_spread_min": "0",
            "quoted_spread_max": "0",
            "spread_volatility": "0",
            "relative_spread_mean": "0"
        }

        return pl.DataFrame({
            "feature": list(zero_features.keys()),
            "value": list(zero_features.values())
        })

    async def extract_batch(
        self,
        data_batch: List[pl.DataFrame],
        metadata_batch: Optional[List[Dict[str, Any]]] = None
    ) -> List[pl.DataFrame]:
        """Extract features for batch of data.

        Args:
            data_batch: List of DataFrames
            metadata_batch: Optional list of metadata dicts

        Returns:
            List of feature DataFrames
        """
        try:
            metadata_batch = metadata_batch or [None] * len(data_batch)

            features_batch = []
            for data, metadata in zip(data_batch, metadata_batch):
                features = await self.extract(data, metadata)
                features_batch.append(features)

            logger.debug(
                "Batch spread features extracted",
                batch_size=len(features_batch)
            )

            return features_batch

        except Exception as e:
            logger.error(
                "Batch spread feature extraction failed",
                error=str(e),
                batch_size=len(data_batch)
            )
            raise
