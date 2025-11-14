"""VWAP (Volume Weighted Average Price) Indicator.

Production-ready implementation of VWAP calculation using Polars for
high-performance data processing.
"""

import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class VWAPCalculator:
    """Calculate Volume Weighted Average Price.

    VWAP is calculated as the cumulative (Price * Volume) / cumulative Volume.
    Typically reset at the beginning of each trading session.

    Attributes:
        config: Configuration dictionary with calculation parameters
        session_start_hour: Hour to reset VWAP (UTC)
        price_precision: Decimal precision for price calculations

    Example:
        >>> config = {"session_start_hour": 0, "price_precision": 8}
        >>> calculator = VWAPCalculator(config)
        >>> df = await calculator.calculate(market_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize VWAP calculator.

        Args:
            config: Configuration dictionary containing:
                - session_start_hour: Hour (0-23) to reset VWAP
                - price_precision: Decimal places for price
                - include_bands: Whether to calculate VWAP bands
                - band_multiplier: Multiplier for standard deviation bands

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.session_start_hour = self.config.get("session_start_hour", 0)
        self.price_precision = self.config.get("price_precision", 8)
        self.include_bands = self.config.get("include_bands", True)
        self.band_multiplier = Decimal(str(self.config.get("band_multiplier", "2.0")))

        logger.info(
            "vwap_calculator_initialized",
            session_start_hour=self.session_start_hour,
            price_precision=self.price_precision,
            include_bands=self.include_bands
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        session_hour = self.config.get("session_start_hour", 0)
        if not (0 <= session_hour < 24):
            raise ValueError(f"session_start_hour must be 0-23, got {session_hour}")

        precision = self.config.get("price_precision", 8)
        if precision < 0 or precision > 18:
            raise ValueError(f"price_precision must be 0-18, got {precision}")

        band_mult = self.config.get("band_multiplier", 2.0)
        if band_mult <= 0:
            raise ValueError(f"band_multiplier must be positive, got {band_mult}")

    async def calculate(
        self,
        data: pl.DataFrame,
        reset_on_session: bool = True
    ) -> pl.DataFrame:
        """Calculate VWAP for market data.

        Args:
            data: Polars DataFrame with columns:
                - timestamp: UTC timestamp
                - high: High price
                - low: Low price
                - close: Close price
                - volume: Trading volume
            reset_on_session: Whether to reset VWAP at session start

        Returns:
            DataFrame with additional columns:
                - vwap: Volume weighted average price
                - vwap_upper: Upper band (if enabled)
                - vwap_lower: Lower band (if enabled)

        Raises:
            ValueError: If data is invalid or missing required columns
        """
        try:
            self._validate_data(data)

            logger.debug(
                "calculating_vwap",
                rows=len(data),
                reset_on_session=reset_on_session
            )

            # Calculate typical price (High + Low + Close) / 3
            result = data.with_columns([
                ((pl.col("high") + pl.col("low") + pl.col("close")) / pl.lit(3.0))
                .alias("typical_price")
            ])

            # Calculate price * volume
            result = result.with_columns([
                (pl.col("typical_price") * pl.col("volume")).alias("pv")
            ])

            if reset_on_session:
                # Add session identifier based on timestamp
                result = result.with_columns([
                    pl.col("timestamp").dt.truncate("1d").alias("session")
                ])

                # Calculate cumulative sums within each session
                result = result.with_columns([
                    pl.col("pv").cum_sum().over("session").alias("cum_pv"),
                    pl.col("volume").cum_sum().over("session").alias("cum_volume")
                ])
            else:
                # Calculate cumulative sums across all data
                result = result.with_columns([
                    pl.col("pv").cum_sum().alias("cum_pv"),
                    pl.col("volume").cum_sum().alias("cum_volume")
                ])

            # Calculate VWAP
            result = result.with_columns([
                (pl.col("cum_pv") / pl.col("cum_volume")).alias("vwap")
            ])

            # Calculate VWAP bands if enabled
            if self.include_bands:
                result = await self._calculate_bands(result, reset_on_session)

            # Clean up intermediate columns
            result = result.drop(["typical_price", "pv", "cum_pv", "cum_volume"])
            if reset_on_session:
                result = result.drop(["session"])

            logger.debug("vwap_calculated", rows=len(result))

            return result

        except Exception as e:
            logger.error("vwap_calculation_failed", error=str(e))
            raise

    async def _calculate_bands(
        self,
        data: pl.DataFrame,
        reset_on_session: bool
    ) -> pl.DataFrame:
        """Calculate VWAP standard deviation bands.

        Args:
            data: DataFrame with VWAP already calculated
            reset_on_session: Whether bands are session-based

        Returns:
            DataFrame with upper and lower bands added
        """
        try:
            # Calculate squared deviations
            result = data.with_columns([
                ((pl.col("close") - pl.col("vwap")) ** 2).alias("sq_dev")
            ])

            # Calculate cumulative variance
            if reset_on_session:
                result = result.with_columns([
                    pl.col("sq_dev").cum_sum().over("session").alias("cum_sq_dev"),
                    pl.col("volume").cum_sum().over("session").alias("count")
                ])
            else:
                result = result.with_columns([
                    pl.col("sq_dev").cum_sum().alias("cum_sq_dev"),
                    pl.len().alias("count")
                ])

            # Calculate standard deviation
            result = result.with_columns([
                (pl.col("cum_sq_dev") / pl.col("count")).sqrt().alias("vwap_std")
            ])

            # Calculate bands
            band_mult_float = float(self.band_multiplier)
            result = result.with_columns([
                (pl.col("vwap") + pl.col("vwap_std") * band_mult_float).alias("vwap_upper"),
                (pl.col("vwap") - pl.col("vwap_std") * band_mult_float).alias("vwap_lower")
            ])

            # Clean up intermediate columns
            result = result.drop(["sq_dev", "cum_sq_dev", "count", "vwap_std"])

            return result

        except Exception as e:
            logger.error("vwap_bands_calculation_failed", error=str(e))
            raise

    def _validate_data(self, data: pl.DataFrame) -> None:
        """Validate input data.

        Args:
            data: Input DataFrame

        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, pl.DataFrame):
            raise ValueError("Data must be a Polars DataFrame")

        if len(data) == 0:
            raise ValueError("Data cannot be empty")

        required_columns = ["timestamp", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Validate no null values in critical columns
        for col in required_columns:
            null_count = data[col].null_count()
            if null_count > 0:
                raise ValueError(f"Column '{col}' contains {null_count} null values")

        # Validate volume is positive
        if (data["volume"] <= 0).any():
            raise ValueError("Volume must be positive")

    async def calculate_anchored_vwap(
        self,
        data: pl.DataFrame,
        anchor_timestamp: datetime
    ) -> pl.DataFrame:
        """Calculate anchored VWAP from a specific timestamp.

        Anchored VWAP starts calculation from a significant event
        (e.g., market open, significant high/low).

        Args:
            data: Market data DataFrame
            anchor_timestamp: Timestamp to anchor VWAP calculation

        Returns:
            DataFrame with anchored_vwap column

        Raises:
            ValueError: If anchor_timestamp is invalid
        """
        try:
            self._validate_data(data)

            if anchor_timestamp.tzinfo is None:
                anchor_timestamp = anchor_timestamp.replace(tzinfo=timezone.utc)

            logger.debug(
                "calculating_anchored_vwap",
                anchor=anchor_timestamp.isoformat(),
                rows=len(data)
            )

            # Filter data from anchor point onwards
            filtered = data.filter(pl.col("timestamp") >= anchor_timestamp)

            if len(filtered) == 0:
                raise ValueError("No data after anchor timestamp")

            # Calculate typical price
            result = filtered.with_columns([
                ((pl.col("high") + pl.col("low") + pl.col("close")) / pl.lit(3.0))
                .alias("typical_price")
            ])

            # Calculate anchored VWAP
            result = result.with_columns([
                (pl.col("typical_price") * pl.col("volume")).alias("pv")
            ])

            result = result.with_columns([
                pl.col("pv").cum_sum().alias("cum_pv"),
                pl.col("volume").cum_sum().alias("cum_volume")
            ])

            result = result.with_columns([
                (pl.col("cum_pv") / pl.col("cum_volume")).alias("anchored_vwap")
            ])

            # Clean up intermediate columns
            result = result.drop(["typical_price", "pv", "cum_pv", "cum_volume"])

            logger.debug("anchored_vwap_calculated", rows=len(result))

            return result

        except Exception as e:
            logger.error("anchored_vwap_calculation_failed", error=str(e))
            raise

    def get_vwap_deviation(
        self,
        data: pl.DataFrame,
        as_percentage: bool = True
    ) -> pl.DataFrame:
        """Calculate price deviation from VWAP.

        Args:
            data: DataFrame with VWAP already calculated
            as_percentage: Return deviation as percentage

        Returns:
            DataFrame with vwap_deviation column

        Raises:
            ValueError: If VWAP not present in data
        """
        try:
            if "vwap" not in data.columns:
                raise ValueError("VWAP must be calculated first")

            if as_percentage:
                result = data.with_columns([
                    ((pl.col("close") - pl.col("vwap")) / pl.col("vwap") * 100.0)
                    .alias("vwap_deviation")
                ])
            else:
                result = data.with_columns([
                    (pl.col("close") - pl.col("vwap")).alias("vwap_deviation")
                ])

            return result

        except Exception as e:
            logger.error("vwap_deviation_calculation_failed", error=str(e))
            raise
