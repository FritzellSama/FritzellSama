"""
Market Profile Indicator - Displays price distribution over time.

Market Profile organizes price and volume data to show where the market spent time,
revealing value areas, support/resistance levels, and market balance/imbalance.
"""

import asyncio
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Tuple, Set
import polars as pl
from structlog import get_logger
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict
from enum import Enum

logger = get_logger(__name__)

# Set high precision for Decimal calculations
getcontext().prec = 28


class ProfileShape(Enum):
    """Market Profile shapes."""
    NORMAL = "normal"  # Bell-shaped distribution
    P_SHAPE = "p_shape"  # Top heavy
    B_SHAPE = "b_shape"  # Bottom heavy
    DOUBLE_DISTRIBUTION = "double_distribution"  # Two peaks
    TREND = "trend"  # Sloping distribution


@dataclass
class ValueArea:
    """Value Area statistics."""
    value_area_high: Decimal
    value_area_low: Decimal
    point_of_control: Decimal
    value_area_volume: Decimal
    total_volume: Decimal
    value_area_percentage: Decimal


@dataclass
class MarketProfileResult:
    """Market Profile analysis result."""
    timestamp: datetime
    value_area: ValueArea
    profile_shape: ProfileShape
    initial_balance_high: Decimal
    initial_balance_low: Decimal
    day_high: Decimal
    day_low: Decimal
    volume_nodes: Dict[str, Decimal]


class MarketProfileIndicator:
    """
    Market Profile indicator for institutional trading.

    Creates a time-price-volume distribution showing:
    - Point of Control (POC): Price level with highest volume
    - Value Area (VA): Price range containing specified % of volume (typically 70%)
    - Initial Balance (IB): Price range of first time period
    - High/Low Volume Nodes

    Attributes:
        config: Configuration dictionary
        tick_size: Minimum price increment
        value_area_percentage: Percentage for value area (default 70%)
        time_period_minutes: Minutes per time period

    Example:
        >>> config = {
        ...     "tick_size": "0.01",
        ...     "value_area_percentage": "70",
        ...     "time_period_minutes": 30
        ... }
        >>> profile = MarketProfileIndicator(config)
        >>> result = await profile.calculate(ohlcv_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Market Profile indicator.

        Args:
            config: Configuration dictionary containing:
                - tick_size: Minimum price increment
                - value_area_percentage: Value area percentage (default 70)
                - time_period_minutes: Minutes per period (default 30)
                - initial_balance_periods: Number of initial periods (default 2)
                - min_volume_threshold: Minimum volume to consider

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.tick_size: Decimal = Decimal(str(config["tick_size"]))
        self.value_area_percentage: Decimal = Decimal(str(
            config.get("value_area_percentage", "70")
        ))
        self.time_period_minutes: int = int(config.get("time_period_minutes", 30))
        self.initial_balance_periods: int = int(config.get("initial_balance_periods", 2))
        self.min_volume_threshold: Decimal = Decimal(str(
            config.get("min_volume_threshold", "0")
        ))

        logger.info(
            "Market Profile indicator initialized",
            tick_size=str(self.tick_size),
            value_area_percentage=str(self.value_area_percentage)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_fields = ["tick_size"]

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")

        if Decimal(str(self.config["tick_size"])) <= Decimal("0"):
            raise ValueError("tick_size must be positive")

        if "value_area_percentage" in self.config:
            va_pct = Decimal(str(self.config["value_area_percentage"]))
            if va_pct <= Decimal("0") or va_pct > Decimal("100"):
                raise ValueError("value_area_percentage must be between 0 and 100")

        logger.debug("Market Profile configuration validated")

    async def calculate(self, data: pl.DataFrame) -> List[MarketProfileResult]:
        """
        Calculate Market Profile for the given data.

        Args:
            data: Polars DataFrame with columns: timestamp, open, high, low, close, volume

        Returns:
            List of MarketProfileResult for each trading session

        Raises:
            ValueError: If data is invalid or insufficient
        """
        try:
            # Validate input data
            self._validate_data(data)

            # Group data by trading sessions
            sessions = await self._group_by_sessions(data)

            # Calculate profile for each session
            results = []
            for session_data in sessions:
                profile = await self._calculate_session_profile(session_data)
                if profile:
                    results.append(profile)

            logger.info(
                "Market Profile calculated",
                sessions=len(results)
            )

            return results

        except Exception as e:
            logger.error("Market Profile calculation failed", error=str(e))
            raise

    async def _group_by_sessions(
        self,
        data: pl.DataFrame
    ) -> List[pl.DataFrame]:
        """
        Group data into trading sessions (typically daily).

        Args:
            data: Input DataFrame

        Returns:
            List of DataFrames, one per session
        """
        try:
            # Group by date
            data_with_date = data.with_columns([
                pl.col("timestamp").cast(pl.Datetime).dt.date().alias("date")
            ])

            sessions = []
            for date in data_with_date["date"].unique().sort():
                session_data = data_with_date.filter(pl.col("date") == date).drop("date")
                if len(session_data) > 0:
                    sessions.append(session_data)

            return sessions

        except Exception as e:
            logger.error("Session grouping failed", error=str(e))
            raise

    async def _calculate_session_profile(
        self,
        data: pl.DataFrame
    ) -> Optional[MarketProfileResult]:
        """
        Calculate Market Profile for a single session.

        Args:
            data: Session data

        Returns:
            MarketProfileResult or None if insufficient data
        """
        try:
            if len(data) == 0:
                return None

            # Build volume distribution
            volume_nodes = await self._build_volume_distribution(data)

            if not volume_nodes:
                return None

            # Calculate Point of Control (highest volume price)
            poc = await self._calculate_poc(volume_nodes)

            # Calculate Value Area
            value_area = await self._calculate_value_area(volume_nodes, poc)

            # Calculate Initial Balance
            ib_high, ib_low = await self._calculate_initial_balance(data)

            # Get day high/low
            day_high = Decimal(str(data["high"].max()))
            day_low = Decimal(str(data["low"].min()))

            # Determine profile shape
            profile_shape = await self._determine_profile_shape(volume_nodes, poc)

            # Get timestamp (end of session)
            timestamp = data["timestamp"][-1]
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            elif not isinstance(timestamp, datetime):
                timestamp = datetime.now(timezone.utc)

            result = MarketProfileResult(
                timestamp=timestamp,
                value_area=value_area,
                profile_shape=profile_shape,
                initial_balance_high=ib_high,
                initial_balance_low=ib_low,
                day_high=day_high,
                day_low=day_low,
                volume_nodes=volume_nodes
            )

            logger.debug(
                "Session profile calculated",
                poc=str(poc),
                va_high=str(value_area.value_area_high),
                va_low=str(value_area.value_area_low)
            )

            return result

        except Exception as e:
            logger.error("Session profile calculation failed", error=str(e))
            return None

    async def _build_volume_distribution(
        self,
        data: pl.DataFrame
    ) -> Dict[str, Decimal]:
        """
        Build volume distribution across price levels.

        Args:
            data: Session data

        Returns:
            Dictionary mapping price levels to volume
        """
        try:
            volume_nodes: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))

            for row in data.iter_rows(named=True):
                high = Decimal(str(row["high"]))
                low = Decimal(str(row["low"]))
                volume = Decimal(str(row["volume"]))

                # Distribute volume across price range
                # Round to tick size
                current_price = low
                price_range = high - low

                if price_range == Decimal("0"):
                    # All volume at one price
                    price_key = str(self._round_to_tick(low))
                    volume_nodes[price_key] += volume
                else:
                    # Distribute volume proportionally
                    num_ticks = int(price_range / self.tick_size) + 1
                    volume_per_tick = volume / Decimal(str(num_ticks))

                    for _ in range(num_ticks):
                        price_key = str(self._round_to_tick(current_price))
                        volume_nodes[price_key] += volume_per_tick
                        current_price += self.tick_size

            # Filter by minimum threshold
            if self.min_volume_threshold > Decimal("0"):
                volume_nodes = {
                    price: vol
                    for price, vol in volume_nodes.items()
                    if vol >= self.min_volume_threshold
                }

            return dict(volume_nodes)

        except Exception as e:
            logger.error("Volume distribution build failed", error=str(e))
            raise

    def _round_to_tick(self, price: Decimal) -> Decimal:
        """Round price to nearest tick size."""
        return (price / self.tick_size).quantize(Decimal("1")) * self.tick_size

    async def _calculate_poc(
        self,
        volume_nodes: Dict[str, Decimal]
    ) -> Decimal:
        """
        Calculate Point of Control (price with highest volume).

        Args:
            volume_nodes: Volume distribution

        Returns:
            POC price level
        """
        try:
            if not volume_nodes:
                raise ValueError("No volume nodes available")

            poc_price = max(volume_nodes.items(), key=lambda x: x[1])[0]
            return Decimal(poc_price)

        except Exception as e:
            logger.error("POC calculation failed", error=str(e))
            raise

    async def _calculate_value_area(
        self,
        volume_nodes: Dict[str, Decimal],
        poc: Decimal
    ) -> ValueArea:
        """
        Calculate Value Area containing specified percentage of volume.

        The Value Area expands from POC until it contains the target percentage.

        Args:
            volume_nodes: Volume distribution
            poc: Point of Control

        Returns:
            ValueArea dataclass
        """
        try:
            total_volume = sum(volume_nodes.values())
            target_volume = total_volume * (self.value_area_percentage / Decimal("100"))

            # Sort prices
            sorted_prices = sorted([Decimal(p) for p in volume_nodes.keys()])

            # Start at POC
            poc_str = str(poc)
            if poc_str not in volume_nodes:
                # Find closest price
                poc = min(sorted_prices, key=lambda x: abs(x - poc))
                poc_str = str(poc)

            # Initialize value area
            va_volume = volume_nodes[poc_str]
            va_high = poc
            va_low = poc

            # Expand value area
            high_idx = sorted_prices.index(poc)
            low_idx = sorted_prices.index(poc)

            while va_volume < target_volume:
                # Check if we can expand
                can_expand_up = high_idx < len(sorted_prices) - 1
                can_expand_down = low_idx > 0

                if not can_expand_up and not can_expand_down:
                    break

                # Calculate volume if we expand up or down
                vol_up = Decimal("0")
                vol_down = Decimal("0")

                if can_expand_up:
                    next_up = sorted_prices[high_idx + 1]
                    vol_up = volume_nodes[str(next_up)]

                if can_expand_down:
                    next_down = sorted_prices[low_idx - 1]
                    vol_down = volume_nodes[str(next_down)]

                # Expand towards higher volume
                if can_expand_up and (not can_expand_down or vol_up >= vol_down):
                    high_idx += 1
                    va_high = sorted_prices[high_idx]
                    va_volume += vol_up
                elif can_expand_down:
                    low_idx -= 1
                    va_low = sorted_prices[low_idx]
                    va_volume += vol_down

            value_area = ValueArea(
                value_area_high=va_high,
                value_area_low=va_low,
                point_of_control=poc,
                value_area_volume=va_volume,
                total_volume=total_volume,
                value_area_percentage=(va_volume / total_volume * Decimal("100"))
            )

            return value_area

        except Exception as e:
            logger.error("Value Area calculation failed", error=str(e))
            raise

    async def _calculate_initial_balance(
        self,
        data: pl.DataFrame
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate Initial Balance (high/low of first periods).

        Args:
            data: Session data

        Returns:
            Tuple of (IB high, IB low)
        """
        try:
            # Take first N periods
            ib_data = data.head(self.initial_balance_periods)

            if len(ib_data) == 0:
                # Fallback to full session
                ib_data = data

            ib_high = Decimal(str(ib_data["high"].max()))
            ib_low = Decimal(str(ib_data["low"].min()))

            return ib_high, ib_low

        except Exception as e:
            logger.error("Initial Balance calculation failed", error=str(e))
            raise

    async def _determine_profile_shape(
        self,
        volume_nodes: Dict[str, Decimal],
        poc: Decimal
    ) -> ProfileShape:
        """
        Determine the shape of the Market Profile.

        Args:
            volume_nodes: Volume distribution
            poc: Point of Control

        Returns:
            ProfileShape enum value
        """
        try:
            sorted_prices = sorted([Decimal(p) for p in volume_nodes.keys()])
            poc_idx = sorted_prices.index(poc)

            # Calculate volume distribution characteristics
            total_volume = sum(volume_nodes.values())

            # Volume above and below POC
            above_volume = sum(
                volume_nodes[str(p)]
                for p in sorted_prices[poc_idx+1:]
            )
            below_volume = sum(
                volume_nodes[str(p)]
                for p in sorted_prices[:poc_idx]
            )

            # Find secondary peaks
            volumes = [volume_nodes[str(p)] for p in sorted_prices]
            max_volume = max(volumes)

            peaks = []
            for i, vol in enumerate(volumes):
                # A peak is a local maximum above 70% of max volume
                if vol > max_volume * Decimal("0.7"):
                    is_peak = True
                    if i > 0 and volumes[i-1] > vol:
                        is_peak = False
                    if i < len(volumes) - 1 and volumes[i+1] > vol:
                        is_peak = False
                    if is_peak:
                        peaks.append(i)

            # Determine shape
            if len(peaks) >= 2:
                return ProfileShape.DOUBLE_DISTRIBUTION

            # P-shape: most volume at top
            if above_volume > below_volume * Decimal("1.5"):
                return ProfileShape.P_SHAPE

            # B-shape: most volume at bottom
            if below_volume > above_volume * Decimal("1.5"):
                return ProfileShape.B_SHAPE

            # Check for trend (sloping distribution)
            # Calculate center of mass
            total_price_volume = sum(
                Decimal(p) * vol
                for p, vol in volume_nodes.items()
            )
            center_of_mass = total_price_volume / total_volume

            # If center of mass is far from POC, it's trending
            if abs(center_of_mass - poc) > (sorted_prices[-1] - sorted_prices[0]) * Decimal("0.2"):
                return ProfileShape.TREND

            return ProfileShape.NORMAL

        except Exception as e:
            logger.error("Profile shape determination failed", error=str(e))
            return ProfileShape.NORMAL

    def _validate_data(self, data: pl.DataFrame) -> None:
        """Validate input data format and content."""
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]

        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        if len(data) == 0:
            raise ValueError("Data is empty")

        # Check for null values
        for col in ["high", "low", "volume"]:
            if data[col].null_count() > 0:
                raise ValueError(f"Data contains null values in '{col}' column")

    async def export_to_dataframe(
        self,
        results: List[MarketProfileResult]
    ) -> pl.DataFrame:
        """
        Export Market Profile results to a DataFrame.

        Args:
            results: List of MarketProfileResult

        Returns:
            DataFrame with profile data
        """
        try:
            records = []
            for result in results:
                records.append({
                    "timestamp": result.timestamp,
                    "poc": str(result.value_area.point_of_control),
                    "va_high": str(result.value_area.value_area_high),
                    "va_low": str(result.value_area.value_area_low),
                    "va_percentage": str(result.value_area.value_area_percentage),
                    "ib_high": str(result.initial_balance_high),
                    "ib_low": str(result.initial_balance_low),
                    "day_high": str(result.day_high),
                    "day_low": str(result.day_low),
                    "profile_shape": result.profile_shape.value
                })

            df = pl.DataFrame(records)
            return df

        except Exception as e:
            logger.error("DataFrame export failed", error=str(e))
            raise


async def create_market_profile_indicator(
    config: Dict[str, Any]
) -> MarketProfileIndicator:
    """
    Factory function to create Market Profile indicator instance.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized Market Profile indicator
    """
    return MarketProfileIndicator(config)
