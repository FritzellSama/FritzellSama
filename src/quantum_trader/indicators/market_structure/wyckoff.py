"""Wyckoff Market Structure Analysis.

Production-ready implementation of Wyckoff methodology for identifying
accumulation, distribution, and market phases using Polars.
"""

import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from datetime import datetime

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class WyckoffPhase(Enum):
    """Wyckoff market phases."""
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class WyckoffEvent(Enum):
    """Key Wyckoff events."""
    PRELIMINARY_SUPPORT = "preliminary_support"  # PS
    SELLING_CLIMAX = "selling_climax"  # SC
    AUTOMATIC_RALLY = "automatic_rally"  # AR
    SECONDARY_TEST = "secondary_test"  # ST
    SPRING = "spring"  # Spring
    SIGN_OF_STRENGTH = "sign_of_strength"  # SOS
    LAST_POINT_OF_SUPPORT = "last_point_of_support"  # LPS
    PRELIMINARY_SUPPLY = "preliminary_supply"  # PSY
    BUYING_CLIMAX = "buying_climax"  # BC
    AUTOMATIC_REACTION = "automatic_reaction"  # AR
    UPTHRUST = "upthrust"  # UT
    SIGN_OF_WEAKNESS = "sign_of_weakness"  # SOW
    LAST_POINT_OF_supply = "last_point_of_supply"  # LPSY
    NONE = "none"


class WyckoffAnalyzer:
    """Analyze market structure using Wyckoff methodology.

    The Wyckoff method identifies market phases and key events through
    price-volume analysis to determine institutional accumulation and distribution.

    Attributes:
        config: Configuration dictionary
        volume_threshold_multiplier: Multiplier for volume significance
        price_range_threshold: Minimum price range for phase detection

    Example:
        >>> config = {
        ...     "volume_threshold_multiplier": 1.5,
        ...     "price_range_threshold": 0.02,
        ...     "lookback_period": 50
        ... }
        >>> analyzer = WyckoffAnalyzer(config)
        >>> df = await analyzer.analyze(market_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Wyckoff analyzer.

        Args:
            config: Configuration dictionary containing:
                - volume_threshold_multiplier: Volume significance multiplier
                - price_range_threshold: Min price range for phases (decimal)
                - lookback_period: Period for context analysis
                - climax_volume_multiplier: Volume threshold for climaxes
                - spring_tolerance: Price tolerance for spring detection

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.volume_threshold_multiplier = Decimal(
            str(self.config.get("volume_threshold_multiplier", "1.5"))
        )
        self.price_range_threshold = Decimal(
            str(self.config.get("price_range_threshold", "0.02"))
        )
        self.lookback_period = self.config.get("lookback_period", 50)
        self.climax_volume_multiplier = Decimal(
            str(self.config.get("climax_volume_multiplier", "2.0"))
        )
        self.spring_tolerance = Decimal(
            str(self.config.get("spring_tolerance", "0.01"))
        )

        logger.info(
            "wyckoff_analyzer_initialized",
            volume_threshold=str(self.volume_threshold_multiplier),
            lookback_period=self.lookback_period
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        vol_mult = self.config.get("volume_threshold_multiplier", 1.5)
        if vol_mult <= 0:
            raise ValueError(f"volume_threshold_multiplier must be positive, got {vol_mult}")

        price_thresh = self.config.get("price_range_threshold", 0.02)
        if price_thresh < 0:
            raise ValueError(f"price_range_threshold must be non-negative, got {price_thresh}")

        lookback = self.config.get("lookback_period", 50)
        if lookback < 10:
            raise ValueError(f"lookback_period must be >= 10, got {lookback}")

    async def analyze(
        self,
        data: pl.DataFrame,
        detect_events: bool = True,
        identify_phases: bool = True
    ) -> pl.DataFrame:
        """Perform Wyckoff analysis on market data.

        Args:
            data: Polars DataFrame with columns:
                - timestamp: UTC timestamp
                - open: Open price
                - high: High price
                - low: Low price
                - close: Close price
                - volume: Trading volume
            detect_events: Whether to detect Wyckoff events
            identify_phases: Whether to identify market phases

        Returns:
            DataFrame with Wyckoff analysis columns:
                - wyckoff_phase: Current market phase
                - wyckoff_event: Detected events
                - effort_vs_result: Effort (volume) vs result (price change)
                - supply_demand_balance: Supply/demand indication

        Raises:
            ValueError: If data is invalid
        """
        try:
            self._validate_data(data)

            logger.debug(
                "starting_wyckoff_analysis",
                rows=len(data),
                detect_events=detect_events,
                identify_phases=identify_phases
            )

            result = data.clone()

            # Calculate basic metrics
            result = await self._calculate_metrics(result)

            # Analyze effort vs result
            result = await self._analyze_effort_vs_result(result)

            # Detect key Wyckoff events
            if detect_events:
                result = await self._detect_events(result)

            # Identify market phases
            if identify_phases:
                result = await self._identify_phases(result)

            logger.debug("wyckoff_analysis_completed", rows=len(result))

            return result

        except Exception as e:
            logger.error("wyckoff_analysis_failed", error=str(e))
            raise

    async def _calculate_metrics(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate supporting metrics for Wyckoff analysis.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with added metric columns
        """
        try:
            # Calculate price range and volume metrics
            result = data.with_columns([
                (pl.col("high") - pl.col("low")).alias("price_range"),
                (pl.col("close") - pl.col("open")).alias("price_change"),
                pl.col("volume")
                .rolling_mean(window_size=self.lookback_period)
                .alias("avg_volume"),
                pl.col("volume")
                .rolling_std(window_size=self.lookback_period)
                .alias("volume_std")
            ])

            # Calculate volume significance
            result = result.with_columns([
                (pl.col("volume") / pl.col("avg_volume")).alias("volume_ratio")
            ])

            # Calculate price momentum
            result = result.with_columns([
                pl.col("close").pct_change(n=1).alias("price_pct_change"),
                pl.col("close")
                .rolling_mean(window_size=20)
                .alias("sma_20"),
                pl.col("close")
                .rolling_mean(window_size=self.lookback_period)
                .alias("sma_long")
            ])

            # Identify swing highs and lows
            result = await self._identify_swing_points(result)

            return result

        except Exception as e:
            logger.error("metrics_calculation_failed", error=str(e))
            raise

    async def _identify_swing_points(
        self,
        data: pl.DataFrame,
        swing_window: int = 5
    ) -> pl.DataFrame:
        """Identify swing highs and lows.

        Args:
            data: DataFrame with OHLC data
            swing_window: Window for swing point detection

        Returns:
            DataFrame with swing point columns
        """
        try:
            result = data.with_columns([
                pl.col("high")
                .rolling_max(window_size=swing_window)
                .alias("swing_high_window"),
                pl.col("low")
                .rolling_min(window_size=swing_window)
                .alias("swing_low_window")
            ])

            # Mark swing highs and lows
            result = result.with_columns([
                (pl.col("high") == pl.col("swing_high_window")).alias("is_swing_high"),
                (pl.col("low") == pl.col("swing_low_window")).alias("is_swing_low")
            ])

            result = result.drop(["swing_high_window", "swing_low_window"])

            return result

        except Exception as e:
            logger.error("swing_point_identification_failed", error=str(e))
            raise

    async def _analyze_effort_vs_result(self, data: pl.DataFrame) -> pl.DataFrame:
        """Analyze effort (volume) versus result (price movement).

        Key principle: High effort with low result suggests accumulation/distribution.

        Args:
            data: DataFrame with calculated metrics

        Returns:
            DataFrame with effort_vs_result analysis
        """
        try:
            volume_mult = float(self.volume_threshold_multiplier)

            result = data.with_columns([
                pl.when(
                    (pl.col("volume_ratio") > volume_mult) &
                    (pl.col("price_range") < pl.col("price_range").rolling_mean(window_size=20))
                )
                .then(pl.lit("high_effort_low_result"))
                .when(
                    (pl.col("volume_ratio") < 1.0 / volume_mult) &
                    (pl.col("price_range") > pl.col("price_range").rolling_mean(window_size=20))
                )
                .then(pl.lit("low_effort_high_result"))
                .when(pl.col("volume_ratio") > volume_mult)
                .then(pl.lit("high_effort"))
                .otherwise(pl.lit("normal"))
                .alias("effort_vs_result")
            ])

            # Analyze supply/demand balance
            result = result.with_columns([
                pl.when(
                    (pl.col("close") > pl.col("open")) &
                    (pl.col("volume_ratio") > volume_mult)
                )
                .then(pl.lit("demand"))
                .when(
                    (pl.col("close") < pl.col("open")) &
                    (pl.col("volume_ratio") > volume_mult)
                )
                .then(pl.lit("supply"))
                .otherwise(pl.lit("neutral"))
                .alias("supply_demand_balance")
            ])

            return result

        except Exception as e:
            logger.error("effort_vs_result_analysis_failed", error=str(e))
            raise

    async def _detect_events(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect key Wyckoff events.

        Args:
            data: DataFrame with metrics calculated

        Returns:
            DataFrame with wyckoff_event column
        """
        try:
            result = data.clone()

            # Initialize event column
            result = result.with_columns([
                pl.lit(WyckoffEvent.NONE.value).alias("wyckoff_event")
            ])

            # Detect selling climax (SC)
            result = await self._detect_selling_climax(result)

            # Detect buying climax (BC)
            result = await self._detect_buying_climax(result)

            # Detect spring
            result = await self._detect_spring(result)

            # Detect upthrust
            result = await self._detect_upthrust(result)

            # Detect sign of strength (SOS)
            result = await self._detect_sign_of_strength(result)

            # Detect sign of weakness (SOW)
            result = await self._detect_sign_of_weakness(result)

            return result

        except Exception as e:
            logger.error("event_detection_failed", error=str(e))
            raise

    async def _detect_selling_climax(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect selling climax - high volume selloff near support.

        Args:
            data: DataFrame with metrics

        Returns:
            DataFrame with selling climax events marked
        """
        try:
            climax_mult = float(self.climax_volume_multiplier)
            lookback = min(20, self.lookback_period)

            result = data.with_columns([
                pl.col("low").rolling_min(window_size=lookback).alias("recent_low")
            ])

            # Selling climax conditions:
            # 1. Very high volume (> 2x average)
            # 2. Large down move
            # 3. Near recent lows
            # 4. Wide price range
            result = result.with_columns([
                pl.when(
                    (pl.col("wyckoff_event") == WyckoffEvent.NONE.value) &
                    (pl.col("volume_ratio") > climax_mult) &
                    (pl.col("close") < pl.col("open")) &
                    (pl.col("low") <= pl.col("recent_low") * 1.01) &
                    (pl.col("price_range") > pl.col("price_range").rolling_mean(window_size=20) * 1.5)
                )
                .then(pl.lit(WyckoffEvent.SELLING_CLIMAX.value))
                .otherwise(pl.col("wyckoff_event"))
                .alias("wyckoff_event")
            ])

            result = result.drop(["recent_low"])

            return result

        except Exception as e:
            logger.error("selling_climax_detection_failed", error=str(e))
            raise

    async def _detect_buying_climax(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect buying climax - high volume rally near resistance.

        Args:
            data: DataFrame with metrics

        Returns:
            DataFrame with buying climax events marked
        """
        try:
            climax_mult = float(self.climax_volume_multiplier)
            lookback = min(20, self.lookback_period)

            result = data.with_columns([
                pl.col("high").rolling_max(window_size=lookback).alias("recent_high")
            ])

            # Buying climax conditions
            result = result.with_columns([
                pl.when(
                    (pl.col("wyckoff_event") == WyckoffEvent.NONE.value) &
                    (pl.col("volume_ratio") > climax_mult) &
                    (pl.col("close") > pl.col("open")) &
                    (pl.col("high") >= pl.col("recent_high") * 0.99) &
                    (pl.col("price_range") > pl.col("price_range").rolling_mean(window_size=20) * 1.5)
                )
                .then(pl.lit(WyckoffEvent.BUYING_CLIMAX.value))
                .otherwise(pl.col("wyckoff_event"))
                .alias("wyckoff_event")
            ])

            result = result.drop(["recent_high"])

            return result

        except Exception as e:
            logger.error("buying_climax_detection_failed", error=str(e))
            raise

    async def _detect_spring(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect spring - false breakdown below support.

        Args:
            data: DataFrame with metrics

        Returns:
            DataFrame with spring events marked
        """
        try:
            spring_tol = float(self.spring_tolerance)
            lookback = min(30, self.lookback_period)

            result = data.with_columns([
                pl.col("low").rolling_min(window_size=lookback).shift(5).alias("prev_support")
            ])

            # Spring conditions:
            # 1. Price breaks below previous support
            # 2. Quickly reverses back above support
            # 3. Volume may be lower (weak hands shaken out)
            result = result.with_columns([
                pl.when(
                    (pl.col("wyckoff_event") == WyckoffEvent.NONE.value) &
                    (pl.col("low") < pl.col("prev_support") * (1 - spring_tol)) &
                    (pl.col("close") > pl.col("prev_support")) &
                    (pl.col("close") > pl.col("open"))
                )
                .then(pl.lit(WyckoffEvent.SPRING.value))
                .otherwise(pl.col("wyckoff_event"))
                .alias("wyckoff_event")
            ])

            result = result.drop(["prev_support"])

            return result

        except Exception as e:
            logger.error("spring_detection_failed", error=str(e))
            raise

    async def _detect_upthrust(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect upthrust - false breakout above resistance.

        Args:
            data: DataFrame with metrics

        Returns:
            DataFrame with upthrust events marked
        """
        try:
            spring_tol = float(self.spring_tolerance)
            lookback = min(30, self.lookback_period)

            result = data.with_columns([
                pl.col("high").rolling_max(window_size=lookback).shift(5).alias("prev_resistance")
            ])

            # Upthrust conditions
            result = result.with_columns([
                pl.when(
                    (pl.col("wyckoff_event") == WyckoffEvent.NONE.value) &
                    (pl.col("high") > pl.col("prev_resistance") * (1 + spring_tol)) &
                    (pl.col("close") < pl.col("prev_resistance")) &
                    (pl.col("close") < pl.col("open"))
                )
                .then(pl.lit(WyckoffEvent.UPTHRUST.value))
                .otherwise(pl.col("wyckoff_event"))
                .alias("wyckoff_event")
            ])

            result = result.drop(["prev_resistance"])

            return result

        except Exception as e:
            logger.error("upthrust_detection_failed", error=str(e))
            raise

    async def _detect_sign_of_strength(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect sign of strength - strong upward move on good volume.

        Args:
            data: DataFrame with metrics

        Returns:
            DataFrame with SOS events marked
        """
        try:
            vol_mult = float(self.volume_threshold_multiplier)

            result = data.with_columns([
                pl.when(
                    (pl.col("wyckoff_event") == WyckoffEvent.NONE.value) &
                    (pl.col("close") > pl.col("open")) &
                    (pl.col("volume_ratio") > vol_mult) &
                    (pl.col("price_change") > pl.col("price_range").rolling_mean(window_size=20)) &
                    (pl.col("close") > pl.col("sma_20"))
                )
                .then(pl.lit(WyckoffEvent.SIGN_OF_STRENGTH.value))
                .otherwise(pl.col("wyckoff_event"))
                .alias("wyckoff_event")
            ])

            return result

        except Exception as e:
            logger.error("sign_of_strength_detection_failed", error=str(e))
            raise

    async def _detect_sign_of_weakness(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect sign of weakness - strong downward move on good volume.

        Args:
            data: DataFrame with metrics

        Returns:
            DataFrame with SOW events marked
        """
        try:
            vol_mult = float(self.volume_threshold_multiplier)

            result = data.with_columns([
                pl.when(
                    (pl.col("wyckoff_event") == WyckoffEvent.NONE.value) &
                    (pl.col("close") < pl.col("open")) &
                    (pl.col("volume_ratio") > vol_mult) &
                    (pl.col("price_change") < -pl.col("price_range").rolling_mean(window_size=20)) &
                    (pl.col("close") < pl.col("sma_20"))
                )
                .then(pl.lit(WyckoffEvent.SIGN_OF_WEAKNESS.value))
                .otherwise(pl.col("wyckoff_event"))
                .alias("wyckoff_event")
            ])

            return result

        except Exception as e:
            logger.error("sign_of_weakness_detection_failed", error=str(e))
            raise

    async def _identify_phases(self, data: pl.DataFrame) -> pl.DataFrame:
        """Identify current Wyckoff phase.

        Args:
            data: DataFrame with events and metrics

        Returns:
            DataFrame with wyckoff_phase column
        """
        try:
            # Simplified phase identification based on price action and volume
            result = data.with_columns([
                pl.when(
                    (pl.col("close") < pl.col("sma_long")) &
                    (pl.col("effort_vs_result") == "high_effort_low_result") &
                    (pl.col("price_range") < pl.col("price_range").rolling_mean(window_size=50))
                )
                .then(pl.lit(WyckoffPhase.ACCUMULATION.value))
                .when(
                    (pl.col("close") > pl.col("sma_long")) &
                    (pl.col("effort_vs_result") == "high_effort_low_result") &
                    (pl.col("price_range") < pl.col("price_range").rolling_mean(window_size=50))
                )
                .then(pl.lit(WyckoffPhase.DISTRIBUTION.value))
                .when(
                    (pl.col("close") > pl.col("sma_long")) &
                    (pl.col("close") > pl.col("sma_20")) &
                    (pl.col("supply_demand_balance") == "demand")
                )
                .then(pl.lit(WyckoffPhase.MARKUP.value))
                .when(
                    (pl.col("close") < pl.col("sma_long")) &
                    (pl.col("close") < pl.col("sma_20")) &
                    (pl.col("supply_demand_balance") == "supply")
                )
                .then(pl.lit(WyckoffPhase.MARKDOWN.value))
                .otherwise(pl.lit(WyckoffPhase.UNKNOWN.value))
                .alias("wyckoff_phase")
            ])

            return result

        except Exception as e:
            logger.error("phase_identification_failed", error=str(e))
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

        if len(data) < self.lookback_period:
            raise ValueError(
                f"Data length ({len(data)}) must be >= lookback_period ({self.lookback_period})"
            )

        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Validate no null values
        for col in required_columns:
            null_count = data[col].null_count()
            if null_count > 0:
                raise ValueError(f"Column '{col}' contains {null_count} null values")
