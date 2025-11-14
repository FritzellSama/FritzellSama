"""
Market Sentiment Custom Indicator.

This module implements a composite sentiment indicator combining multiple
sentiment sources for institutional trading decision-making.
"""

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class SentimentError(Exception):
    """Base exception for Sentiment indicator errors."""
    pass


class SentimentValidator:
    """Validates Sentiment configuration and input data."""

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate Sentiment configuration.

        Args:
            config: Configuration dictionary

        Raises:
            SentimentError: If configuration is invalid
        """
        required_keys = ['weights', 'smoothing_period']
        for key in required_keys:
            if key not in config:
                raise SentimentError(f"Missing required config key: {key}")

        if not isinstance(config['weights'], dict):
            raise SentimentError("weights must be a dictionary")

        # Validate weights sum to 1
        weights = config['weights']
        total_weight = sum(Decimal(str(v)) for v in weights.values())
        if abs(total_weight - Decimal('1.0')) > Decimal('0.001'):
            raise SentimentError(f"Weights must sum to 1.0, got {total_weight}")

    @staticmethod
    def validate_dataframe(df: pl.DataFrame, required_columns: List[str]) -> None:
        """Validate input dataframe structure.

        Args:
            df: Input dataframe
            required_columns: List of required column names

        Raises:
            SentimentError: If dataframe is invalid
        """
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise SentimentError(f"Missing required columns: {missing}")

        if len(df) < 1:
            raise SentimentError("Dataframe must have at least 1 row")


class SentimentIndicator:
    """Composite market sentiment indicator.

    Combines multiple sentiment sources including:
    - Price momentum sentiment
    - Volume sentiment
    - Volatility sentiment
    - Funding rate sentiment (for crypto)
    - Open interest sentiment
    - Long/short ratio sentiment

    Attributes:
        config: Configuration dictionary
        weights: Weights for each sentiment component
        smoothing_period: Period for EMA smoothing

    Example:
        >>> config = {
        ...     'weights': {
        ...         'price': '0.3',
        ...         'volume': '0.2',
        ...         'volatility': '0.2',
        ...         'funding': '0.15',
        ...         'oi': '0.15'
        ...     },
        ...     'smoothing_period': 10,
        ...     'momentum_period': 14
        ... }
        >>> sentiment = SentimentIndicator(config)
        >>> result = await sentiment.calculate(df)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Sentiment indicator.

        Args:
            config: Configuration dictionary

        Raises:
            SentimentError: If configuration is invalid
        """
        SentimentValidator.validate_config(config)

        self.config = config
        self.weights = {k: Decimal(str(v)) for k, v in config['weights'].items()}
        self.smoothing_period = int(config['smoothing_period'])

        logger.info(
            "SentimentIndicator initialized",
            weights=str(self.weights),
            smoothing_period=self.smoothing_period
        )

    async def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate composite sentiment score.

        Args:
            df: Input dataframe with market data

        Returns:
            DataFrame with additional columns:
                - sentiment_composite: Weighted composite sentiment (-100 to 100)
                - sentiment_smoothed: EMA-smoothed sentiment
                - sentiment_signal: Trading signal (1, -1, or 0)

        Raises:
            SentimentError: If calculation fails

        Example:
            >>> df = pl.DataFrame({
            ...     'close': [100.0, 102.0, 101.0],
            ...     'volume': [1000000, 1200000, 950000]
            ... })
            >>> result = await sentiment.calculate(df)
        """
        try:
            logger.debug("Calculating sentiment indicator", rows=len(df))

            # Calculate individual sentiment components
            df_with_components = await self._calculate_components(df)

            # Calculate composite sentiment
            result = await self._calculate_composite(df_with_components)

            logger.info("Sentiment indicator calculated successfully", rows=len(result))

            return result

        except SentimentError:
            raise
        except Exception as e:
            logger.error("Sentiment calculation failed", error=str(e))
            raise SentimentError(f"Calculation failed: {e}")

    async def _calculate_components(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate individual sentiment components.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with individual sentiment scores
        """
        result = df

        # Price momentum sentiment
        if 'close' in df.columns:
            result = await self._calculate_price_sentiment(result)

        # Volume sentiment
        if 'volume' in df.columns:
            result = await self._calculate_volume_sentiment(result)

        # Volatility sentiment
        if 'high' in df.columns and 'low' in df.columns:
            result = await self._calculate_volatility_sentiment(result)

        # Funding rate sentiment (if available)
        if 'funding_rate' in df.columns:
            result = await self._calculate_funding_sentiment(result)

        # Open interest sentiment (if available)
        if 'open_interest' in df.columns:
            result = await self._calculate_oi_sentiment(result)

        # Long/short ratio sentiment (if available)
        if 'long_short_ratio' in df.columns:
            result = await self._calculate_ls_ratio_sentiment(result)

        return result

    async def _calculate_price_sentiment(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate price momentum sentiment.

        Uses rate of change to determine price sentiment.
        """
        period = int(self.config.get('momentum_period', 14))
        closes = [Decimal(str(x)) for x in df['close'].to_list()]

        price_sentiment = []

        for i in range(len(closes)):
            if i < period:
                price_sentiment.append(Decimal('0'))
            else:
                # Calculate ROC
                roc = ((closes[i] - closes[i - period]) / closes[i - period]) * Decimal('100')

                # Normalize to -100 to 100 scale (cap at ±50% change)
                normalized = max(Decimal('-100'), min(Decimal('100'), roc * Decimal('2')))
                price_sentiment.append(normalized)

        return df.with_columns([
            pl.Series('sentiment_price', [float(x) for x in price_sentiment])
        ])

    async def _calculate_volume_sentiment(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate volume sentiment.

        Compares current volume to average volume.
        """
        period = int(self.config.get('volume_period', 20))
        volumes = [Decimal(str(x)) for x in df['volume'].to_list()]

        volume_sentiment = []

        for i in range(len(volumes)):
            if i < period:
                volume_sentiment.append(Decimal('0'))
            else:
                # Calculate average volume
                avg_volume = sum(volumes[i - period:i]) / Decimal(str(period))

                if avg_volume > Decimal('0'):
                    # Calculate volume ratio
                    ratio = (volumes[i] - avg_volume) / avg_volume

                    # Normalize to -100 to 100 scale (cap at ±200% change)
                    normalized = max(Decimal('-100'), min(Decimal('100'), ratio * Decimal('50')))
                    volume_sentiment.append(normalized)
                else:
                    volume_sentiment.append(Decimal('0'))

        return df.with_columns([
            pl.Series('sentiment_volume', [float(x) for x in volume_sentiment])
        ])

    async def _calculate_volatility_sentiment(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate volatility sentiment.

        Lower volatility = positive sentiment, higher volatility = negative sentiment.
        """
        period = int(self.config.get('volatility_period', 14))
        highs = [Decimal(str(x)) for x in df['high'].to_list()]
        lows = [Decimal(str(x)) for x in df['low'].to_list()]
        closes = [Decimal(str(x)) for x in df['close'].to_list()]

        volatility_sentiment = []

        for i in range(len(closes)):
            if i < period:
                volatility_sentiment.append(Decimal('0'))
            else:
                # Calculate Average True Range
                tr_sum = Decimal('0')
                for j in range(i - period + 1, i + 1):
                    high_low = highs[j] - lows[j]
                    if j > 0:
                        high_close = abs(highs[j] - closes[j-1])
                        low_close = abs(lows[j] - closes[j-1])
                        tr = max(high_low, high_close, low_close)
                    else:
                        tr = high_low
                    tr_sum += tr

                atr = tr_sum / Decimal(str(period))

                # Normalize ATR as percentage of price
                if closes[i] > Decimal('0'):
                    atr_pct = (atr / closes[i]) * Decimal('100')

                    # Invert (lower volatility = positive sentiment)
                    # Normalize around typical 2% ATR
                    normalized = Decimal('50') - (atr_pct * Decimal('25'))
                    normalized = max(Decimal('-100'), min(Decimal('100'), normalized))
                    volatility_sentiment.append(normalized)
                else:
                    volatility_sentiment.append(Decimal('0'))

        return df.with_columns([
            pl.Series('sentiment_volatility', [float(x) for x in volatility_sentiment])
        ])

    async def _calculate_funding_sentiment(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate funding rate sentiment.

        High positive funding = overbought (negative sentiment)
        High negative funding = oversold (positive sentiment)
        """
        funding_rates = [Decimal(str(x)) for x in df['funding_rate'].to_list()]

        funding_sentiment = []

        for fr in funding_rates:
            # Typical funding rates are -0.1% to 0.1%
            # Normalize to -100 to 100 scale
            normalized = -fr * Decimal('1000')  # Invert because high funding = bearish
            normalized = max(Decimal('-100'), min(Decimal('100'), normalized))
            funding_sentiment.append(normalized)

        return df.with_columns([
            pl.Series('sentiment_funding', [float(x) for x in funding_sentiment])
        ])

    async def _calculate_oi_sentiment(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate open interest sentiment.

        Rising OI with rising price = bullish
        Rising OI with falling price = bearish
        """
        if 'close' not in df.columns:
            return df

        oi_values = [Decimal(str(x)) for x in df['open_interest'].to_list()]
        closes = [Decimal(str(x)) for x in df['close'].to_list()]

        oi_sentiment = []

        for i in range(len(oi_values)):
            if i < 1:
                oi_sentiment.append(Decimal('0'))
            else:
                oi_change = (oi_values[i] - oi_values[i-1]) / max(oi_values[i-1], Decimal('0.000001'))
                price_change = (closes[i] - closes[i-1]) / max(closes[i-1], Decimal('0.000001'))

                # If both positive or both negative = strong signal
                if oi_change > Decimal('0') and price_change > Decimal('0'):
                    sentiment = Decimal('50')  # Bullish
                elif oi_change > Decimal('0') and price_change < Decimal('0'):
                    sentiment = Decimal('-50')  # Bearish
                elif oi_change < Decimal('0') and price_change > Decimal('0'):
                    sentiment = Decimal('25')  # Weak bullish
                elif oi_change < Decimal('0') and price_change < Decimal('0'):
                    sentiment = Decimal('-25')  # Weak bearish
                else:
                    sentiment = Decimal('0')

                oi_sentiment.append(sentiment)

        return df.with_columns([
            pl.Series('sentiment_oi', [float(x) for x in oi_sentiment])
        ])

    async def _calculate_ls_ratio_sentiment(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate long/short ratio sentiment.

        Extreme ratios indicate contrarian opportunities.
        """
        ls_ratios = [Decimal(str(x)) for x in df['long_short_ratio'].to_list()]

        ls_sentiment = []

        for ratio in ls_ratios:
            # Ratio > 2 = too bullish (contrarian bearish)
            # Ratio < 0.5 = too bearish (contrarian bullish)
            if ratio > Decimal('2'):
                sentiment = -(ratio - Decimal('1')) * Decimal('50')
            elif ratio < Decimal('0.5'):
                sentiment = (Decimal('1') - ratio) * Decimal('100')
            else:
                sentiment = Decimal('0')

            normalized = max(Decimal('-100'), min(Decimal('100'), sentiment))
            ls_sentiment.append(normalized)

        return df.with_columns([
            pl.Series('sentiment_ls_ratio', [float(x) for x in ls_sentiment])
        ])

    async def _calculate_composite(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate weighted composite sentiment score.

        Args:
            df: DataFrame with individual sentiment components

        Returns:
            DataFrame with composite sentiment
        """
        composite_scores = []

        for i in range(len(df)):
            total_score = Decimal('0')
            total_weight = Decimal('0')

            # Sum weighted components
            for component, weight in self.weights.items():
                col_name = f'sentiment_{component}'
                if col_name in df.columns:
                    value = df[col_name][i]
                    if value is not None:
                        total_score += Decimal(str(value)) * weight
                        total_weight += weight

            # Normalize if not all components available
            if total_weight > Decimal('0'):
                composite = total_score / total_weight
            else:
                composite = Decimal('0')

            composite_scores.append(float(composite))

        result = df.with_columns([
            pl.Series('sentiment_composite', composite_scores)
        ])

        # Calculate smoothed sentiment
        result = await self._smooth_sentiment(result)

        # Generate signals
        result = self._generate_signals(result)

        return result

    async def _smooth_sentiment(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply EMA smoothing to composite sentiment.

        Args:
            df: DataFrame with composite sentiment

        Returns:
            DataFrame with smoothed sentiment
        """
        sentiment_values = df['sentiment_composite'].to_list()

        multiplier = Decimal('2') / Decimal(str(self.smoothing_period + 1))
        smoothed = []
        ema = None

        for val in sentiment_values:
            if val is None:
                smoothed.append(None)
            else:
                val_decimal = Decimal(str(val))
                if ema is None:
                    ema = val_decimal
                else:
                    ema = (val_decimal - ema) * multiplier + ema
                smoothed.append(float(ema))

        return df.with_columns([
            pl.Series('sentiment_smoothed', smoothed)
        ])

    def _generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate trading signals from sentiment.

        Args:
            df: DataFrame with sentiment scores

        Returns:
            DataFrame with signal column
        """
        sentiment_values = df['sentiment_smoothed'].to_list()

        threshold_bullish = Decimal(str(self.config.get('threshold_bullish', '30')))
        threshold_bearish = Decimal(str(self.config.get('threshold_bearish', '-30')))

        signals = []

        for val in sentiment_values:
            if val is None:
                signals.append(0)
            else:
                val_decimal = Decimal(str(val))
                if val_decimal > threshold_bullish:
                    signals.append(1)
                elif val_decimal < threshold_bearish:
                    signals.append(-1)
                else:
                    signals.append(0)

        return df.with_columns([
            pl.Series('sentiment_signal', signals)
        ])


async def calculate_sentiment(
    df: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate sentiment indicator.

    Args:
        df: Input dataframe with market data
        config: Configuration dictionary

    Returns:
        DataFrame with sentiment calculations

    Example:
        >>> config = {
        ...     'weights': {'price': '0.5', 'volume': '0.5'},
        ...     'smoothing_period': 10
        ... }
        >>> result = await calculate_sentiment(df, config)
    """
    sentiment = SentimentIndicator(config)
    return await sentiment.calculate(df)
