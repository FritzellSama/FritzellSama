"""High-performance volume data processor using Polars for institutional trading.

Processes millions of trades to calculate volume metrics, VWAP, and volume profiles
with checkpoint/recovery for fault tolerance.
"""

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class VolumeMetrics:
    """Calculated volume metrics for a symbol."""
    symbol: str
    exchange: str
    period_start: datetime
    period_end: datetime
    total_volume: Decimal
    buy_volume: Decimal
    sell_volume: Decimal
    vwap: Decimal
    num_trades: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class VolumeProcessor:
    """Stream-based volume processor with batch optimization and parallel processing.

    Attributes:
        config: Configuration dictionary
        checkpoint_manager: Manages processing checkpoints
        _running: Processor status flag
        _batch_buffer: Buffered trades for batch processing
        _checkpoint_interval: Checkpointing frequency
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize volume processor with configuration.

        Args:
            config: Must contain:
                - batch_size: Number of trades per batch
                - checkpoint_interval_seconds: Checkpoint frequency
                - aggregation_windows: List of time windows (e.g., ['1m', '5m', '1h'])
                - parallel_workers: Number of parallel processing workers
                - memory_limit_mb: Maximum memory usage
                - db_url: PostgreSQL connection for checkpoints

        Raises:
            ValueError: If required config parameters missing
        """
        self.config = config
        self._validate_config()

        self._running = False
        self._batch_buffer: List[pl.DataFrame] = []
        self._checkpoint_state: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self.db_pool: Optional[Any] = None

        self.batch_size = config['batch_size']
        self.checkpoint_interval = config['checkpoint_interval_seconds']
        self.aggregation_windows = config['aggregation_windows']
        self.parallel_workers = config['parallel_workers']
        self.memory_limit = config['memory_limit_mb'] * 1024 * 1024

        logger.info(
            "VolumeProcessor initialized",
            batch_size=self.batch_size,
            windows=self.aggregation_windows,
            workers=self.parallel_workers
        )

    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required = [
            'batch_size', 'checkpoint_interval_seconds', 'aggregation_windows',
            'parallel_workers', 'memory_limit_mb', 'db_url'
        ]
        missing = [key for key in required if key not in self.config]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")

        if not isinstance(self.config['aggregation_windows'], list):
            raise ValueError("'aggregation_windows' must be a list")

    async def connect(self) -> None:
        """Initialize database connection for checkpoint management."""
        import asyncpg

        try:
            self.db_pool = await asyncpg.create_pool(
                self.config['db_url'],
                min_size=self.config.get('db_pool_min', 2),
                max_size=self.config.get('db_pool_max', 5),
                command_timeout=self.config.get('db_timeout', 30)
            )
            logger.info("VolumeProcessor connected to PostgreSQL")

            # Load last checkpoint
            await self._load_checkpoint()
            self._running = True

        except Exception as e:
            logger.error("Failed to connect VolumeProcessor", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Gracefully shutdown processor and save checkpoint."""
        self._running = False

        # Process remaining buffered data
        if self._batch_buffer:
            await self._process_batch()

        # Save final checkpoint
        await self._save_checkpoint()

        if self.db_pool:
            await self.db_pool.close()
            logger.info("VolumeProcessor disconnected")

    async def process_trades(
        self,
        trades_df: pl.DataFrame,
        output_callback: Optional[Any] = None
    ) -> List[VolumeMetrics]:
        """Process trades to calculate volume metrics.

        Args:
            trades_df: Polars DataFrame with columns:
                - symbol, price, quantity, side, exchange, trade_id, timestamp
            output_callback: Optional async callback(metrics) for results

        Returns:
            List of calculated VolumeMetrics

        Raises:
            ValueError: If required columns missing
        """
        if not self._running:
            raise RuntimeError("VolumeProcessor not connected")

        self._validate_trades_dataframe(trades_df)

        try:
            # Add to batch buffer
            async with self._lock:
                self._batch_buffer.append(trades_df)
                total_rows = sum(len(df) for df in self._batch_buffer)

                # Process batch if size threshold reached
                if total_rows >= self.batch_size:
                    metrics = await self._process_batch()

                    if output_callback and metrics:
                        await output_callback(metrics)

                    return metrics

            return []

        except Exception as e:
            logger.error("Trade processing failed", error=str(e))
            raise

    def _validate_trades_dataframe(self, df: pl.DataFrame) -> None:
        """Validate trades DataFrame has required columns.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If required columns missing
        """
        required_columns = {'symbol', 'price', 'quantity', 'side', 'exchange', 'timestamp'}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    async def _process_batch(self) -> List[VolumeMetrics]:
        """Process buffered trades in parallel batches.

        Returns:
            List of calculated volume metrics
        """
        async with self._lock:
            if not self._batch_buffer:
                return []

            # Combine all buffered DataFrames
            combined_df = pl.concat(self._batch_buffer)
            self._batch_buffer.clear()

        logger.info("Processing batch", rows=len(combined_df))

        # Convert price and quantity to Decimal for accurate calculations
        combined_df = combined_df.with_columns([
            pl.col('price').cast(pl.Utf8).alias('price_str'),
            pl.col('quantity').cast(pl.Utf8).alias('quantity_str')
        ])

        # Process each aggregation window in parallel
        tasks = [
            self._calculate_window_metrics(combined_df, window)
            for window in self.aggregation_windows
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and filter out errors
        all_metrics = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("Window calculation failed", error=str(result))
            elif result:
                all_metrics.extend(result)

        logger.info("Batch processed", metrics_calculated=len(all_metrics))
        return all_metrics

    async def _calculate_window_metrics(
        self,
        trades_df: pl.DataFrame,
        window: str
    ) -> List[VolumeMetrics]:
        """Calculate volume metrics for a specific time window.

        Args:
            trades_df: Trades DataFrame
            window: Time window (e.g., '1m', '5m', '1h')

        Returns:
            List of VolumeMetrics for each symbol/exchange/period
        """
        try:
            # Parse window duration
            window_duration = self._parse_window_duration(window)

            # Group by symbol, exchange, and time window
            grouped = trades_df.group_by_dynamic(
                'timestamp',
                every=window_duration,
                by=['symbol', 'exchange']
            ).agg([
                pl.count().alias('num_trades'),
                pl.col('quantity_str').alias('quantities'),
                pl.col('price_str').alias('prices'),
                pl.col('side').alias('sides'),
                pl.min('timestamp').alias('period_start'),
                pl.max('timestamp').alias('period_end')
            ])

            metrics = []

            # Calculate metrics for each group
            for row in grouped.iter_rows(named=True):
                try:
                    quantities = [Decimal(q) for q in row['quantities']]
                    prices = [Decimal(p) for p in row['prices']]
                    sides = row['sides']

                    # Total volume
                    total_volume = sum(quantities)

                    # Buy/Sell volume
                    buy_volume = sum(
                        q for q, s in zip(quantities, sides) if s == 'BUY'
                    )
                    sell_volume = sum(
                        q for q, s in zip(quantities, sides) if s == 'SELL'
                    )

                    # VWAP calculation
                    vwap = self._calculate_vwap(prices, quantities)

                    metrics.append(VolumeMetrics(
                        symbol=row['symbol'],
                        exchange=row['exchange'],
                        period_start=row['period_start'],
                        period_end=row['period_end'],
                        total_volume=total_volume,
                        buy_volume=buy_volume,
                        sell_volume=sell_volume,
                        vwap=vwap,
                        num_trades=row['num_trades'],
                        metadata={'window': window}
                    ))

                except Exception as e:
                    logger.error(
                        "Metric calculation failed",
                        symbol=row.get('symbol'),
                        error=str(e)
                    )
                    continue

            return metrics

        except Exception as e:
            logger.error("Window metrics calculation failed", window=window, error=str(e))
            raise

    def _parse_window_duration(self, window: str) -> str:
        """Parse window string to Polars duration format.

        Args:
            window: Window string (e.g., '1m', '5m', '1h', '1d')

        Returns:
            Polars duration string
        """
        mapping = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
            '1w': '1w'
        }

        if window not in mapping:
            raise ValueError(f"Unsupported window: {window}")

        return mapping[window]

    def _calculate_vwap(
        self,
        prices: List[Decimal],
        quantities: List[Decimal]
    ) -> Decimal:
        """Calculate Volume Weighted Average Price.

        Args:
            prices: List of trade prices
            quantities: List of trade quantities

        Returns:
            VWAP as Decimal
        """
        if not prices or not quantities:
            return Decimal('0')

        total_value = sum(p * q for p, q in zip(prices, quantities))
        total_volume = sum(quantities)

        if total_volume == Decimal('0'):
            return Decimal('0')

        vwap = total_value / total_volume
        return vwap.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)

    async def calculate_volume_profile(
        self,
        symbol: str,
        exchange: str,
        start_time: datetime,
        end_time: datetime,
        price_buckets: int = 100
    ) -> pl.DataFrame:
        """Calculate volume profile (volume at price levels).

        Args:
            symbol: Trading pair
            exchange: Exchange name
            start_time: Start of time range
            end_time: End of time range
            price_buckets: Number of price levels to bucket

        Returns:
            Polars DataFrame with columns: price_level, volume, num_trades
        """
        if not self._running:
            raise RuntimeError("VolumeProcessor not connected")

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT price, quantity
                    FROM trades
                    WHERE symbol = $1
                      AND exchange = $2
                      AND timestamp >= $3
                      AND timestamp <= $4
                    ORDER BY price
                    """,
                    symbol, exchange, start_time, end_time
                )

            if not rows:
                return pl.DataFrame({
                    'price_level': [],
                    'volume': [],
                    'num_trades': []
                })

            # Convert to Polars DataFrame
            df = pl.DataFrame({
                'price': [Decimal(r['price']) for r in rows],
                'quantity': [Decimal(r['quantity']) for r in rows]
            })

            # Calculate price range and bucket size
            min_price = df['price'].min()
            max_price = df['price'].max()
            bucket_size = (max_price - min_price) / Decimal(price_buckets)

            # Bucket prices and aggregate volume
            df = df.with_columns([
                ((pl.col('price') - min_price) / bucket_size)
                .cast(pl.Int64)
                .alias('bucket')
            ])

            profile = df.group_by('bucket').agg([
                pl.col('quantity').sum().alias('volume'),
                pl.count().alias('num_trades'),
                pl.col('price').mean().alias('price_level')
            ]).sort('bucket')

            logger.info(
                "Volume profile calculated",
                symbol=symbol,
                buckets=len(profile),
                total_volume=str(profile['volume'].sum())
            )

            return profile.select(['price_level', 'volume', 'num_trades'])

        except Exception as e:
            logger.error("Volume profile calculation failed", error=str(e))
            raise

    async def _load_checkpoint(self) -> None:
        """Load last processing checkpoint from database."""
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT state, updated_at
                    FROM processing_checkpoints
                    WHERE processor_name = 'volume_processor'
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """
                )

                if row:
                    import json
                    self._checkpoint_state = json.loads(row['state'])
                    logger.info(
                        "Checkpoint loaded",
                        timestamp=row['updated_at'].isoformat()
                    )
                else:
                    self._checkpoint_state = {'last_processed_time': None}

        except Exception as e:
            logger.warning("Failed to load checkpoint", error=str(e))
            self._checkpoint_state = {'last_processed_time': None}

    async def _save_checkpoint(self) -> None:
        """Save current processing checkpoint to database."""
        try:
            import json

            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO processing_checkpoints (processor_name, state, updated_at)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (processor_name)
                    DO UPDATE SET state = $2, updated_at = $3
                    """,
                    'volume_processor',
                    json.dumps(self._checkpoint_state),
                    datetime.now(timezone.utc)
                )

                logger.debug("Checkpoint saved")

        except Exception as e:
            logger.error("Failed to save checkpoint", error=str(e))
