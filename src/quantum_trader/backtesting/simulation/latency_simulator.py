"""Latency Simulator for Backtesting.

Simulates realistic network and exchange latency for accurate backtesting.
"""

import asyncio
import random
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyProfile:
    """Latency profile for different operations.

    Attributes:
        min_latency_ms: Minimum latency in milliseconds
        avg_latency_ms: Average latency in milliseconds
        max_latency_ms: Maximum latency in milliseconds
        std_dev_ms: Standard deviation in milliseconds
        jitter_pct: Jitter percentage (0-100)
    """
    min_latency_ms: Decimal
    avg_latency_ms: Decimal
    max_latency_ms: Decimal
    std_dev_ms: Decimal
    jitter_pct: Decimal


class LatencySimulator:
    """Simulate realistic latency for trading operations.

    Simulates network latency, exchange processing delays, and market data delays
    for realistic backtesting.

    Attributes:
        config: Simulator configuration from config files
        profiles: Latency profiles for different operation types
        enable_jitter: Whether to apply random jitter
        seed: Random seed for reproducibility
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize latency simulator.

        Args:
            config: Configuration dictionary with latency profiles

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = {
            ...     "order_placement": {"min_ms": 5, "avg_ms": 15, "max_ms": 50, ...},
            ...     "order_cancel": {"min_ms": 3, "avg_ms": 10, "max_ms": 30, ...},
            ...     "market_data": {"min_ms": 1, "avg_ms": 5, "max_ms": 20, ...}
            ... }
            >>> simulator = LatencySimulator(config)
        """
        self.config = config
        self._validate_config()

        self.enable_jitter = config.get("enable_jitter", True)
        self.seed = config.get("random_seed")

        if self.seed is not None:
            random.seed(self.seed)

        # Load latency profiles
        self.profiles = self._load_latency_profiles()

        # Statistics tracking
        self.latency_stats: Dict[str, List[Decimal]] = {
            "order_placement": [],
            "order_cancel": [],
            "market_data": [],
            "order_fill": []
        }

        logger.info(
            "LatencySimulator initialized",
            profiles=list(self.profiles.keys()),
            jitter_enabled=self.enable_jitter
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config parameters missing or invalid
        """
        required_operation_types = [
            "order_placement",
            "order_cancel",
            "market_data",
            "order_fill"
        ]

        for op_type in required_operation_types:
            if op_type not in self.config:
                error_msg = f"Missing latency config for operation: {op_type}"
                logger.error("Config validation failed", operation=op_type)
                raise ValueError(error_msg)

            op_config = self.config[op_type]
            required_keys = ["min_ms", "avg_ms", "max_ms", "std_dev_ms", "jitter_pct"]
            missing_keys = [key for key in required_keys if key not in op_config]

            if missing_keys:
                error_msg = f"Missing keys in {op_type} config: {missing_keys}"
                logger.error("Config validation failed", operation=op_type, missing_keys=missing_keys)
                raise ValueError(error_msg)

    def _load_latency_profiles(self) -> Dict[str, LatencyProfile]:
        """Load latency profiles from configuration.

        Returns:
            Dictionary mapping operation types to latency profiles
        """
        profiles = {}

        operation_types = ["order_placement", "order_cancel", "market_data", "order_fill"]

        for op_type in operation_types:
            op_config = self.config[op_type]

            profiles[op_type] = LatencyProfile(
                min_latency_ms=Decimal(str(op_config["min_ms"])),
                avg_latency_ms=Decimal(str(op_config["avg_ms"])),
                max_latency_ms=Decimal(str(op_config["max_ms"])),
                std_dev_ms=Decimal(str(op_config["std_dev_ms"])),
                jitter_pct=Decimal(str(op_config["jitter_pct"]))
            )

        return profiles

    async def simulate_order_placement_latency(self) -> Decimal:
        """Simulate latency for order placement.

        Returns:
            Latency in milliseconds

        Example:
            >>> simulator = LatencySimulator(config)
            >>> latency_ms = await simulator.simulate_order_placement_latency()
            >>> print(f"Order placement latency: {latency_ms}ms")
        """
        latency = await self._calculate_latency("order_placement")
        self.latency_stats["order_placement"].append(latency)

        logger.debug("Simulated order placement latency", latency_ms=float(latency))
        return latency

    async def simulate_order_cancel_latency(self) -> Decimal:
        """Simulate latency for order cancellation.

        Returns:
            Latency in milliseconds
        """
        latency = await self._calculate_latency("order_cancel")
        self.latency_stats["order_cancel"].append(latency)

        logger.debug("Simulated order cancel latency", latency_ms=float(latency))
        return latency

    async def simulate_market_data_latency(self) -> Decimal:
        """Simulate latency for market data updates.

        Returns:
            Latency in milliseconds
        """
        latency = await self._calculate_latency("market_data")
        self.latency_stats["market_data"].append(latency)

        logger.debug("Simulated market data latency", latency_ms=float(latency))
        return latency

    async def simulate_order_fill_latency(self) -> Decimal:
        """Simulate latency for order fill confirmation.

        Returns:
            Latency in milliseconds
        """
        latency = await self._calculate_latency("order_fill")
        self.latency_stats["order_fill"].append(latency)

        logger.debug("Simulated order fill latency", latency_ms=float(latency))
        return latency

    async def _calculate_latency(self, operation_type: str) -> Decimal:
        """Calculate latency for a given operation type.

        Args:
            operation_type: Type of operation (order_placement, order_cancel, etc.)

        Returns:
            Latency in milliseconds

        Raises:
            ValueError: If operation type is unknown
        """
        if operation_type not in self.profiles:
            error_msg = f"Unknown operation type: {operation_type}"
            logger.error("Invalid operation type", operation_type=operation_type)
            raise ValueError(error_msg)

        profile = self.profiles[operation_type]

        # Generate base latency using normal distribution
        base_latency = self._generate_normal_latency(
            float(profile.avg_latency_ms),
            float(profile.std_dev_ms)
        )

        # Clamp to min/max bounds
        base_latency = max(float(profile.min_latency_ms), base_latency)
        base_latency = min(float(profile.max_latency_ms), base_latency)

        # Apply jitter if enabled
        if self.enable_jitter:
            jitter_factor = Decimal("1.0") + (
                Decimal(str(random.uniform(-1, 1))) *
                profile.jitter_pct / Decimal("100")
            )
            base_latency = float(Decimal(str(base_latency)) * jitter_factor)

        # Ensure non-negative
        final_latency = max(Decimal("0"), Decimal(str(base_latency)))

        return final_latency

    def _generate_normal_latency(self, mean: float, std_dev: float) -> float:
        """Generate latency value from normal distribution.

        Args:
            mean: Mean latency
            std_dev: Standard deviation

        Returns:
            Generated latency value
        """
        return random.gauss(mean, std_dev)

    async def apply_latency_delay(self, latency_ms: Decimal) -> None:
        """Apply actual delay for given latency.

        Args:
            latency_ms: Latency to apply in milliseconds

        Example:
            >>> simulator = LatencySimulator(config)
            >>> latency = await simulator.simulate_order_placement_latency()
            >>> await simulator.apply_latency_delay(latency)
        """
        if latency_ms > Decimal("0"):
            delay_seconds = float(latency_ms) / 1000.0
            await asyncio.sleep(delay_seconds)
            logger.debug("Applied latency delay", latency_ms=float(latency_ms))

    async def get_timestamp_with_latency(
        self,
        base_timestamp: datetime,
        operation_type: str
    ) -> datetime:
        """Get timestamp adjusted for latency.

        Args:
            base_timestamp: Base timestamp
            operation_type: Type of operation

        Returns:
            Timestamp adjusted for simulated latency

        Example:
            >>> simulator = LatencySimulator(config)
            >>> now = datetime.utcnow()
            >>> adjusted_time = await simulator.get_timestamp_with_latency(now, "order_placement")
        """
        latency_ms = await self._calculate_latency(operation_type)
        latency_delta = timedelta(milliseconds=float(latency_ms))

        adjusted_timestamp = base_timestamp + latency_delta

        logger.debug(
            "Timestamp adjusted for latency",
            base_timestamp=base_timestamp.isoformat(),
            latency_ms=float(latency_ms),
            adjusted_timestamp=adjusted_timestamp.isoformat()
        )

        return adjusted_timestamp

    def get_latency_statistics(self) -> Dict[str, Dict[str, Decimal]]:
        """Get latency statistics for all operation types.

        Returns:
            Dictionary with latency statistics (min, max, avg, p50, p95, p99)

        Example:
            >>> simulator = LatencySimulator(config)
            >>> # ... perform operations ...
            >>> stats = simulator.get_latency_statistics()
            >>> print(f"Average order placement latency: {stats['order_placement']['avg']}ms")
        """
        statistics = {}

        for op_type, latencies in self.latency_stats.items():
            if not latencies:
                statistics[op_type] = {
                    "count": 0,
                    "min": Decimal("0"),
                    "max": Decimal("0"),
                    "avg": Decimal("0"),
                    "p50": Decimal("0"),
                    "p95": Decimal("0"),
                    "p99": Decimal("0")
                }
                continue

            sorted_latencies = sorted(latencies)
            count = len(sorted_latencies)

            statistics[op_type] = {
                "count": count,
                "min": sorted_latencies[0],
                "max": sorted_latencies[-1],
                "avg": sum(sorted_latencies) / Decimal(str(count)),
                "p50": sorted_latencies[int(count * 0.50)],
                "p95": sorted_latencies[int(count * 0.95)] if count > 1 else sorted_latencies[0],
                "p99": sorted_latencies[int(count * 0.99)] if count > 1 else sorted_latencies[0]
            }

        logger.info("Latency statistics calculated", operation_types=list(statistics.keys()))
        return statistics

    def reset_statistics(self) -> None:
        """Reset latency statistics.

        Example:
            >>> simulator = LatencySimulator(config)
            >>> simulator.reset_statistics()
        """
        for op_type in self.latency_stats:
            self.latency_stats[op_type] = []

        logger.info("Latency statistics reset")
