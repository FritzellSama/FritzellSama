"""Slippage Model for Order Execution Simulation.

Models realistic slippage in order execution based on market conditions,
order size, and volatility.
"""

import asyncio
import random
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class SlippageModel(Enum):
    """Slippage model types."""
    FIXED = "FIXED"
    PROPORTIONAL = "PROPORTIONAL"
    VOLUME_BASED = "VOLUME_BASED"
    VOLATILITY_ADJUSTED = "VOLATILITY_ADJUSTED"


@dataclass
class SlippageResult:
    """Result of slippage calculation.

    Attributes:
        original_price: Price before slippage
        slipped_price: Price after slippage
        slippage_amount: Absolute slippage
        slippage_bps: Slippage in basis points
        model_used: Slippage model used
    """
    original_price: Decimal
    slipped_price: Decimal
    slippage_amount: Decimal
    slippage_bps: Decimal
    model_used: SlippageModel


class SlippageSimulator:
    """Simulate realistic order execution slippage.

    Models slippage based on:
    - Order size relative to market
    - Market volatility
    - Spread conditions
    - Liquidity availability

    Attributes:
        config: Simulator configuration from config files
        model_type: Slippage model to use
        base_slippage_bps: Base slippage in basis points
        seed: Random seed for reproducibility
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize slippage simulator.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = {
            ...     "model_type": "VOLATILITY_ADJUSTED",
            ...     "base_slippage_bps": 5,
            ...     "volatility_multiplier": 2.0
            ... }
            >>> simulator = SlippageSimulator(config)
        """
        self.config = config
        self._validate_config()

        # Load model type
        model_str = config.get("model_type", "PROPORTIONAL").upper()
        self.model_type = SlippageModel[model_str]

        self.base_slippage_bps = Decimal(str(config.get("base_slippage_bps", 5)))
        self.volatility_multiplier = Decimal(str(config.get("volatility_multiplier", 2.0)))
        self.volume_impact_factor = Decimal(str(config.get("volume_impact_factor", 0.5)))

        # Random seed
        self.seed = config.get("random_seed")
        if self.seed is not None:
            random.seed(self.seed)

        # Statistics
        self.slippage_stats = {
            "total_orders": 0,
            "total_slippage_bps": Decimal("0"),
            "max_slippage_bps": Decimal("0"),
            "min_slippage_bps": Decimal("999999")
        }

        logger.info(
            "SlippageSimulator initialized",
            model_type=self.model_type.value,
            base_slippage_bps=float(self.base_slippage_bps)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("config must be a dictionary")

        # Validate model type if provided
        if "model_type" in self.config:
            model_str = self.config["model_type"].upper()
            if model_str not in [m.value for m in SlippageModel]:
                error_msg = f"Invalid model_type: {model_str}"
                logger.error("Invalid slippage model", model=model_str)
                raise ValueError(error_msg)

    async def calculate_slippage(
        self,
        order_price: Decimal,
        order_quantity: Decimal,
        is_buy: bool,
        market_volatility: Optional[Decimal] = None,
        market_volume: Optional[Decimal] = None,
        spread_bps: Optional[Decimal] = None
    ) -> SlippageResult:
        """Calculate slippage for an order.

        Args:
            order_price: Original order price
            order_quantity: Order size
            is_buy: True for buy order, False for sell order
            market_volatility: Current market volatility (optional)
            market_volume: Recent market volume (optional)
            spread_bps: Current spread in basis points (optional)

        Returns:
            Slippage calculation result

        Raises:
            ValueError: If inputs are invalid

        Example:
            >>> simulator = SlippageSimulator(config)
            >>> result = await simulator.calculate_slippage(
            ...     Decimal("50000"),
            ...     Decimal("10"),
            ...     is_buy=True,
            ...     market_volatility=Decimal("0.02")
            ... )
            >>> print(f"Slippage: {result.slippage_bps} bps")
        """
        if order_price <= Decimal("0"):
            error_msg = f"Invalid order_price: {order_price}"
            logger.error("Invalid order price", price=str(order_price))
            raise ValueError(error_msg)

        if order_quantity <= Decimal("0"):
            error_msg = f"Invalid order_quantity: {order_quantity}"
            logger.error("Invalid order quantity", quantity=str(order_quantity))
            raise ValueError(error_msg)

        try:
            self.slippage_stats["total_orders"] += 1

            # Calculate slippage based on model
            if self.model_type == SlippageModel.FIXED:
                slippage_bps = await self._calculate_fixed_slippage()

            elif self.model_type == SlippageModel.PROPORTIONAL:
                slippage_bps = await self._calculate_proportional_slippage(
                    order_quantity,
                    spread_bps
                )

            elif self.model_type == SlippageModel.VOLUME_BASED:
                slippage_bps = await self._calculate_volume_based_slippage(
                    order_quantity,
                    market_volume
                )

            elif self.model_type == SlippageModel.VOLATILITY_ADJUSTED:
                slippage_bps = await self._calculate_volatility_adjusted_slippage(
                    market_volatility,
                    order_quantity,
                    market_volume
                )

            else:
                slippage_bps = self.base_slippage_bps

            # Add random component for realism
            slippage_bps = await self._add_randomness(slippage_bps)

            # Calculate slippage amount
            slippage_amount = order_price * slippage_bps / Decimal("10000")

            # Apply direction
            if is_buy:
                slipped_price = order_price + slippage_amount
            else:
                slipped_price = order_price - slippage_amount

            # Update statistics
            self._update_statistics(slippage_bps)

            result = SlippageResult(
                original_price=order_price,
                slipped_price=slipped_price,
                slippage_amount=slippage_amount,
                slippage_bps=slippage_bps,
                model_used=self.model_type
            )

            logger.debug(
                "Slippage calculated",
                slippage_bps=float(slippage_bps),
                model=self.model_type.value
            )

            return result

        except Exception as e:
            logger.error("Failed to calculate slippage", error=str(e))
            raise

    async def _calculate_fixed_slippage(self) -> Decimal:
        """Calculate fixed slippage.

        Returns:
            Slippage in basis points
        """
        return self.base_slippage_bps

    async def _calculate_proportional_slippage(
        self,
        order_quantity: Decimal,
        spread_bps: Optional[Decimal]
    ) -> Decimal:
        """Calculate proportional slippage based on spread.

        Args:
            order_quantity: Order size
            spread_bps: Current spread

        Returns:
            Slippage in basis points
        """
        # Start with base slippage
        slippage = self.base_slippage_bps

        # Add spread component if provided
        if spread_bps is not None:
            spread_component = spread_bps * Decimal("0.5")  # Half the spread
            slippage += spread_component

        return slippage

    async def _calculate_volume_based_slippage(
        self,
        order_quantity: Decimal,
        market_volume: Optional[Decimal]
    ) -> Decimal:
        """Calculate slippage based on order size vs market volume.

        Args:
            order_quantity: Order size
            market_volume: Recent market volume

        Returns:
            Slippage in basis points
        """
        slippage = self.base_slippage_bps

        if market_volume is not None and market_volume > Decimal("0"):
            # Calculate participation rate
            participation = order_quantity / market_volume

            # Slippage increases with participation
            volume_impact = participation * self.volume_impact_factor * Decimal("100")
            slippage += volume_impact

        return slippage

    async def _calculate_volatility_adjusted_slippage(
        self,
        market_volatility: Optional[Decimal],
        order_quantity: Decimal,
        market_volume: Optional[Decimal]
    ) -> Decimal:
        """Calculate volatility-adjusted slippage.

        Args:
            market_volatility: Current volatility
            order_quantity: Order size
            market_volume: Recent volume

        Returns:
            Slippage in basis points
        """
        slippage = self.base_slippage_bps

        # Volatility adjustment
        if market_volatility is not None:
            volatility_component = market_volatility * self.volatility_multiplier * Decimal("100")
            slippage += volatility_component

        # Volume component
        if market_volume is not None and market_volume > Decimal("0"):
            participation = order_quantity / market_volume
            volume_component = participation * self.volume_impact_factor * Decimal("50")
            slippage += volume_component

        return slippage

    async def _add_randomness(self, base_slippage: Decimal) -> Decimal:
        """Add random component to slippage for realism.

        Args:
            base_slippage: Base slippage value

        Returns:
            Slippage with randomness
        """
        randomness_pct = self.config.get("randomness_pct", 20)
        variance = float(base_slippage) * (randomness_pct / 100)

        random_component = Decimal(str(random.uniform(-variance, variance)))
        final_slippage = base_slippage + random_component

        # Ensure non-negative
        return max(Decimal("0"), final_slippage)

    def _update_statistics(self, slippage_bps: Decimal) -> None:
        """Update slippage statistics.

        Args:
            slippage_bps: Slippage value
        """
        self.slippage_stats["total_slippage_bps"] += slippage_bps
        self.slippage_stats["max_slippage_bps"] = max(
            self.slippage_stats["max_slippage_bps"],
            slippage_bps
        )
        self.slippage_stats["min_slippage_bps"] = min(
            self.slippage_stats["min_slippage_bps"],
            slippage_bps
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get slippage statistics.

        Returns:
            Dictionary with slippage stats

        Example:
            >>> simulator = SlippageSimulator(config)
            >>> # ... simulate orders ...
            >>> stats = simulator.get_statistics()
            >>> print(f"Avg slippage: {stats['avg_slippage_bps']} bps")
        """
        total_orders = self.slippage_stats["total_orders"]

        if total_orders > 0:
            avg_slippage = self.slippage_stats["total_slippage_bps"] / Decimal(str(total_orders))
        else:
            avg_slippage = Decimal("0")

        return {
            "total_orders": total_orders,
            "avg_slippage_bps": float(avg_slippage),
            "max_slippage_bps": float(self.slippage_stats["max_slippage_bps"]),
            "min_slippage_bps": float(self.slippage_stats["min_slippage_bps"]),
            "model_type": self.model_type.value
        }

    def reset_statistics(self) -> None:
        """Reset slippage statistics.

        Example:
            >>> simulator = SlippageSimulator(config)
            >>> simulator.reset_statistics()
        """
        self.slippage_stats = {
            "total_orders": 0,
            "total_slippage_bps": Decimal("0"),
            "max_slippage_bps": Decimal("0"),
            "min_slippage_bps": Decimal("999999")
        }

        logger.info("Slippage statistics reset")

    async def estimate_execution_price(
        self,
        order_price: Decimal,
        order_quantity: Decimal,
        is_buy: bool,
        **kwargs
    ) -> Decimal:
        """Estimate execution price including slippage.

        Args:
            order_price: Original order price
            order_quantity: Order size
            is_buy: Order direction
            **kwargs: Additional market parameters

        Returns:
            Estimated execution price

        Example:
            >>> simulator = SlippageSimulator(config)
            >>> exec_price = await simulator.estimate_execution_price(
            ...     Decimal("50000"),
            ...     Decimal("10"),
            ...     is_buy=True,
            ...     market_volatility=Decimal("0.02")
            ... )
        """
        result = await self.calculate_slippage(
            order_price,
            order_quantity,
            is_buy,
            market_volatility=kwargs.get("market_volatility"),
            market_volume=kwargs.get("market_volume"),
            spread_bps=kwargs.get("spread_bps")
        )

        return result.slipped_price
