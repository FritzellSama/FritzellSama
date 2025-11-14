"""Market Impact Model for Order Simulation.

Models the price impact of orders on the market for realistic backtesting.
"""

import asyncio
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class ImpactModel(Enum):
    """Market impact model types."""
    LINEAR = "LINEAR"
    SQUARE_ROOT = "SQUARE_ROOT"
    POWER_LAW = "POWER_LAW"
    ALMGREN_CHRISS = "ALMGREN_CHRISS"


@dataclass
class ImpactParameters:
    """Parameters for market impact calculation.

    Attributes:
        model_type: Type of impact model to use
        temporary_impact_factor: Factor for temporary impact
        permanent_impact_factor: Factor for permanent impact
        liquidity_factor: Market liquidity factor
        volatility_factor: Price volatility factor
    """
    model_type: ImpactModel
    temporary_impact_factor: Decimal
    permanent_impact_factor: Decimal
    liquidity_factor: Decimal
    volatility_factor: Decimal


@dataclass
class ImpactResult:
    """Result of market impact calculation.

    Attributes:
        original_price: Price before impact
        impacted_price: Price after impact
        temporary_impact: Temporary price impact
        permanent_impact: Permanent price impact
        total_impact: Total price impact
        impact_bps: Impact in basis points
    """
    original_price: Decimal
    impacted_price: Decimal
    temporary_impact: Decimal
    permanent_impact: Decimal
    total_impact: Decimal

    @property
    def impact_bps(self) -> Decimal:
        """Calculate impact in basis points."""
        if self.original_price > Decimal("0"):
            return (self.total_impact / self.original_price) * Decimal("10000")
        return Decimal("0")


class MarketImpactModel:
    """Model market impact of orders for backtesting.

    Simulates realistic price impact based on order size, market liquidity,
    and volatility using various impact models.

    Attributes:
        config: Model configuration from config files
        symbol: Trading symbol
        impact_params: Impact calculation parameters
    """

    def __init__(self, config: Dict[str, Any], symbol: str) -> None:
        """Initialize market impact model.

        Args:
            config: Configuration dictionary with impact parameters
            symbol: Trading symbol

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = {
            ...     "model_type": "SQUARE_ROOT",
            ...     "temporary_impact_factor": 0.05,
            ...     "permanent_impact_factor": 0.02,
            ...     "liquidity_factor": {"BTC/USDT": 1000000},
            ...     "volatility_factor": 1.0
            ... }
            >>> model = MarketImpactModel(config, "BTC/USDT")
        """
        self.config = config
        self.symbol = symbol
        self._validate_config()

        # Load impact parameters
        self.impact_params = self._load_impact_parameters()

        # Track cumulative permanent impact
        self.cumulative_permanent_impact = Decimal("0")

        logger.info(
            "MarketImpactModel initialized",
            symbol=symbol,
            model_type=self.impact_params.model_type.value
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config parameters missing or invalid
        """
        required_keys = [
            "model_type",
            "temporary_impact_factor",
            "permanent_impact_factor",
            "liquidity_factor"
        ]

        missing_keys = [key for key in required_keys if key not in self.config]

        if missing_keys:
            error_msg = f"Missing required config keys: {missing_keys}"
            logger.error("Config validation failed", missing_keys=missing_keys)
            raise ValueError(error_msg)

        # Validate model type
        model_type_str = self.config["model_type"].upper()
        if model_type_str not in [m.value for m in ImpactModel]:
            error_msg = f"Invalid model_type: {model_type_str}"
            logger.error("Invalid model type", model_type=model_type_str)
            raise ValueError(error_msg)

    def _load_impact_parameters(self) -> ImpactParameters:
        """Load impact parameters from configuration.

        Returns:
            Impact parameters
        """
        model_type = ImpactModel[self.config["model_type"].upper()]

        # Get symbol-specific liquidity or default
        liquidity_config = self.config["liquidity_factor"]
        if isinstance(liquidity_config, dict):
            liquidity_factor = Decimal(str(liquidity_config.get(
                self.symbol,
                liquidity_config.get("default", 100000)
            )))
        else:
            liquidity_factor = Decimal(str(liquidity_config))

        return ImpactParameters(
            model_type=model_type,
            temporary_impact_factor=Decimal(str(self.config["temporary_impact_factor"])),
            permanent_impact_factor=Decimal(str(self.config["permanent_impact_factor"])),
            liquidity_factor=liquidity_factor,
            volatility_factor=Decimal(str(self.config.get("volatility_factor", 1.0)))
        )

    async def calculate_impact(
        self,
        order_quantity: Decimal,
        order_price: Decimal,
        is_buy: bool,
        market_volume: Optional[Decimal] = None,
        volatility: Optional[Decimal] = None
    ) -> ImpactResult:
        """Calculate market impact of an order.

        Args:
            order_quantity: Size of the order
            order_price: Current market price
            is_buy: True for buy order, False for sell order
            market_volume: Recent market volume (optional)
            volatility: Current volatility (optional)

        Returns:
            Impact calculation result

        Raises:
            ValueError: If inputs are invalid

        Example:
            >>> model = MarketImpactModel(config, "BTC/USDT")
            >>> result = await model.calculate_impact(
            ...     Decimal("10.0"),
            ...     Decimal("50000.0"),
            ...     is_buy=True,
            ...     market_volume=Decimal("1000000")
            ... )
            >>> print(f"Impact: {result.impact_bps:.2f} bps")
        """
        if order_quantity <= Decimal("0"):
            error_msg = f"Invalid order_quantity: {order_quantity}"
            logger.error("Invalid order quantity", order_quantity=str(order_quantity))
            raise ValueError(error_msg)

        if order_price <= Decimal("0"):
            error_msg = f"Invalid order_price: {order_price}"
            logger.error("Invalid order price", order_price=str(order_price))
            raise ValueError(error_msg)

        try:
            # Calculate temporary impact
            temporary_impact = await self._calculate_temporary_impact(
                order_quantity,
                order_price,
                market_volume,
                volatility
            )

            # Calculate permanent impact
            permanent_impact = await self._calculate_permanent_impact(
                order_quantity,
                order_price,
                market_volume
            )

            # Total impact depends on order direction
            total_impact = temporary_impact + permanent_impact

            if is_buy:
                impacted_price = order_price + total_impact
            else:
                impacted_price = order_price - total_impact

            # Update cumulative permanent impact
            self.cumulative_permanent_impact += permanent_impact

            result = ImpactResult(
                original_price=order_price,
                impacted_price=impacted_price,
                temporary_impact=temporary_impact,
                permanent_impact=permanent_impact,
                total_impact=total_impact
            )

            logger.debug(
                "Market impact calculated",
                symbol=self.symbol,
                quantity=str(order_quantity),
                is_buy=is_buy,
                impact_bps=float(result.impact_bps)
            )

            return result

        except Exception as e:
            logger.error("Failed to calculate market impact", error=str(e))
            raise

    async def _calculate_temporary_impact(
        self,
        order_quantity: Decimal,
        order_price: Decimal,
        market_volume: Optional[Decimal],
        volatility: Optional[Decimal]
    ) -> Decimal:
        """Calculate temporary price impact.

        Args:
            order_quantity: Size of the order
            order_price: Current market price
            market_volume: Recent market volume
            volatility: Current volatility

        Returns:
            Temporary impact amount
        """
        # Effective volatility factor
        vol_factor = volatility if volatility is not None else self.impact_params.volatility_factor

        # Calculate participation rate (order size / market volume)
        if market_volume is not None and market_volume > Decimal("0"):
            participation_rate = order_quantity / market_volume
        else:
            # Use liquidity factor as proxy for market volume
            participation_rate = order_quantity / self.impact_params.liquidity_factor

        # Apply impact model
        if self.impact_params.model_type == ImpactModel.LINEAR:
            impact_ratio = participation_rate

        elif self.impact_params.model_type == ImpactModel.SQUARE_ROOT:
            # Square root model (commonly used)
            impact_ratio = participation_rate.sqrt() if participation_rate > Decimal("0") else Decimal("0")

        elif self.impact_params.model_type == ImpactModel.POWER_LAW:
            # Power law with exponent 0.6
            if participation_rate > Decimal("0"):
                impact_ratio = participation_rate ** Decimal("0.6")
            else:
                impact_ratio = Decimal("0")

        elif self.impact_params.model_type == ImpactModel.ALMGREN_CHRISS:
            # Almgren-Chriss model
            sigma = vol_factor * order_price
            impact_ratio = (participation_rate ** Decimal("0.5")) * sigma / order_price

        else:
            impact_ratio = participation_rate

        # Apply impact factor and volatility adjustment
        temporary_impact = (
            order_price *
            impact_ratio *
            self.impact_params.temporary_impact_factor *
            vol_factor
        )

        return temporary_impact

    async def _calculate_permanent_impact(
        self,
        order_quantity: Decimal,
        order_price: Decimal,
        market_volume: Optional[Decimal]
    ) -> Decimal:
        """Calculate permanent price impact.

        Args:
            order_quantity: Size of the order
            order_price: Current market price
            market_volume: Recent market volume

        Returns:
            Permanent impact amount
        """
        # Permanent impact is typically smaller than temporary
        # and based on information content of the trade

        if market_volume is not None and market_volume > Decimal("0"):
            participation_rate = order_quantity / market_volume
        else:
            participation_rate = order_quantity / self.impact_params.liquidity_factor

        # Permanent impact grows more slowly with size
        if self.impact_params.model_type == ImpactModel.LINEAR:
            impact_ratio = participation_rate * Decimal("0.5")
        else:
            # Square root for most models
            if participation_rate > Decimal("0"):
                impact_ratio = participation_rate.sqrt()
            else:
                impact_ratio = Decimal("0")

        permanent_impact = (
            order_price *
            impact_ratio *
            self.impact_params.permanent_impact_factor
        )

        return permanent_impact

    async def calculate_vwap_impact(
        self,
        total_quantity: Decimal,
        num_slices: int,
        initial_price: Decimal,
        market_volume: Decimal,
        time_horizon_seconds: int
    ) -> Tuple[Decimal, List[Decimal]]:
        """Calculate VWAP execution impact with order slicing.

        Args:
            total_quantity: Total order quantity
            num_slices: Number of child orders
            initial_price: Starting price
            market_volume: Total market volume over time horizon
            time_horizon_seconds: Execution time window

        Returns:
            Tuple of (average_execution_price, slice_prices)

        Example:
            >>> model = MarketImpactModel(config, "BTC/USDT")
            >>> avg_price, prices = await model.calculate_vwap_impact(
            ...     Decimal("100"),
            ...     10,
            ...     Decimal("50000"),
            ...     Decimal("10000000"),
            ...     3600
            ... )
        """
        if num_slices <= 0:
            raise ValueError("num_slices must be positive")

        if total_quantity <= Decimal("0"):
            raise ValueError("total_quantity must be positive")

        try:
            slice_quantity = total_quantity / Decimal(str(num_slices))
            slice_volume = market_volume / Decimal(str(num_slices))

            current_price = initial_price
            slice_prices = []
            total_cost = Decimal("0")

            # Calculate impact for each slice
            for i in range(num_slices):
                # Calculate impact for this slice
                impact_result = await self.calculate_impact(
                    slice_quantity,
                    current_price,
                    is_buy=True,  # Assuming buy side
                    market_volume=slice_volume
                )

                execution_price = impact_result.impacted_price
                slice_prices.append(execution_price)

                total_cost += execution_price * slice_quantity

                # Update price for next slice (permanent impact persists)
                current_price += impact_result.permanent_impact

                # Temporary impact decays between slices
                decay_factor = Decimal(str(self.config.get("impact_decay_factor", 0.5)))
                current_price += impact_result.temporary_impact * decay_factor

            # Calculate average execution price
            average_price = total_cost / total_quantity

            logger.info(
                "VWAP impact calculated",
                total_quantity=str(total_quantity),
                num_slices=num_slices,
                avg_price=str(average_price),
                initial_price=str(initial_price),
                total_impact_bps=float(((average_price - initial_price) / initial_price) * Decimal("10000"))
            )

            return average_price, slice_prices

        except Exception as e:
            logger.error("Failed to calculate VWAP impact", error=str(e))
            raise

    def get_cumulative_impact(self) -> Decimal:
        """Get cumulative permanent price impact.

        Returns:
            Total permanent impact accumulated

        Example:
            >>> model = MarketImpactModel(config, "BTC/USDT")
            >>> # ... execute multiple orders ...
            >>> total_impact = model.get_cumulative_impact()
        """
        return self.cumulative_permanent_impact

    def reset_cumulative_impact(self) -> None:
        """Reset cumulative permanent impact to zero.

        Example:
            >>> model = MarketImpactModel(config, "BTC/USDT")
            >>> model.reset_cumulative_impact()
        """
        self.cumulative_permanent_impact = Decimal("0")
        logger.debug("Cumulative impact reset", symbol=self.symbol)

    async def estimate_optimal_slicing(
        self,
        total_quantity: Decimal,
        total_time_seconds: int,
        market_volume_per_second: Decimal,
        max_participation_rate: Decimal = Decimal("0.1")
    ) -> Dict[str, Any]:
        """Estimate optimal order slicing parameters.

        Args:
            total_quantity: Total order size
            total_time_seconds: Available time window
            market_volume_per_second: Expected market volume rate
            max_participation_rate: Maximum allowed participation rate

        Returns:
            Dictionary with slicing recommendations

        Example:
            >>> model = MarketImpactModel(config, "BTC/USDT")
            >>> recommendation = await model.estimate_optimal_slicing(
            ...     Decimal("100"),
            ...     3600,
            ...     Decimal("1000"),
            ...     Decimal("0.1")
            ... )
            >>> print(f"Recommended slices: {recommendation['recommended_slices']}")
        """
        try:
            # Calculate total expected volume
            total_market_volume = market_volume_per_second * Decimal(str(total_time_seconds))

            # Calculate minimum slices based on participation rate
            min_slices = int((total_quantity / (total_market_volume * max_participation_rate)).to_integral_value())
            min_slices = max(1, min_slices)

            # Calculate optimal slice size to minimize total cost
            # Using simplified cost model: cost = permanent + temporary
            # Optimal slices balances these two components

            recommended_slices = min_slices * 2  # Conservative approach

            slice_quantity = total_quantity / Decimal(str(recommended_slices))
            slice_interval_seconds = total_time_seconds // recommended_slices

            recommendation = {
                "recommended_slices": recommended_slices,
                "minimum_slices": min_slices,
                "slice_quantity": float(slice_quantity),
                "slice_interval_seconds": slice_interval_seconds,
                "participation_rate": float(slice_quantity / (market_volume_per_second * Decimal(str(slice_interval_seconds))))
            }

            logger.info(
                "Optimal slicing estimated",
                symbol=self.symbol,
                recommended_slices=recommended_slices,
                participation_rate=recommendation["participation_rate"]
            )

            return recommendation

        except Exception as e:
            logger.error("Failed to estimate optimal slicing", error=str(e))
            raise
