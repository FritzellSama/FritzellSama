"""
Inventory Manager for Market Making Strategies.

Manages inventory risk and position skew for market making operations,
adjusting quotes based on current inventory levels.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import deque
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class InventoryState(Enum):
    """Inventory position state."""
    NEUTRAL = "NEUTRAL"
    LONG_SKEW = "LONG_SKEW"
    SHORT_SKEW = "SHORT_SKEW"
    CRITICAL_LONG = "CRITICAL_LONG"
    CRITICAL_SHORT = "CRITICAL_SHORT"


@dataclass
class InventorySnapshot:
    """Snapshot of inventory state at a point in time."""
    timestamp: datetime
    position: Decimal
    position_pct: Decimal
    value_usd: Decimal
    unrealized_pnl: Decimal
    state: InventoryState


@dataclass
class QuoteAdjustment:
    """Recommended quote adjustment based on inventory."""
    bid_skew: Decimal  # Adjustment to bid price (bps)
    ask_skew: Decimal  # Adjustment to ask price (bps)
    bid_size_multiplier: Decimal  # Multiplier for bid size
    ask_size_multiplier: Decimal  # Multiplier for ask size
    urgency: Decimal  # Urgency score (0-1)


class InventoryManager:
    """
    Inventory Manager for Market Making.

    Monitors position inventory and provides quote adjustments
    to maintain balanced inventory within risk limits.

    Attributes:
        config: Configuration dictionary
        max_position_size: Maximum allowed position
        target_position: Target neutral position
        current_position: Current inventory position
        inventory_history: Historical inventory snapshots
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Inventory Manager.

        Args:
            config: Configuration dictionary with parameters

        Raises:
            ValueError: If required config parameters missing
        """
        self.config = config
        self._validate_config()

        # Load parameters from config
        self.max_position_size = Decimal(str(config["max_position_size"]))
        self.target_position = Decimal(str(config.get("target_position", "0")))
        self.skew_threshold_pct = Decimal(str(config["skew_threshold_pct"]))
        self.critical_threshold_pct = Decimal(str(config["critical_threshold_pct"]))
        self.max_skew_bps = Decimal(str(config["max_skew_bps"]))
        self.size_adjustment_factor = Decimal(str(config["size_adjustment_factor"]))
        self.inventory_halflife_seconds = Decimal(str(config["inventory_halflife_seconds"]))

        # State tracking
        self.current_position = Decimal("0")
        self.avg_entry_price = Decimal("0")
        self.inventory_history: deque = deque(maxlen=int(config.get("history_length", 100)))
        self.last_snapshot_time: Optional[datetime] = None

        logger.info(
            "inventory_manager_initialized",
            max_position=float(self.max_position_size),
            target=float(self.target_position)
        )

    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required_params = [
            "max_position_size",
            "skew_threshold_pct",
            "critical_threshold_pct",
            "max_skew_bps",
            "size_adjustment_factor",
            "inventory_halflife_seconds"
        ]

        missing = [p for p in required_params if p not in self.config]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")

    async def calculate_quote_adjustment(
        self,
        current_price: Decimal,
        timestamp: datetime
    ) -> QuoteAdjustment:
        """
        Calculate quote adjustments based on current inventory.

        Args:
            current_price: Current market price
            timestamp: Current timestamp

        Returns:
            QuoteAdjustment with recommended skews and sizes

        Raises:
            ValueError: If current_price invalid
        """
        try:
            if current_price <= Decimal("0"):
                raise ValueError("Current price must be positive")

            # Calculate inventory state
            inventory_state = await self._get_inventory_state()

            # Calculate position percentage
            if self.max_position_size > Decimal("0"):
                position_pct = (
                    (self.current_position - self.target_position) / self.max_position_size
                )
            else:
                position_pct = Decimal("0")

            # Calculate price skews
            bid_skew, ask_skew = await self._calculate_price_skews(
                position_pct=position_pct,
                inventory_state=inventory_state
            )

            # Calculate size adjustments
            bid_size_mult, ask_size_mult = await self._calculate_size_adjustments(
                position_pct=position_pct,
                inventory_state=inventory_state
            )

            # Calculate urgency (how aggressive should we be in reducing position)
            urgency = await self._calculate_urgency(
                position_pct=position_pct,
                inventory_state=inventory_state,
                timestamp=timestamp
            )

            adjustment = QuoteAdjustment(
                bid_skew=bid_skew,
                ask_skew=ask_skew,
                bid_size_multiplier=bid_size_mult,
                ask_size_multiplier=ask_size_mult,
                urgency=urgency
            )

            # Store snapshot
            await self._store_snapshot(
                current_price=current_price,
                timestamp=timestamp,
                state=inventory_state
            )

            logger.debug(
                "quote_adjustment_calculated",
                position=float(self.current_position),
                position_pct=float(position_pct),
                bid_skew=float(bid_skew),
                ask_skew=float(ask_skew),
                urgency=float(urgency)
            )

            return adjustment

        except Exception as e:
            logger.error("quote_adjustment_calculation_failed", error=str(e), exc_info=True)
            raise

    async def _get_inventory_state(self) -> InventoryState:
        """
        Determine current inventory state.

        Returns:
            InventoryState enum value
        """
        if self.max_position_size == Decimal("0"):
            return InventoryState.NEUTRAL

        position_pct = abs(
            (self.current_position - self.target_position) / self.max_position_size
        ) * Decimal("100")

        is_long = self.current_position > self.target_position

        # Determine state based on thresholds
        if position_pct >= self.critical_threshold_pct:
            return InventoryState.CRITICAL_LONG if is_long else InventoryState.CRITICAL_SHORT
        elif position_pct >= self.skew_threshold_pct:
            return InventoryState.LONG_SKEW if is_long else InventoryState.SHORT_SKEW
        else:
            return InventoryState.NEUTRAL

    async def _calculate_price_skews(
        self,
        position_pct: Decimal,
        inventory_state: InventoryState
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate bid/ask price skews based on inventory.

        Args:
            position_pct: Position as percentage of max (-1 to +1)
            inventory_state: Current inventory state

        Returns:
            Tuple of (bid_skew_bps, ask_skew_bps)
        """
        # If we're long (positive position), we want to:
        # - Widen asks (increase ask price) to collect more premium
        # - Tighten bids (decrease bid price) to discourage buying
        # This is represented as negative bid skew and positive ask skew

        if inventory_state == InventoryState.NEUTRAL:
            return Decimal("0"), Decimal("0")

        # Calculate base skew magnitude based on position percentage
        skew_magnitude = min(
            self.max_skew_bps,
            abs(position_pct) * self.max_skew_bps
        )

        # Apply skew direction
        if position_pct > Decimal("0"):  # Long position
            bid_skew = -skew_magnitude  # Make bids less competitive
            ask_skew = skew_magnitude * Decimal("0.5")  # Make asks slightly more competitive
        else:  # Short position
            bid_skew = skew_magnitude * Decimal("0.5")  # Make bids slightly more competitive
            ask_skew = -skew_magnitude  # Make asks less competitive

        # Amplify skew in critical states
        if inventory_state in [InventoryState.CRITICAL_LONG, InventoryState.CRITICAL_SHORT]:
            bid_skew *= Decimal("1.5")
            ask_skew *= Decimal("1.5")

        return bid_skew, ask_skew

    async def _calculate_size_adjustments(
        self,
        position_pct: Decimal,
        inventory_state: InventoryState
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate bid/ask size multipliers based on inventory.

        Args:
            position_pct: Position as percentage of max
            inventory_state: Current inventory state

        Returns:
            Tuple of (bid_size_multiplier, ask_size_multiplier)
        """
        # Default multipliers
        bid_mult = Decimal("1.0")
        ask_mult = Decimal("1.0")

        if inventory_state == InventoryState.NEUTRAL:
            return bid_mult, ask_mult

        # Calculate adjustment factor
        adjustment = Decimal("1") + (abs(position_pct) * self.size_adjustment_factor)

        # If long, increase ask size and decrease bid size
        if position_pct > Decimal("0"):
            bid_mult = Decimal("1") / adjustment  # Reduce bid size
            ask_mult = adjustment  # Increase ask size
        else:
            bid_mult = adjustment  # Increase bid size
            ask_mult = Decimal("1") / adjustment  # Reduce ask size

        # In critical states, be more aggressive
        if inventory_state in [InventoryState.CRITICAL_LONG, InventoryState.CRITICAL_SHORT]:
            if position_pct > Decimal("0"):
                bid_mult *= Decimal("0.5")  # Drastically reduce bid size
                ask_mult *= Decimal("2.0")  # Double ask size
            else:
                bid_mult *= Decimal("2.0")  # Double bid size
                ask_mult *= Decimal("0.5")  # Drastically reduce ask size

        # Ensure multipliers are within reasonable bounds
        bid_mult = max(Decimal("0.1"), min(Decimal("3.0"), bid_mult))
        ask_mult = max(Decimal("0.1"), min(Decimal("3.0"), ask_mult))

        return bid_mult, ask_mult

    async def _calculate_urgency(
        self,
        position_pct: Decimal,
        inventory_state: InventoryState,
        timestamp: datetime
    ) -> Decimal:
        """
        Calculate urgency score for inventory reduction.

        Args:
            position_pct: Position as percentage of max
            inventory_state: Current inventory state
            timestamp: Current timestamp

        Returns:
            Urgency score (0.0 to 1.0)
        """
        base_urgency = abs(position_pct)

        # Increase urgency based on state
        if inventory_state == InventoryState.NEUTRAL:
            urgency = Decimal("0.0")
        elif inventory_state in [InventoryState.LONG_SKEW, InventoryState.SHORT_SKEW]:
            urgency = base_urgency * Decimal("0.5")
        else:  # Critical states
            urgency = base_urgency * Decimal("0.9")

        # Increase urgency if position has been held for long time
        if self.inventory_history:
            time_in_position = await self._calculate_time_in_position(timestamp)
            time_factor = time_in_position / self.inventory_halflife_seconds
            urgency += min(Decimal("0.3"), time_factor)

        return min(Decimal("1.0"), urgency)

    async def _calculate_time_in_position(self, current_time: datetime) -> Decimal:
        """
        Calculate how long we've been in current position.

        Args:
            current_time: Current timestamp

        Returns:
            Time in seconds
        """
        if not self.inventory_history:
            return Decimal("0")

        # Find when position crossed into current state
        current_state = await self._get_inventory_state()

        for i in range(len(self.inventory_history) - 1, -1, -1):
            snapshot = list(self.inventory_history)[i]
            if snapshot.state != current_state:
                # Found transition point
                time_delta = (current_time - snapshot.timestamp).total_seconds()
                return Decimal(str(time_delta))

        # Been in this state since first snapshot
        first_snapshot = list(self.inventory_history)[0]
        time_delta = (current_time - first_snapshot.timestamp).total_seconds()
        return Decimal(str(time_delta))

    async def _store_snapshot(
        self,
        current_price: Decimal,
        timestamp: datetime,
        state: InventoryState
    ) -> None:
        """
        Store inventory snapshot for historical tracking.

        Args:
            current_price: Current market price
            timestamp: Snapshot timestamp
            state: Current inventory state
        """
        # Calculate position percentage
        if self.max_position_size > Decimal("0"):
            position_pct = (self.current_position / self.max_position_size) * Decimal("100")
        else:
            position_pct = Decimal("0")

        # Calculate position value
        value_usd = self.current_position * current_price

        # Calculate unrealized PnL
        if self.avg_entry_price > Decimal("0") and self.current_position != Decimal("0"):
            unrealized_pnl = (current_price - self.avg_entry_price) * self.current_position
        else:
            unrealized_pnl = Decimal("0")

        snapshot = InventorySnapshot(
            timestamp=timestamp,
            position=self.current_position,
            position_pct=position_pct,
            value_usd=value_usd,
            unrealized_pnl=unrealized_pnl,
            state=state
        )

        self.inventory_history.append(snapshot)
        self.last_snapshot_time = timestamp

    async def update_position(
        self,
        quantity: Decimal,
        price: Decimal,
        side: OrderSide,
        timestamp: datetime
    ) -> None:
        """
        Update inventory position after trade execution.

        Args:
            quantity: Trade quantity (always positive)
            price: Execution price
            side: Order side (BUY/SELL)
            timestamp: Execution timestamp

        Raises:
            ValueError: If quantity or price invalid
        """
        try:
            if quantity <= Decimal("0"):
                raise ValueError("Quantity must be positive")
            if price <= Decimal("0"):
                raise ValueError("Price must be positive")

            # Calculate position change
            position_change = quantity if side == OrderSide.BUY else -quantity

            # Update average entry price
            old_position = self.current_position
            new_position = old_position + position_change

            # Calculate new average entry price
            if new_position == Decimal("0"):
                self.avg_entry_price = Decimal("0")
            elif (old_position >= Decimal("0") and new_position >= Decimal("0")) or \
                 (old_position <= Decimal("0") and new_position <= Decimal("0")):
                # Adding to position
                total_cost = (old_position * self.avg_entry_price) + (position_change * price)
                self.avg_entry_price = total_cost / new_position if new_position != Decimal("0") else Decimal("0")
            else:
                # Reducing or flipping position
                if abs(new_position) < abs(old_position):
                    # Reducing: keep old avg price
                    pass
                else:
                    # Flipping: new avg price is the execution price
                    self.avg_entry_price = price

            # Update position
            self.current_position = new_position

            # Check limits
            await self._check_position_limits()

            logger.info(
                "inventory_updated",
                side=side.value,
                quantity=float(quantity),
                price=float(price),
                new_position=float(self.current_position),
                avg_entry_price=float(self.avg_entry_price)
            )

        except Exception as e:
            logger.error("position_update_failed", error=str(e), exc_info=True)
            raise

    async def _check_position_limits(self) -> None:
        """
        Check if position exceeds risk limits.

        Raises:
            RuntimeError: If position exceeds critical limits
        """
        position_magnitude = abs(self.current_position)

        if position_magnitude > self.max_position_size:
            logger.error(
                "position_limit_exceeded",
                current=float(self.current_position),
                max_allowed=float(self.max_position_size)
            )
            raise RuntimeError(
                f"Position {self.current_position} exceeds limit {self.max_position_size}"
            )

    async def get_inventory_metrics(self) -> Dict[str, Any]:
        """
        Get current inventory metrics.

        Returns:
            Dictionary of inventory metrics
        """
        state = await self._get_inventory_state()

        position_pct = Decimal("0")
        if self.max_position_size > Decimal("0"):
            position_pct = (self.current_position / self.max_position_size) * Decimal("100")

        return {
            "position": str(self.current_position),
            "position_pct": str(position_pct),
            "avg_entry_price": str(self.avg_entry_price),
            "max_position_size": str(self.max_position_size),
            "target_position": str(self.target_position),
            "state": state.value,
            "snapshots_count": len(self.inventory_history)
        }

    async def reset_position(self) -> None:
        """Reset inventory position (use with caution)."""
        logger.warning("inventory_reset", old_position=float(self.current_position))
        self.current_position = Decimal("0")
        self.avg_entry_price = Decimal("0")
        self.inventory_history.clear()
