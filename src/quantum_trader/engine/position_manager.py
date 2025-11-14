"""Position Manager - CRITICAL PRODUCTION COMPONENT"""
from decimal import Decimal
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import os, logging
import polars as pl

logger = logging.getLogger(__name__)

class PositionSide(Enum):
    """Position side enumeration"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

class PositionStatus(Enum):
    """Position status enumeration"""
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"
    FAILED = "failed"

class Position:
    """Represents a trading position"""

    def __init__(
        self,
        position_id: str,
        symbol: str,
        side: PositionSide,
        size: Decimal,
        entry_price: Decimal,
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.position_id = position_id
        self.symbol = symbol
        self.side = side
        self.size = size
        self.entry_price = entry_price
        self.timestamp = timestamp
        self.metadata = metadata or {}

        self.status = PositionStatus.OPEN
        self.exit_price: Optional[Decimal] = None
        self.exit_timestamp: Optional[datetime] = None
        self.realized_pnl: Optional[Decimal] = None

    def calculate_unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L"""
        try:
            price_diff = current_price - self.entry_price

            if self.side == PositionSide.SHORT:
                price_diff = -price_diff

            pnl = price_diff * self.size
            return pnl

        except Exception as e:
            logger.error(f"PnL calculation error: {e}")
            return Decimal('0')

    def close(
        self,
        exit_price: Decimal,
        exit_timestamp: Optional[datetime] = None
    ) -> Decimal:
        """Close position and calculate realized P&L"""
        try:
            self.exit_price = exit_price
            self.exit_timestamp = exit_timestamp or datetime.utcnow()
            self.realized_pnl = self.calculate_unrealized_pnl(exit_price)
            self.status = PositionStatus.CLOSED

            return self.realized_pnl

        except Exception as e:
            logger.error(f"Position close error: {e}")
            self.status = PositionStatus.FAILED
            return Decimal('0')

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'size': self.size,
            'entry_price': self.entry_price,
            'entry_timestamp': self.timestamp,
            'status': self.status.value,
            'exit_price': self.exit_price,
            'exit_timestamp': self.exit_timestamp,
            'realized_pnl': self.realized_pnl,
            'metadata': self.metadata
        }

class PositionManager:
    """
    CRITICAL: Manages all trading positions with risk controls

    Responsibilities:
    - Track all open positions
    - Calculate portfolio metrics
    - Enforce position size limits
    - Monitor exposure and leverage
    - Provide position analytics
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Risk limits
        self.max_position_size = Decimal(str(config.get(
            'max_position_size',
            os.getenv('PM_MAX_POSITION_SIZE', '100000')
        )))
        self.max_total_exposure = Decimal(str(config.get(
            'max_total_exposure',
            os.getenv('PM_MAX_EXPOSURE', '500000')
        )))
        self.max_leverage = Decimal(str(config.get(
            'max_leverage',
            os.getenv('PM_MAX_LEVERAGE', '3.0')
        )))
        self.max_positions_per_symbol = int(config.get(
            'max_positions_per_symbol',
            os.getenv('PM_MAX_PER_SYMBOL', '5')
        ))

        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.position_count = 0

        # Portfolio state
        self.total_capital = Decimal(str(config.get(
            'total_capital',
            os.getenv('PM_TOTAL_CAPITAL', '1000000')
        )))
        self.available_capital = self.total_capital

        self.logger.info(
            f"PositionManager initialized: max_position={self.max_position_size}, "
            f"max_exposure={self.max_total_exposure}, "
            f"max_leverage={self.max_leverage}"
        )

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        size: Decimal,
        entry_price: Decimal,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Position]:
        """Open a new position with risk checks"""
        try:
            # Risk checks
            if not self._validate_new_position(symbol, size):
                return None

            # Create position
            position_id = f"pos_{self.position_count}_{datetime.utcnow().timestamp()}"
            position = Position(
                position_id=position_id,
                symbol=symbol,
                side=side,
                size=size,
                entry_price=entry_price,
                timestamp=datetime.utcnow(),
                metadata=metadata
            )

            # Update tracking
            self.positions[position_id] = position
            self.position_count += 1

            # Update capital
            position_value = size * entry_price
            self.available_capital -= position_value

            self.logger.info(
                f"Position opened: {position_id}, {symbol}, {side.value}, "
                f"size={size}, price={entry_price}"
            )

            return position

        except Exception as e:
            self.logger.error(f"Open position error: {e}", exc_info=True)
            return None

    def close_position(
        self,
        position_id: str,
        exit_price: Decimal
    ) -> Optional[Decimal]:
        """Close an existing position"""
        try:
            if position_id not in self.positions:
                self.logger.warning(f"Position not found: {position_id}")
                return None

            position = self.positions[position_id]

            # Close position
            realized_pnl = position.close(exit_price)

            # Update capital
            position_value = position.size * exit_price
            self.available_capital += position_value + realized_pnl

            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[position_id]

            self.logger.info(
                f"Position closed: {position_id}, exit_price={exit_price}, "
                f"pnl={realized_pnl}"
            )

            return realized_pnl

        except Exception as e:
            self.logger.error(f"Close position error: {e}", exc_info=True)
            return None

    def _validate_new_position(
        self,
        symbol: str,
        size: Decimal
    ) -> bool:
        """Validate if new position can be opened"""
        try:
            # Check position size limit
            if size > self.max_position_size:
                self.logger.warning(
                    f"Position size {size} exceeds max {self.max_position_size}"
                )
                return False

            # Check per-symbol limit
            symbol_positions = [
                p for p in self.positions.values()
                if p.symbol == symbol and p.status == PositionStatus.OPEN
            ]
            if len(symbol_positions) >= self.max_positions_per_symbol:
                self.logger.warning(
                    f"Max positions for {symbol} reached: {self.max_positions_per_symbol}"
                )
                return False

            # Check total exposure
            current_exposure = self.get_total_exposure()
            if current_exposure >= self.max_total_exposure:
                self.logger.warning(
                    f"Total exposure {current_exposure} exceeds max {self.max_total_exposure}"
                )
                return False

            # Check available capital
            if size > self.available_capital:
                self.logger.warning(
                    f"Insufficient capital: need {size}, have {self.available_capital}"
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False

    def get_total_exposure(self) -> Decimal:
        """Calculate total exposure across all positions"""
        try:
            total = Decimal('0')
            for position in self.positions.values():
                if position.status == PositionStatus.OPEN:
                    total += position.size * position.entry_price

            return total

        except Exception as e:
            self.logger.error(f"Exposure calculation error: {e}")
            return Decimal('0')

    def calculate_portfolio_pnl(
        self,
        current_prices: Dict[str, Decimal]
    ) -> Dict[str, Any]:
        """Calculate total portfolio P&L"""
        try:
            total_unrealized_pnl = Decimal('0')
            total_realized_pnl = Decimal('0')

            # Unrealized P&L from open positions
            for position in self.positions.values():
                if position.status == PositionStatus.OPEN:
                    current_price = current_prices.get(position.symbol)
                    if current_price:
                        pnl = position.calculate_unrealized_pnl(current_price)
                        total_unrealized_pnl += pnl

            # Realized P&L from closed positions
            for position in self.closed_positions:
                if position.realized_pnl:
                    total_realized_pnl += position.realized_pnl

            total_pnl = total_unrealized_pnl + total_realized_pnl
            total_value = self.available_capital + self.get_total_exposure() + total_unrealized_pnl

            return {
                'total_pnl': total_pnl,
                'unrealized_pnl': total_unrealized_pnl,
                'realized_pnl': total_realized_pnl,
                'total_value': total_value,
                'return_pct': (total_pnl / self.total_capital) * Decimal('100'),
                'timestamp': datetime.utcnow()
            }

        except Exception as e:
            self.logger.error(f"Portfolio PnL error: {e}")
            return {'error': str(e)}

    def get_position_statistics(self) -> Dict[str, Any]:
        """Get position statistics"""
        try:
            open_positions = len(self.positions)
            total_positions = self.position_count

            winning_positions = len([
                p for p in self.closed_positions
                if p.realized_pnl and p.realized_pnl > 0
            ])
            losing_positions = len([
                p for p in self.closed_positions
                if p.realized_pnl and p.realized_pnl < 0
            ])

            win_rate = Decimal('0')
            if len(self.closed_positions) > 0:
                win_rate = Decimal(str(winning_positions)) / Decimal(str(len(self.closed_positions)))

            avg_win = Decimal('0')
            if winning_positions > 0:
                total_wins = sum([
                    p.realized_pnl for p in self.closed_positions
                    if p.realized_pnl and p.realized_pnl > 0
                ])
                avg_win = total_wins / Decimal(str(winning_positions))

            avg_loss = Decimal('0')
            if losing_positions > 0:
                total_losses = sum([
                    abs(p.realized_pnl) for p in self.closed_positions
                    if p.realized_pnl and p.realized_pnl < 0
                ])
                avg_loss = total_losses / Decimal(str(losing_positions))

            return {
                'open_positions': open_positions,
                'closed_positions': len(self.closed_positions),
                'total_positions': total_positions,
                'winning_positions': winning_positions,
                'losing_positions': losing_positions,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_exposure': self.get_total_exposure(),
                'available_capital': self.available_capital,
                'timestamp': datetime.utcnow()
            }

        except Exception as e:
            self.logger.error(f"Statistics error: {e}")
            return {'error': str(e)}

    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get all open positions for a symbol"""
        return [
            p for p in self.positions.values()
            if p.symbol == symbol and p.status == PositionStatus.OPEN
        ]

    def get_all_open_positions(self) -> List[Position]:
        """Get all open positions"""
        return [
            p for p in self.positions.values()
            if p.status == PositionStatus.OPEN
        ]

    def close_all_positions(
        self,
        current_prices: Dict[str, Decimal]
    ) -> List[Decimal]:
        """Emergency close all positions"""
        try:
            self.logger.warning("CLOSING ALL POSITIONS")

            pnls = []
            positions_to_close = list(self.positions.keys())

            for position_id in positions_to_close:
                position = self.positions[position_id]
                exit_price = current_prices.get(position.symbol)

                if exit_price:
                    pnl = self.close_position(position_id, exit_price)
                    if pnl is not None:
                        pnls.append(pnl)

            self.logger.warning(f"Closed {len(pnls)} positions")
            return pnls

        except Exception as e:
            self.logger.error(f"Emergency close error: {e}", exc_info=True)
            return []
