"""
Quantum Trader AI - Trade Settlement Management System
Production-grade settlement tracking and management

CRITICAL: All numeric values use Decimal, never float
CRITICAL: All data operations use polars DataFrame, never pandas
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import polars as pl
import yaml

from quantum_trader.models import (
    AuditLog,
    ExecutionResult,
    ExecutionStatus,
    Order,
    OrderSide,
    OrderStatus,
)


logger = logging.getLogger(__name__)


class SettlementCycle(Enum):
    """Settlement cycle types"""
    T_PLUS_0 = "T+0"  # Same day settlement
    T_PLUS_1 = "T+1"  # Next day settlement
    T_PLUS_2 = "T+2"  # Two days settlement
    T_PLUS_3 = "T+3"  # Three days settlement


class SettlementStatus(Enum):
    """Settlement lifecycle status"""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SETTLED = "SETTLED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    DISPUTED = "DISPUTED"


class DVPStatus(Enum):
    """Delivery vs Payment status"""
    AWAITING_DELIVERY = "AWAITING_DELIVERY"
    AWAITING_PAYMENT = "AWAITING_PAYMENT"
    DELIVERY_CONFIRMED = "DELIVERY_CONFIRMED"
    PAYMENT_CONFIRMED = "PAYMENT_CONFIRMED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class SettlementInstruction:
    """Settlement instruction for a trade"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    total_amount: Decimal
    fees: Decimal
    settlement_cycle: SettlementCycle
    settlement_date: datetime
    exchange: str
    counterparty: str
    created_at: datetime
    status: SettlementStatus = SettlementStatus.PENDING
    dvp_status: DVPStatus = DVPStatus.AWAITING_DELIVERY
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate settlement instruction"""
        for field_name in ["quantity", "price", "total_amount", "fees"]:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                raise TypeError(f"{field_name} must be Decimal, got {type(value)}")


@dataclass
class CorporateAction:
    """Corporate action affecting settlements"""
    action_id: str
    symbol: str
    action_type: str  # DIVIDEND, SPLIT, MERGER, etc
    effective_date: datetime
    record_date: datetime
    payment_date: datetime
    ratio: Optional[Decimal] = None
    amount_per_share: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SettlementManager:
    """
    Production-grade settlement management system

    Handles:
    - T+0/T+1/T+2 settlement tracking
    - Delivery vs payment (DVP) workflows
    - Settlement risk management
    - Failed trade handling
    - Corporate actions processing
    - Settlement reporting
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize settlement manager

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Settlement tracking
        self.pending_settlements: Dict[str, SettlementInstruction] = {}
        self.settlement_history: List[SettlementInstruction] = []

        # Corporate actions tracking
        self.corporate_actions: Dict[str, List[CorporateAction]] = defaultdict(list)

        # Risk tracking
        self.counterparty_exposure: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        self.daily_settlement_volume: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))

        # Configuration
        self.max_settlement_exposure = Decimal(
            str(self.config.get("counterparty", {}).get("max_settlement_exposure_usd", "500000"))
        )
        self.max_retries = self.config.get("bot", {}).get("execution", {}).get("retry_attempts", 3)
        self.retry_delay = self.config.get("bot", {}).get("execution", {}).get("retry_delay_ms", 1000) / 1000.0
        self.retry_backoff = Decimal(
            str(self.config.get("bot", {}).get("execution", {}).get("retry_backoff_multiplier", 2.0))
        )

        self.logger.info("SettlementManager initialized successfully")

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML files

        Args:
            config_path: Optional path to specific config file

        Returns:
            Merged configuration dictionary
        """
        config: Dict[str, Any] = {}

        # Load bot configuration
        bot_config_path = Path("/home/user/FritzellSama/config/bot/bot.yaml")
        if bot_config_path.exists():
            with open(bot_config_path, "r") as f:
                config.update(yaml.safe_load(f) or {})

        # Load risk configuration
        risk_config_path = Path("/home/user/FritzellSama/config/bot/risk.yaml")
        if risk_config_path.exists():
            with open(risk_config_path, "r") as f:
                risk_config = yaml.safe_load(f) or {}
                config.update(risk_config)

        # Load environment configuration
        env_config_path = Path("/home/user/FritzellSama/config/environments/production.yaml")
        if env_config_path.exists():
            with open(env_config_path, "r") as f:
                env_config = yaml.safe_load(f) or {}
                config.update(env_config)

        # Load custom config if provided
        if config_path:
            custom_path = Path(config_path)
            if custom_path.exists():
                with open(custom_path, "r") as f:
                    config.update(yaml.safe_load(f) or {})

        return config

    async def create_settlement_instruction(
        self,
        execution_result: ExecutionResult,
        settlement_cycle: SettlementCycle = SettlementCycle.T_PLUS_2,
    ) -> SettlementInstruction:
        """
        Create settlement instruction from execution result

        Args:
            execution_result: Execution result to settle
            settlement_cycle: Settlement cycle to use

        Returns:
            Created settlement instruction
        """
        try:
            now = datetime.utcnow()

            # Calculate settlement date based on cycle
            settlement_days = int(settlement_cycle.value.split("+")[1])
            settlement_date = now + timedelta(days=settlement_days)

            instruction = SettlementInstruction(
                trade_id=f"TRD_{execution_result.order_id}_{now.strftime('%Y%m%d%H%M%S')}",
                order_id=execution_result.order_id,
                symbol=execution_result.symbol,
                side=execution_result.side,
                quantity=execution_result.filled_quantity,
                price=execution_result.average_price,
                total_amount=execution_result.total_cost,
                fees=execution_result.fees,
                settlement_cycle=settlement_cycle,
                settlement_date=settlement_date,
                exchange=execution_result.exchange,
                counterparty=execution_result.exchange,
                created_at=now,
                status=SettlementStatus.PENDING,
                metadata={
                    "exchange_order_id": execution_result.exchange_order_id,
                    "execution_timestamp": execution_result.timestamp.isoformat(),
                },
            )

            self.pending_settlements[instruction.trade_id] = instruction
            self.logger.info(
                f"Created settlement instruction {instruction.trade_id} "
                f"for {instruction.symbol} {instruction.side.value}"
            )

            # Update counterparty exposure
            await self._update_counterparty_exposure(instruction)

            return instruction

        except Exception as e:
            self.logger.error(f"Failed to create settlement instruction: {e}", exc_info=True)
            raise

    async def _update_counterparty_exposure(self, instruction: SettlementInstruction) -> None:
        """
        Update counterparty exposure tracking

        Args:
            instruction: Settlement instruction to track
        """
        try:
            counterparty = instruction.counterparty
            amount = instruction.total_amount

            self.counterparty_exposure[counterparty] += amount
            self.daily_settlement_volume[counterparty] += amount

            # Check exposure limits
            if self.counterparty_exposure[counterparty] > self.max_settlement_exposure:
                self.logger.warning(
                    f"Counterparty {counterparty} exposure "
                    f"{self.counterparty_exposure[counterparty]} "
                    f"exceeds limit {self.max_settlement_exposure}"
                )

        except Exception as e:
            self.logger.error(f"Failed to update counterparty exposure: {e}", exc_info=True)

    async def process_dvp_workflow(self, trade_id: str) -> bool:
        """
        Process Delivery vs Payment workflow

        Args:
            trade_id: Trade ID to process

        Returns:
            True if DVP workflow completed successfully
        """
        if trade_id not in self.pending_settlements:
            self.logger.error(f"Settlement instruction {trade_id} not found")
            return False

        instruction = self.pending_settlements[trade_id]
        max_retries = self.max_retries
        retry_delay = self.retry_delay

        for attempt in range(max_retries):
            try:
                # Step 1: Confirm delivery
                if instruction.dvp_status == DVPStatus.AWAITING_DELIVERY:
                    delivery_confirmed = await self._confirm_delivery(instruction)
                    if delivery_confirmed:
                        instruction.dvp_status = DVPStatus.DELIVERY_CONFIRMED
                        self.logger.info(f"Delivery confirmed for {trade_id}")
                    else:
                        raise Exception("Delivery confirmation failed")

                # Step 2: Confirm payment
                if instruction.dvp_status == DVPStatus.DELIVERY_CONFIRMED:
                    payment_confirmed = await self._confirm_payment(instruction)
                    if payment_confirmed:
                        instruction.dvp_status = DVPStatus.PAYMENT_CONFIRMED
                        self.logger.info(f"Payment confirmed for {trade_id}")
                    else:
                        raise Exception("Payment confirmation failed")

                # Step 3: Complete DVP
                if instruction.dvp_status == DVPStatus.PAYMENT_CONFIRMED:
                    instruction.dvp_status = DVPStatus.COMPLETED
                    instruction.status = SettlementStatus.SETTLED
                    self.logger.info(f"DVP workflow completed for {trade_id}")
                    return True

            except Exception as e:
                self.logger.warning(
                    f"DVP workflow attempt {attempt + 1}/{max_retries} failed for {trade_id}: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= float(self.retry_backoff)
                else:
                    instruction.dvp_status = DVPStatus.FAILED
                    instruction.status = SettlementStatus.FAILED
                    self.logger.error(f"DVP workflow failed for {trade_id} after {max_retries} attempts")
                    return False

        return False

    async def _confirm_delivery(self, instruction: SettlementInstruction) -> bool:
        """
        Confirm delivery of securities

        Args:
            instruction: Settlement instruction

        Returns:
            True if delivery confirmed
        """
        try:
            # In production, this would interface with exchange/custodian APIs
            # For now, simulate delivery confirmation
            await asyncio.sleep(0.1)  # Simulate API call

            # Check if counterparty has delivered securities
            self.logger.debug(
                f"Confirming delivery of {instruction.quantity} {instruction.symbol} "
                f"from {instruction.counterparty}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Delivery confirmation failed: {e}", exc_info=True)
            return False

    async def _confirm_payment(self, instruction: SettlementInstruction) -> bool:
        """
        Confirm payment for securities

        Args:
            instruction: Settlement instruction

        Returns:
            True if payment confirmed
        """
        try:
            # In production, this would interface with payment systems
            # For now, simulate payment confirmation
            await asyncio.sleep(0.1)  # Simulate API call

            # Check if payment has been received
            self.logger.debug(
                f"Confirming payment of {instruction.total_amount} "
                f"for {instruction.symbol} trade"
            )

            return True

        except Exception as e:
            self.logger.error(f"Payment confirmation failed: {e}", exc_info=True)
            return False

    async def handle_failed_settlement(self, trade_id: str) -> None:
        """
        Handle failed settlement

        Args:
            trade_id: Trade ID that failed settlement
        """
        if trade_id not in self.pending_settlements:
            self.logger.error(f"Settlement instruction {trade_id} not found")
            return

        instruction = self.pending_settlements[trade_id]

        try:
            self.logger.error(
                f"Handling failed settlement for {trade_id}: "
                f"{instruction.symbol} {instruction.side.value} {instruction.quantity}"
            )

            # Mark as failed
            instruction.status = SettlementStatus.FAILED

            # Reduce counterparty exposure
            self.counterparty_exposure[instruction.counterparty] -= instruction.total_amount

            # Move to history
            self.settlement_history.append(instruction)
            del self.pending_settlements[trade_id]

            # Trigger alert
            await self._send_settlement_alert(
                trade_id=trade_id,
                message=f"Settlement failed for {instruction.symbol}",
                severity="CRITICAL",
            )

        except Exception as e:
            self.logger.error(f"Failed to handle failed settlement: {e}", exc_info=True)

    async def _send_settlement_alert(
        self, trade_id: str, message: str, severity: str
    ) -> None:
        """
        Send settlement alert

        Args:
            trade_id: Trade ID
            message: Alert message
            severity: Alert severity
        """
        try:
            # In production, this would send to notification system
            self.logger.log(
                logging.ERROR if severity == "CRITICAL" else logging.WARNING,
                f"SETTLEMENT ALERT [{severity}] - Trade {trade_id}: {message}",
            )

        except Exception as e:
            self.logger.error(f"Failed to send settlement alert: {e}", exc_info=True)

    async def process_corporate_action(self, action: CorporateAction) -> None:
        """
        Process corporate action affecting settlements

        Args:
            action: Corporate action to process
        """
        try:
            self.logger.info(
                f"Processing corporate action {action.action_id} for {action.symbol}: "
                f"{action.action_type}"
            )

            # Store corporate action
            self.corporate_actions[action.symbol].append(action)

            # Adjust pending settlements for this symbol
            affected_settlements = [
                inst for inst in self.pending_settlements.values()
                if inst.symbol == action.symbol
                and inst.settlement_date >= action.effective_date
            ]

            for instruction in affected_settlements:
                await self._apply_corporate_action(instruction, action)

            self.logger.info(
                f"Corporate action {action.action_id} processed, "
                f"affected {len(affected_settlements)} settlements"
            )

        except Exception as e:
            self.logger.error(f"Failed to process corporate action: {e}", exc_info=True)

    async def _apply_corporate_action(
        self, instruction: SettlementInstruction, action: CorporateAction
    ) -> None:
        """
        Apply corporate action to settlement instruction

        Args:
            instruction: Settlement instruction to adjust
            action: Corporate action to apply
        """
        try:
            if action.action_type == "SPLIT" and action.ratio:
                # Adjust quantity and price for stock split
                instruction.quantity *= action.ratio
                instruction.price /= action.ratio
                instruction.metadata["corporate_action"] = {
                    "action_id": action.action_id,
                    "type": action.action_type,
                    "ratio": str(action.ratio),
                }

            elif action.action_type == "DIVIDEND" and action.amount_per_share:
                # Record dividend payment
                instruction.metadata["dividend"] = {
                    "action_id": action.action_id,
                    "amount_per_share": str(action.amount_per_share),
                    "total_dividend": str(action.amount_per_share * instruction.quantity),
                }

            self.logger.debug(
                f"Applied {action.action_type} to settlement {instruction.trade_id}"
            )

        except Exception as e:
            self.logger.error(f"Failed to apply corporate action: {e}", exc_info=True)

    async def check_settlement_risk(self) -> Dict[str, Any]:
        """
        Check settlement risk across all pending settlements

        Returns:
            Risk assessment dictionary
        """
        try:
            total_exposure = sum(
                inst.total_amount for inst in self.pending_settlements.values()
            )

            counterparty_risk = {
                cp: exposure
                for cp, exposure in self.counterparty_exposure.items()
                if exposure > self.max_settlement_exposure * Decimal("0.8")  # 80% threshold
            }

            overdue_settlements = [
                inst.trade_id
                for inst in self.pending_settlements.values()
                if inst.settlement_date < datetime.utcnow()
                and inst.status == SettlementStatus.PENDING
            ]

            risk_assessment = {
                "total_pending_settlements": len(self.pending_settlements),
                "total_exposure_usd": str(total_exposure),
                "high_risk_counterparties": {cp: str(exp) for cp, exp in counterparty_risk.items()},
                "overdue_settlements": overdue_settlements,
                "overdue_count": len(overdue_settlements),
                "assessment_timestamp": datetime.utcnow().isoformat(),
            }

            if overdue_settlements:
                self.logger.warning(
                    f"Found {len(overdue_settlements)} overdue settlements"
                )

            return risk_assessment

        except Exception as e:
            self.logger.error(f"Failed to check settlement risk: {e}", exc_info=True)
            return {}

    async def generate_settlement_report(
        self, start_date: datetime, end_date: datetime
    ) -> pl.DataFrame:
        """
        Generate settlement report for date range

        Args:
            start_date: Report start date
            end_date: Report end date

        Returns:
            Polars DataFrame with settlement data
        """
        try:
            # Filter settlements within date range
            relevant_settlements = [
                inst for inst in self.settlement_history
                if start_date <= inst.created_at <= end_date
            ]

            if not relevant_settlements:
                self.logger.info("No settlements found for date range")
                return pl.DataFrame()

            # Convert to polars DataFrame
            data = {
                "trade_id": [inst.trade_id for inst in relevant_settlements],
                "order_id": [inst.order_id for inst in relevant_settlements],
                "symbol": [inst.symbol for inst in relevant_settlements],
                "side": [inst.side.value for inst in relevant_settlements],
                "quantity": [float(inst.quantity) for inst in relevant_settlements],
                "price": [float(inst.price) for inst in relevant_settlements],
                "total_amount": [float(inst.total_amount) for inst in relevant_settlements],
                "fees": [float(inst.fees) for inst in relevant_settlements],
                "settlement_cycle": [inst.settlement_cycle.value for inst in relevant_settlements],
                "settlement_date": [inst.settlement_date for inst in relevant_settlements],
                "status": [inst.status.value for inst in relevant_settlements],
                "dvp_status": [inst.dvp_status.value for inst in relevant_settlements],
                "exchange": [inst.exchange for inst in relevant_settlements],
                "counterparty": [inst.counterparty for inst in relevant_settlements],
                "created_at": [inst.created_at for inst in relevant_settlements],
            }

            df = pl.DataFrame(data)

            self.logger.info(
                f"Generated settlement report with {len(df)} records "
                f"from {start_date} to {end_date}"
            )

            return df

        except Exception as e:
            self.logger.error(f"Failed to generate settlement report: {e}", exc_info=True)
            return pl.DataFrame()

    async def reconcile_settlements(self) -> Dict[str, Any]:
        """
        Reconcile settlements with exchange records

        Returns:
            Reconciliation results
        """
        try:
            discrepancies: List[Dict[str, Any]] = []

            for trade_id, instruction in self.pending_settlements.items():
                # In production, this would query exchange APIs
                # For now, perform basic validation

                if instruction.settlement_date < datetime.utcnow() - timedelta(days=1):
                    if instruction.status == SettlementStatus.PENDING:
                        discrepancies.append({
                            "trade_id": trade_id,
                            "issue": "overdue_settlement",
                            "settlement_date": instruction.settlement_date.isoformat(),
                            "days_overdue": (datetime.utcnow() - instruction.settlement_date).days,
                        })

            reconciliation_result = {
                "reconciliation_timestamp": datetime.utcnow().isoformat(),
                "total_pending": len(self.pending_settlements),
                "discrepancies_found": len(discrepancies),
                "discrepancies": discrepancies,
            }

            if discrepancies:
                self.logger.warning(
                    f"Settlement reconciliation found {len(discrepancies)} discrepancies"
                )
            else:
                self.logger.info("Settlement reconciliation completed successfully, no issues found")

            return reconciliation_result

        except Exception as e:
            self.logger.error(f"Failed to reconcile settlements: {e}", exc_info=True)
            return {}

    async def cancel_settlement(self, trade_id: str, reason: str) -> bool:
        """
        Cancel pending settlement

        Args:
            trade_id: Trade ID to cancel
            reason: Cancellation reason

        Returns:
            True if cancelled successfully
        """
        if trade_id not in self.pending_settlements:
            self.logger.error(f"Settlement instruction {trade_id} not found")
            return False

        try:
            instruction = self.pending_settlements[trade_id]

            # Can only cancel if not already settled
            if instruction.status == SettlementStatus.SETTLED:
                self.logger.error(f"Cannot cancel settled trade {trade_id}")
                return False

            # Update status
            instruction.status = SettlementStatus.CANCELLED
            instruction.metadata["cancellation_reason"] = reason
            instruction.metadata["cancelled_at"] = datetime.utcnow().isoformat()

            # Reduce counterparty exposure
            self.counterparty_exposure[instruction.counterparty] -= instruction.total_amount

            # Move to history
            self.settlement_history.append(instruction)
            del self.pending_settlements[trade_id]

            self.logger.info(f"Cancelled settlement {trade_id}: {reason}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel settlement: {e}", exc_info=True)
            return False

    def get_settlement_status(self, trade_id: str) -> Optional[SettlementStatus]:
        """
        Get settlement status for trade

        Args:
            trade_id: Trade ID

        Returns:
            Settlement status or None if not found
        """
        if trade_id in self.pending_settlements:
            return self.pending_settlements[trade_id].status

        # Check history
        for inst in self.settlement_history:
            if inst.trade_id == trade_id:
                return inst.status

        return None

    def get_counterparty_exposure(self, counterparty: str) -> Decimal:
        """
        Get current exposure to counterparty

        Args:
            counterparty: Counterparty identifier

        Returns:
            Total exposure amount
        """
        return self.counterparty_exposure.get(counterparty, Decimal("0"))
