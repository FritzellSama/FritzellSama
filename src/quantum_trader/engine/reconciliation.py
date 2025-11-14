"""
Quantum Trader AI - Post-Trade Reconciliation System
Production-grade reconciliation and break resolution

CRITICAL: All numeric values use Decimal, never float
CRITICAL: Uses polars DataFrame for data operations
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4
import yaml

import polars as pl

from quantum_trader.models import (
    Order,
    OrderSide,
    Position,
    ExecutionResult,
    AuditLog,
)


logger = logging.getLogger(__name__)


class DiscrepancyType(Enum):
    """Types of reconciliation discrepancies"""

    POSITION_MISMATCH = "POSITION_MISMATCH"
    TRADE_BREAK = "TRADE_BREAK"
    FEE_MISMATCH = "FEE_MISMATCH"
    PNL_DISCREPANCY = "PNL_DISCREPANCY"
    MISSING_TRADE = "MISSING_TRADE"
    DUPLICATE_TRADE = "DUPLICATE_TRADE"
    PRICE_DEVIATION = "PRICE_DEVIATION"
    QUANTITY_MISMATCH = "QUANTITY_MISMATCH"


class DiscrepancySeverity(Enum):
    """Severity levels for discrepancies"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ResolutionStatus(Enum):
    """Status of discrepancy resolution"""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"
    FAILED = "FAILED"


@dataclass
class Discrepancy:
    """Represents a reconciliation discrepancy"""

    discrepancy_id: str
    discrepancy_type: DiscrepancyType
    severity: DiscrepancySeverity
    symbol: str
    exchange: str
    internal_value: Decimal
    external_value: Decimal
    difference: Decimal
    timestamp: datetime
    details: Dict
    resolution_status: ResolutionStatus = ResolutionStatus.PENDING
    resolution_details: Optional[str] = None
    resolved_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Validate discrepancy data"""
        if not isinstance(self.internal_value, Decimal):
            raise TypeError(f"Internal value must be Decimal, got {type(self.internal_value)}")
        if not isinstance(self.external_value, Decimal):
            raise TypeError(f"External value must be Decimal, got {type(self.external_value)}")
        if not isinstance(self.difference, Decimal):
            raise TypeError(f"Difference must be Decimal, got {type(self.difference)}")


class ReconciliationEngine:
    """
    Post-trade reconciliation system

    Features:
    - Exchange vs internal position reconciliation
    - Trade break detection and resolution
    - Daily P&L reconciliation
    - Fee reconciliation
    - Automated discrepancy detection
    - Multi-level alert system
    - Automated correction workflows
    - Comprehensive audit trail
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize reconciliation engine

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)

        # Discrepancies stored as polars DataFrame
        self.discrepancies = pl.DataFrame(
            schema={
                "discrepancy_id": pl.Utf8,
                "type": pl.Utf8,
                "severity": pl.Utf8,
                "symbol": pl.Utf8,
                "exchange": pl.Utf8,
                "internal_value": pl.Utf8,
                "external_value": pl.Utf8,
                "difference": pl.Utf8,
                "timestamp": pl.Datetime,
                "status": pl.Utf8,
                "resolved_at": pl.Datetime,
            }
        )

        # Position snapshots stored as polars DataFrame
        self.position_snapshots = pl.DataFrame(
            schema={
                "snapshot_id": pl.Utf8,
                "symbol": pl.Utf8,
                "exchange": pl.Utf8,
                "internal_quantity": pl.Utf8,
                "external_quantity": pl.Utf8,
                "internal_value": pl.Utf8,
                "external_value": pl.Utf8,
                "timestamp": pl.Datetime,
            }
        )

        # P&L reconciliation data stored as polars DataFrame
        self.pnl_reconciliation = pl.DataFrame(
            schema={
                "date": pl.Date,
                "symbol": pl.Utf8,
                "internal_pnl": pl.Utf8,
                "external_pnl": pl.Utf8,
                "difference": pl.Utf8,
                "reconciled": pl.Boolean,
                "timestamp": pl.Datetime,
            }
        )

        # Fee reconciliation data stored as polars DataFrame
        self.fee_reconciliation = pl.DataFrame(
            schema={
                "date": pl.Date,
                "exchange": pl.Utf8,
                "internal_fees": pl.Utf8,
                "external_fees": pl.Utf8,
                "difference": pl.Utf8,
                "reconciled": pl.Boolean,
                "timestamp": pl.Datetime,
            }
        )

        # Trade break tracking
        self.trade_breaks: Dict[str, Discrepancy] = {}

        # Alerts tracking
        self.active_alerts: Set[str] = set()

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Metrics
        self.metrics = {
            "total_reconciliations": 0,
            "total_discrepancies": 0,
            "resolved_discrepancies": 0,
            "active_discrepancies": 0,
            "total_alerts": 0,
        }

        # Start background tasks
        self._background_tasks: List[asyncio.Task] = []

        logger.info(
            "ReconciliationEngine initialized",
            extra={
                "reconciliation_interval": self.config["reconciliation_interval_seconds"],
                "auto_correction_enabled": self.config["auto_correction_enabled"],
            },
        )

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """
        Load configuration from YAML files

        Args:
            config_path: Override config path

        Returns:
            Configuration dictionary
        """
        default_config = {
            "reconciliation_interval_seconds": 300,  # 5 minutes
            "position_tolerance_percent": Decimal("0.01"),  # 0.01% tolerance
            "pnl_tolerance_usd": Decimal("1.00"),  # $1 tolerance
            "fee_tolerance_percent": Decimal("1.0"),  # 1% tolerance
            "auto_correction_enabled": True,
            "auto_correction_threshold_usd": Decimal("100.00"),
            "alert_on_discrepancy": True,
            "critical_threshold_usd": Decimal("10000.00"),
            "max_retry_attempts": 3,
            "retry_delay_seconds": 60,
            "daily_reconciliation_hour": 1,  # 1 AM UTC
        }

        if config_path is None:
            # Try to load from default locations
            base_path = Path(__file__).parent.parent.parent.parent
            config_files = [
                base_path / "config" / "bot" / "bot.yaml",
                base_path / "config" / "environments" / "production.yaml",
            ]
        else:
            config_files = [Path(config_path)]

        loaded_config = default_config.copy()

        for config_file in config_files:
            try:
                if config_file.exists():
                    with open(config_file, "r") as f:
                        file_config = yaml.safe_load(f)

                        # Extract monitoring settings
                        if "monitoring" in file_config:
                            mon_config = file_config["monitoring"]
                            if "metrics_export_interval" in mon_config:
                                loaded_config["reconciliation_interval_seconds"] = mon_config[
                                    "metrics_export_interval"
                                ]

                        # Extract backup settings
                        if "backup" in file_config:
                            backup_config = file_config["backup"]
                            loaded_config["backup_enabled"] = backup_config.get("enabled", True)

            except Exception as e:
                logger.warning(f"Failed to load config from {config_file}: {e}")
                continue

        return loaded_config

    async def reconcile_positions(
        self,
        internal_positions: List[Position],
        exchange_positions: Dict[str, Dict[str, Decimal]],
    ) -> List[Discrepancy]:
        """
        Reconcile internal positions against exchange positions

        Args:
            internal_positions: List of internal positions
            exchange_positions: Exchange positions {exchange: {symbol: quantity}}

        Returns:
            List of discrepancies found
        """
        async with self._lock:
            try:
                discrepancies: List[Discrepancy] = []

                # Build internal position map
                internal_map: Dict[str, Dict[str, Decimal]] = {}
                for pos in internal_positions:
                    if pos.exchange not in internal_map:
                        internal_map[pos.exchange] = {}
                    internal_map[pos.exchange][pos.symbol] = pos.quantity

                # Compare positions
                all_exchanges = set(internal_map.keys()) | set(exchange_positions.keys())

                for exchange in all_exchanges:
                    internal_exch = internal_map.get(exchange, {})
                    external_exch = exchange_positions.get(exchange, {})

                    all_symbols = set(internal_exch.keys()) | set(external_exch.keys())

                    for symbol in all_symbols:
                        internal_qty = internal_exch.get(symbol, Decimal("0"))
                        external_qty = external_exch.get(symbol, Decimal("0"))

                        # Check if quantities match within tolerance
                        if not self._is_within_tolerance(
                            internal_qty, external_qty, self.config["position_tolerance_percent"]
                        ):
                            difference = internal_qty - external_qty

                            # Determine severity
                            severity = self._calculate_severity(abs(difference))

                            # Create discrepancy
                            discrepancy = Discrepancy(
                                discrepancy_id=self._generate_discrepancy_id(),
                                discrepancy_type=DiscrepancyType.POSITION_MISMATCH,
                                severity=severity,
                                symbol=symbol,
                                exchange=exchange,
                                internal_value=internal_qty,
                                external_value=external_qty,
                                difference=difference,
                                timestamp=datetime.utcnow(),
                                details={
                                    "internal_quantity": str(internal_qty),
                                    "external_quantity": str(external_qty),
                                    "difference": str(difference),
                                },
                            )

                            discrepancies.append(discrepancy)

                            # Record discrepancy
                            await self._record_discrepancy(discrepancy)

                            # Create alert if enabled
                            if self.config["alert_on_discrepancy"]:
                                await self._create_alert(discrepancy)

                            # Attempt auto-correction if enabled
                            if self.config["auto_correction_enabled"]:
                                await self._auto_correct_position(discrepancy)

                # Create position snapshot
                await self._create_position_snapshot(internal_map, exchange_positions)

                # Update metrics
                self.metrics["total_reconciliations"] += 1
                self.metrics["total_discrepancies"] += len(discrepancies)
                self.metrics["active_discrepancies"] = len(
                    [d for d in discrepancies if d.resolution_status == ResolutionStatus.PENDING]
                )

                logger.info(
                    f"Position reconciliation completed: {len(discrepancies)} discrepancies found"
                )

                return discrepancies

            except Exception as e:
                logger.error(f"Position reconciliation failed: {e}", exc_info=True)
                await self._create_audit_log(
                    operation="reconcile_positions_failed",
                    component="reconciliation_engine",
                    severity="ERROR",
                    details={"error": str(e)},
                )
                return []

    async def reconcile_pnl(
        self,
        date: datetime,
        internal_pnl: Dict[str, Decimal],
        external_pnl: Dict[str, Decimal],
    ) -> List[Discrepancy]:
        """
        Reconcile daily P&L

        Args:
            date: Reconciliation date
            internal_pnl: Internal P&L by symbol
            external_pnl: Exchange P&L by symbol

        Returns:
            List of discrepancies found
        """
        async with self._lock:
            try:
                discrepancies: List[Discrepancy] = []

                all_symbols = set(internal_pnl.keys()) | set(external_pnl.keys())

                for symbol in all_symbols:
                    internal = internal_pnl.get(symbol, Decimal("0"))
                    external = external_pnl.get(symbol, Decimal("0"))

                    # Check if P&L matches within tolerance
                    tolerance = self.config["pnl_tolerance_usd"]
                    difference = internal - external

                    if abs(difference) > tolerance:
                        # Determine severity
                        severity = self._calculate_severity(abs(difference))

                        # Create discrepancy
                        discrepancy = Discrepancy(
                            discrepancy_id=self._generate_discrepancy_id(),
                            discrepancy_type=DiscrepancyType.PNL_DISCREPANCY,
                            severity=severity,
                            symbol=symbol,
                            exchange="ALL",
                            internal_value=internal,
                            external_value=external,
                            difference=difference,
                            timestamp=datetime.utcnow(),
                            details={
                                "date": date.strftime("%Y-%m-%d"),
                                "internal_pnl": str(internal),
                                "external_pnl": str(external),
                                "difference": str(difference),
                            },
                        )

                        discrepancies.append(discrepancy)

                        # Record discrepancy
                        await self._record_discrepancy(discrepancy)

                        # Create alert
                        if self.config["alert_on_discrepancy"]:
                            await self._create_alert(discrepancy)

                    # Record P&L reconciliation
                    await self._record_pnl_reconciliation(
                        date=date,
                        symbol=symbol,
                        internal_pnl=internal,
                        external_pnl=external,
                        difference=difference,
                        reconciled=abs(difference) <= tolerance,
                    )

                logger.info(f"P&L reconciliation completed: {len(discrepancies)} discrepancies found")

                return discrepancies

            except Exception as e:
                logger.error(f"P&L reconciliation failed: {e}", exc_info=True)
                return []

    async def reconcile_fees(
        self,
        date: datetime,
        internal_fees: Dict[str, Decimal],
        external_fees: Dict[str, Decimal],
    ) -> List[Discrepancy]:
        """
        Reconcile trading fees

        Args:
            date: Reconciliation date
            internal_fees: Internal fees by exchange
            external_fees: Exchange-reported fees by exchange

        Returns:
            List of discrepancies found
        """
        async with self._lock:
            try:
                discrepancies: List[Discrepancy] = []

                all_exchanges = set(internal_fees.keys()) | set(external_fees.keys())

                for exchange in all_exchanges:
                    internal = internal_fees.get(exchange, Decimal("0"))
                    external = external_fees.get(exchange, Decimal("0"))

                    # Check if fees match within tolerance percentage
                    if external > Decimal("0"):
                        tolerance_pct = self.config["fee_tolerance_percent"]
                        tolerance = external * (tolerance_pct / Decimal("100"))
                    else:
                        tolerance = Decimal("0.01")  # Minimal tolerance for zero fees

                    difference = internal - external

                    if abs(difference) > tolerance:
                        # Determine severity
                        severity = self._calculate_severity(abs(difference))

                        # Create discrepancy
                        discrepancy = Discrepancy(
                            discrepancy_id=self._generate_discrepancy_id(),
                            discrepancy_type=DiscrepancyType.FEE_MISMATCH,
                            severity=severity,
                            symbol="ALL",
                            exchange=exchange,
                            internal_value=internal,
                            external_value=external,
                            difference=difference,
                            timestamp=datetime.utcnow(),
                            details={
                                "date": date.strftime("%Y-%m-%d"),
                                "internal_fees": str(internal),
                                "external_fees": str(external),
                                "difference": str(difference),
                            },
                        )

                        discrepancies.append(discrepancy)

                        # Record discrepancy
                        await self._record_discrepancy(discrepancy)

                        # Create alert
                        if self.config["alert_on_discrepancy"]:
                            await self._create_alert(discrepancy)

                    # Record fee reconciliation
                    await self._record_fee_reconciliation(
                        date=date,
                        exchange=exchange,
                        internal_fees=internal,
                        external_fees=external,
                        difference=difference,
                        reconciled=abs(difference) <= tolerance,
                    )

                logger.info(f"Fee reconciliation completed: {len(discrepancies)} discrepancies found")

                return discrepancies

            except Exception as e:
                logger.error(f"Fee reconciliation failed: {e}", exc_info=True)
                return []

    async def detect_trade_breaks(
        self,
        internal_trades: pl.DataFrame,
        external_trades: pl.DataFrame,
    ) -> List[Discrepancy]:
        """
        Detect trade breaks between internal and exchange trades

        Args:
            internal_trades: Internal trade records
            external_trades: Exchange trade records

        Returns:
            List of trade break discrepancies
        """
        async with self._lock:
            try:
                discrepancies: List[Discrepancy] = []

                # Find missing trades (in internal but not external)
                internal_ids = set(internal_trades["trade_id"].to_list())
                external_ids = set(external_trades["trade_id"].to_list())

                missing_in_external = internal_ids - external_ids
                missing_in_internal = external_ids - internal_ids

                # Create discrepancies for missing trades
                for trade_id in missing_in_external:
                    trade = internal_trades.filter(pl.col("trade_id") == trade_id).row(0, named=True)

                    discrepancy = Discrepancy(
                        discrepancy_id=self._generate_discrepancy_id(),
                        discrepancy_type=DiscrepancyType.MISSING_TRADE,
                        severity=DiscrepancySeverity.HIGH,
                        symbol=trade["symbol"],
                        exchange=trade["exchange"],
                        internal_value=Decimal(trade["quantity"]),
                        external_value=Decimal("0"),
                        difference=Decimal(trade["quantity"]),
                        timestamp=datetime.utcnow(),
                        details={
                            "trade_id": trade_id,
                            "direction": "missing_in_external",
                            "quantity": trade["quantity"],
                            "price": trade["price"],
                        },
                    )

                    discrepancies.append(discrepancy)
                    self.trade_breaks[discrepancy.discrepancy_id] = discrepancy

                    await self._record_discrepancy(discrepancy)
                    await self._create_alert(discrepancy)

                for trade_id in missing_in_internal:
                    trade = external_trades.filter(pl.col("trade_id") == trade_id).row(0, named=True)

                    discrepancy = Discrepancy(
                        discrepancy_id=self._generate_discrepancy_id(),
                        discrepancy_type=DiscrepancyType.MISSING_TRADE,
                        severity=DiscrepancySeverity.HIGH,
                        symbol=trade["symbol"],
                        exchange=trade["exchange"],
                        internal_value=Decimal("0"),
                        external_value=Decimal(trade["quantity"]),
                        difference=Decimal(trade["quantity"]) * Decimal("-1"),
                        timestamp=datetime.utcnow(),
                        details={
                            "trade_id": trade_id,
                            "direction": "missing_in_internal",
                            "quantity": trade["quantity"],
                            "price": trade["price"],
                        },
                    )

                    discrepancies.append(discrepancy)
                    self.trade_breaks[discrepancy.discrepancy_id] = discrepancy

                    await self._record_discrepancy(discrepancy)
                    await self._create_alert(discrepancy)

                logger.info(f"Trade break detection completed: {len(discrepancies)} breaks found")

                return discrepancies

            except Exception as e:
                logger.error(f"Trade break detection failed: {e}", exc_info=True)
                return []

    async def resolve_discrepancy(
        self, discrepancy_id: str, resolution_details: str, auto: bool = False
    ) -> bool:
        """
        Resolve a discrepancy

        Args:
            discrepancy_id: Discrepancy ID
            resolution_details: Details of resolution
            auto: Whether this is an auto-resolution

        Returns:
            True if successful
        """
        async with self._lock:
            try:
                # Update discrepancy status in DataFrame
                mask = pl.col("discrepancy_id") == discrepancy_id

                if self.discrepancies.filter(mask).height == 0:
                    logger.error(f"Discrepancy {discrepancy_id} not found")
                    return False

                # Update status
                self.discrepancies = self.discrepancies.with_columns(
                    [
                        pl.when(mask)
                        .then(pl.lit(ResolutionStatus.RESOLVED.value))
                        .otherwise(pl.col("status"))
                        .alias("status"),
                        pl.when(mask)
                        .then(pl.lit(datetime.utcnow()))
                        .otherwise(pl.col("resolved_at"))
                        .alias("resolved_at"),
                    ]
                )

                # Update trade breaks if applicable
                if discrepancy_id in self.trade_breaks:
                    self.trade_breaks[discrepancy_id].resolution_status = ResolutionStatus.RESOLVED
                    self.trade_breaks[discrepancy_id].resolution_details = resolution_details
                    self.trade_breaks[discrepancy_id].resolved_at = datetime.utcnow()

                # Update metrics
                self.metrics["resolved_discrepancies"] += 1
                self.metrics["active_discrepancies"] -= 1

                # Create audit log
                await self._create_audit_log(
                    operation="resolve_discrepancy",
                    component="reconciliation_engine",
                    severity="INFO",
                    details={
                        "discrepancy_id": discrepancy_id,
                        "resolution_details": resolution_details,
                        "auto_resolved": auto,
                    },
                )

                logger.info(f"Discrepancy {discrepancy_id} resolved: {resolution_details}")

                return True

            except Exception as e:
                logger.error(f"Failed to resolve discrepancy {discrepancy_id}: {e}", exc_info=True)
                return False

    async def _auto_correct_position(self, discrepancy: Discrepancy) -> bool:
        """
        Attempt to auto-correct position discrepancy

        Args:
            discrepancy: Position discrepancy

        Returns:
            True if correction successful
        """
        try:
            # Only auto-correct if below threshold
            if abs(discrepancy.difference) > self.config["auto_correction_threshold_usd"]:
                logger.info(
                    f"Discrepancy {discrepancy.discrepancy_id} exceeds auto-correction threshold"
                )
                return False

            # Retry logic with exponential backoff
            max_retries = self.config["max_retry_attempts"]
            retry_delay = self.config["retry_delay_seconds"]

            for attempt in range(max_retries):
                try:
                    # Simulate correction (in production, would call exchange API)
                    success = await self._execute_correction(discrepancy)

                    if success:
                        await self.resolve_discrepancy(
                            discrepancy.discrepancy_id,
                            f"Auto-corrected: adjusted position by {discrepancy.difference}",
                            auto=True,
                        )
                        return True

                except Exception as e:
                    logger.error(f"Auto-correction attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

            # Auto-correction failed
            logger.warning(f"Auto-correction failed for discrepancy {discrepancy.discrepancy_id}")
            return False

        except Exception as e:
            logger.error(f"Auto-correction error: {e}", exc_info=True)
            return False

    async def _execute_correction(self, discrepancy: Discrepancy) -> bool:
        """
        Execute position correction (placeholder for actual implementation)

        Args:
            discrepancy: Discrepancy to correct

        Returns:
            True if successful
        """
        # In production, this would call exchange API to adjust position
        await asyncio.sleep(0.01)  # Simulate API call
        return True

    def _is_within_tolerance(
        self, internal: Decimal, external: Decimal, tolerance_pct: Decimal
    ) -> bool:
        """
        Check if values are within tolerance

        Args:
            internal: Internal value
            external: External value
            tolerance_pct: Tolerance percentage

        Returns:
            True if within tolerance
        """
        if internal == external:
            return True

        if external == Decimal("0"):
            return internal == Decimal("0")

        deviation = abs((internal - external) / external * Decimal("100"))
        return deviation <= tolerance_pct

    def _calculate_severity(self, amount: Decimal) -> DiscrepancySeverity:
        """
        Calculate discrepancy severity based on amount

        Args:
            amount: Discrepancy amount

        Returns:
            Severity level
        """
        critical_threshold = self.config["critical_threshold_usd"]

        if amount >= critical_threshold:
            return DiscrepancySeverity.CRITICAL
        elif amount >= critical_threshold / Decimal("2"):
            return DiscrepancySeverity.HIGH
        elif amount >= critical_threshold / Decimal("10"):
            return DiscrepancySeverity.MEDIUM
        else:
            return DiscrepancySeverity.LOW

    async def _record_discrepancy(self, discrepancy: Discrepancy) -> None:
        """
        Record discrepancy in database

        Args:
            discrepancy: Discrepancy to record
        """
        try:
            new_row = pl.DataFrame(
                {
                    "discrepancy_id": [discrepancy.discrepancy_id],
                    "type": [discrepancy.discrepancy_type.value],
                    "severity": [discrepancy.severity.value],
                    "symbol": [discrepancy.symbol],
                    "exchange": [discrepancy.exchange],
                    "internal_value": [str(discrepancy.internal_value)],
                    "external_value": [str(discrepancy.external_value)],
                    "difference": [str(discrepancy.difference)],
                    "timestamp": [discrepancy.timestamp],
                    "status": [discrepancy.resolution_status.value],
                    "resolved_at": [discrepancy.resolved_at],
                }
            )

            self.discrepancies = pl.concat([self.discrepancies, new_row])

        except Exception as e:
            logger.error(f"Failed to record discrepancy: {e}", exc_info=True)

    async def _create_position_snapshot(
        self,
        internal_map: Dict[str, Dict[str, Decimal]],
        external_map: Dict[str, Dict[str, Decimal]],
    ) -> None:
        """
        Create position snapshot for historical tracking

        Args:
            internal_map: Internal positions
            external_map: External positions
        """
        try:
            timestamp = datetime.utcnow()
            all_exchanges = set(internal_map.keys()) | set(external_map.keys())

            rows = []
            for exchange in all_exchanges:
                internal_exch = internal_map.get(exchange, {})
                external_exch = external_map.get(exchange, {})

                all_symbols = set(internal_exch.keys()) | set(external_exch.keys())

                for symbol in all_symbols:
                    internal_qty = internal_exch.get(symbol, Decimal("0"))
                    external_qty = external_exch.get(symbol, Decimal("0"))

                    rows.append(
                        {
                            "snapshot_id": str(uuid4()),
                            "symbol": symbol,
                            "exchange": exchange,
                            "internal_quantity": str(internal_qty),
                            "external_quantity": str(external_qty),
                            "internal_value": str(internal_qty),
                            "external_value": str(external_qty),
                            "timestamp": timestamp,
                        }
                    )

            if rows:
                new_snapshot = pl.DataFrame(rows)
                self.position_snapshots = pl.concat([self.position_snapshots, new_snapshot])

        except Exception as e:
            logger.error(f"Failed to create position snapshot: {e}", exc_info=True)

    async def _record_pnl_reconciliation(
        self,
        date: datetime,
        symbol: str,
        internal_pnl: Decimal,
        external_pnl: Decimal,
        difference: Decimal,
        reconciled: bool,
    ) -> None:
        """
        Record P&L reconciliation

        Args:
            date: Reconciliation date
            symbol: Trading symbol
            internal_pnl: Internal P&L
            external_pnl: External P&L
            difference: Difference
            reconciled: Whether reconciled
        """
        try:
            new_row = pl.DataFrame(
                {
                    "date": [date.date()],
                    "symbol": [symbol],
                    "internal_pnl": [str(internal_pnl)],
                    "external_pnl": [str(external_pnl)],
                    "difference": [str(difference)],
                    "reconciled": [reconciled],
                    "timestamp": [datetime.utcnow()],
                }
            )

            self.pnl_reconciliation = pl.concat([self.pnl_reconciliation, new_row])

        except Exception as e:
            logger.error(f"Failed to record P&L reconciliation: {e}", exc_info=True)

    async def _record_fee_reconciliation(
        self,
        date: datetime,
        exchange: str,
        internal_fees: Decimal,
        external_fees: Decimal,
        difference: Decimal,
        reconciled: bool,
    ) -> None:
        """
        Record fee reconciliation

        Args:
            date: Reconciliation date
            exchange: Exchange name
            internal_fees: Internal fees
            external_fees: External fees
            difference: Difference
            reconciled: Whether reconciled
        """
        try:
            new_row = pl.DataFrame(
                {
                    "date": [date.date()],
                    "exchange": [exchange],
                    "internal_fees": [str(internal_fees)],
                    "external_fees": [str(external_fees)],
                    "difference": [str(difference)],
                    "reconciled": [reconciled],
                    "timestamp": [datetime.utcnow()],
                }
            )

            self.fee_reconciliation = pl.concat([self.fee_reconciliation, new_row])

        except Exception as e:
            logger.error(f"Failed to record fee reconciliation: {e}", exc_info=True)

    async def _create_alert(self, discrepancy: Discrepancy) -> None:
        """
        Create alert for discrepancy

        Args:
            discrepancy: Discrepancy to alert on
        """
        try:
            alert_id = f"ALERT_{discrepancy.discrepancy_id}"

            if alert_id in self.active_alerts:
                return

            self.active_alerts.add(alert_id)
            self.metrics["total_alerts"] += 1

            # Create audit log for alert
            await self._create_audit_log(
                operation="create_alert",
                component="reconciliation_engine",
                severity=discrepancy.severity.value,
                details={
                    "alert_id": alert_id,
                    "discrepancy_id": discrepancy.discrepancy_id,
                    "discrepancy_type": discrepancy.discrepancy_type.value,
                    "symbol": discrepancy.symbol,
                    "exchange": discrepancy.exchange,
                    "difference": str(discrepancy.difference),
                },
            )

            logger.warning(
                f"Alert created: {discrepancy.discrepancy_type.value} - "
                f"{discrepancy.symbol} on {discrepancy.exchange}, "
                f"difference: {discrepancy.difference}"
            )

        except Exception as e:
            logger.error(f"Failed to create alert: {e}", exc_info=True)

    async def _create_audit_log(
        self, operation: str, component: str, severity: str, details: Dict
    ) -> None:
        """
        Create audit log entry

        Args:
            operation: Operation name
            component: Component name
            severity: Log severity
            details: Additional details
        """
        try:
            audit_log = AuditLog(
                timestamp=datetime.utcnow(),
                operation=operation,
                user_id="reconciliation_engine",
                component=component,
                severity=severity,
                details=details,
            )

            logger.log(
                getattr(logging, severity),
                f"Audit: {operation}",
                extra={"audit_log": audit_log},
            )

        except Exception as e:
            logger.error(f"Failed to create audit log: {e}", exc_info=True)

    def _generate_discrepancy_id(self) -> str:
        """
        Generate unique discrepancy ID

        Returns:
            Discrepancy ID
        """
        return f"DISC_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"

    def get_discrepancies(
        self,
        status: Optional[ResolutionStatus] = None,
        severity: Optional[DiscrepancySeverity] = None,
        limit: int = 100,
    ) -> pl.DataFrame:
        """
        Get discrepancies with optional filters

        Args:
            status: Filter by resolution status
            severity: Filter by severity
            limit: Maximum number to return

        Returns:
            DataFrame of discrepancies
        """
        df = self.discrepancies

        if status:
            df = df.filter(pl.col("status") == status.value)

        if severity:
            df = df.filter(pl.col("severity") == severity.value)

        return df.sort("timestamp", descending=True).head(limit)

    def get_position_snapshots(
        self, symbol: Optional[str] = None, limit: int = 100
    ) -> pl.DataFrame:
        """
        Get position snapshots

        Args:
            symbol: Filter by symbol
            limit: Maximum number to return

        Returns:
            DataFrame of position snapshots
        """
        df = self.position_snapshots

        if symbol:
            df = df.filter(pl.col("symbol") == symbol)

        return df.sort("timestamp", descending=True).head(limit)

    def get_pnl_reconciliation(self, date: Optional[datetime] = None) -> pl.DataFrame:
        """
        Get P&L reconciliation data

        Args:
            date: Filter by date

        Returns:
            DataFrame of P&L reconciliation
        """
        df = self.pnl_reconciliation

        if date:
            df = df.filter(pl.col("date") == date.date())

        return df.sort("timestamp", descending=True)

    def get_fee_reconciliation(self, date: Optional[datetime] = None) -> pl.DataFrame:
        """
        Get fee reconciliation data

        Args:
            date: Filter by date

        Returns:
            DataFrame of fee reconciliation
        """
        df = self.fee_reconciliation

        if date:
            df = df.filter(pl.col("date") == date.date())

        return df.sort("timestamp", descending=True)

    def get_metrics(self) -> Dict:
        """
        Get reconciliation metrics

        Returns:
            Dictionary of metrics
        """
        return {
            "total_reconciliations": self.metrics["total_reconciliations"],
            "total_discrepancies": self.metrics["total_discrepancies"],
            "resolved_discrepancies": self.metrics["resolved_discrepancies"],
            "active_discrepancies": self.metrics["active_discrepancies"],
            "total_alerts": self.metrics["total_alerts"],
            "active_alerts": len(self.active_alerts),
        }
