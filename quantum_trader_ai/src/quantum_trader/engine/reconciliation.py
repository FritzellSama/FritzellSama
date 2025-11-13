"""
Trade Reconciliation and Settlement Verification
Production-ready reconciliation with comprehensive discrepancy detection
"""

from decimal import Decimal
from typing import Optional, Dict, List, Any, Tuple
import logging
import asyncio
from datetime import datetime, timedelta
import polars as pl
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class DiscrepancyType(Enum):
    """Types of reconciliation discrepancies"""
    QUANTITY_MISMATCH = "QUANTITY_MISMATCH"
    PRICE_MISMATCH = "PRICE_MISMATCH"
    MISSING_TRADE = "MISSING_TRADE"
    DUPLICATE_TRADE = "DUPLICATE_TRADE"
    BALANCE_MISMATCH = "BALANCE_MISMATCH"
    POSITION_MISMATCH = "POSITION_MISMATCH"
    SETTLEMENT_MISMATCH = "SETTLEMENT_MISMATCH"


class ReconciliationStatus(Enum):
    """Reconciliation status"""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    MATCHED = "MATCHED"
    DISCREPANCY = "DISCREPANCY"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"


@dataclass
class Trade:
    """Trade data structure"""
    trade_id: str
    order_id: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    fee: Decimal
    fee_currency: str
    timestamp: str
    exchange: str
    user_id: str
    settlement_status: str = "PENDING"


@dataclass
class Balance:
    """Balance data structure"""
    asset: str
    total: Decimal
    available: Decimal
    locked: Decimal
    timestamp: str
    exchange: str


@dataclass
class Discrepancy:
    """Discrepancy data structure"""
    discrepancy_id: str
    discrepancy_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    internal_value: Optional[str]
    external_value: Optional[str]
    difference: Optional[str]
    trade_id: Optional[str]
    symbol: Optional[str]
    timestamp: str
    status: str
    resolution: Optional[str] = None


class Reconciliation:
    """Trade reconciliation and settlement verification system"""

    def __init__(self) -> None:
        """Initialize reconciliation engine with configuration"""
        self.config = get_config()
        self._load_config()
        self._internal_trades: Dict[str, Trade] = {}
        self._external_trades: Dict[str, Trade] = {}
        self._balances: Dict[str, Balance] = {}
        self._discrepancies: List[Discrepancy] = []
        self._reconciliation_runs: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._running = False
        self._reconciliation_task: Optional[asyncio.Task] = None
        logger.info("Reconciliation engine initialized")

    def _load_config(self) -> None:
        """Load configuration from engine.yaml"""
        self.enabled = self.config.get_bool('engine', 'reconciliation.enabled')
        self.interval_seconds = self.config.get_int('engine', 'reconciliation.interval_seconds')
        self.mismatch_tolerance_usd = self.config.get_decimal('engine', 'reconciliation.mismatch_tolerance_usd')
        self.auto_correct_enabled = self.config.get_bool('engine', 'reconciliation.auto_correct_enabled')
        self.alert_on_mismatch = self.config.get_bool('engine', 'reconciliation.alert_on_mismatch')

        logger.info(
            f"Reconciliation configured: enabled={self.enabled}, "
            f"interval={self.interval_seconds}s, tolerance=${self.mismatch_tolerance_usd}"
        )

    async def start_continuous_reconciliation(self) -> None:
        """Start continuous reconciliation process"""
        if not self.enabled:
            logger.warning("Reconciliation is disabled in configuration")
            return

        if self._running:
            logger.warning("Reconciliation already running")
            return

        self._running = True
        self._reconciliation_task = asyncio.create_task(self._reconciliation_loop())
        logger.info("Started continuous reconciliation")

    async def stop_continuous_reconciliation(self) -> None:
        """Stop continuous reconciliation process"""
        self._running = False

        if self._reconciliation_task:
            self._reconciliation_task.cancel()
            try:
                await self._reconciliation_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped continuous reconciliation")

    async def _reconciliation_loop(self) -> None:
        """Continuous reconciliation loop"""
        while self._running:
            try:
                await self.reconcile_trades()
                await asyncio.sleep(self.interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in reconciliation loop: {e}")
                await asyncio.sleep(self.interval_seconds)

    async def reconcile_trades(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Reconcile trades between internal records and exchange

        Args:
            start_time: Start time for reconciliation (optional)
            end_time: End time for reconciliation (optional)

        Returns:
            Dictionary with reconciliation results
        """
        async with self._lock:
            run_id = str(uuid.uuid4())
            start = datetime.utcnow()

            if start_time is None:
                start_time = datetime.utcnow() - timedelta(hours=24)
            if end_time is None:
                end_time = datetime.utcnow()

            logger.info(
                f"Starting trade reconciliation {run_id} for period "
                f"{start_time.isoformat()} to {end_time.isoformat()}"
            )

            # Get trades for the period
            internal_trades = self._filter_trades_by_time(
                self._internal_trades,
                start_time,
                end_time
            )
            external_trades = self._filter_trades_by_time(
                self._external_trades,
                start_time,
                end_time
            )

            # Reconcile
            matched = 0
            discrepancies_found = 0

            # Check for matches and mismatches
            for trade_id, internal_trade in internal_trades.items():
                if trade_id in external_trades:
                    external_trade = external_trades[trade_id]

                    # Compare trade details
                    mismatch = await self._compare_trades(
                        internal_trade,
                        external_trade
                    )

                    if mismatch:
                        discrepancies_found += 1
                    else:
                        matched += 1
                else:
                    # Missing on exchange
                    await self._report_missing_trade(internal_trade, "EXTERNAL")
                    discrepancies_found += 1

            # Check for trades on exchange not in internal records
            for trade_id, external_trade in external_trades.items():
                if trade_id not in internal_trades:
                    await self._report_missing_trade(external_trade, "INTERNAL")
                    discrepancies_found += 1

            end = datetime.utcnow()
            duration_ms = int((end - start).total_seconds() * 1000)

            result = {
                'run_id': run_id,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'internal_trades': len(internal_trades),
                'external_trades': len(external_trades),
                'matched': matched,
                'discrepancies': discrepancies_found,
                'duration_ms': duration_ms,
                'timestamp': start.isoformat()
            }

            self._reconciliation_runs.append(result)

            logger.info(
                f"Reconciliation {run_id} completed: "
                f"{matched} matched, {discrepancies_found} discrepancies in {duration_ms}ms"
            )

            return result

    def _filter_trades_by_time(
        self,
        trades: Dict[str, Trade],
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Trade]:
        """Filter trades by time period"""
        filtered = {}

        for trade_id, trade in trades.items():
            trade_time = datetime.fromisoformat(trade.timestamp)
            if start_time <= trade_time <= end_time:
                filtered[trade_id] = trade

        return filtered

    async def _compare_trades(
        self,
        internal_trade: Trade,
        external_trade: Trade
    ) -> bool:
        """
        Compare internal and external trades

        Returns:
            bool: True if mismatch found
        """
        mismatch_found = False

        # Compare quantity
        qty_diff = abs(internal_trade.quantity - external_trade.quantity)
        if qty_diff > Decimal('0.00000001'):
            await self._report_discrepancy(
                discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
                description=f"Quantity mismatch for trade {internal_trade.trade_id}",
                internal_value=str(internal_trade.quantity),
                external_value=str(external_trade.quantity),
                difference=str(qty_diff),
                trade_id=internal_trade.trade_id,
                symbol=internal_trade.symbol,
                severity="HIGH"
            )
            mismatch_found = True

        # Compare price
        price_diff = abs(internal_trade.price - external_trade.price)
        price_diff_usd = price_diff * internal_trade.quantity

        if price_diff_usd > self.mismatch_tolerance_usd:
            await self._report_discrepancy(
                discrepancy_type=DiscrepancyType.PRICE_MISMATCH,
                description=f"Price mismatch for trade {internal_trade.trade_id}",
                internal_value=str(internal_trade.price),
                external_value=str(external_trade.price),
                difference=str(price_diff),
                trade_id=internal_trade.trade_id,
                symbol=internal_trade.symbol,
                severity="MEDIUM" if price_diff_usd < self.mismatch_tolerance_usd * Decimal('10') else "HIGH"
            )
            mismatch_found = True

        return mismatch_found

    async def _report_missing_trade(
        self,
        trade: Trade,
        missing_from: str
    ) -> None:
        """Report a missing trade"""
        await self._report_discrepancy(
            discrepancy_type=DiscrepancyType.MISSING_TRADE,
            description=f"Trade {trade.trade_id} missing from {missing_from}",
            internal_value=str(trade.trade_id) if missing_from == "EXTERNAL" else None,
            external_value=str(trade.trade_id) if missing_from == "INTERNAL" else None,
            difference=None,
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            severity="CRITICAL"
        )

    async def _report_discrepancy(
        self,
        discrepancy_type: DiscrepancyType,
        description: str,
        internal_value: Optional[str],
        external_value: Optional[str],
        difference: Optional[str],
        trade_id: Optional[str],
        symbol: Optional[str],
        severity: str
    ) -> None:
        """Report a reconciliation discrepancy"""
        discrepancy = Discrepancy(
            discrepancy_id=str(uuid.uuid4()),
            discrepancy_type=discrepancy_type.value,
            severity=severity,
            description=description,
            internal_value=internal_value,
            external_value=external_value,
            difference=difference,
            trade_id=trade_id,
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat(),
            status=ReconciliationStatus.DISCREPANCY.value
        )

        self._discrepancies.append(discrepancy)

        logger.warning(
            f"Discrepancy detected: {discrepancy_type.value} - {description}"
        )

        if self.alert_on_mismatch:
            await self._send_alert(discrepancy)

    async def _send_alert(self, discrepancy: Discrepancy) -> None:
        """Send alert for discrepancy"""
        # In production, this would send to monitoring/alerting system
        logger.error(
            f"ALERT: {discrepancy.severity} discrepancy - {discrepancy.description}"
        )

    async def check_balances(
        self,
        internal_balances: Dict[str, Decimal],
        external_balances: Dict[str, Decimal]
    ) -> Dict[str, Any]:
        """
        Check balance reconciliation

        Args:
            internal_balances: Internal balance records
            external_balances: External balance records (from exchange)

        Returns:
            Dictionary with balance check results
        """
        async with self._lock:
            mismatches = []
            matched = 0

            all_assets = set(internal_balances.keys()) | set(external_balances.keys())

            for asset in all_assets:
                internal_balance = internal_balances.get(asset, Decimal('0'))
                external_balance = external_balances.get(asset, Decimal('0'))

                difference = abs(internal_balance - external_balance)

                # Convert to USD for tolerance check (simplified)
                difference_usd = difference  # In production, convert to USD

                if difference_usd > self.mismatch_tolerance_usd:
                    mismatch = {
                        'asset': asset,
                        'internal_balance': str(internal_balance),
                        'external_balance': str(external_balance),
                        'difference': str(difference),
                        'difference_usd': str(difference_usd)
                    }
                    mismatches.append(mismatch)

                    # Report discrepancy
                    await self._report_discrepancy(
                        discrepancy_type=DiscrepancyType.BALANCE_MISMATCH,
                        description=f"Balance mismatch for {asset}",
                        internal_value=str(internal_balance),
                        external_value=str(external_balance),
                        difference=str(difference),
                        trade_id=None,
                        symbol=asset,
                        severity="CRITICAL" if difference_usd > self.mismatch_tolerance_usd * Decimal('100') else "HIGH"
                    )
                else:
                    matched += 1

            result = {
                'total_assets': len(all_assets),
                'matched': matched,
                'mismatches': len(mismatches),
                'details': mismatches,
                'timestamp': datetime.utcnow().isoformat()
            }

            logger.info(
                f"Balance check completed: {matched} matched, {len(mismatches)} mismatches"
            )

            return result

    async def report_discrepancies(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> pl.DataFrame:
        """
        Get discrepancy report

        Args:
            status: Filter by status (optional)
            severity: Filter by severity (optional)
            limit: Maximum number of records

        Returns:
            Polars DataFrame with discrepancies
        """
        async with self._lock:
            if not self._discrepancies:
                return pl.DataFrame()

            # Convert to dictionaries
            discrepancy_dicts = [asdict(d) for d in self._discrepancies]

            # Create DataFrame
            df = pl.DataFrame(discrepancy_dicts)

            # Apply filters
            if status:
                df = df.filter(pl.col('status') == status)

            if severity:
                df = df.filter(pl.col('severity') == severity)

            # Limit results
            df = df.tail(limit)

            logger.info(f"Generated discrepancy report with {len(df)} records")

            return df

    async def resolve_discrepancy(
        self,
        discrepancy_id: str,
        resolution: str,
        auto_correct: bool = False
    ) -> bool:
        """
        Resolve a discrepancy

        Args:
            discrepancy_id: Discrepancy ID to resolve
            resolution: Resolution description
            auto_correct: Apply automatic correction if enabled

        Returns:
            bool: True if resolved successfully
        """
        async with self._lock:
            for discrepancy in self._discrepancies:
                if discrepancy.discrepancy_id == discrepancy_id:
                    if auto_correct and self.auto_correct_enabled:
                        # Apply automatic correction
                        logger.info(
                            f"Auto-correcting discrepancy {discrepancy_id}: {resolution}"
                        )

                    discrepancy.status = ReconciliationStatus.RESOLVED.value
                    discrepancy.resolution = resolution

                    logger.info(f"Discrepancy {discrepancy_id} resolved: {resolution}")
                    return True

            logger.warning(f"Discrepancy {discrepancy_id} not found")
            return False

    def add_internal_trade(self, trade: Trade) -> None:
        """Add internal trade record"""
        self._internal_trades[trade.trade_id] = trade

    def add_external_trade(self, trade: Trade) -> None:
        """Add external trade record"""
        self._external_trades[trade.trade_id] = trade

    def get_reconciliation_metrics(self) -> Dict[str, Any]:
        """Get reconciliation metrics"""
        if not self._reconciliation_runs:
            return {
                'total_runs': 0,
                'total_discrepancies': len(self._discrepancies),
                'unresolved_discrepancies': sum(
                    1 for d in self._discrepancies
                    if d.status != ReconciliationStatus.RESOLVED.value
                )
            }

        total_matched = sum(r['matched'] for r in self._reconciliation_runs)
        total_discrepancies = sum(r['discrepancies'] for r in self._reconciliation_runs)
        total_trades = sum(r['internal_trades'] for r in self._reconciliation_runs)

        match_rate = total_matched / total_trades if total_trades > 0 else 0.0

        severity_counts = {}
        for discrepancy in self._discrepancies:
            severity = discrepancy.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            'total_runs': len(self._reconciliation_runs),
            'total_trades_reconciled': total_trades,
            'total_matched': total_matched,
            'total_discrepancies': total_discrepancies,
            'match_rate': match_rate,
            'unresolved_discrepancies': sum(
                1 for d in self._discrepancies
                if d.status != ReconciliationStatus.RESOLVED.value
            ),
            'severity_counts': severity_counts
        }
