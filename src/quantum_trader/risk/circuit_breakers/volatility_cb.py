"""
Quantum Trader AI - Volatility Circuit Breaker
Production-grade volatility-based circuit breaker system

CRITICAL CONSTRAINTS:
- All numeric values use Decimal, NEVER float
- All data operations use polars DataFrame
- All external calls wrapped in try/except with retry logic
- Complete type hints everywhere
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
from enum import Enum
import yaml
import polars as pl
import numpy as np

from quantum_trader.models import Position, AuditLog
from quantum_trader.database.timeseries import TimeSeriesDB
from quantum_trader.notifications.alerting import AlertManager


logger = logging.getLogger(__name__)


class CircuitBreakerStatus(Enum):
    """Circuit breaker status"""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    TRIGGERED = "TRIGGERED"
    COOLING_DOWN = "COOLING_DOWN"


class VolatilityCircuitBreaker:
    """
    Volatility-based circuit breaker system

    Features:
    - Volatility spike detection
    - VIX threshold monitoring
    - Market volatility regime changes
    - Automatic position reduction
    """

    def __init__(
        self,
        config_path: Path = Path("/home/user/FritzellSama/config/bot/risk.yaml"),
        env_config_path: Path = Path("/home/user/FritzellSama/config/environments/production.yaml")
    ) -> None:
        """Initialize volatility circuit breaker with configuration"""
        self.config = self._load_config(config_path)
        self.env_config = self._load_config(env_config_path)

        # Market conditions configuration
        market_conditions = self.config.get("market_conditions", {})
        self.reduce_on_high_volatility = market_conditions.get("reduce_on_high_volatility", True)
        self.volatility_vix_threshold = Decimal(str(market_conditions.get("volatility_vix_threshold", 30.0)))
        self.volatility_position_multiplier = Decimal(str(market_conditions.get("volatility_position_multiplier", 0.5)))
        self.disable_new_trades_on_stress = market_conditions.get("disable_new_trades_on_stress", True)
        self.stress_indicator_threshold = Decimal(str(market_conditions.get("stress_indicator_threshold", 0.75)))
        self.gap_threshold_percent = Decimal(str(market_conditions.get("gap_threshold_percent", 2.0)))

        # Global risk settings
        global_config = self.config.get("global", {})
        self.loss_pause_duration_seconds = int(global_config.get("loss_pause_duration_seconds", 300))

        # Circuit breaker thresholds
        self.vol_spike_threshold = Decimal('2.0')  # 2x normal volatility triggers warning
        self.vol_extreme_threshold = Decimal('3.0')  # 3x triggers circuit breaker
        self.vol_spike_window = 5  # Days to detect spike
        self.vol_baseline_window = 30  # Days for baseline calculation

        # Position reduction settings
        self.reduction_levels = {
            'WARNING': Decimal('0.8'),   # Reduce to 80% of normal size
            'TRIGGERED': Decimal('0.5')  # Reduce to 50% of normal size
        }

        # Cooldown settings
        self.cooldown_duration_seconds = 1800  # 30 minutes
        self.reset_threshold = Decimal('1.2')  # Reset when vol drops to 1.2x baseline

        # Internal state
        self._status: CircuitBreakerStatus = CircuitBreakerStatus.NORMAL
        self._triggered_at: Optional[datetime] = None
        self._trigger_reason: Optional[str] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_running: bool = False
        self._callbacks: List[Callable] = []

        # Database and alerting
        self.db: Optional[TimeSeriesDB] = None
        self.alert_manager: Optional[AlertManager] = None

        # Retry configuration
        self.retry_attempts = 3
        self.retry_delay_ms = 1000

        logger.info("VolatilityCircuitBreaker initialized")

    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}

    async def initialize(self) -> None:
        """Initialize database connections and components"""
        try:
            self.db = TimeSeriesDB()
            await self.db.connect()

            self.alert_manager = AlertManager()
            await self.alert_manager.initialize()

            logger.info("VolatilityCircuitBreaker initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize VolatilityCircuitBreaker: {e}")
            raise

    async def start_monitoring(self) -> None:
        """Start continuous volatility monitoring"""
        if self._is_running:
            logger.warning("Circuit breaker monitoring already running")
            return

        self._is_running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Volatility circuit breaker monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop volatility monitoring"""
        self._is_running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Volatility circuit breaker monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        check_interval = 60  # Check every minute

        while self._is_running:
            try:
                # Check volatility conditions
                await self.check_volatility_conditions()

                # Check if cooldown period ended
                if self._status == CircuitBreakerStatus.COOLING_DOWN:
                    await self._check_cooldown_completion()

                await asyncio.sleep(check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in circuit breaker monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(check_interval)

    async def check_volatility_conditions(self) -> CircuitBreakerStatus:
        """
        Check current volatility conditions and update circuit breaker status

        Returns:
            Current circuit breaker status
        """
        try:
            # Get major market indices/symbols to monitor
            monitor_symbols = await self._get_monitor_symbols()

            # Calculate volatility metrics
            vol_metrics = await self._calculate_volatility_metrics(monitor_symbols)

            # Detect volatility spikes
            spike_detected, spike_ratio = await self._detect_volatility_spike(vol_metrics)

            # Check VIX levels (if available)
            vix_elevated = await self._check_vix_levels()

            # Detect market gaps
            gap_detected = await self._detect_market_gaps(monitor_symbols)

            # Evaluate conditions and update status
            previous_status = self._status

            if spike_ratio >= self.vol_extreme_threshold or (vix_elevated and spike_ratio >= self.vol_spike_threshold):
                # Trigger circuit breaker
                self._status = CircuitBreakerStatus.TRIGGERED
                self._triggered_at = datetime.utcnow()
                self._trigger_reason = f"Volatility spike {spike_ratio:.2f}x baseline"

                if previous_status != CircuitBreakerStatus.TRIGGERED:
                    await self._on_circuit_breaker_triggered()

            elif spike_ratio >= self.vol_spike_threshold or vix_elevated or gap_detected:
                # Warning level
                if self._status == CircuitBreakerStatus.NORMAL:
                    self._status = CircuitBreakerStatus.WARNING
                    await self._on_warning_triggered()

            elif self._status == CircuitBreakerStatus.WARNING:
                # Check if conditions normalized
                if spike_ratio < self.reset_threshold:
                    self._status = CircuitBreakerStatus.NORMAL
                    await self._on_status_normalized()

            return self._status

        except Exception as e:
            logger.error(f"Error checking volatility conditions: {e}")
            return self._status

    async def _get_monitor_symbols(self) -> List[str]:
        """Get symbols to monitor for circuit breaker"""
        # Monitor major indices/symbols
        # In production, this would be configurable
        return ["SPY", "QQQ", "VIX", "BTC/USDT", "ETH/USDT"]

    async def _calculate_volatility_metrics(
        self,
        symbols: List[str]
    ) -> Dict[str, Dict[str, Decimal]]:
        """
        Calculate volatility metrics for symbols

        Returns:
            Dictionary mapping symbol to metrics
        """
        metrics = {}

        for symbol in symbols:
            try:
                # Calculate current volatility
                current_vol = await self._calculate_current_volatility(symbol, self.vol_spike_window)

                # Calculate baseline volatility
                baseline_vol = await self._calculate_current_volatility(symbol, self.vol_baseline_window)

                # Calculate ratio
                vol_ratio = current_vol / baseline_vol if baseline_vol > 0 else Decimal('1')

                metrics[symbol] = {
                    'current_vol': current_vol,
                    'baseline_vol': baseline_vol,
                    'vol_ratio': vol_ratio
                }

            except Exception as e:
                logger.error(f"Error calculating volatility for {symbol}: {e}")
                continue

        return metrics

    async def _calculate_current_volatility(
        self,
        symbol: str,
        window_days: int
    ) -> Decimal:
        """Calculate volatility over specified window"""
        if not self.db:
            raise RuntimeError("Database not initialized")

        try:
            lookback_date = datetime.utcnow() - timedelta(days=window_days + 5)
            price_df = await self.db.query_ohlcv(
                symbol=symbol,
                start_date=lookback_date,
                end_date=datetime.utcnow(),
                timeframe="1d"
            )

            if len(price_df) < window_days:
                return Decimal('0')

            # Calculate log returns
            close_prices = price_df["close"].to_numpy()
            log_returns = np.log(close_prices[1:] / close_prices[:-1])

            # Calculate standard deviation (annualized)
            std_dev = np.std(log_returns, ddof=1)
            annualized_vol = Decimal(str(std_dev)) * Decimal('252').sqrt()

            return annualized_vol

        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return Decimal('0')

    async def _detect_volatility_spike(
        self,
        vol_metrics: Dict[str, Dict[str, Decimal]]
    ) -> Tuple[bool, Decimal]:
        """
        Detect volatility spikes across monitored symbols

        Returns:
            Tuple of (spike_detected, max_spike_ratio)
        """
        try:
            if not vol_metrics:
                return False, Decimal('1')

            # Get maximum volatility ratio across all symbols
            max_ratio = Decimal('0')
            for symbol, metrics in vol_metrics.items():
                vol_ratio = metrics.get('vol_ratio', Decimal('1'))
                if vol_ratio > max_ratio:
                    max_ratio = vol_ratio

            # Detect spike
            spike_detected = max_ratio >= self.vol_spike_threshold

            return spike_detected, max_ratio

        except Exception as e:
            logger.error(f"Error detecting volatility spike: {e}")
            return False, Decimal('1')

    async def _check_vix_levels(self) -> bool:
        """
        Check if VIX is elevated

        Returns:
            True if VIX is above threshold
        """
        try:
            if not self.db:
                return False

            # Fetch latest VIX data
            vix_df = await self.db.query_latest_price("VIX")

            if len(vix_df) == 0:
                return False

            current_vix = Decimal(str(vix_df["close"][0]))

            return current_vix > self.volatility_vix_threshold

        except Exception as e:
            logger.error(f"Error checking VIX levels: {e}")
            return False

    async def _detect_market_gaps(
        self,
        symbols: List[str]
    ) -> bool:
        """
        Detect significant market gaps

        Returns:
            True if significant gap detected
        """
        try:
            for symbol in symbols:
                if not self.db:
                    continue

                # Get last 2 closes
                recent_df = await self.db.query_ohlcv(
                    symbol=symbol,
                    start_date=datetime.utcnow() - timedelta(days=5),
                    end_date=datetime.utcnow(),
                    timeframe="1d"
                )

                if len(recent_df) < 2:
                    continue

                # Calculate gap
                prev_close = Decimal(str(recent_df["close"][-2]))
                current_open = Decimal(str(recent_df["open"][-1]))

                gap_percent = abs((current_open - prev_close) / prev_close * Decimal('100'))

                if gap_percent > self.gap_threshold_percent:
                    logger.warning(f"Significant gap detected in {symbol}: {gap_percent:.2f}%")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error detecting market gaps: {e}")
            return False

    async def _on_circuit_breaker_triggered(self) -> None:
        """Handle circuit breaker triggered event"""
        logger.critical(f"CIRCUIT BREAKER TRIGGERED: {self._trigger_reason}")

        # Send critical alert
        await self._send_alert(
            "CRITICAL",
            f"Circuit breaker triggered: {self._trigger_reason}",
            {
                "status": self._status.value,
                "triggered_at": self._triggered_at.isoformat() if self._triggered_at else None,
                "reason": self._trigger_reason
            }
        )

        # Create audit log
        await self._create_audit_log(
            "CIRCUIT_BREAKER_TRIGGERED",
            "CRITICAL",
            {"reason": self._trigger_reason}
        )

        # Trigger callbacks
        for callback in self._callbacks:
            try:
                await callback(self._status, self._trigger_reason)
            except Exception as e:
                logger.error(f"Error in circuit breaker callback: {e}")

        # Initiate cooldown
        await self._start_cooldown()

    async def _on_warning_triggered(self) -> None:
        """Handle warning level triggered"""
        logger.warning("Circuit breaker WARNING level activated")

        await self._send_alert(
            "WARNING",
            "High volatility detected - circuit breaker on warning",
            {"status": self._status.value}
        )

        await self._create_audit_log(
            "CIRCUIT_BREAKER_WARNING",
            "WARNING",
            {"status": self._status.value}
        )

    async def _on_status_normalized(self) -> None:
        """Handle status returning to normal"""
        logger.info("Circuit breaker status normalized")

        await self._send_alert(
            "INFO",
            "Volatility conditions normalized - circuit breaker reset",
            {"status": self._status.value}
        )

        await self._create_audit_log(
            "CIRCUIT_BREAKER_NORMALIZED",
            "INFO",
            {"status": self._status.value}
        )

    async def _start_cooldown(self) -> None:
        """Start cooldown period"""
        self._status = CircuitBreakerStatus.COOLING_DOWN
        logger.info(f"Circuit breaker entering cooldown for {self.cooldown_duration_seconds}s")

    async def _check_cooldown_completion(self) -> None:
        """Check if cooldown period is complete"""
        if not self._triggered_at:
            return

        cooldown_end = self._triggered_at + timedelta(seconds=self.cooldown_duration_seconds)

        if datetime.utcnow() >= cooldown_end:
            # Check if conditions normalized
            monitor_symbols = await self._get_monitor_symbols()
            vol_metrics = await self._calculate_volatility_metrics(monitor_symbols)
            _, max_ratio = await self._detect_volatility_spike(vol_metrics)

            if max_ratio < self.reset_threshold:
                self._status = CircuitBreakerStatus.NORMAL
                self._triggered_at = None
                self._trigger_reason = None

                logger.info("Circuit breaker cooldown complete - reset to NORMAL")

                await self._send_alert(
                    "INFO",
                    "Circuit breaker cooldown complete - trading resumed",
                    {"status": self._status.value}
                )
            else:
                # Extend cooldown
                self._triggered_at = datetime.utcnow()
                logger.warning(f"Volatility still elevated ({max_ratio:.2f}x) - extending cooldown")

    async def _send_alert(
        self,
        severity: str,
        message: str,
        metadata: Dict
    ) -> None:
        """Send alert through alert manager"""
        if self.alert_manager:
            try:
                await self.alert_manager.send_alert(
                    severity=severity,
                    message=message,
                    component="VolatilityCircuitBreaker",
                    metadata=metadata
                )
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")

    async def _create_audit_log(
        self,
        operation: str,
        severity: str,
        details: Dict
    ) -> None:
        """Create audit log entry"""
        try:
            audit_log = AuditLog(
                timestamp=datetime.utcnow(),
                operation=operation,
                user_id="system",
                component="VolatilityCircuitBreaker",
                severity=severity,
                details=details,
                result="SUCCESS"
            )

            if self.db:
                await self.db.insert_audit_log(audit_log)

        except Exception as e:
            logger.error(f"Failed to create audit log: {e}")

    def register_callback(self, callback: Callable) -> None:
        """Register callback for circuit breaker events"""
        self._callbacks.append(callback)

    def get_status(self) -> CircuitBreakerStatus:
        """Get current circuit breaker status"""
        return self._status

    def get_position_size_multiplier(self) -> Decimal:
        """
        Get position size multiplier based on current status

        Returns:
            Multiplier to apply to position sizes
        """
        if self._status == CircuitBreakerStatus.TRIGGERED:
            return self.reduction_levels['TRIGGERED']
        elif self._status == CircuitBreakerStatus.WARNING:
            return self.reduction_levels['WARNING']
        elif self._status == CircuitBreakerStatus.COOLING_DOWN:
            return self.reduction_levels['TRIGGERED']  # Stay reduced during cooldown
        else:
            return Decimal('1.0')

    def is_trading_allowed(self) -> bool:
        """
        Check if new trades are allowed

        Returns:
            True if new trades allowed, False if circuit breaker active
        """
        if self._status == CircuitBreakerStatus.TRIGGERED:
            return False
        elif self._status == CircuitBreakerStatus.COOLING_DOWN:
            return False
        else:
            return True

    async def get_circuit_breaker_report(self) -> Dict:
        """
        Generate circuit breaker status report

        Returns:
            Comprehensive status report
        """
        try:
            # Get current volatility metrics
            monitor_symbols = await self._get_monitor_symbols()
            vol_metrics = await self._calculate_volatility_metrics(monitor_symbols)

            # Get VIX level
            vix_elevated = await self._check_vix_levels()

            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'status': self._status.value,
                'trading_allowed': self.is_trading_allowed(),
                'position_size_multiplier': float(self.get_position_size_multiplier()),
                'triggered_at': self._triggered_at.isoformat() if self._triggered_at else None,
                'trigger_reason': self._trigger_reason,
                'vix_elevated': vix_elevated,
                'volatility_metrics': {
                    symbol: {
                        'current_vol': float(metrics['current_vol']),
                        'baseline_vol': float(metrics['baseline_vol']),
                        'vol_ratio': float(metrics['vol_ratio'])
                    }
                    for symbol, metrics in vol_metrics.items()
                }
            }

            return report

        except Exception as e:
            logger.error(f"Error generating circuit breaker report: {e}")
            return {'error': str(e)}

    async def force_trigger(self, reason: str) -> None:
        """
        Manually trigger circuit breaker

        Args:
            reason: Reason for manual trigger
        """
        logger.warning(f"Circuit breaker manually triggered: {reason}")

        self._status = CircuitBreakerStatus.TRIGGERED
        self._triggered_at = datetime.utcnow()
        self._trigger_reason = f"Manual trigger: {reason}"

        await self._on_circuit_breaker_triggered()

    async def force_reset(self) -> None:
        """Manually reset circuit breaker to normal"""
        logger.info("Circuit breaker manually reset")

        self._status = CircuitBreakerStatus.NORMAL
        self._triggered_at = None
        self._trigger_reason = None

        await self._send_alert(
            "WARNING",
            "Circuit breaker manually reset",
            {"status": self._status.value}
        )

    async def cleanup(self) -> None:
        """Cleanup resources"""
        await self.stop_monitoring()

        if self.db:
            await self.db.disconnect()

        if self.alert_manager:
            await self.alert_manager.cleanup()

        logger.info("VolatilityCircuitBreaker cleanup complete")
