"""
Real-Time Risk Monitor
CRITICAL: Real-time risk monitoring dashboard with alerts
"""

import logging
import asyncio
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from collections import deque

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class RiskMonitor:
    """Real-time risk monitoring with alerts"""

    def __init__(self):
        """Initialize risk monitor with config"""
        self.config = get_config()

        # Load monitoring parameters
        self.check_interval_seconds = self.config.get_int("risk", "monitoring.check_interval_seconds")
        self.alert_cooldown_seconds = self.config.get_int("risk", "monitoring.alert_cooldown_seconds")
        self.metrics_retention_days = self.config.get_int("risk", "monitoring.metrics_retention_days")

        # Monitoring state
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None

        # Metrics history (using deque for efficient append/pop)
        self.metrics_history: deque = deque(maxlen=10000)

        # Alert tracking
        self.last_alert_time: Dict[str, float] = {}
        self.alert_callbacks: List[Callable] = []

        # Performance tracking
        self.monitoring_stats = {
            'start_time': None,
            'checks_performed': 0,
            'alerts_sent': 0,
            'errors_encountered': 0
        }

        logger.info(
            f"RiskMonitor initialized: check_interval={self.check_interval_seconds}s, "
            f"alert_cooldown={self.alert_cooldown_seconds}s"
        )

    async def start_monitoring(
        self,
        portfolio_data_callback: Callable,
        market_data_callback: Callable,
        alert_callback: Optional[Callable] = None
    ) -> None:
        """
        Start real-time risk monitoring

        Args:
            portfolio_data_callback: Async function that returns current portfolio data
            market_data_callback: Async function that returns current market data
            alert_callback: Optional async function called when alerts are triggered
        """
        try:
            if self.is_monitoring:
                logger.warning("Risk monitoring already running")
                return

            logger.info("Starting real-time risk monitoring")

            # Register alert callback
            if alert_callback:
                self.alert_callbacks.append(alert_callback)

            # Start monitoring
            self.is_monitoring = True
            self.monitoring_stats['start_time'] = datetime.now().timestamp()

            # Create monitoring task
            self.monitor_task = asyncio.create_task(
                self._monitoring_loop(portfolio_data_callback, market_data_callback)
            )

            logger.info("Risk monitoring started successfully")

        except Exception as e:
            logger.error(f"Error starting risk monitoring: {e}", exc_info=True)
            self.is_monitoring = False
            raise

    async def stop_monitoring(self) -> None:
        """Stop real-time risk monitoring"""
        try:
            if not self.is_monitoring:
                logger.warning("Risk monitoring not running")
                return

            logger.info("Stopping real-time risk monitoring")

            self.is_monitoring = False

            # Cancel monitoring task
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    logger.debug("Monitoring task cancelled")

            # Log final stats
            uptime = datetime.now().timestamp() - self.monitoring_stats['start_time']
            logger.info(
                f"Risk monitoring stopped: "
                f"uptime={uptime:.1f}s, "
                f"checks={self.monitoring_stats['checks_performed']}, "
                f"alerts={self.monitoring_stats['alerts_sent']}, "
                f"errors={self.monitoring_stats['errors_encountered']}"
            )

        except Exception as e:
            logger.error(f"Error stopping risk monitoring: {e}", exc_info=True)
            raise

    async def get_real_time_metrics(self) -> Dict[str, any]:
        """
        Get current real-time risk metrics

        Returns:
            Dictionary with current metrics and status
        """
        try:
            if not self.metrics_history:
                logger.warning("No metrics available yet")
                return {
                    'status': 'no_data',
                    'message': 'No metrics collected yet',
                    'timestamp': datetime.now().timestamp()
                }

            # Get latest metrics
            latest_metrics = self.metrics_history[-1]

            # Calculate monitoring stats
            uptime = None
            if self.monitoring_stats['start_time']:
                uptime = datetime.now().timestamp() - self.monitoring_stats['start_time']

            return {
                'status': 'active' if self.is_monitoring else 'stopped',
                'latest_metrics': latest_metrics,
                'monitoring_stats': {
                    'uptime_seconds': uptime,
                    'checks_performed': self.monitoring_stats['checks_performed'],
                    'alerts_sent': self.monitoring_stats['alerts_sent'],
                    'errors_encountered': self.monitoring_stats['errors_encountered'],
                    'metrics_count': len(self.metrics_history)
                },
                'timestamp': datetime.now().timestamp()
            }

        except Exception as e:
            logger.error(f"Error getting real-time metrics: {e}", exc_info=True)
            raise

    async def send_alerts(
        self,
        alert_type: str,
        severity: str,
        message: str,
        data: Optional[Dict] = None
    ) -> None:
        """
        Send risk alerts to registered callbacks

        Args:
            alert_type: Type of alert (e.g., 'limit_breach', 'volatility_spike')
            severity: Alert severity ('info', 'warning', 'critical', 'halt')
            message: Human-readable alert message
            data: Optional additional data
        """
        try:
            # Check alert cooldown
            current_time = datetime.now().timestamp()
            last_alert = self.last_alert_time.get(alert_type, 0)

            if current_time - last_alert < self.alert_cooldown_seconds:
                logger.debug(
                    f"Alert {alert_type} suppressed due to cooldown "
                    f"({current_time - last_alert:.1f}s < {self.alert_cooldown_seconds}s)"
                )
                return

            # Create alert
            alert = {
                'alert_type': alert_type,
                'severity': severity,
                'message': message,
                'data': data or {},
                'timestamp': current_time
            }

            # Log alert
            log_func = {
                'info': logger.info,
                'warning': logger.warning,
                'critical': logger.error,
                'halt': logger.critical
            }.get(severity, logger.info)

            log_func(f"RISK ALERT [{severity.upper()}] {alert_type}: {message}")

            # Send to callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}", exc_info=True)

            # Update tracking
            self.last_alert_time[alert_type] = current_time
            self.monitoring_stats['alerts_sent'] += 1

        except Exception as e:
            logger.error(f"Error sending alerts: {e}", exc_info=True)
            raise

    async def _monitoring_loop(
        self,
        portfolio_data_callback: Callable,
        market_data_callback: Callable
    ) -> None:
        """Main monitoring loop"""
        try:
            logger.info("Risk monitoring loop started")

            while self.is_monitoring:
                try:
                    # Get current data
                    portfolio_data = await portfolio_data_callback()
                    market_data = await market_data_callback()

                    # Calculate metrics
                    metrics = await self._calculate_monitoring_metrics(
                        portfolio_data,
                        market_data
                    )

                    # Store metrics
                    self.metrics_history.append(metrics)

                    # Check for alerts
                    await self._check_for_alerts(metrics)

                    # Update stats
                    self.monitoring_stats['checks_performed'] += 1

                    # Sleep until next check
                    await asyncio.sleep(self.check_interval_seconds)

                except asyncio.CancelledError:
                    logger.info("Monitoring loop cancelled")
                    break

                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                    self.monitoring_stats['errors_encountered'] += 1

                    # Continue monitoring despite errors
                    await asyncio.sleep(self.check_interval_seconds)

        except Exception as e:
            logger.error(f"Fatal error in monitoring loop: {e}", exc_info=True)
            self.is_monitoring = False
            raise

    async def _calculate_monitoring_metrics(
        self,
        portfolio_data: Dict[str, any],
        market_data: pl.DataFrame
    ) -> Dict[str, any]:
        """Calculate metrics for monitoring"""
        try:
            portfolio_value = portfolio_data.get('portfolio_value', Decimal("0"))
            positions = portfolio_data.get('positions', {})
            prices = portfolio_data.get('prices', {})

            # Calculate basic metrics
            total_position_value = sum(
                abs(qty * prices.get(symbol, Decimal("0")))
                for symbol, qty in positions.items()
            )

            leverage = total_position_value / portfolio_value if portfolio_value > 0 else Decimal("0")

            # Calculate concentration
            max_concentration = Decimal("0")
            if positions and portfolio_value > 0:
                for symbol, qty in positions.items():
                    price = prices.get(symbol, Decimal("0"))
                    position_value = abs(qty * price)
                    concentration = position_value / portfolio_value
                    max_concentration = max(max_concentration, concentration)

            # Calculate volatility from recent market data
            volatility = Decimal("0")
            if not market_data.is_empty() and 'returns' in market_data.columns:
                returns = market_data.select('returns').to_numpy().flatten()
                if len(returns) > 1:
                    vol = float(np.std(returns, ddof=1))
                    volatility = Decimal(str(vol)).quantize(Decimal("0.000001"))

            metrics = {
                'timestamp': datetime.now().timestamp(),
                'portfolio_value': portfolio_value.quantize(Decimal("0.01")),
                'total_position_value': total_position_value.quantize(Decimal("0.01")),
                'leverage': leverage.quantize(Decimal("0.01")),
                'num_positions': len(positions),
                'max_concentration': max_concentration.quantize(Decimal("0.0001")),
                'volatility': volatility
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating monitoring metrics: {e}", exc_info=True)
            return {
                'timestamp': datetime.now().timestamp(),
                'error': str(e)
            }

    async def _check_for_alerts(self, metrics: Dict[str, any]) -> None:
        """Check metrics and trigger alerts if needed"""
        try:
            # Check leverage
            leverage = metrics.get('leverage', Decimal("0"))
            max_leverage = self.config.get_decimal("risk", "position_limits.max_leverage")

            if leverage >= max_leverage:
                await self.send_alerts(
                    alert_type='leverage_breach',
                    severity='critical',
                    message=f"Leverage {leverage:.2f}x exceeds maximum {max_leverage:.2f}x",
                    data={'leverage': float(leverage), 'max_leverage': float(max_leverage)}
                )
            elif leverage >= max_leverage * Decimal("0.9"):
                await self.send_alerts(
                    alert_type='leverage_warning',
                    severity='warning',
                    message=f"Leverage {leverage:.2f}x approaching maximum {max_leverage:.2f}x",
                    data={'leverage': float(leverage), 'max_leverage': float(max_leverage)}
                )

            # Check concentration
            max_concentration = metrics.get('max_concentration', Decimal("0"))
            max_concentration_limit = self.config.get_decimal("risk", "position_limits.max_concentration_pct")

            if max_concentration >= max_concentration_limit:
                await self.send_alerts(
                    alert_type='concentration_breach',
                    severity='critical',
                    message=f"Position concentration {max_concentration:.2%} exceeds limit {max_concentration_limit:.2%}",
                    data={'concentration': float(max_concentration), 'limit': float(max_concentration_limit)}
                )

            # Check volatility
            volatility = metrics.get('volatility', Decimal("0"))
            if len(self.metrics_history) >= 20:
                # Calculate average volatility
                recent_vols = [m.get('volatility', Decimal("0")) for m in list(self.metrics_history)[-20:]]
                avg_vol = sum(recent_vols) / len(recent_vols) if recent_vols else Decimal("0")

                # Alert if current volatility is 2x average
                if avg_vol > 0 and volatility > avg_vol * Decimal("2.0"):
                    await self.send_alerts(
                        alert_type='volatility_spike',
                        severity='warning',
                        message=f"Volatility spike detected: {volatility:.4f} (avg: {avg_vol:.4f})",
                        data={'current_volatility': float(volatility), 'average_volatility': float(avg_vol)}
                    )

        except Exception as e:
            logger.error(f"Error checking for alerts: {e}", exc_info=True)

    async def get_metrics_history(
        self,
        minutes: int = 60
    ) -> List[Dict]:
        """
        Get metrics history for specified time period

        Args:
            minutes: Number of minutes of history to return

        Returns:
            List of metrics dictionaries
        """
        try:
            if not self.metrics_history:
                return []

            cutoff_time = datetime.now().timestamp() - (minutes * 60)

            # Filter metrics by timestamp
            recent_metrics = [
                m for m in self.metrics_history
                if m.get('timestamp', 0) >= cutoff_time
            ]

            logger.debug(f"Retrieved {len(recent_metrics)} metrics from last {minutes} minutes")

            return recent_metrics

        except Exception as e:
            logger.error(f"Error getting metrics history: {e}", exc_info=True)
            return []

    async def register_alert_callback(self, callback: Callable) -> None:
        """
        Register a callback function for alerts

        Args:
            callback: Async function that accepts alert dictionary
        """
        try:
            if callback not in self.alert_callbacks:
                self.alert_callbacks.append(callback)
                logger.info(f"Registered alert callback: {callback.__name__}")

        except Exception as e:
            logger.error(f"Error registering alert callback: {e}", exc_info=True)
            raise

    async def unregister_alert_callback(self, callback: Callable) -> None:
        """
        Unregister an alert callback

        Args:
            callback: Callback function to remove
        """
        try:
            if callback in self.alert_callbacks:
                self.alert_callbacks.remove(callback)
                logger.info(f"Unregistered alert callback: {callback.__name__}")

        except Exception as e:
            logger.error(f"Error unregistering alert callback: {e}", exc_info=True)
            raise


# Need numpy import for volatility calculation
import numpy as np
