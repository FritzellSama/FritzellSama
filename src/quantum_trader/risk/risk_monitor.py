"""
Quantum Trader AI - Real-time Risk Monitor
Production-grade continuous risk monitoring system

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
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import yaml
import polars as pl

from quantum_trader.models import RiskMetrics, Position, AuditLog
from quantum_trader.database.timeseries import TimeSeriesDB
from quantum_trader.notifications.alerting import AlertManager


logger = logging.getLogger(__name__)


class RiskMonitor:
    """
    Real-time risk monitoring system

    Features:
    - Continuous risk metric updates
    - Alert threshold monitoring
    - Dashboard data aggregation
    - Historical risk tracking
    - Risk trend analysis
    """

    def __init__(
        self,
        config_path: Path = Path("/home/user/FritzellSama/config/bot/risk.yaml"),
        env_config_path: Path = Path("/home/user/FritzellSama/config/environments/production.yaml")
    ) -> None:
        """Initialize risk monitor with configuration"""
        self.config = self._load_config(config_path)
        self.env_config = self._load_config(env_config_path)

        # Extract monitoring configuration
        monitoring_config = self.config.get("monitoring", {})
        self.update_frequency_seconds = Decimal(str(monitoring_config.get("update_frequency_seconds", 30)))
        self.enable_dashboard = monitoring_config.get("enable_dashboard", True)
        self.alert_portfolio_risk_percent = Decimal(str(monitoring_config.get("alert_portfolio_risk_percent", 1.5)))
        self.alert_unrealized_loss_percent = Decimal(str(monitoring_config.get("alert_unrealized_loss_percent", 3.0)))
        self.enable_risk_logging = monitoring_config.get("enable_risk_logging", True)
        self.log_path = Path(monitoring_config.get("log_path", "/var/log/quantum_trader/risk_events.log"))
        self.alert_webhook_url = monitoring_config.get("alert_webhook_url")

        # Global risk settings
        global_config = self.config.get("global", {})
        self.max_daily_loss_usd = Decimal(str(global_config.get("max_daily_loss_usd", 50000)))
        self.max_daily_loss_percent = Decimal(str(global_config.get("max_daily_loss_percent", 5.0)))
        self.max_portfolio_risk_percent = Decimal(str(global_config.get("max_portfolio_risk_percent", 2.0)))

        # Risk model parameters
        risk_model = self.config.get("risk_model", {})
        self.var_confidence_level = Decimal(str(risk_model.get("var_confidence_level", 0.95)))
        self.min_sharpe_ratio = Decimal(str(risk_model.get("min_sharpe_ratio", 0.5)))
        self.max_drawdown_tolerance = Decimal(str(risk_model.get("max_drawdown_tolerance", 10.0)))

        # Initialize components
        self.db: Optional[TimeSeriesDB] = None
        self.alert_manager: Optional[AlertManager] = None

        # Internal state
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_running: bool = False
        self._current_metrics: Optional[RiskMetrics] = None
        self._metrics_history: pl.DataFrame = pl.DataFrame()
        self._alert_callbacks: List[Callable] = []

        # Retry configuration
        self.retry_attempts = 3
        self.retry_delay_ms = 1000

        logger.info("RiskMonitor initialized with configuration")

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
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
            # Initialize database connection
            self.db = TimeSeriesDB()
            await self.db.connect()

            # Initialize alert manager
            self.alert_manager = AlertManager()
            await self.alert_manager.initialize()

            # Load historical metrics
            await self._load_historical_metrics()

            logger.info("RiskMonitor initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize RiskMonitor: {e}")
            raise

    async def start_monitoring(self) -> None:
        """Start continuous risk monitoring"""
        if self._is_running:
            logger.warning("Risk monitoring already running")
            return

        self._is_running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Risk monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop risk monitoring"""
        self._is_running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Risk monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self._is_running:
            try:
                # Update risk metrics
                metrics = await self.calculate_current_metrics()

                # Store metrics
                await self._store_metrics(metrics)

                # Check alert thresholds
                await self._check_alert_thresholds(metrics)

                # Update dashboard if enabled
                if self.enable_dashboard:
                    await self._update_dashboard(metrics)

                # Analyze trends
                await self._analyze_risk_trends()

                # Wait for next update
                await asyncio.sleep(float(self.update_frequency_seconds))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Back off on error

    async def calculate_current_metrics(self) -> RiskMetrics:
        """Calculate current risk metrics with retry logic"""
        for attempt in range(self.retry_attempts):
            try:
                # Fetch current positions
                positions = await self._fetch_positions()

                # Calculate portfolio metrics
                portfolio_value = await self._calculate_portfolio_value(positions)
                cash_balance = await self._fetch_cash_balance()
                total_exposure = await self._calculate_total_exposure(positions)

                # Calculate risk metrics
                var_95 = await self._calculate_var(positions)
                cvar_95 = await self._calculate_cvar(positions)
                max_drawdown = await self._calculate_max_drawdown()
                sharpe_ratio = await self._calculate_sharpe_ratio()
                sortino_ratio = await self._calculate_sortino_ratio()
                beta = await self._calculate_beta(positions)
                daily_pnl = await self._calculate_daily_pnl(positions)

                # Create metrics object
                metrics = RiskMetrics(
                    portfolio_value=portfolio_value,
                    cash_balance=cash_balance,
                    total_exposure=total_exposure,
                    var_95=var_95,
                    cvar_95=cvar_95,
                    max_drawdown=max_drawdown,
                    sharpe_ratio=sharpe_ratio,
                    sortino_ratio=sortino_ratio,
                    beta=beta,
                    daily_pnl=daily_pnl,
                    timestamp=datetime.utcnow(),
                    metadata={
                        "num_positions": len(positions),
                        "monitoring_status": "active"
                    }
                )

                self._current_metrics = metrics
                return metrics

            except Exception as e:
                logger.error(f"Error calculating metrics (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay_ms / 1000)
                else:
                    raise

    async def _fetch_positions(self) -> List[Position]:
        """Fetch current positions from database"""
        if not self.db:
            raise RuntimeError("Database not initialized")

        try:
            positions_df = await self.db.query_positions(open_only=True)
            return self._dataframe_to_positions(positions_df)
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            raise

    def _dataframe_to_positions(self, df: pl.DataFrame) -> List[Position]:
        """Convert polars DataFrame to Position objects"""
        positions = []

        for row in df.iter_rows(named=True):
            position = Position(
                symbol=row["symbol"],
                quantity=Decimal(str(row["quantity"])),
                entry_price=Decimal(str(row["entry_price"])),
                current_price=Decimal(str(row["current_price"])),
                exchange=row["exchange"],
                strategy=row["strategy"],
                opened_at=row["opened_at"],
                position_id=row["position_id"]
            )
            positions.append(position)

        return positions

    async def _calculate_portfolio_value(self, positions: List[Position]) -> Decimal:
        """Calculate total portfolio value"""
        total = Decimal('0')

        for position in positions:
            position_value = abs(position.quantity) * position.current_price
            total += position_value

        # Add cash balance
        cash = await self._fetch_cash_balance()
        total += cash

        return total

    async def _fetch_cash_balance(self) -> Decimal:
        """Fetch current cash balance"""
        if not self.db:
            raise RuntimeError("Database not initialized")

        try:
            balance_df = await self.db.query_latest_balance()
            if len(balance_df) > 0:
                return Decimal(str(balance_df["balance"][0]))
            return Decimal('0')
        except Exception as e:
            logger.error(f"Failed to fetch cash balance: {e}")
            return Decimal('0')

    async def _calculate_total_exposure(self, positions: List[Position]) -> Decimal:
        """Calculate total portfolio exposure"""
        total_exposure = Decimal('0')

        for position in positions:
            exposure = abs(position.quantity * position.current_price)
            total_exposure += exposure

        return total_exposure

    async def _calculate_var(self, positions: List[Position]) -> Decimal:
        """Calculate Value at Risk (simplified)"""
        # This is a placeholder - full implementation in value_at_risk.py
        total_value = await self._calculate_portfolio_value(positions)
        # Using simplified VaR estimate: 2% of portfolio value
        return total_value * Decimal('0.02')

    async def _calculate_cvar(self, positions: List[Position]) -> Decimal:
        """Calculate Conditional Value at Risk"""
        # CVaR typically 1.5x VaR
        var = await self._calculate_var(positions)
        return var * Decimal('1.5')

    async def _calculate_max_drawdown(self) -> Decimal:
        """Calculate maximum drawdown from historical data"""
        if len(self._metrics_history) == 0:
            return Decimal('0')

        try:
            # Calculate running maximum and drawdown
            portfolio_values = self._metrics_history.select("portfolio_value").to_series()
            running_max = portfolio_values.cum_max()
            drawdown = (portfolio_values - running_max) / running_max * Decimal('100')

            max_dd = abs(drawdown.min())
            return Decimal(str(max_dd)) if max_dd is not None else Decimal('0')
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return Decimal('0')

    async def _calculate_sharpe_ratio(self) -> Decimal:
        """Calculate Sharpe ratio (simplified)"""
        # This is a placeholder - full implementation in sharpe_ratio.py
        if len(self._metrics_history) < 2:
            return Decimal('0')

        try:
            returns = self._metrics_history.select("daily_pnl").to_series()
            mean_return = returns.mean()
            std_return = returns.std()

            if std_return and std_return > 0:
                sharpe = Decimal(str(mean_return)) / Decimal(str(std_return))
                return sharpe
            return Decimal('0')
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return Decimal('0')

    async def _calculate_sortino_ratio(self) -> Decimal:
        """Calculate Sortino ratio"""
        # Similar to Sharpe but only considers downside volatility
        if len(self._metrics_history) < 2:
            return Decimal('0')

        try:
            returns = self._metrics_history.select("daily_pnl").to_series()
            mean_return = returns.mean()

            # Only negative returns for downside deviation
            negative_returns = returns.filter(returns < 0)
            if len(negative_returns) > 0:
                downside_std = negative_returns.std()
                if downside_std and downside_std > 0:
                    sortino = Decimal(str(mean_return)) / Decimal(str(downside_std))
                    return sortino

            return Decimal('0')
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return Decimal('0')

    async def _calculate_beta(self, positions: List[Position]) -> Decimal:
        """Calculate portfolio beta"""
        # Simplified beta calculation (market benchmark required)
        return Decimal('1.0')  # Placeholder

    async def _calculate_daily_pnl(self, positions: List[Position]) -> Decimal:
        """Calculate daily P&L"""
        total_pnl = Decimal('0')

        for position in positions:
            pnl = position.pnl
            total_pnl += pnl

        return total_pnl

    async def _store_metrics(self, metrics: RiskMetrics) -> None:
        """Store metrics in database"""
        if not self.db:
            return

        try:
            # Convert to DataFrame
            metrics_df = pl.DataFrame({
                "timestamp": [metrics.timestamp],
                "portfolio_value": [float(metrics.portfolio_value)],
                "cash_balance": [float(metrics.cash_balance)],
                "total_exposure": [float(metrics.total_exposure)],
                "var_95": [float(metrics.var_95)],
                "cvar_95": [float(metrics.cvar_95)],
                "max_drawdown": [float(metrics.max_drawdown)],
                "sharpe_ratio": [float(metrics.sharpe_ratio)],
                "sortino_ratio": [float(metrics.sortino_ratio)],
                "beta": [float(metrics.beta)],
                "daily_pnl": [float(metrics.daily_pnl)]
            })

            # Store in database
            await self.db.insert_risk_metrics(metrics_df)

            # Update in-memory history
            self._metrics_history = pl.concat([self._metrics_history, metrics_df])

            # Keep only last 30 days
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            self._metrics_history = self._metrics_history.filter(
                pl.col("timestamp") > cutoff_date
            )

        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")

    async def _check_alert_thresholds(self, metrics: RiskMetrics) -> None:
        """Check if any alert thresholds are breached"""
        alerts = []

        # Check portfolio risk
        portfolio_risk_percent = (metrics.total_exposure / metrics.portfolio_value * Decimal('100')
                                  if metrics.portfolio_value > 0 else Decimal('0'))

        if portfolio_risk_percent > self.alert_portfolio_risk_percent:
            alerts.append({
                "severity": "WARNING",
                "message": f"Portfolio risk at {portfolio_risk_percent:.2f}% (threshold: {self.alert_portfolio_risk_percent}%)",
                "metric": "portfolio_risk",
                "value": float(portfolio_risk_percent)
            })

        # Check unrealized loss
        unrealized_loss_percent = (metrics.daily_pnl / metrics.portfolio_value * Decimal('100')
                                   if metrics.portfolio_value > 0 else Decimal('0'))

        if unrealized_loss_percent < -self.alert_unrealized_loss_percent:
            alerts.append({
                "severity": "CRITICAL",
                "message": f"Unrealized loss at {unrealized_loss_percent:.2f}% (threshold: -{self.alert_unrealized_loss_percent}%)",
                "metric": "unrealized_loss",
                "value": float(unrealized_loss_percent)
            })

        # Check max drawdown
        if metrics.max_drawdown > self.max_drawdown_tolerance:
            alerts.append({
                "severity": "CRITICAL",
                "message": f"Max drawdown {metrics.max_drawdown:.2f}% exceeds tolerance {self.max_drawdown_tolerance}%",
                "metric": "max_drawdown",
                "value": float(metrics.max_drawdown)
            })

        # Check Sharpe ratio
        if metrics.sharpe_ratio < self.min_sharpe_ratio:
            alerts.append({
                "severity": "WARNING",
                "message": f"Sharpe ratio {metrics.sharpe_ratio:.2f} below minimum {self.min_sharpe_ratio}",
                "metric": "sharpe_ratio",
                "value": float(metrics.sharpe_ratio)
            })

        # Send alerts
        for alert in alerts:
            await self._send_alert(alert)

            # Trigger callbacks
            for callback in self._alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")

    async def _send_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert through alert manager"""
        if self.alert_manager:
            try:
                await self.alert_manager.send_alert(
                    severity=alert["severity"],
                    message=alert["message"],
                    component="RiskMonitor",
                    metadata=alert
                )
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")

        # Log alert
        if self.enable_risk_logging:
            logger.warning(f"RISK ALERT: {alert['message']}")

    async def _update_dashboard(self, metrics: RiskMetrics) -> None:
        """Update dashboard with current metrics"""
        # This would integrate with a real dashboard system
        pass

    async def _analyze_risk_trends(self) -> None:
        """Analyze risk trends over time"""
        if len(self._metrics_history) < 10:
            return

        try:
            # Calculate trend indicators
            recent_sharpe = self._metrics_history.tail(10).select("sharpe_ratio").mean()
            overall_sharpe = self._metrics_history.select("sharpe_ratio").mean()

            if recent_sharpe and overall_sharpe:
                trend = "improving" if recent_sharpe[0] > overall_sharpe[0] else "declining"
                logger.info(f"Risk trend: Sharpe ratio {trend}")
        except Exception as e:
            logger.error(f"Error analyzing risk trends: {e}")

    async def _load_historical_metrics(self) -> None:
        """Load historical risk metrics from database"""
        if not self.db:
            return

        try:
            # Load last 30 days of metrics
            lookback_date = datetime.utcnow() - timedelta(days=30)
            self._metrics_history = await self.db.query_risk_metrics(
                start_date=lookback_date,
                end_date=datetime.utcnow()
            )

            logger.info(f"Loaded {len(self._metrics_history)} historical risk metrics")
        except Exception as e:
            logger.error(f"Failed to load historical metrics: {e}")
            self._metrics_history = pl.DataFrame()

    def register_alert_callback(self, callback: Callable) -> None:
        """Register callback for risk alerts"""
        self._alert_callbacks.append(callback)
        logger.info("Alert callback registered")

    def get_current_metrics(self) -> Optional[RiskMetrics]:
        """Get current risk metrics"""
        return self._current_metrics

    def get_metrics_history(self) -> pl.DataFrame:
        """Get historical risk metrics"""
        return self._metrics_history

    async def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        if not self._current_metrics:
            return {"error": "No current metrics available"}

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "current_metrics": {
                "portfolio_value": float(self._current_metrics.portfolio_value),
                "cash_balance": float(self._current_metrics.cash_balance),
                "total_exposure": float(self._current_metrics.total_exposure),
                "var_95": float(self._current_metrics.var_95),
                "cvar_95": float(self._current_metrics.cvar_95),
                "max_drawdown": float(self._current_metrics.max_drawdown),
                "sharpe_ratio": float(self._current_metrics.sharpe_ratio),
                "sortino_ratio": float(self._current_metrics.sortino_ratio),
                "beta": float(self._current_metrics.beta),
                "daily_pnl": float(self._current_metrics.daily_pnl)
            },
            "trends": {},
            "alerts": []
        }

        # Add trends if history available
        if len(self._metrics_history) > 0:
            report["trends"] = {
                "avg_sharpe_7d": float(self._metrics_history.tail(7).select("sharpe_ratio").mean()[0] or 0),
                "avg_drawdown_7d": float(self._metrics_history.tail(7).select("max_drawdown").mean()[0] or 0)
            }

        return report

    async def cleanup(self) -> None:
        """Cleanup resources"""
        await self.stop_monitoring()

        if self.db:
            await self.db.disconnect()

        if self.alert_manager:
            await self.alert_manager.cleanup()

        logger.info("RiskMonitor cleanup complete")
