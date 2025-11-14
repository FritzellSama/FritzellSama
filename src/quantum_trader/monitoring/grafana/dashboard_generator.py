"""Grafana Dashboard Generator for Quantum Trader AI.

Production-ready dashboard generation system that creates comprehensive
monitoring dashboards for trading performance, system health, and risk metrics.
"""

import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import json
from pathlib import Path

from prometheus_client import Counter
from structlog import get_logger

logger = get_logger(__name__)


class DashboardGenerator:
    """Generate Grafana dashboards programmatically.

    Creates pre-configured dashboards for:
    - Trading performance overview
    - Risk management
    - System health monitoring
    - Exchange connectivity
    - Strategy performance
    - ML model monitoring

    Attributes:
        config: Configuration dictionary
        datasource_uid: Prometheus datasource UID
        dashboards: Generated dashboard definitions

    Example:
        >>> config = {
        ...     "datasource_uid": "prometheus",
        ...     "refresh_interval": "30s",
        ...     "timezone": "UTC"
        ... }
        >>> generator = DashboardGenerator(config)
        >>> dashboards = await generator.generate_all_dashboards()
        >>> await generator.export_to_file("dashboards.json")
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize dashboard generator.

        Args:
            config: Configuration dictionary containing:
                - datasource_uid: Prometheus datasource UID
                - refresh_interval: Dashboard refresh interval
                - timezone: Dashboard timezone
                - default_time_range: Default time range (e.g., "6h")
                - theme: Dashboard theme (dark/light)

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.datasource_uid = self.config.get("datasource_uid", "prometheus")
        self.refresh_interval = self.config.get("refresh_interval", "30s")
        self.timezone = self.config.get("timezone", "UTC")
        self.default_time_range = self.config.get("default_time_range", "6h")
        self.theme = self.config.get("theme", "dark")

        self.dashboards: List[Dict[str, Any]] = []
        self._panel_id_counter = 1

        # Metrics
        self._dashboards_generated = Counter(
            "quantum_trader_dashboards_generated_total",
            "Total dashboards generated"
        )

        logger.info(
            "dashboard_generator_initialized",
            datasource=self.datasource_uid,
            refresh=self.refresh_interval,
            timezone=self.timezone
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        datasource = self.config.get("datasource_uid", "prometheus")
        if not datasource:
            raise ValueError("datasource_uid cannot be empty")

    def _get_next_panel_id(self) -> int:
        """Get next panel ID.

        Returns:
            Panel ID
        """
        panel_id = self._panel_id_counter
        self._panel_id_counter += 1
        return panel_id

    def _reset_panel_counter(self) -> None:
        """Reset panel ID counter for new dashboard."""
        self._panel_id_counter = 1

    async def generate_all_dashboards(self) -> List[Dict[str, Any]]:
        """Generate all dashboards.

        Returns:
            List of dashboard definitions

        Raises:
            Exception: If generation fails
        """
        try:
            logger.info("generating_all_dashboards")

            self.dashboards = []

            # Generate different dashboard types
            self.dashboards.append(await self._generate_trading_overview())
            self.dashboards.append(await self._generate_risk_dashboard())
            self.dashboards.append(await self._generate_system_health())
            self.dashboards.append(await self._generate_exchange_dashboard())
            self.dashboards.append(await self._generate_strategy_dashboard())

            self._dashboards_generated.inc(len(self.dashboards))

            logger.info(
                "dashboards_generated",
                total=len(self.dashboards)
            )

            return self.dashboards

        except Exception as e:
            logger.error("dashboard_generation_failed", error=str(e))
            raise

    async def _generate_trading_overview(self) -> Dict[str, Any]:
        """Generate trading overview dashboard.

        Returns:
            Dashboard definition
        """
        try:
            self._reset_panel_counter()

            dashboard = {
                "dashboard": {
                    "title": "Quantum Trader - Trading Overview",
                    "uid": "quantum_trader_trading_overview",
                    "timezone": self.timezone,
                    "schemaVersion": 16,
                    "version": 0,
                    "refresh": self.refresh_interval,
                    "time": {
                        "from": f"now-{self.default_time_range}",
                        "to": "now"
                    },
                    "tags": ["quantum_trader", "trading"],
                    "panels": []
                },
                "overwrite": True
            }

            panels = []

            # Row 1: Summary Stats
            panels.append(self._create_row("Summary Statistics", 0))

            # Total PnL stat
            panels.append(self._create_stat_panel(
                title="Total PnL (USD)",
                targets=[{
                    "expr": "sum(quantum_trader_pnl_total)",
                    "refId": "A"
                }],
                x=0, y=1, w=6, h=4,
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 0},
                    {"color": "green", "value": 1000}
                ],
                unit="currencyUSD"
            ))

            # Win Rate stat
            panels.append(self._create_stat_panel(
                title="Win Rate (%)",
                targets=[{
                    "expr": "(sum(rate(quantum_trader_trades_won_total[1h])) / sum(rate(quantum_trader_trades_total[1h]))) * 100",
                    "refId": "A"
                }],
                x=6, y=1, w=6, h=4,
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 40},
                    {"color": "green", "value": 55}
                ],
                unit="percent"
            ))

            # Total Trades stat
            panels.append(self._create_stat_panel(
                title="Total Trades (24h)",
                targets=[{
                    "expr": "sum(increase(quantum_trader_trades_total[24h]))",
                    "refId": "A"
                }],
                x=12, y=1, w=6, h=4,
                unit="short"
            ))

            # Active Positions stat
            panels.append(self._create_stat_panel(
                title="Active Positions",
                targets=[{
                    "expr": "sum(quantum_trader_open_positions)",
                    "refId": "A"
                }],
                x=18, y=1, w=6, h=4,
                unit="short"
            ))

            # Row 2: Charts
            panels.append(self._create_row("Performance Charts", 5))

            # PnL over time
            panels.append(self._create_time_series_panel(
                title="PnL Over Time",
                targets=[{
                    "expr": "sum(quantum_trader_pnl_total) by (exchange)",
                    "legendFormat": "{{exchange}}",
                    "refId": "A"
                }],
                x=0, y=6, w=12, h=8,
                unit="currencyUSD"
            ))

            # Trade volume
            panels.append(self._create_time_series_panel(
                title="Trading Volume (USD)",
                targets=[{
                    "expr": "sum(rate(quantum_trader_volume_traded_total[5m])) by (exchange)",
                    "legendFormat": "{{exchange}}",
                    "refId": "A"
                }],
                x=12, y=6, w=12, h=8,
                unit="currencyUSD"
            ))

            # Orders submitted vs filled
            panels.append(self._create_time_series_panel(
                title="Orders: Submitted vs Filled",
                targets=[
                    {
                        "expr": "sum(rate(quantum_trader_orders_submitted_total[5m]))",
                        "legendFormat": "Submitted",
                        "refId": "A"
                    },
                    {
                        "expr": "sum(rate(quantum_trader_orders_filled_total[5m]))",
                        "legendFormat": "Filled",
                        "refId": "B"
                    }
                ],
                x=0, y=14, w=12, h=8,
                unit="ops"
            ))

            # PnL distribution heatmap
            panels.append(self._create_heatmap_panel(
                title="PnL Distribution",
                target={
                    "expr": "sum(rate(quantum_trader_pnl_per_trade_bucket[5m])) by (le)",
                    "format": "heatmap",
                    "refId": "A"
                },
                x=12, y=14, w=12, h=8
            ))

            dashboard["dashboard"]["panels"] = panels

            return dashboard

        except Exception as e:
            logger.error("trading_overview_generation_failed", error=str(e))
            raise

    async def _generate_risk_dashboard(self) -> Dict[str, Any]:
        """Generate risk management dashboard.

        Returns:
            Dashboard definition
        """
        try:
            self._reset_panel_counter()

            dashboard = {
                "dashboard": {
                    "title": "Quantum Trader - Risk Management",
                    "uid": "quantum_trader_risk",
                    "timezone": self.timezone,
                    "schemaVersion": 16,
                    "version": 0,
                    "refresh": self.refresh_interval,
                    "time": {
                        "from": f"now-{self.default_time_range}",
                        "to": "now"
                    },
                    "tags": ["quantum_trader", "risk"],
                    "panels": []
                },
                "overwrite": True
            }

            panels = []

            # Risk metrics row
            panels.append(self._create_row("Risk Metrics", 0))

            # Max Drawdown
            panels.append(self._create_stat_panel(
                title="Max Drawdown (%)",
                targets=[{
                    "expr": "max(quantum_trader_max_drawdown_pct)",
                    "refId": "A"
                }],
                x=0, y=1, w=6, h=4,
                thresholds=[
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 5},
                    {"color": "red", "value": 10}
                ],
                unit="percent"
            ))

            # Current Leverage
            panels.append(self._create_stat_panel(
                title="Current Leverage",
                targets=[{
                    "expr": "max(quantum_trader_current_leverage)",
                    "refId": "A"
                }],
                x=6, y=1, w=6, h=4,
                thresholds=[
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 2},
                    {"color": "red", "value": 3}
                ],
                unit="short"
            ))

            # VaR
            panels.append(self._create_stat_panel(
                title="Value at Risk (95%)",
                targets=[{
                    "expr": "max(quantum_trader_var_usd)",
                    "refId": "A"
                }],
                x=12, y=1, w=6, h=4,
                unit="currencyUSD"
            ))

            # Sharpe Ratio
            panels.append(self._create_stat_panel(
                title="Sharpe Ratio",
                targets=[{
                    "expr": "avg(quantum_trader_sharpe_ratio)",
                    "refId": "A"
                }],
                x=18, y=1, w=6, h=4,
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 1},
                    {"color": "green", "value": 2}
                ],
                unit="short"
            ))

            # Risk over time
            panels.append(self._create_time_series_panel(
                title="Drawdown Over Time",
                targets=[{
                    "expr": "quantum_trader_max_drawdown_pct",
                    "legendFormat": "{{exchange}}",
                    "refId": "A"
                }],
                x=0, y=5, w=12, h=8,
                unit="percent"
            ))

            # Exposure by exchange
            panels.append(self._create_time_series_panel(
                title="Total Exposure by Exchange",
                targets=[{
                    "expr": "sum(quantum_trader_exposure_total_usd) by (exchange)",
                    "legendFormat": "{{exchange}}",
                    "refId": "A"
                }],
                x=12, y=5, w=12, h=8,
                unit="currencyUSD"
            ))

            dashboard["dashboard"]["panels"] = panels

            return dashboard

        except Exception as e:
            logger.error("risk_dashboard_generation_failed", error=str(e))
            raise

    async def _generate_system_health(self) -> Dict[str, Any]:
        """Generate system health dashboard.

        Returns:
            Dashboard definition
        """
        try:
            self._reset_panel_counter()

            dashboard = {
                "dashboard": {
                    "title": "Quantum Trader - System Health",
                    "uid": "quantum_trader_system_health",
                    "timezone": self.timezone,
                    "schemaVersion": 16,
                    "version": 0,
                    "refresh": self.refresh_interval,
                    "time": {
                        "from": f"now-{self.default_time_range}",
                        "to": "now"
                    },
                    "tags": ["quantum_trader", "system"],
                    "panels": []
                },
                "overwrite": True
            }

            panels = []

            # System status row
            panels.append(self._create_row("System Status", 0))

            # Service uptime
            panels.append(self._create_stat_panel(
                title="Service Status",
                targets=[{
                    "expr": "up{job='quantum_trader'}",
                    "refId": "A"
                }],
                x=0, y=1, w=6, h=4,
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "green", "value": 1}
                ],
                mappings=[
                    {"type": "value", "value": "0", "text": "DOWN"},
                    {"type": "value", "value": "1", "text": "UP"}
                ]
            ))

            # Processing latency
            panels.append(self._create_stat_panel(
                title="P99 Latency (ms)",
                targets=[{
                    "expr": "histogram_quantile(0.99, quantum_trader_processing_latency_seconds_bucket) * 1000",
                    "refId": "A"
                }],
                x=6, y=1, w=6, h=4,
                thresholds=[
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 50},
                    {"color": "red", "value": 100}
                ],
                unit="ms"
            ))

            # Error rate
            panels.append(self._create_stat_panel(
                title="Error Rate",
                targets=[{
                    "expr": "sum(rate(quantum_trader_processing_errors_total[5m]))",
                    "refId": "A"
                }],
                x=12, y=1, w=6, h=4,
                thresholds=[
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 1},
                    {"color": "red", "value": 10}
                ],
                unit="ops"
            ))

            # Queue size
            panels.append(self._create_stat_panel(
                title="Event Queue Size",
                targets=[{
                    "expr": "sum(quantum_trader_queue_size)",
                    "refId": "A"
                }],
                x=18, y=1, w=6, h=4,
                unit="short"
            ))

            # Latency over time
            panels.append(self._create_time_series_panel(
                title="Order Latency (P95, P99)",
                targets=[
                    {
                        "expr": "histogram_quantile(0.95, sum(rate(quantum_trader_order_latency_seconds_bucket[5m])) by (le)) * 1000",
                        "legendFormat": "P95",
                        "refId": "A"
                    },
                    {
                        "expr": "histogram_quantile(0.99, sum(rate(quantum_trader_order_latency_seconds_bucket[5m])) by (le)) * 1000",
                        "legendFormat": "P99",
                        "refId": "B"
                    }
                ],
                x=0, y=5, w=12, h=8,
                unit="ms"
            ))

            # Events processed
            panels.append(self._create_time_series_panel(
                title="Events Processed",
                targets=[{
                    "expr": "sum(rate(quantum_trader_events_processed_total[5m])) by (event_type)",
                    "legendFormat": "{{event_type}}",
                    "refId": "A"
                }],
                x=12, y=5, w=12, h=8,
                unit="ops"
            ))

            dashboard["dashboard"]["panels"] = panels

            return dashboard

        except Exception as e:
            logger.error("system_health_generation_failed", error=str(e))
            raise

    async def _generate_exchange_dashboard(self) -> Dict[str, Any]:
        """Generate exchange connectivity dashboard.

        Returns:
            Dashboard definition
        """
        try:
            self._reset_panel_counter()

            dashboard = {
                "dashboard": {
                    "title": "Quantum Trader - Exchange Connectivity",
                    "uid": "quantum_trader_exchanges",
                    "timezone": self.timezone,
                    "schemaVersion": 16,
                    "version": 0,
                    "refresh": self.refresh_interval,
                    "time": {
                        "from": f"now-{self.default_time_range}",
                        "to": "now"
                    },
                    "tags": ["quantum_trader", "exchanges"],
                    "panels": []
                },
                "overwrite": True
            }

            panels = []

            # Exchange status
            panels.append(self._create_row("Exchange Status", 0))

            # Connection status
            panels.append(self._create_stat_panel(
                title="Exchanges Connected",
                targets=[{
                    "expr": "sum(quantum_trader_exchange_connected)",
                    "refId": "A"
                }],
                x=0, y=1, w=6, h=4,
                unit="short"
            ))

            # API latency
            panels.append(self._create_time_series_panel(
                title="API Latency by Exchange",
                targets=[{
                    "expr": "avg(quantum_trader_exchange_api_latency_ms) by (exchange)",
                    "legendFormat": "{{exchange}}",
                    "refId": "A"
                }],
                x=6, y=1, w=18, h=8,
                unit="ms"
            ))

            # API requests
            panels.append(self._create_time_series_panel(
                title="API Requests by Exchange",
                targets=[{
                    "expr": "sum(rate(quantum_trader_exchange_requests_total[5m])) by (exchange)",
                    "legendFormat": "{{exchange}}",
                    "refId": "A"
                }],
                x=0, y=9, w=12, h=8,
                unit="reqps"
            ))

            # Errors by exchange
            panels.append(self._create_time_series_panel(
                title="Exchange Errors",
                targets=[{
                    "expr": "sum(rate(quantum_trader_exchange_errors_total[5m])) by (exchange, error_type)",
                    "legendFormat": "{{exchange}} - {{error_type}}",
                    "refId": "A"
                }],
                x=12, y=9, w=12, h=8,
                unit="ops"
            ))

            dashboard["dashboard"]["panels"] = panels

            return dashboard

        except Exception as e:
            logger.error("exchange_dashboard_generation_failed", error=str(e))
            raise

    async def _generate_strategy_dashboard(self) -> Dict[str, Any]:
        """Generate strategy performance dashboard.

        Returns:
            Dashboard definition
        """
        try:
            self._reset_panel_counter()

            dashboard = {
                "dashboard": {
                    "title": "Quantum Trader - Strategy Performance",
                    "uid": "quantum_trader_strategies",
                    "timezone": self.timezone,
                    "schemaVersion": 16,
                    "version": 0,
                    "refresh": self.refresh_interval,
                    "time": {
                        "from": f"now-{self.default_time_range}",
                        "to": "now"
                    },
                    "tags": ["quantum_trader", "strategies"],
                    "panels": []
                },
                "overwrite": True
            }

            panels = []

            # Strategy overview
            panels.append(self._create_row("Strategy Overview", 0))

            # Active strategies
            panels.append(self._create_stat_panel(
                title="Active Strategies",
                targets=[{
                    "expr": "sum(quantum_trader_strategy_active)",
                    "refId": "A"
                }],
                x=0, y=1, w=6, h=4,
                unit="short"
            ))

            # Strategy PnL
            panels.append(self._create_time_series_panel(
                title="Strategy PnL",
                targets=[{
                    "expr": "quantum_trader_strategy_pnl_usd",
                    "legendFormat": "{{strategy}}",
                    "refId": "A"
                }],
                x=6, y=1, w=18, h=8,
                unit="currencyUSD"
            ))

            # Signals generated
            panels.append(self._create_time_series_panel(
                title="Signals Generated",
                targets=[{
                    "expr": "sum(rate(quantum_trader_strategy_signals_total[5m])) by (strategy, signal_type)",
                    "legendFormat": "{{strategy}} - {{signal_type}}",
                    "refId": "A"
                }],
                x=0, y=9, w=24, h=8,
                unit="ops"
            ))

            dashboard["dashboard"]["panels"] = panels

            return dashboard

        except Exception as e:
            logger.error("strategy_dashboard_generation_failed", error=str(e))
            raise

    def _create_row(self, title: str, y: int) -> Dict[str, Any]:
        """Create row panel.

        Args:
            title: Row title
            y: Y position

        Returns:
            Row panel definition
        """
        return {
            "type": "row",
            "id": self._get_next_panel_id(),
            "title": title,
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
            "collapsed": False
        }

    def _create_stat_panel(
        self,
        title: str,
        targets: List[Dict[str, str]],
        x: int,
        y: int,
        w: int,
        h: int,
        thresholds: Optional[List[Dict[str, Any]]] = None,
        unit: str = "short",
        mappings: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Create stat panel.

        Args:
            title: Panel title
            targets: Query targets
            x, y, w, h: Grid position and size
            thresholds: Threshold configuration
            unit: Value unit
            mappings: Value mappings

        Returns:
            Stat panel definition
        """
        for target in targets:
            target["datasource"] = {"uid": self.datasource_uid}

        panel = {
            "type": "stat",
            "id": self._get_next_panel_id(),
            "title": title,
            "targets": targets,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "options": {
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"]
                },
                "orientation": "auto",
                "textMode": "auto",
                "colorMode": "value"
            },
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": thresholds or [
                            {"color": "green", "value": None}
                        ]
                    }
                }
            }
        }

        if mappings:
            panel["fieldConfig"]["defaults"]["mappings"] = mappings

        return panel

    def _create_time_series_panel(
        self,
        title: str,
        targets: List[Dict[str, str]],
        x: int,
        y: int,
        w: int,
        h: int,
        unit: str = "short"
    ) -> Dict[str, Any]:
        """Create time series panel.

        Args:
            title: Panel title
            targets: Query targets
            x, y, w, h: Grid position and size
            unit: Value unit

        Returns:
            Time series panel definition
        """
        for target in targets:
            target["datasource"] = {"uid": self.datasource_uid}

        return {
            "type": "timeseries",
            "id": self._get_next_panel_id(),
            "title": title,
            "targets": targets,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "linear",
                        "fillOpacity": 10,
                        "showPoints": "never"
                    }
                }
            },
            "options": {
                "tooltip": {"mode": "multi"},
                "legend": {"displayMode": "list", "placement": "bottom"}
            }
        }

    def _create_heatmap_panel(
        self,
        title: str,
        target: Dict[str, str],
        x: int,
        y: int,
        w: int,
        h: int
    ) -> Dict[str, Any]:
        """Create heatmap panel.

        Args:
            title: Panel title
            target: Query target
            x, y, w, h: Grid position and size

        Returns:
            Heatmap panel definition
        """
        target["datasource"] = {"uid": self.datasource_uid}

        return {
            "type": "heatmap",
            "id": self._get_next_panel_id(),
            "title": title,
            "targets": [target],
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "options": {
                "calculate": True,
                "calculation": {},
                "cellGap": 2,
                "color": {
                    "mode": "scheme",
                    "scheme": "Spectral"
                },
                "yAxis": {"decimals": 0}
            }
        }

    async def export_to_file(self, file_path: str) -> None:
        """Export dashboards to JSON file.

        Args:
            file_path: Path to export file

        Raises:
            IOError: If file write fails
        """
        try:
            if not self.dashboards:
                await self.generate_all_dashboards()

            export_path = Path(file_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            with open(export_path, 'w') as f:
                json.dump(self.dashboards, f, indent=2)

            logger.info("dashboards_exported", file_path=file_path, count=len(self.dashboards))

        except Exception as e:
            logger.error("export_failed", error=str(e), file_path=file_path)
            raise

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get summary of generated dashboards.

        Returns:
            Summary dictionary
        """
        return {
            "total_dashboards": len(self.dashboards),
            "dashboards": [
                {
                    "title": d["dashboard"]["title"],
                    "uid": d["dashboard"]["uid"],
                    "panels": len(d["dashboard"]["panels"])
                }
                for d in self.dashboards
            ]
        }
