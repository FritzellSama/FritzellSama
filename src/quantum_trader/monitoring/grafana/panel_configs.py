"""Grafana dashboard panel configurations for Quantum Trader AI.

This module provides pre-configured Grafana panel definitions for visualizing
trading system metrics, including trading performance, system health, and
risk metrics.
"""

import os
from typing import Dict, Any, List, Optional
from structlog import get_logger

logger = get_logger(__name__)


class PanelConfigs:
    """Grafana panel configuration generator.

    Generates JSON panel configurations for Grafana dashboards that visualize
    trading system metrics from Prometheus.

    Attributes:
        config: Configuration dictionary
        namespace: Metrics namespace
        datasource: Prometheus datasource name

    Example:
        >>> config = {
        ...     "namespace": "quantum_trader",
        ...     "datasource": "Prometheus"
        ... }
        >>> panels = PanelConfigs(config)
        >>> trading_panel = panels.get_trading_overview_panel()
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize panel configurations.

        Args:
            config: Configuration dictionary containing:
                - namespace: Metrics namespace
                - datasource: Prometheus datasource name
                - refresh_interval: Dashboard refresh interval

        Raises:
            ValueError: If required config parameters are missing
        """
        self.config = config
        self._validate_config()

        self.namespace = self.config.get(
            "namespace", os.getenv("METRICS_NAMESPACE", "quantum_trader")
        )
        self.datasource = self.config.get(
            "datasource", os.getenv("GRAFANA_DATASOURCE", "Prometheus")
        )
        self.refresh_interval = self.config.get(
            "refresh_interval", os.getenv("GRAFANA_REFRESH_INTERVAL", "5s")
        )

        logger.info(
            "PanelConfigs initialized",
            namespace=self.namespace,
            datasource=self.datasource
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        logger.debug("Config validation passed")

    def get_trading_overview_panel(self) -> Dict[str, Any]:
        """Get trading overview panel configuration.

        Returns:
            Grafana panel configuration dict
        """
        return {
            "id": 1,
            "title": "Trading Overview",
            "type": "row",
            "collapsed": False,
            "panels": [
                self._create_stat_panel(
                    id=2,
                    title="Total Orders",
                    query=f"sum(rate({self.namespace}_trading_orders_total[5m]))",
                    unit="ops",
                    gridPos={"h": 4, "w": 6, "x": 0, "y": 0}
                ),
                self._create_stat_panel(
                    id=3,
                    title="Total Trades",
                    query=f"sum(rate({self.namespace}_trading_trades_total[5m]))",
                    unit="ops",
                    gridPos={"h": 4, "w": 6, "x": 6, "y": 0}
                ),
                self._create_stat_panel(
                    id=4,
                    title="Total Volume (24h)",
                    query=f"sum(increase({self.namespace}_trading_trade_volume_usd[24h]))",
                    unit="currencyUSD",
                    gridPos={"h": 4, "w": 6, "x": 12, "y": 0}
                ),
                self._create_stat_panel(
                    id=5,
                    title="Open Orders",
                    query=f"sum({self.namespace}_trading_open_orders)",
                    unit="short",
                    gridPos={"h": 4, "w": 6, "x": 18, "y": 0}
                ),
            ]
        }

    def get_performance_panel(self) -> Dict[str, Any]:
        """Get performance metrics panel configuration.

        Returns:
            Grafana panel configuration dict
        """
        return {
            "id": 10,
            "title": "Performance Metrics",
            "type": "row",
            "collapsed": False,
            "panels": [
                self._create_graph_panel(
                    id=11,
                    title="Total PnL",
                    queries=[
                        {
                            "expr": f"sum({self.namespace}_trading_total_pnl_usd) by (strategy)",
                            "legendFormat": "{{strategy}}"
                        }
                    ],
                    unit="currencyUSD",
                    gridPos={"h": 8, "w": 12, "x": 0, "y": 4}
                ),
                self._create_graph_panel(
                    id=12,
                    title="Win Rate",
                    queries=[
                        {
                            "expr": f"{self.namespace}_trading_win_rate_percent",
                            "legendFormat": "{{strategy}} - {{timeframe}}"
                        }
                    ],
                    unit="percent",
                    gridPos={"h": 8, "w": 12, "x": 12, "y": 4}
                ),
            ]
        }

    def get_risk_panel(self) -> Dict[str, Any]:
        """Get risk metrics panel configuration.

        Returns:
            Grafana panel configuration dict
        """
        return {
            "id": 20,
            "title": "Risk Metrics",
            "type": "row",
            "collapsed": False,
            "panels": [
                self._create_gauge_panel(
                    id=21,
                    title="Current Drawdown",
                    query=f"max({self.namespace}_trading_drawdown_percent)",
                    unit="percent",
                    thresholds=[
                        {"color": "green", "value": 0},
                        {"color": "yellow", "value": 5},
                        {"color": "red", "value": 10}
                    ],
                    gridPos={"h": 6, "w": 6, "x": 0, "y": 12}
                ),
                self._create_stat_panel(
                    id=22,
                    title="Max Drawdown",
                    query=f"max({self.namespace}_trading_max_drawdown_percent)",
                    unit="percent",
                    gridPos={"h": 6, "w": 6, "x": 6, "y": 12}
                ),
                self._create_stat_panel(
                    id=23,
                    title="VaR (95%)",
                    query=f"max({self.namespace}_trading_value_at_risk_95_usd)",
                    unit="currencyUSD",
                    gridPos={"h": 6, "w": 6, "x": 12, "y": 12}
                ),
                self._create_stat_panel(
                    id=24,
                    title="Sharpe Ratio",
                    query=f"avg({self.namespace}_trading_sharpe_ratio)",
                    unit="short",
                    gridPos={"h": 6, "w": 6, "x": 18, "y": 12}
                ),
            ]
        }

    def get_latency_panel(self) -> Dict[str, Any]:
        """Get latency metrics panel configuration.

        Returns:
            Grafana panel configuration dict
        """
        return {
            "id": 30,
            "title": "Latency Metrics",
            "type": "row",
            "collapsed": False,
            "panels": [
                self._create_heatmap_panel(
                    id=31,
                    title="Order Latency Heatmap",
                    query=f"rate({self.namespace}_trading_order_latency_milliseconds_bucket[5m])",
                    gridPos={"h": 8, "w": 12, "x": 0, "y": 18}
                ),
                self._create_graph_panel(
                    id=32,
                    title="Execution Latency (p50, p95, p99)",
                    queries=[
                        {
                            "expr": f"histogram_quantile(0.50, rate({self.namespace}_trading_execution_latency_milliseconds_bucket[5m]))",
                            "legendFormat": "p50"
                        },
                        {
                            "expr": f"histogram_quantile(0.95, rate({self.namespace}_trading_execution_latency_milliseconds_bucket[5m]))",
                            "legendFormat": "p95"
                        },
                        {
                            "expr": f"histogram_quantile(0.99, rate({self.namespace}_trading_execution_latency_milliseconds_bucket[5m]))",
                            "legendFormat": "p99"
                        }
                    ],
                    unit="ms",
                    gridPos={"h": 8, "w": 12, "x": 12, "y": 18}
                ),
            ]
        }

    def get_system_panel(self) -> Dict[str, Any]:
        """Get system health panel configuration.

        Returns:
            Grafana panel configuration dict
        """
        return {
            "id": 40,
            "title": "System Health",
            "type": "row",
            "collapsed": False,
            "panels": [
                self._create_graph_panel(
                    id=41,
                    title="CPU Usage",
                    queries=[
                        {
                            "expr": f"{self.namespace}_trading_cpu_usage_percent",
                            "legendFormat": "{{component}}"
                        }
                    ],
                    unit="percent",
                    gridPos={"h": 6, "w": 12, "x": 0, "y": 26}
                ),
                self._create_graph_panel(
                    id=42,
                    title="Memory Usage",
                    queries=[
                        {
                            "expr": f"{self.namespace}_trading_memory_usage_bytes",
                            "legendFormat": "{{component}}"
                        }
                    ],
                    unit="bytes",
                    gridPos={"h": 6, "w": 12, "x": 12, "y": 26}
                ),
            ]
        }

    def get_exchange_panel(self) -> Dict[str, Any]:
        """Get exchange connectivity panel configuration.

        Returns:
            Grafana panel configuration dict
        """
        return {
            "id": 50,
            "title": "Exchange Connectivity",
            "type": "row",
            "collapsed": False,
            "panels": [
                self._create_stat_panel(
                    id=51,
                    title="Connected Exchanges",
                    query=f"sum({self.namespace}_trading_exchange_connected)",
                    unit="short",
                    gridPos={"h": 4, "w": 6, "x": 0, "y": 32}
                ),
                self._create_graph_panel(
                    id=52,
                    title="API Call Rate",
                    queries=[
                        {
                            "expr": f"sum(rate({self.namespace}_trading_api_calls_total[5m])) by (exchange)",
                            "legendFormat": "{{exchange}}"
                        }
                    ],
                    unit="ops",
                    gridPos={"h": 8, "w": 12, "x": 6, "y": 32}
                ),
                self._create_graph_panel(
                    id=53,
                    title="Rate Limit Remaining",
                    queries=[
                        {
                            "expr": f"{self.namespace}_trading_rate_limit_remaining",
                            "legendFormat": "{{exchange}} - {{limit_type}}"
                        }
                    ],
                    unit="short",
                    gridPos={"h": 8, "w": 6, "x": 18, "y": 32}
                ),
            ]
        }

    def _create_stat_panel(
        self,
        id: int,
        title: str,
        query: str,
        unit: str,
        gridPos: Dict[str, int]
    ) -> Dict[str, Any]:
        """Create a stat panel configuration.

        Args:
            id: Panel ID
            title: Panel title
            query: Prometheus query
            unit: Unit type
            gridPos: Grid position dictionary

        Returns:
            Panel configuration dict
        """
        return {
            "id": id,
            "title": title,
            "type": "stat",
            "datasource": self.datasource,
            "gridPos": gridPos,
            "targets": [
                {
                    "expr": query,
                    "refId": "A"
                }
            ],
            "options": {
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"]
                }
            },
            "fieldConfig": {
                "defaults": {
                    "unit": unit
                }
            }
        }

    def _create_graph_panel(
        self,
        id: int,
        title: str,
        queries: List[Dict[str, str]],
        unit: str,
        gridPos: Dict[str, int]
    ) -> Dict[str, Any]:
        """Create a graph panel configuration.

        Args:
            id: Panel ID
            title: Panel title
            queries: List of query dictionaries
            unit: Unit type
            gridPos: Grid position dictionary

        Returns:
            Panel configuration dict
        """
        targets = []
        for i, query in enumerate(queries):
            targets.append({
                "expr": query["expr"],
                "legendFormat": query.get("legendFormat", ""),
                "refId": chr(65 + i)  # A, B, C, etc.
            })

        return {
            "id": id,
            "title": title,
            "type": "timeseries",
            "datasource": self.datasource,
            "gridPos": gridPos,
            "targets": targets,
            "fieldConfig": {
                "defaults": {
                    "unit": unit
                }
            }
        }

    def _create_gauge_panel(
        self,
        id: int,
        title: str,
        query: str,
        unit: str,
        thresholds: List[Dict[str, Any]],
        gridPos: Dict[str, int]
    ) -> Dict[str, Any]:
        """Create a gauge panel configuration.

        Args:
            id: Panel ID
            title: Panel title
            query: Prometheus query
            unit: Unit type
            thresholds: List of threshold dictionaries
            gridPos: Grid position dictionary

        Returns:
            Panel configuration dict
        """
        return {
            "id": id,
            "title": title,
            "type": "gauge",
            "datasource": self.datasource,
            "gridPos": gridPos,
            "targets": [
                {
                    "expr": query,
                    "refId": "A"
                }
            ],
            "options": {
                "showThresholdLabels": True,
                "showThresholdMarkers": True
            },
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": thresholds
                    }
                }
            }
        }

    def _create_heatmap_panel(
        self,
        id: int,
        title: str,
        query: str,
        gridPos: Dict[str, int]
    ) -> Dict[str, Any]:
        """Create a heatmap panel configuration.

        Args:
            id: Panel ID
            title: Panel title
            query: Prometheus query
            gridPos: Grid position dictionary

        Returns:
            Panel configuration dict
        """
        return {
            "id": id,
            "title": title,
            "type": "heatmap",
            "datasource": self.datasource,
            "gridPos": gridPos,
            "targets": [
                {
                    "expr": query,
                    "format": "heatmap",
                    "refId": "A"
                }
            ],
            "options": {
                "calculate": True,
                "calculation": {}
            }
        }

    def generate_full_dashboard(self) -> Dict[str, Any]:
        """Generate complete dashboard configuration.

        Returns:
            Full Grafana dashboard JSON configuration
        """
        dashboard = {
            "title": "Quantum Trader AI - Trading Dashboard",
            "uid": "quantum-trader-main",
            "timezone": "utc",
            "refresh": self.refresh_interval,
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "panels": [
                self.get_trading_overview_panel(),
                self.get_performance_panel(),
                self.get_risk_panel(),
                self.get_latency_panel(),
                self.get_system_panel(),
                self.get_exchange_panel(),
            ]
        }

        logger.info("Full dashboard configuration generated")
        return dashboard
