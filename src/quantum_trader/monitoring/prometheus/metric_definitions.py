"""Prometheus metric definitions for Quantum Trader AI.

This module defines all Prometheus metrics used throughout the trading system,
including counters, gauges, histograms, and summaries for tracking trading
performance, system health, and operational metrics.
"""

import os
from decimal import Decimal
from typing import Dict, Any, Optional, List
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    Info,
    CollectorRegistry,
)
from structlog import get_logger

logger = get_logger(__name__)


class MetricDefinitions:
    """Centralized Prometheus metric definitions for the trading system.

    Attributes:
        registry: Prometheus collector registry
        namespace: Metric namespace prefix
        subsystem: Metric subsystem prefix

    Example:
        >>> config = {"namespace": "quantum_trader", "subsystem": "trading"}
        >>> metrics = MetricDefinitions(config)
        >>> metrics.order_total.inc()
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize metric definitions.

        Args:
            config: Configuration dictionary containing:
                - namespace: Metric namespace (from env/config)
                - subsystem: Metric subsystem (from env/config)
                - enabled_metrics: List of metric categories to enable

        Raises:
            ValueError: If required config parameters are missing
        """
        self.config = config
        self._validate_config()

        self.namespace = self.config.get("namespace", os.getenv("METRICS_NAMESPACE", "quantum_trader"))
        self.subsystem = self.config.get("subsystem", os.getenv("METRICS_SUBSYSTEM", "trading"))
        self.registry = CollectorRegistry()

        # Initialize all metric definitions
        self._init_order_metrics()
        self._init_trade_metrics()
        self._init_position_metrics()
        self._init_pnl_metrics()
        self._init_risk_metrics()
        self._init_exchange_metrics()
        self._init_strategy_metrics()
        self._init_system_metrics()
        self._init_latency_metrics()

        logger.info(
            "Metric definitions initialized",
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry_size=len(list(self.registry.collect()))
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        # Validation passed
        logger.debug("Config validation passed", config_keys=list(self.config.keys()))

    def _init_order_metrics(self) -> None:
        """Initialize order-related metrics."""
        # Order counters
        self.order_total = Counter(
            "orders_total",
            "Total number of orders placed",
            ["exchange", "order_type", "side", "status"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.order_errors = Counter(
            "order_errors_total",
            "Total number of order errors",
            ["exchange", "error_type", "order_type"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.order_cancellations = Counter(
            "order_cancellations_total",
            "Total number of order cancellations",
            ["exchange", "reason"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        # Order gauges
        self.open_orders = Gauge(
            "open_orders",
            "Current number of open orders",
            ["exchange", "order_type"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        # Order histograms
        buckets = self._get_histogram_buckets("order_size")
        self.order_size = Histogram(
            "order_size_usd",
            "Order size in USD",
            ["exchange", "order_type", "side"],
            buckets=buckets,
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

    def _init_trade_metrics(self) -> None:
        """Initialize trade execution metrics."""
        self.trades_total = Counter(
            "trades_total",
            "Total number of trades executed",
            ["exchange", "side", "symbol"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.trade_volume = Counter(
            "trade_volume_usd",
            "Total trade volume in USD",
            ["exchange", "symbol"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.trade_fees = Counter(
            "trade_fees_usd",
            "Total trading fees in USD",
            ["exchange", "fee_type"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        buckets = self._get_histogram_buckets("slippage")
        self.trade_slippage = Histogram(
            "trade_slippage_bps",
            "Trade slippage in basis points",
            ["exchange", "order_type"],
            buckets=buckets,
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

    def _init_position_metrics(self) -> None:
        """Initialize position tracking metrics."""
        self.position_size = Gauge(
            "position_size_usd",
            "Current position size in USD",
            ["exchange", "symbol", "side"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.position_count = Gauge(
            "position_count",
            "Number of open positions",
            ["exchange"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.leverage_ratio = Gauge(
            "leverage_ratio",
            "Current leverage ratio",
            ["exchange", "symbol"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

    def _init_pnl_metrics(self) -> None:
        """Initialize profit and loss metrics."""
        self.realized_pnl = Gauge(
            "realized_pnl_usd",
            "Realized profit and loss in USD",
            ["exchange", "symbol", "strategy"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.unrealized_pnl = Gauge(
            "unrealized_pnl_usd",
            "Unrealized profit and loss in USD",
            ["exchange", "symbol"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.total_pnl = Gauge(
            "total_pnl_usd",
            "Total profit and loss in USD",
            ["strategy"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.win_rate = Gauge(
            "win_rate_percent",
            "Win rate percentage",
            ["strategy", "timeframe"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

    def _init_risk_metrics(self) -> None:
        """Initialize risk management metrics."""
        self.drawdown = Gauge(
            "drawdown_percent",
            "Current drawdown percentage",
            ["strategy"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.max_drawdown = Gauge(
            "max_drawdown_percent",
            "Maximum drawdown percentage",
            ["strategy", "timeframe"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.var_95 = Gauge(
            "value_at_risk_95_usd",
            "Value at Risk (95% confidence) in USD",
            ["strategy"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.sharpe_ratio = Gauge(
            "sharpe_ratio",
            "Sharpe ratio",
            ["strategy", "timeframe"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.risk_violations = Counter(
            "risk_violations_total",
            "Total number of risk limit violations",
            ["violation_type", "severity"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

    def _init_exchange_metrics(self) -> None:
        """Initialize exchange connectivity metrics."""
        self.exchange_connected = Gauge(
            "exchange_connected",
            "Exchange connection status (1=connected, 0=disconnected)",
            ["exchange"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.exchange_errors = Counter(
            "exchange_errors_total",
            "Total number of exchange errors",
            ["exchange", "error_type"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.api_calls = Counter(
            "api_calls_total",
            "Total number of API calls",
            ["exchange", "endpoint", "method"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.rate_limit_remaining = Gauge(
            "rate_limit_remaining",
            "Remaining API rate limit",
            ["exchange", "limit_type"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

    def _init_strategy_metrics(self) -> None:
        """Initialize strategy performance metrics."""
        self.signals_generated = Counter(
            "signals_generated_total",
            "Total number of trading signals generated",
            ["strategy", "signal_type", "symbol"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.strategy_state = Gauge(
            "strategy_state",
            "Strategy state (1=active, 0=inactive)",
            ["strategy"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.model_predictions = Counter(
            "model_predictions_total",
            "Total number of ML model predictions",
            ["model_name", "prediction_type"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        buckets = self._get_histogram_buckets("prediction_confidence")
        self.prediction_confidence = Histogram(
            "prediction_confidence",
            "ML model prediction confidence",
            ["model_name"],
            buckets=buckets,
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

    def _init_system_metrics(self) -> None:
        """Initialize system health metrics."""
        self.cpu_usage = Gauge(
            "cpu_usage_percent",
            "CPU usage percentage",
            ["component"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.memory_usage = Gauge(
            "memory_usage_bytes",
            "Memory usage in bytes",
            ["component"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.queue_size = Gauge(
            "queue_size",
            "Current queue size",
            ["queue_name", "queue_type"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        self.data_processing_lag = Gauge(
            "data_processing_lag_seconds",
            "Data processing lag in seconds",
            ["data_type"],
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

    def _init_latency_metrics(self) -> None:
        """Initialize latency tracking metrics."""
        buckets = self._get_histogram_buckets("order_latency")
        self.order_latency = Histogram(
            "order_latency_milliseconds",
            "Order placement latency in milliseconds",
            ["exchange", "order_type"],
            buckets=buckets,
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        buckets = self._get_histogram_buckets("execution_latency")
        self.execution_latency = Histogram(
            "execution_latency_milliseconds",
            "Order execution latency in milliseconds",
            ["exchange"],
            buckets=buckets,
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

        buckets = self._get_histogram_buckets("market_data_latency")
        self.market_data_latency = Histogram(
            "market_data_latency_milliseconds",
            "Market data processing latency in milliseconds",
            ["exchange", "data_type"],
            buckets=buckets,
            namespace=self.namespace,
            subsystem=self.subsystem,
            registry=self.registry,
        )

    def _get_histogram_buckets(self, metric_type: str) -> List[float]:
        """Get histogram buckets from config.

        Args:
            metric_type: Type of metric to get buckets for

        Returns:
            List of bucket boundaries
        """
        # Get from config or use environment variable
        config_key = f"{metric_type}_buckets"
        env_key = f"METRICS_{metric_type.upper()}_BUCKETS"

        buckets_config = self.config.get(config_key, os.getenv(env_key))

        if buckets_config:
            try:
                if isinstance(buckets_config, str):
                    return [float(b) for b in buckets_config.split(",")]
                elif isinstance(buckets_config, list):
                    return [float(b) for b in buckets_config]
            except (ValueError, AttributeError) as e:
                logger.warning(
                    "Invalid bucket configuration, using defaults",
                    metric_type=metric_type,
                    error=str(e)
                )

        # Default buckets based on metric type
        default_buckets = {
            "order_size": [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
            "slippage": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0],
            "prediction_confidence": [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
            "order_latency": [1, 2, 5, 10, 20, 50, 100, 200, 500],
            "execution_latency": [5, 10, 25, 50, 100, 250, 500, 1000, 2000],
            "market_data_latency": [0.5, 1, 2, 5, 10, 20, 50, 100, 200],
        }

        return default_buckets.get(metric_type, [0.1, 0.5, 1.0, 2.5, 5.0, 10.0])

    def get_registry(self) -> CollectorRegistry:
        """Get the Prometheus collector registry.

        Returns:
            CollectorRegistry instance
        """
        return self.registry

    def reset_all(self) -> None:
        """Reset all metrics (used for testing)."""
        logger.warning("Resetting all metrics")
        # Create new registry to clear all metrics
        self.registry = CollectorRegistry()
        self._init_order_metrics()
        self._init_trade_metrics()
        self._init_position_metrics()
        self._init_pnl_metrics()
        self._init_risk_metrics()
        self._init_exchange_metrics()
        self._init_strategy_metrics()
        self._init_system_metrics()
        self._init_latency_metrics()
