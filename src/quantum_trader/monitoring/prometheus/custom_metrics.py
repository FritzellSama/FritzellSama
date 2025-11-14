"""Custom Prometheus Metrics for Quantum Trader AI.

Production-ready custom metrics collection for comprehensive
trading system monitoring and observability.
"""

import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timezone
from enum import Enum
import threading

from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry, generate_latest,
    CONTENT_TYPE_LATEST
)
from structlog import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class CustomMetricsCollector:
    """Collect and manage custom Prometheus metrics.

    Provides a centralized registry for all trading-specific metrics
    with proper labeling and categorization.

    Attributes:
        config: Configuration dictionary
        registry: Prometheus metrics registry
        trading_metrics: Trading-specific metrics
        system_metrics: System performance metrics
        risk_metrics: Risk management metrics

    Example:
        >>> config = {
        ...     "namespace": "quantum_trader",
        ...     "enable_detailed_metrics": True
        ... }
        >>> collector = CustomMetricsCollector(config)
        >>> collector.record_trade(
        ...     exchange="binance",
        ...     symbol="BTC/USDT",
        ...     side="buy",
        ...     pnl=Decimal("150.50")
        ... )
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize custom metrics collector.

        Args:
            config: Configuration dictionary containing:
                - namespace: Metric namespace prefix
                - enable_detailed_metrics: Enable fine-grained metrics
                - histogram_buckets: Custom histogram buckets
                - summary_quantiles: Summary quantile configuration

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.namespace = self.config.get("namespace", "quantum_trader")
        self.enable_detailed = self.config.get("enable_detailed_metrics", True)

        # Custom histogram buckets for latency metrics (in seconds)
        self.latency_buckets = self.config.get(
            "latency_buckets",
            [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )

        # Custom histogram buckets for financial metrics
        self.pnl_buckets = self.config.get(
            "pnl_buckets",
            [-10000, -5000, -1000, -100, 0, 100, 1000, 5000, 10000, 50000, 100000]
        )

        # Thread safety
        self._lock = threading.RLock()

        # Create custom registry
        self.registry = CollectorRegistry()

        # Initialize metric groups
        self._init_trading_metrics()
        self._init_system_metrics()
        self._init_risk_metrics()
        self._init_exchange_metrics()
        self._init_strategy_metrics()
        self._init_ml_metrics()

        logger.info(
            "custom_metrics_collector_initialized",
            namespace=self.namespace,
            detailed_metrics=self.enable_detailed
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        namespace = self.config.get("namespace", "quantum_trader")
        if not namespace:
            raise ValueError("namespace cannot be empty")

    def _init_trading_metrics(self) -> None:
        """Initialize trading-related metrics."""
        # Order metrics
        self.orders_submitted = Counter(
            f"{self.namespace}_orders_submitted_total",
            "Total orders submitted",
            ["exchange", "symbol", "order_type", "side"],
            registry=self.registry
        )

        self.orders_filled = Counter(
            f"{self.namespace}_orders_filled_total",
            "Total orders filled",
            ["exchange", "symbol", "order_type", "side"],
            registry=self.registry
        )

        self.orders_cancelled = Counter(
            f"{self.namespace}_orders_cancelled_total",
            "Total orders cancelled",
            ["exchange", "symbol", "reason"],
            registry=self.registry
        )

        self.orders_rejected = Counter(
            f"{self.namespace}_orders_rejected_total",
            "Total orders rejected",
            ["exchange", "symbol", "reason"],
            registry=self.registry
        )

        self.active_orders = Gauge(
            f"{self.namespace}_active_orders",
            "Number of active orders",
            ["exchange", "symbol"],
            registry=self.registry
        )

        # Trade metrics
        self.trades_total = Counter(
            f"{self.namespace}_trades_total",
            "Total trades executed",
            ["exchange", "symbol", "side"],
            registry=self.registry
        )

        self.trades_won = Counter(
            f"{self.namespace}_trades_won_total",
            "Total winning trades",
            ["exchange", "symbol"],
            registry=self.registry
        )

        self.trades_lost = Counter(
            f"{self.namespace}_trades_lost_total",
            "Total losing trades",
            ["exchange", "symbol"],
            registry=self.registry
        )

        # PnL metrics
        self.pnl_total = Gauge(
            f"{self.namespace}_pnl_total",
            "Total profit and loss (USD)",
            ["exchange"],
            registry=self.registry
        )

        self.pnl_realized = Counter(
            f"{self.namespace}_pnl_realized_total",
            "Total realized PnL (USD)",
            ["exchange", "symbol"],
            registry=self.registry
        )

        self.pnl_unrealized = Gauge(
            f"{self.namespace}_pnl_unrealized",
            "Current unrealized PnL (USD)",
            ["exchange", "symbol"],
            registry=self.registry
        )

        self.pnl_per_trade = Histogram(
            f"{self.namespace}_pnl_per_trade",
            "PnL distribution per trade",
            ["exchange", "symbol"],
            buckets=self.pnl_buckets,
            registry=self.registry
        )

        # Volume metrics
        self.volume_traded = Counter(
            f"{self.namespace}_volume_traded_total",
            "Total volume traded (USD)",
            ["exchange", "symbol"],
            registry=self.registry
        )

        self.trade_size = Histogram(
            f"{self.namespace}_trade_size_usd",
            "Trade size distribution (USD)",
            ["exchange", "symbol"],
            buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
            registry=self.registry
        )

        # Position metrics
        self.position_count = Gauge(
            f"{self.namespace}_open_positions",
            "Number of open positions",
            ["exchange"],
            registry=self.registry
        )

        self.position_size = Gauge(
            f"{self.namespace}_position_size_usd",
            "Position size in USD",
            ["exchange", "symbol", "side"],
            registry=self.registry
        )

        self.position_duration = Histogram(
            f"{self.namespace}_position_duration_seconds",
            "Position hold duration",
            ["exchange", "symbol"],
            buckets=[60, 300, 900, 1800, 3600, 7200, 14400, 28800, 86400],
            registry=self.registry
        )

    def _init_system_metrics(self) -> None:
        """Initialize system performance metrics."""
        # Latency metrics
        self.order_latency = Histogram(
            f"{self.namespace}_order_latency_seconds",
            "Order submission latency",
            ["exchange", "order_type"],
            buckets=self.latency_buckets,
            registry=self.registry
        )

        self.fill_latency = Histogram(
            f"{self.namespace}_fill_latency_seconds",
            "Order fill latency",
            ["exchange"],
            buckets=self.latency_buckets,
            registry=self.registry
        )

        self.market_data_latency = Histogram(
            f"{self.namespace}_market_data_latency_seconds",
            "Market data processing latency",
            ["exchange", "data_type"],
            buckets=self.latency_buckets,
            registry=self.registry
        )

        # Processing metrics
        self.events_processed = Counter(
            f"{self.namespace}_events_processed_total",
            "Total events processed",
            ["event_type"],
            registry=self.registry
        )

        self.processing_errors = Counter(
            f"{self.namespace}_processing_errors_total",
            "Total processing errors",
            ["component", "error_type"],
            registry=self.registry
        )

        self.queue_size = Gauge(
            f"{self.namespace}_queue_size",
            "Event queue size",
            ["queue_name"],
            registry=self.registry
        )

        # Cache metrics
        self.cache_hits = Counter(
            f"{self.namespace}_cache_hits_total",
            "Cache hit count",
            ["cache_name"],
            registry=self.registry
        )

        self.cache_misses = Counter(
            f"{self.namespace}_cache_misses_total",
            "Cache miss count",
            ["cache_name"],
            registry=self.registry
        )

    def _init_risk_metrics(self) -> None:
        """Initialize risk management metrics."""
        self.risk_limit_breaches = Counter(
            f"{self.namespace}_risk_limit_breaches_total",
            "Risk limit breaches",
            ["limit_type", "severity"],
            registry=self.registry
        )

        self.max_drawdown_pct = Gauge(
            f"{self.namespace}_max_drawdown_pct",
            "Maximum drawdown percentage",
            ["exchange"],
            registry=self.registry
        )

        self.current_leverage = Gauge(
            f"{self.namespace}_current_leverage",
            "Current leverage ratio",
            ["exchange"],
            registry=self.registry
        )

        self.var_value = Gauge(
            f"{self.namespace}_var_usd",
            "Value at Risk (USD)",
            ["exchange", "confidence_level"],
            registry=self.registry
        )

        self.exposure_total = Gauge(
            f"{self.namespace}_exposure_total_usd",
            "Total market exposure (USD)",
            ["exchange", "asset_class"],
            registry=self.registry
        )

        self.sharpe_ratio = Gauge(
            f"{self.namespace}_sharpe_ratio",
            "Sharpe ratio",
            ["exchange", "timeframe"],
            registry=self.registry
        )

        self.win_rate = Gauge(
            f"{self.namespace}_win_rate",
            "Win rate percentage",
            ["exchange", "strategy"],
            registry=self.registry
        )

    def _init_exchange_metrics(self) -> None:
        """Initialize exchange connectivity metrics."""
        self.exchange_connected = Gauge(
            f"{self.namespace}_exchange_connected",
            "Exchange connection status (1=connected, 0=disconnected)",
            ["exchange"],
            registry=self.registry
        )

        self.exchange_api_latency = Histogram(
            f"{self.namespace}_exchange_api_latency_ms",
            "Exchange API latency (milliseconds)",
            ["exchange", "endpoint"],
            buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
            registry=self.registry
        )

        self.exchange_requests = Counter(
            f"{self.namespace}_exchange_requests_total",
            "Total API requests",
            ["exchange", "endpoint", "method"],
            registry=self.registry
        )

        self.exchange_errors = Counter(
            f"{self.namespace}_exchange_errors_total",
            "Exchange API errors",
            ["exchange", "error_type"],
            registry=self.registry
        )

        self.rate_limit_hits = Counter(
            f"{self.namespace}_rate_limit_hits_total",
            "Rate limit hits",
            ["exchange", "endpoint"],
            registry=self.registry
        )

        self.websocket_reconnections = Counter(
            f"{self.namespace}_websocket_reconnections_total",
            "WebSocket reconnections",
            ["exchange"],
            registry=self.registry
        )

    def _init_strategy_metrics(self) -> None:
        """Initialize strategy performance metrics."""
        self.strategy_signals = Counter(
            f"{self.namespace}_strategy_signals_total",
            "Strategy signals generated",
            ["strategy", "signal_type"],
            registry=self.registry
        )

        self.strategy_pnl = Gauge(
            f"{self.namespace}_strategy_pnl_usd",
            "Strategy PnL (USD)",
            ["strategy"],
            registry=self.registry
        )

        self.strategy_active = Gauge(
            f"{self.namespace}_strategy_active",
            "Strategy active status (1=active, 0=inactive)",
            ["strategy"],
            registry=self.registry
        )

        self.backtest_results = Gauge(
            f"{self.namespace}_backtest_sharpe_ratio",
            "Backtest Sharpe ratio",
            ["strategy"],
            registry=self.registry
        )

    def _init_ml_metrics(self) -> None:
        """Initialize ML model metrics."""
        self.model_predictions = Counter(
            f"{self.namespace}_ml_predictions_total",
            "ML model predictions",
            ["model_name", "prediction_class"],
            registry=self.registry
        )

        self.model_accuracy = Gauge(
            f"{self.namespace}_ml_accuracy",
            "ML model accuracy",
            ["model_name"],
            registry=self.registry
        )

        self.model_inference_time = Histogram(
            f"{self.namespace}_ml_inference_seconds",
            "ML model inference time",
            ["model_name"],
            buckets=self.latency_buckets,
            registry=self.registry
        )

        self.feature_importance = Gauge(
            f"{self.namespace}_ml_feature_importance",
            "Feature importance scores",
            ["model_name", "feature_name"],
            registry=self.registry
        )

    # Trading metric update methods
    def record_order_submitted(
        self,
        exchange: str,
        symbol: str,
        order_type: str,
        side: str
    ) -> None:
        """Record order submission."""
        self.orders_submitted.labels(
            exchange=exchange,
            symbol=symbol,
            order_type=order_type,
            side=side
        ).inc()

    def record_order_filled(
        self,
        exchange: str,
        symbol: str,
        order_type: str,
        side: str,
        fill_time: float
    ) -> None:
        """Record order fill."""
        self.orders_filled.labels(
            exchange=exchange,
            symbol=symbol,
            order_type=order_type,
            side=side
        ).inc()

        self.fill_latency.labels(exchange=exchange).observe(fill_time)

    def record_trade(
        self,
        exchange: str,
        symbol: str,
        side: str,
        size_usd: Decimal,
        pnl: Optional[Decimal] = None
    ) -> None:
        """Record completed trade."""
        self.trades_total.labels(
            exchange=exchange,
            symbol=symbol,
            side=side
        ).inc()

        size_float = float(size_usd)
        self.volume_traded.labels(exchange=exchange, symbol=symbol).inc(size_float)
        self.trade_size.labels(exchange=exchange, symbol=symbol).observe(size_float)

        if pnl is not None:
            pnl_float = float(pnl)
            self.pnl_per_trade.labels(exchange=exchange, symbol=symbol).observe(pnl_float)

            if pnl > 0:
                self.trades_won.labels(exchange=exchange, symbol=symbol).inc()
            elif pnl < 0:
                self.trades_lost.labels(exchange=exchange, symbol=symbol).inc()

    def update_pnl(
        self,
        exchange: str,
        total_pnl: Decimal,
        symbol: Optional[str] = None,
        unrealized_pnl: Optional[Decimal] = None
    ) -> None:
        """Update PnL metrics."""
        self.pnl_total.labels(exchange=exchange).set(float(total_pnl))

        if symbol and unrealized_pnl is not None:
            self.pnl_unrealized.labels(
                exchange=exchange,
                symbol=symbol
            ).set(float(unrealized_pnl))

    def record_latency(
        self,
        metric_type: str,
        latency_seconds: float,
        **labels
    ) -> None:
        """Record latency metric."""
        if metric_type == "order":
            self.order_latency.labels(**labels).observe(latency_seconds)
        elif metric_type == "market_data":
            self.market_data_latency.labels(**labels).observe(latency_seconds)
        elif metric_type == "exchange_api":
            self.exchange_api_latency.labels(**labels).observe(latency_seconds * 1000)

    def update_risk_metrics(
        self,
        exchange: str,
        drawdown_pct: Optional[float] = None,
        leverage: Optional[float] = None,
        var_usd: Optional[Decimal] = None,
        sharpe: Optional[float] = None
    ) -> None:
        """Update risk metrics."""
        if drawdown_pct is not None:
            self.max_drawdown_pct.labels(exchange=exchange).set(drawdown_pct)

        if leverage is not None:
            self.current_leverage.labels(exchange=exchange).set(leverage)

        if var_usd is not None:
            self.var_value.labels(
                exchange=exchange,
                confidence_level="95"
            ).set(float(var_usd))

        if sharpe is not None:
            self.sharpe_ratio.labels(
                exchange=exchange,
                timeframe="daily"
            ).set(sharpe)

    def set_exchange_status(self, exchange: str, connected: bool) -> None:
        """Set exchange connection status."""
        self.exchange_connected.labels(exchange=exchange).set(1 if connected else 0)

    def record_exchange_error(self, exchange: str, error_type: str) -> None:
        """Record exchange error."""
        self.exchange_errors.labels(
            exchange=exchange,
            error_type=error_type
        ).inc()

    def get_metrics(self) -> bytes:
        """Get current metrics in Prometheus format.

        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get content type for metrics endpoint.

        Returns:
            Content type string
        """
        return CONTENT_TYPE_LATEST

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check.

        Returns:
            Health status dictionary
        """
        return {
            "healthy": True,
            "namespace": self.namespace,
            "detailed_metrics": self.enable_detailed,
            "registry_collectors": len(self.registry._collector_to_names)
        }
