"""Grafana Alert Rules Generator for Quantum Trader AI.

Production-ready alert rule generation for Grafana monitoring.
Creates alerting rules for trading performance, system health, and risk metrics.
"""

import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import json

from prometheus_client import Counter, Gauge
from structlog import get_logger

logger = get_logger(__name__)


class AlertRuleGenerator:
    """Generate Grafana alert rules for trading system monitoring.

    Creates alert rules for:
    - Trading performance degradation
    - System health issues
    - Risk threshold breaches
    - Exchange connectivity
    - Order execution anomalies

    Attributes:
        config: Configuration dictionary
        rule_groups: Generated rule groups

    Example:
        >>> config = {
        ...     "datasource_uid": "prometheus",
        ...     "namespace": "quantum_trader",
        ...     "eval_interval": "1m"
        ... }
        >>> generator = AlertRuleGenerator(config)
        >>> rules = await generator.generate_all_rules()
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize alert rule generator.

        Args:
            config: Configuration dictionary containing:
                - datasource_uid: Prometheus datasource UID
                - namespace: Alert rule namespace
                - eval_interval: Evaluation interval (e.g., "1m")
                - notification_channels: List of notification channel UIDs
                - default_for: Duration before alert fires
                - thresholds: Dict of threshold values

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.datasource_uid = self.config.get("datasource_uid", "prometheus")
        self.namespace = self.config.get("namespace", "quantum_trader")
        self.eval_interval = self.config.get("eval_interval", "1m")
        self.notification_channels = self.config.get("notification_channels", [])
        self.default_for = self.config.get("default_for", "5m")
        self.thresholds = self.config.get("thresholds", {})

        self.rule_groups: List[Dict[str, Any]] = []

        # Metrics
        self._rules_generated = Counter(
            "quantum_trader_alert_rules_generated_total",
            "Total alert rules generated",
            ["rule_group"]
        )

        logger.info(
            "alert_rule_generator_initialized",
            datasource=self.datasource_uid,
            namespace=self.namespace,
            eval_interval=self.eval_interval
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

        namespace = self.config.get("namespace", "quantum_trader")
        if not namespace:
            raise ValueError("namespace cannot be empty")

    async def generate_all_rules(self) -> List[Dict[str, Any]]:
        """Generate all alert rule groups.

        Returns:
            List of alert rule groups

        Raises:
            Exception: If rule generation fails
        """
        try:
            logger.info("generating_all_alert_rules")

            self.rule_groups = []

            # Generate different rule categories
            await self._generate_trading_rules()
            await self._generate_system_health_rules()
            await self._generate_risk_rules()
            await self._generate_exchange_rules()
            await self._generate_order_rules()
            await self._generate_performance_rules()

            logger.info(
                "alert_rules_generated",
                total_groups=len(self.rule_groups),
                total_rules=sum(len(g.get("rules", [])) for g in self.rule_groups)
            )

            return self.rule_groups

        except Exception as e:
            logger.error("alert_rule_generation_failed", error=str(e))
            raise

    async def _generate_trading_rules(self) -> None:
        """Generate trading performance alert rules."""
        try:
            pnl_threshold = self.thresholds.get("daily_loss_threshold", -10000)
            win_rate_threshold = self.thresholds.get("min_win_rate", 0.45)
            drawdown_threshold = self.thresholds.get("max_drawdown_pct", 10)

            rules = [
                {
                    "uid": f"{self.namespace}_trading_daily_loss",
                    "title": "Daily Loss Threshold Exceeded",
                    "condition": "C",
                    "data": [
                        {
                            "refId": "A",
                            "queryType": "range",
                            "relativeTimeRange": {
                                "from": 86400,  # 24 hours
                                "to": 0
                            },
                            "datasourceUid": self.datasource_uid,
                            "model": {
                                "expr": "sum(quantum_trader_pnl_total)",
                                "intervalMs": 1000,
                                "maxDataPoints": 43200
                            }
                        },
                        {
                            "refId": "C",
                            "queryType": "",
                            "relativeTimeRange": {
                                "from": 0,
                                "to": 0
                            },
                            "datasourceUid": "-100",
                            "model": {
                                "type": "classic_conditions",
                                "conditions": [
                                    {
                                        "evaluator": {
                                            "params": [pnl_threshold],
                                            "type": "lt"
                                        },
                                        "operator": {"type": "and"},
                                        "query": {"params": ["A"]},
                                        "type": "query"
                                    }
                                ]
                            }
                        }
                    ],
                    "noDataState": "NoData",
                    "execErrState": "Error",
                    "for": self.default_for,
                    "annotations": {
                        "description": f"Daily PnL has fallen below ${pnl_threshold}",
                        "summary": "Critical trading loss detected"
                    },
                    "labels": {
                        "severity": "critical",
                        "category": "trading"
                    }
                },
                {
                    "uid": f"{self.namespace}_trading_win_rate_low",
                    "title": "Win Rate Below Threshold",
                    "condition": "C",
                    "data": [
                        {
                            "refId": "A",
                            "queryType": "range",
                            "relativeTimeRange": {
                                "from": 3600,  # 1 hour
                                "to": 0
                            },
                            "datasourceUid": self.datasource_uid,
                            "model": {
                                "expr": "sum(rate(quantum_trader_trades_won_total[1h])) / sum(rate(quantum_trader_trades_total[1h]))",
                                "intervalMs": 1000,
                                "maxDataPoints": 43200
                            }
                        },
                        {
                            "refId": "C",
                            "queryType": "",
                            "datasourceUid": "-100",
                            "model": {
                                "type": "classic_conditions",
                                "conditions": [
                                    {
                                        "evaluator": {
                                            "params": [win_rate_threshold],
                                            "type": "lt"
                                        },
                                        "operator": {"type": "and"},
                                        "query": {"params": ["A"]},
                                        "type": "query"
                                    }
                                ]
                            }
                        }
                    ],
                    "noDataState": "NoData",
                    "execErrState": "Error",
                    "for": "10m",
                    "annotations": {
                        "description": f"Win rate has dropped below {win_rate_threshold * 100}%",
                        "summary": "Trading strategy performance degraded"
                    },
                    "labels": {
                        "severity": "warning",
                        "category": "trading"
                    }
                },
                {
                    "uid": f"{self.namespace}_trading_max_drawdown",
                    "title": "Maximum Drawdown Exceeded",
                    "condition": "C",
                    "data": [
                        {
                            "refId": "A",
                            "queryType": "range",
                            "relativeTimeRange": {
                                "from": 86400,
                                "to": 0
                            },
                            "datasourceUid": self.datasource_uid,
                            "model": {
                                "expr": "quantum_trader_max_drawdown_pct",
                                "intervalMs": 1000,
                                "maxDataPoints": 43200
                            }
                        },
                        {
                            "refId": "C",
                            "queryType": "",
                            "datasourceUid": "-100",
                            "model": {
                                "type": "classic_conditions",
                                "conditions": [
                                    {
                                        "evaluator": {
                                            "params": [drawdown_threshold],
                                            "type": "gt"
                                        },
                                        "operator": {"type": "and"},
                                        "query": {"params": ["A"]},
                                        "type": "query"
                                    }
                                ]
                            }
                        }
                    ],
                    "noDataState": "NoData",
                    "execErrState": "Error",
                    "for": "5m",
                    "annotations": {
                        "description": f"Drawdown exceeded {drawdown_threshold}%",
                        "summary": "Risk limit breach - excessive drawdown"
                    },
                    "labels": {
                        "severity": "critical",
                        "category": "risk"
                    }
                }
            ]

            self.rule_groups.append({
                "name": "Trading Performance",
                "interval": self.eval_interval,
                "rules": rules
            })

            self._rules_generated.labels(rule_group="trading_performance").inc(len(rules))

        except Exception as e:
            logger.error("trading_rules_generation_failed", error=str(e))
            raise

    async def _generate_system_health_rules(self) -> None:
        """Generate system health alert rules."""
        try:
            cpu_threshold = self.thresholds.get("cpu_threshold_pct", 80)
            memory_threshold = self.thresholds.get("memory_threshold_pct", 85)
            disk_threshold = self.thresholds.get("disk_threshold_pct", 90)

            rules = [
                {
                    "uid": f"{self.namespace}_system_cpu_high",
                    "title": "High CPU Usage",
                    "condition": "C",
                    "data": [
                        {
                            "refId": "A",
                            "queryType": "range",
                            "relativeTimeRange": {
                                "from": 300,
                                "to": 0
                            },
                            "datasourceUid": self.datasource_uid,
                            "model": {
                                "expr": "avg(rate(process_cpu_seconds_total{job='quantum_trader'}[5m])) * 100",
                                "intervalMs": 1000,
                                "maxDataPoints": 43200
                            }
                        },
                        {
                            "refId": "C",
                            "queryType": "",
                            "datasourceUid": "-100",
                            "model": {
                                "type": "classic_conditions",
                                "conditions": [
                                    {
                                        "evaluator": {
                                            "params": [cpu_threshold],
                                            "type": "gt"
                                        },
                                        "operator": {"type": "and"},
                                        "query": {"params": ["A"]},
                                        "type": "query"
                                    }
                                ]
                            }
                        }
                    ],
                    "noDataState": "NoData",
                    "execErrState": "Error",
                    "for": "5m",
                    "annotations": {
                        "description": f"CPU usage above {cpu_threshold}%",
                        "summary": "System resource constraint"
                    },
                    "labels": {
                        "severity": "warning",
                        "category": "system"
                    }
                },
                {
                    "uid": f"{self.namespace}_system_memory_high",
                    "title": "High Memory Usage",
                    "condition": "C",
                    "data": [
                        {
                            "refId": "A",
                            "queryType": "range",
                            "relativeTimeRange": {
                                "from": 300,
                                "to": 0
                            },
                            "datasourceUid": self.datasource_uid,
                            "model": {
                                "expr": "(process_resident_memory_bytes{job='quantum_trader'} / process_virtual_memory_max_bytes{job='quantum_trader'}) * 100",
                                "intervalMs": 1000,
                                "maxDataPoints": 43200
                            }
                        },
                        {
                            "refId": "C",
                            "queryType": "",
                            "datasourceUid": "-100",
                            "model": {
                                "type": "classic_conditions",
                                "conditions": [
                                    {
                                        "evaluator": {
                                            "params": [memory_threshold],
                                            "type": "gt"
                                        },
                                        "operator": {"type": "and"},
                                        "query": {"params": ["A"]},
                                        "type": "query"
                                    }
                                ]
                            }
                        }
                    ],
                    "noDataState": "NoData",
                    "execErrState": "Error",
                    "for": "5m",
                    "annotations": {
                        "description": f"Memory usage above {memory_threshold}%",
                        "summary": "Memory pressure detected"
                    },
                    "labels": {
                        "severity": "warning",
                        "category": "system"
                    }
                },
                {
                    "uid": f"{self.namespace}_system_service_down",
                    "title": "Service Down",
                    "condition": "C",
                    "data": [
                        {
                            "refId": "A",
                            "queryType": "range",
                            "relativeTimeRange": {
                                "from": 60,
                                "to": 0
                            },
                            "datasourceUid": self.datasource_uid,
                            "model": {
                                "expr": "up{job='quantum_trader'}",
                                "intervalMs": 1000,
                                "maxDataPoints": 43200
                            }
                        },
                        {
                            "refId": "C",
                            "queryType": "",
                            "datasourceUid": "-100",
                            "model": {
                                "type": "classic_conditions",
                                "conditions": [
                                    {
                                        "evaluator": {
                                            "params": [1],
                                            "type": "lt"
                                        },
                                        "operator": {"type": "and"},
                                        "query": {"params": ["A"]},
                                        "type": "query"
                                    }
                                ]
                            }
                        }
                    ],
                    "noDataState": "Alerting",
                    "execErrState": "Error",
                    "for": "1m",
                    "annotations": {
                        "description": "Quantum Trader service is down",
                        "summary": "CRITICAL: Service unavailable"
                    },
                    "labels": {
                        "severity": "critical",
                        "category": "system"
                    }
                }
            ]

            self.rule_groups.append({
                "name": "System Health",
                "interval": self.eval_interval,
                "rules": rules
            })

            self._rules_generated.labels(rule_group="system_health").inc(len(rules))

        except Exception as e:
            logger.error("system_health_rules_generation_failed", error=str(e))
            raise

    async def _generate_risk_rules(self) -> None:
        """Generate risk management alert rules."""
        try:
            position_limit = self.thresholds.get("position_size_limit", 1000000)
            leverage_limit = self.thresholds.get("max_leverage", 3)
            var_threshold = self.thresholds.get("var_threshold", 50000)

            rules = [
                {
                    "uid": f"{self.namespace}_risk_position_limit",
                    "title": "Position Size Limit Exceeded",
                    "condition": "C",
                    "data": [
                        {
                            "refId": "A",
                            "queryType": "range",
                            "relativeTimeRange": {
                                "from": 60,
                                "to": 0
                            },
                            "datasourceUid": self.datasource_uid,
                            "model": {
                                "expr": "max(quantum_trader_position_size_usd)",
                                "intervalMs": 1000,
                                "maxDataPoints": 43200
                            }
                        },
                        {
                            "refId": "C",
                            "queryType": "",
                            "datasourceUid": "-100",
                            "model": {
                                "type": "classic_conditions",
                                "conditions": [
                                    {
                                        "evaluator": {
                                            "params": [position_limit],
                                            "type": "gt"
                                        },
                                        "operator": {"type": "and"},
                                        "query": {"params": ["A"]},
                                        "type": "query"
                                    }
                                ]
                            }
                        }
                    ],
                    "noDataState": "NoData",
                    "execErrState": "Error",
                    "for": "1m",
                    "annotations": {
                        "description": f"Position size exceeded ${position_limit}",
                        "summary": "Risk limit breach"
                    },
                    "labels": {
                        "severity": "critical",
                        "category": "risk"
                    }
                },
                {
                    "uid": f"{self.namespace}_risk_leverage_high",
                    "title": "High Leverage Detected",
                    "condition": "C",
                    "data": [
                        {
                            "refId": "A",
                            "queryType": "range",
                            "relativeTimeRange": {
                                "from": 60,
                                "to": 0
                            },
                            "datasourceUid": self.datasource_uid,
                            "model": {
                                "expr": "quantum_trader_current_leverage",
                                "intervalMs": 1000,
                                "maxDataPoints": 43200
                            }
                        },
                        {
                            "refId": "C",
                            "queryType": "",
                            "datasourceUid": "-100",
                            "model": {
                                "type": "classic_conditions",
                                "conditions": [
                                    {
                                        "evaluator": {
                                            "params": [leverage_limit],
                                            "type": "gt"
                                        },
                                        "operator": {"type": "and"},
                                        "query": {"params": ["A"]},
                                        "type": "query"
                                    }
                                ]
                            }
                        }
                    ],
                    "noDataState": "NoData",
                    "execErrState": "Error",
                    "for": "2m",
                    "annotations": {
                        "description": f"Leverage exceeded {leverage_limit}x",
                        "summary": "Excessive leverage warning"
                    },
                    "labels": {
                        "severity": "warning",
                        "category": "risk"
                    }
                }
            ]

            self.rule_groups.append({
                "name": "Risk Management",
                "interval": self.eval_interval,
                "rules": rules
            })

            self._rules_generated.labels(rule_group="risk_management").inc(len(rules))

        except Exception as e:
            logger.error("risk_rules_generation_failed", error=str(e))
            raise

    async def _generate_exchange_rules(self) -> None:
        """Generate exchange connectivity alert rules."""
        try:
            latency_threshold = self.thresholds.get("api_latency_ms", 500)
            error_rate_threshold = self.thresholds.get("error_rate_pct", 5)

            rules = [
                {
                    "uid": f"{self.namespace}_exchange_disconnected",
                    "title": "Exchange Disconnected",
                    "condition": "C",
                    "data": [
                        {
                            "refId": "A",
                            "queryType": "range",
                            "relativeTimeRange": {
                                "from": 120,
                                "to": 0
                            },
                            "datasourceUid": self.datasource_uid,
                            "model": {
                                "expr": "quantum_trader_exchange_connected",
                                "intervalMs": 1000,
                                "maxDataPoints": 43200
                            }
                        },
                        {
                            "refId": "C",
                            "queryType": "",
                            "datasourceUid": "-100",
                            "model": {
                                "type": "classic_conditions",
                                "conditions": [
                                    {
                                        "evaluator": {
                                            "params": [1],
                                            "type": "lt"
                                        },
                                        "operator": {"type": "and"},
                                        "query": {"params": ["A"]},
                                        "type": "query"
                                    }
                                ]
                            }
                        }
                    ],
                    "noDataState": "Alerting",
                    "execErrState": "Error",
                    "for": "2m",
                    "annotations": {
                        "description": "Lost connection to exchange",
                        "summary": "Exchange connectivity issue"
                    },
                    "labels": {
                        "severity": "critical",
                        "category": "exchange"
                    }
                },
                {
                    "uid": f"{self.namespace}_exchange_high_latency",
                    "title": "High Exchange API Latency",
                    "condition": "C",
                    "data": [
                        {
                            "refId": "A",
                            "queryType": "range",
                            "relativeTimeRange": {
                                "from": 300,
                                "to": 0
                            },
                            "datasourceUid": self.datasource_uid,
                            "model": {
                                "expr": "avg(quantum_trader_exchange_api_latency_ms)",
                                "intervalMs": 1000,
                                "maxDataPoints": 43200
                            }
                        },
                        {
                            "refId": "C",
                            "queryType": "",
                            "datasourceUid": "-100",
                            "model": {
                                "type": "classic_conditions",
                                "conditions": [
                                    {
                                        "evaluator": {
                                            "params": [latency_threshold],
                                            "type": "gt"
                                        },
                                        "operator": {"type": "and"},
                                        "query": {"params": ["A"]},
                                        "type": "query"
                                    }
                                ]
                            }
                        }
                    ],
                    "noDataState": "NoData",
                    "execErrState": "Error",
                    "for": "5m",
                    "annotations": {
                        "description": f"API latency above {latency_threshold}ms",
                        "summary": "Exchange performance degraded"
                    },
                    "labels": {
                        "severity": "warning",
                        "category": "exchange"
                    }
                }
            ]

            self.rule_groups.append({
                "name": "Exchange Connectivity",
                "interval": self.eval_interval,
                "rules": rules
            })

            self._rules_generated.labels(rule_group="exchange_connectivity").inc(len(rules))

        except Exception as e:
            logger.error("exchange_rules_generation_failed", error=str(e))
            raise

    async def _generate_order_rules(self) -> None:
        """Generate order execution alert rules."""
        try:
            rejection_rate_threshold = self.thresholds.get("order_rejection_rate_pct", 10)
            fill_delay_threshold = self.thresholds.get("order_fill_delay_sec", 5)

            rules = [
                {
                    "uid": f"{self.namespace}_order_rejection_high",
                    "title": "High Order Rejection Rate",
                    "condition": "C",
                    "data": [
                        {
                            "refId": "A",
                            "queryType": "range",
                            "relativeTimeRange": {
                                "from": 600,
                                "to": 0
                            },
                            "datasourceUid": self.datasource_uid,
                            "model": {
                                "expr": "(sum(rate(quantum_trader_orders_rejected_total[10m])) / sum(rate(quantum_trader_orders_submitted_total[10m]))) * 100",
                                "intervalMs": 1000,
                                "maxDataPoints": 43200
                            }
                        },
                        {
                            "refId": "C",
                            "queryType": "",
                            "datasourceUid": "-100",
                            "model": {
                                "type": "classic_conditions",
                                "conditions": [
                                    {
                                        "evaluator": {
                                            "params": [rejection_rate_threshold],
                                            "type": "gt"
                                        },
                                        "operator": {"type": "and"},
                                        "query": {"params": ["A"]},
                                        "type": "query"
                                    }
                                ]
                            }
                        }
                    ],
                    "noDataState": "NoData",
                    "execErrState": "Error",
                    "for": "5m",
                    "annotations": {
                        "description": f"Order rejection rate above {rejection_rate_threshold}%",
                        "summary": "Order execution issues"
                    },
                    "labels": {
                        "severity": "warning",
                        "category": "orders"
                    }
                }
            ]

            self.rule_groups.append({
                "name": "Order Execution",
                "interval": self.eval_interval,
                "rules": rules
            })

            self._rules_generated.labels(rule_group="order_execution").inc(len(rules))

        except Exception as e:
            logger.error("order_rules_generation_failed", error=str(e))
            raise

    async def _generate_performance_rules(self) -> None:
        """Generate performance monitoring alert rules."""
        try:
            latency_p99_threshold = self.thresholds.get("latency_p99_ms", 100)
            throughput_min = self.thresholds.get("min_throughput_tps", 10)

            rules = [
                {
                    "uid": f"{self.namespace}_performance_latency_high",
                    "title": "High Processing Latency (P99)",
                    "condition": "C",
                    "data": [
                        {
                            "refId": "A",
                            "queryType": "range",
                            "relativeTimeRange": {
                                "from": 300,
                                "to": 0
                            },
                            "datasourceUid": self.datasource_uid,
                            "model": {
                                "expr": "histogram_quantile(0.99, quantum_trader_processing_latency_seconds_bucket) * 1000",
                                "intervalMs": 1000,
                                "maxDataPoints": 43200
                            }
                        },
                        {
                            "refId": "C",
                            "queryType": "",
                            "datasourceUid": "-100",
                            "model": {
                                "type": "classic_conditions",
                                "conditions": [
                                    {
                                        "evaluator": {
                                            "params": [latency_p99_threshold],
                                            "type": "gt"
                                        },
                                        "operator": {"type": "and"},
                                        "query": {"params": ["A"]},
                                        "type": "query"
                                    }
                                ]
                            }
                        }
                    ],
                    "noDataState": "NoData",
                    "execErrState": "Error",
                    "for": "5m",
                    "annotations": {
                        "description": f"P99 latency above {latency_p99_threshold}ms",
                        "summary": "Performance degradation detected"
                    },
                    "labels": {
                        "severity": "warning",
                        "category": "performance"
                    }
                }
            ]

            self.rule_groups.append({
                "name": "Performance Metrics",
                "interval": self.eval_interval,
                "rules": rules
            })

            self._rules_generated.labels(rule_group="performance_metrics").inc(len(rules))

        except Exception as e:
            logger.error("performance_rules_generation_failed", error=str(e))
            raise

    async def export_to_file(self, file_path: str) -> None:
        """Export rule groups to JSON file.

        Args:
            file_path: Path to export file

        Raises:
            IOError: If file write fails
        """
        try:
            if not self.rule_groups:
                await self.generate_all_rules()

            export_data = {
                "apiVersion": 1,
                "groups": self.rule_groups
            }

            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)

            logger.info("alert_rules_exported", file_path=file_path)

        except Exception as e:
            logger.error("export_failed", error=str(e), file_path=file_path)
            raise

    def get_rule_summary(self) -> Dict[str, Any]:
        """Get summary of generated rules.

        Returns:
            Summary dictionary
        """
        return {
            "total_groups": len(self.rule_groups),
            "total_rules": sum(len(g.get("rules", [])) for g in self.rule_groups),
            "groups": [
                {
                    "name": g["name"],
                    "interval": g["interval"],
                    "rule_count": len(g.get("rules", []))
                }
                for g in self.rule_groups
            ]
        }
