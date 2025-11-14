"""Prometheus recording rules configuration for Quantum Trader AI.

This module generates Prometheus recording rules for pre-computing frequently
needed expressions and aggregations to improve query performance.
"""

import os
from typing import Dict, Any, List
from structlog import get_logger

logger = get_logger(__name__)


class RecordingRules:
    """Generator for Prometheus recording rules.

    Recording rules allow you to precompute frequently needed or computationally
    expensive expressions and save their result as a new set of time series.

    Attributes:
        config: Configuration dictionary
        namespace: Metrics namespace
        evaluation_interval: How often rules are evaluated

    Example:
        >>> config = {
        ...     "namespace": "quantum_trader",
        ...     "evaluation_interval": "15s"
        ... }
        >>> rules = RecordingRules(config)
        >>> yaml_config = rules.generate_rules_yaml()
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize recording rules generator.

        Args:
            config: Configuration dictionary containing:
                - namespace: Metrics namespace
                - evaluation_interval: Rule evaluation interval
                - aggregation_periods: List of aggregation periods

        Raises:
            ValueError: If required config parameters are missing
        """
        self.config = config
        self._validate_config()

        self.namespace = self.config.get(
            "namespace", os.getenv("METRICS_NAMESPACE", "quantum_trader")
        )
        self.evaluation_interval = self.config.get(
            "evaluation_interval", os.getenv("RECORDING_RULES_INTERVAL", "15s")
        )
        self.aggregation_periods = self.config.get(
            "aggregation_periods",
            os.getenv("RECORDING_RULES_PERIODS", "1m,5m,15m,1h,24h").split(",")
        )

        logger.info(
            "RecordingRules initialized",
            namespace=self.namespace,
            evaluation_interval=self.evaluation_interval
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        logger.debug("Config validation passed")

    def get_trading_rules(self) -> List[Dict[str, Any]]:
        """Get trading-related recording rules.

        Returns:
            List of rule dictionaries
        """
        rules = []
        metric_prefix = f"{self.namespace}_trading"

        for period in self.aggregation_periods:
            # Order rate by exchange
            rules.append({
                "record": f"{metric_prefix}:order_rate:{period}:by_exchange",
                "expr": f'sum(rate({metric_prefix}_orders_total[{period}])) by (exchange)',
            })

            # Trade volume by exchange
            rules.append({
                "record": f"{metric_prefix}:trade_volume:{period}:by_exchange",
                "expr": f'sum(rate({metric_prefix}_trade_volume_usd[{period}])) by (exchange)',
            })

            # Error rate
            rules.append({
                "record": f"{metric_prefix}:error_rate:{period}",
                "expr": f'sum(rate({metric_prefix}_order_errors_total[{period}])) / sum(rate({metric_prefix}_orders_total[{period}]))',
            })

        # Current positions summary
        rules.append({
            "record": f"{metric_prefix}:total_position_size:usd",
            "expr": f'sum({metric_prefix}_position_size_usd)',
        })

        return rules

    def get_performance_rules(self) -> List[Dict[str, Any]]:
        """Get performance-related recording rules.

        Returns:
            List of rule dictionaries
        """
        rules = []
        metric_prefix = f"{self.namespace}_trading"

        for period in self.aggregation_periods:
            # Latency percentiles
            for percentile in [50, 95, 99]:
                rules.append({
                    "record": f"{metric_prefix}:order_latency_ms:p{percentile}:{period}",
                    "expr": f'histogram_quantile(0.{percentile}, sum(rate({metric_prefix}_order_latency_milliseconds_bucket[{period}])) by (le, exchange))',
                })

                rules.append({
                    "record": f"{metric_prefix}:execution_latency_ms:p{percentile}:{period}",
                    "expr": f'histogram_quantile(0.{percentile}, sum(rate({metric_prefix}_execution_latency_milliseconds_bucket[{period}])) by (le, exchange))',
                })

            # Slippage average
            rules.append({
                "record": f"{metric_prefix}:avg_slippage_bps:{period}",
                "expr": f'sum(rate({metric_prefix}_trade_slippage_bps_sum[{period}])) / sum(rate({metric_prefix}_trade_slippage_bps_count[{period}]))',
            })

        return rules

    def get_pnl_rules(self) -> List[Dict[str, Any]]:
        """Get PnL-related recording rules.

        Returns:
            List of rule dictionaries
        """
        rules = []
        metric_prefix = f"{self.namespace}_trading"

        # Total PnL by strategy
        rules.append({
            "record": f"{metric_prefix}:total_pnl:by_strategy",
            "expr": f'sum({metric_prefix}_total_pnl_usd) by (strategy)',
        })

        # Aggregate PnL
        rules.append({
            "record": f"{metric_prefix}:aggregate_pnl:usd",
            "expr": f'sum({metric_prefix}_total_pnl_usd)',
        })

        # Win rate by strategy
        rules.append({
            "record": f"{metric_prefix}:win_rate:by_strategy",
            "expr": f'avg({metric_prefix}_win_rate_percent) by (strategy)',
        })

        return rules

    def get_risk_rules(self) -> List[Dict[str, Any]]:
        """Get risk-related recording rules.

        Returns:
            List of rule dictionaries
        """
        rules = []
        metric_prefix = f"{self.namespace}_trading"

        # Maximum drawdown across strategies
        rules.append({
            "record": f"{metric_prefix}:max_drawdown:percent",
            "expr": f'max({metric_prefix}_drawdown_percent)',
        })

        # Total VaR
        rules.append({
            "record": f"{metric_prefix}:total_var_95:usd",
            "expr": f'sum({metric_prefix}_value_at_risk_95_usd)',
        })

        # Average Sharpe ratio
        rules.append({
            "record": f"{metric_prefix}:avg_sharpe_ratio",
            "expr": f'avg({metric_prefix}_sharpe_ratio)',
        })

        for period in ["1h", "24h"]:
            # Risk violation rate
            rules.append({
                "record": f"{metric_prefix}:risk_violation_rate:{period}",
                "expr": f'sum(rate({metric_prefix}_risk_violations_total[{period}]))',
            })

        return rules

    def get_system_rules(self) -> List[Dict[str, Any]]:
        """Get system health recording rules.

        Returns:
            List of rule dictionaries
        """
        rules = []
        metric_prefix = f"{self.namespace}_trading"

        for period in ["5m", "1h"]:
            # Average CPU usage
            rules.append({
                "record": f"{metric_prefix}:avg_cpu_usage:{period}",
                "expr": f'avg({metric_prefix}_cpu_usage_percent)',
            })

            # Average memory usage
            rules.append({
                "record": f"{metric_prefix}:avg_memory_usage:{period}",
                "expr": f'avg({metric_prefix}_memory_usage_bytes)',
            })

        # Connected exchanges count
        rules.append({
            "record": f"{metric_prefix}:connected_exchanges:count",
            "expr": f'sum({metric_prefix}_exchange_connected)',
        })

        return rules

    def get_exchange_rules(self) -> List[Dict[str, Any]]:
        """Get exchange connectivity recording rules.

        Returns:
            List of rule dictionaries
        """
        rules = []
        metric_prefix = f"{self.namespace}_trading"

        for period in self.aggregation_periods:
            # API call rate by exchange
            rules.append({
                "record": f"{metric_prefix}:api_call_rate:{period}:by_exchange",
                "expr": f'sum(rate({metric_prefix}_api_calls_total[{period}])) by (exchange)',
            })

            # Exchange error rate
            rules.append({
                "record": f"{metric_prefix}:exchange_error_rate:{period}",
                "expr": f'sum(rate({metric_prefix}_exchange_errors_total[{period}])) by (exchange)',
            })

        return rules

    def get_all_rules(self) -> List[Dict[str, Any]]:
        """Get all recording rules.

        Returns:
            List of all rule dictionaries
        """
        all_rules = []
        all_rules.extend(self.get_trading_rules())
        all_rules.extend(self.get_performance_rules())
        all_rules.extend(self.get_pnl_rules())
        all_rules.extend(self.get_risk_rules())
        all_rules.extend(self.get_system_rules())
        all_rules.extend(self.get_exchange_rules())

        logger.info("Generated all recording rules", count=len(all_rules))
        return all_rules

    def generate_rules_yaml(self) -> str:
        """Generate Prometheus recording rules in YAML format.

        Returns:
            YAML string with recording rules configuration
        """
        rules = self.get_all_rules()

        yaml_output = "# Quantum Trader AI - Prometheus Recording Rules\n"
        yaml_output += f"# Generated for namespace: {self.namespace}\n"
        yaml_output += f"# Evaluation interval: {self.evaluation_interval}\n\n"

        yaml_output += "groups:\n"

        # Group rules by category
        categories = {
            "trading": [],
            "performance": [],
            "pnl": [],
            "risk": [],
            "system": [],
            "exchange": [],
        }

        for rule in rules:
            record = rule["record"]
            if ":order_" in record or ":trade_" in record:
                categories["trading"].append(rule)
            elif ":latency_" in record or ":slippage_" in record:
                categories["performance"].append(rule)
            elif ":pnl" in record or ":win_rate" in record:
                categories["pnl"].append(rule)
            elif ":drawdown" in record or ":var_" in record or ":sharpe" in record or ":risk_" in record:
                categories["risk"].append(rule)
            elif ":cpu_" in record or ":memory_" in record:
                categories["system"].append(rule)
            elif ":exchange_" in record or ":api_" in record or ":connected_" in record:
                categories["exchange"].append(rule)

        # Generate YAML for each category
        for category, category_rules in categories.items():
            if not category_rules:
                continue

            yaml_output += f"  - name: {self.namespace}_{category}_rules\n"
            yaml_output += f"    interval: {self.evaluation_interval}\n"
            yaml_output += "    rules:\n"

            for rule in category_rules:
                yaml_output += f"      - record: {rule['record']}\n"
                yaml_output += f"        expr: |\n"
                yaml_output += f"          {rule['expr']}\n"

        logger.info("Generated recording rules YAML configuration")
        return yaml_output

    def generate_rules_dict(self) -> Dict[str, Any]:
        """Generate recording rules as a dictionary structure.

        Returns:
            Dictionary with recording rules configuration
        """
        rules = self.get_all_rules()

        return {
            "groups": [
                {
                    "name": f"{self.namespace}_recording_rules",
                    "interval": self.evaluation_interval,
                    "rules": rules,
                }
            ]
        }

    def save_rules_yaml(self, filepath: str) -> None:
        """Save recording rules to a YAML file.

        Args:
            filepath: Path to save YAML file

        Raises:
            IOError: If file cannot be written
        """
        try:
            yaml_content = self.generate_rules_yaml()

            with open(filepath, "w") as f:
                f.write(yaml_content)

            logger.info("Recording rules saved to file", filepath=filepath)

        except Exception as e:
            logger.error(
                "Error saving recording rules",
                filepath=filepath,
                error=str(e),
                exc_info=True
            )
            raise
