"""
Report Templates - Performance and system report formatting.

Generates formatted reports for daily/weekly/monthly performance,
system health, risk metrics, and portfolio analytics.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
import os
from structlog import get_logger

logger = get_logger(__name__)


class ReportTemplates:
    """
    Report formatting templates for notifications.

    Generates formatted text and HTML reports for various metrics
    including performance, risk, and system health.

    Attributes:
        config: Configuration dictionary from environment
        currency_symbol: Currency symbol for formatting
        decimal_places: Decimal places for price formatting
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize report templates.

        Args:
            config: Optional configuration override. If None, loads from environment.
        """
        self.config = config or self._load_config()
        self._validate_config()

        self.currency_symbol = self.config.get(
            "currency_symbol",
            os.getenv("REPORT_CURRENCY_SYMBOL", "$")
        )
        self.decimal_places = int(self.config.get(
            "decimal_places",
            os.getenv("REPORT_DECIMAL_PLACES", "2")
        ))

        logger.info("report_templates_initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            "currency_symbol": os.getenv("REPORT_CURRENCY_SYMBOL", "$"),
            "decimal_places": os.getenv("REPORT_DECIMAL_PLACES", "2"),
            "timezone": os.getenv("REPORT_TIMEZONE", "UTC"),
            "date_format": os.getenv("REPORT_DATE_FORMAT", "%Y-%m-%d %H:%M:%S UTC"),
            "include_footer": os.getenv("REPORT_INCLUDE_FOOTER", "true").lower() == "true",
            "footer_text": os.getenv("REPORT_FOOTER_TEXT", "Quantum Trader AI - Institutional Trading Platform"),
        }

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if int(self.config.get("decimal_places", 2)) < 0:
            raise ValueError("decimal_places must be >= 0")

    def format_decimal(self, value: Decimal, places: Optional[int] = None) -> str:
        """
        Format Decimal value for display.

        Args:
            value: Decimal value to format
            places: Optional decimal places override

        Returns:
            Formatted string
        """
        if places is None:
            places = self.decimal_places

        return f"{value:.{places}f}"

    def format_currency(self, value: Decimal, places: Optional[int] = None) -> str:
        """
        Format Decimal value as currency.

        Args:
            value: Decimal value to format
            places: Optional decimal places override

        Returns:
            Formatted currency string
        """
        formatted = self.format_decimal(value, places)
        return f"{self.currency_symbol}{formatted}"

    def format_percentage(self, value: Decimal, places: int = 2) -> str:
        """
        Format Decimal value as percentage.

        Args:
            value: Decimal value to format (e.g., 0.15 for 15%)
            places: Decimal places for percentage

        Returns:
            Formatted percentage string
        """
        percentage = value * Decimal("100")
        return f"{self.format_decimal(percentage, places)}%"

    def format_timestamp(self, timestamp: datetime) -> str:
        """
        Format timestamp for display.

        Args:
            timestamp: Datetime to format

        Returns:
            Formatted timestamp string
        """
        date_format = self.config.get("date_format", "%Y-%m-%d %H:%M:%S UTC")
        return timestamp.strftime(date_format)

    def daily_performance_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generate daily performance report.

        Args:
            metrics: Performance metrics dictionary containing:
                - date: Report date
                - pnl: Daily P&L (Decimal)
                - pnl_percentage: Daily P&L % (Decimal)
                - trades_count: Number of trades
                - win_rate: Win rate (Decimal)
                - total_volume: Trading volume (Decimal)
                - sharpe_ratio: Sharpe ratio (Decimal)
                - max_drawdown: Maximum drawdown (Decimal)
                - top_performers: List of top performing trades
                - worst_performers: List of worst performing trades

        Returns:
            Formatted report string
        """
        try:
            date = metrics.get("date", datetime.utcnow())
            pnl = Decimal(str(metrics.get("pnl", 0)))
            pnl_pct = Decimal(str(metrics.get("pnl_percentage", 0)))
            trades_count = metrics.get("trades_count", 0)
            win_rate = Decimal(str(metrics.get("win_rate", 0)))
            volume = Decimal(str(metrics.get("total_volume", 0)))
            sharpe = Decimal(str(metrics.get("sharpe_ratio", 0)))
            max_dd = Decimal(str(metrics.get("max_drawdown", 0)))

            pnl_emoji = "📈" if pnl >= 0 else "📉"
            pnl_sign = "+" if pnl >= 0 else ""

            report = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{pnl_emoji} DAILY PERFORMANCE REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 Date: {self.format_timestamp(date)}

💰 P&L: {pnl_sign}{self.format_currency(pnl)} ({pnl_sign}{self.format_percentage(pnl_pct)})
📊 Trades: {trades_count}
🎯 Win Rate: {self.format_percentage(win_rate)}
💵 Volume: {self.format_currency(volume)}

📈 Performance Metrics:
  • Sharpe Ratio: {self.format_decimal(sharpe, 3)}
  • Max Drawdown: {self.format_percentage(max_dd)}
"""

            # Add top performers if available
            if metrics.get("top_performers"):
                report += "\n🏆 Top Performers:\n"
                for i, trade in enumerate(metrics["top_performers"][:3], 1):
                    symbol = trade.get("symbol", "N/A")
                    pnl_val = Decimal(str(trade.get("pnl", 0)))
                    report += f"  {i}. {symbol}: {self.format_currency(pnl_val)}\n"

            # Add worst performers if available
            if metrics.get("worst_performers"):
                report += "\n⚠️ Worst Performers:\n"
                for i, trade in enumerate(metrics["worst_performers"][:3], 1):
                    symbol = trade.get("symbol", "N/A")
                    pnl_val = Decimal(str(trade.get("pnl", 0)))
                    report += f"  {i}. {symbol}: {self.format_currency(pnl_val)}\n"

            if self.config.get("include_footer", True):
                report += f"\n{self.config.get('footer_text', '')}\n"

            report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            return report.strip()

        except Exception as e:
            logger.error("failed_to_generate_daily_report", error=str(e))
            raise

    def weekly_performance_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generate weekly performance report.

        Args:
            metrics: Weekly performance metrics

        Returns:
            Formatted report string
        """
        try:
            start_date = metrics.get("start_date", datetime.utcnow() - timedelta(days=7))
            end_date = metrics.get("end_date", datetime.utcnow())
            pnl = Decimal(str(metrics.get("pnl", 0)))
            pnl_pct = Decimal(str(metrics.get("pnl_percentage", 0)))
            trades_count = metrics.get("trades_count", 0)
            win_rate = Decimal(str(metrics.get("win_rate", 0)))
            avg_daily_pnl = Decimal(str(metrics.get("avg_daily_pnl", 0)))
            best_day_pnl = Decimal(str(metrics.get("best_day_pnl", 0)))
            worst_day_pnl = Decimal(str(metrics.get("worst_day_pnl", 0)))

            pnl_emoji = "📈" if pnl >= 0 else "📉"
            pnl_sign = "+" if pnl >= 0 else ""

            report = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{pnl_emoji} WEEKLY PERFORMANCE REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 Period: {self.format_timestamp(start_date)} - {self.format_timestamp(end_date)}

💰 Total P&L: {pnl_sign}{self.format_currency(pnl)} ({pnl_sign}{self.format_percentage(pnl_pct)})
📊 Total Trades: {trades_count}
🎯 Win Rate: {self.format_percentage(win_rate)}

📈 Daily Metrics:
  • Avg Daily P&L: {self.format_currency(avg_daily_pnl)}
  • Best Day: {self.format_currency(best_day_pnl)}
  • Worst Day: {self.format_currency(worst_day_pnl)}
"""

            # Add daily breakdown if available
            if metrics.get("daily_breakdown"):
                report += "\n📊 Daily Breakdown:\n"
                for day_data in metrics["daily_breakdown"]:
                    day = day_data.get("date", "N/A")
                    day_pnl = Decimal(str(day_data.get("pnl", 0)))
                    day_emoji = "+" if day_pnl >= 0 else "-"
                    report += f"  {day_emoji} {day}: {self.format_currency(day_pnl)}\n"

            if self.config.get("include_footer", True):
                report += f"\n{self.config.get('footer_text', '')}\n"

            report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            return report.strip()

        except Exception as e:
            logger.error("failed_to_generate_weekly_report", error=str(e))
            raise

    def risk_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generate risk metrics report.

        Args:
            metrics: Risk metrics dictionary

        Returns:
            Formatted report string
        """
        try:
            timestamp = metrics.get("timestamp", datetime.utcnow())
            portfolio_value = Decimal(str(metrics.get("portfolio_value", 0)))
            var_95 = Decimal(str(metrics.get("var_95", 0)))
            var_99 = Decimal(str(metrics.get("var_99", 0)))
            max_drawdown = Decimal(str(metrics.get("max_drawdown", 0)))
            current_drawdown = Decimal(str(metrics.get("current_drawdown", 0)))
            leverage = Decimal(str(metrics.get("leverage", 1)))
            exposure = Decimal(str(metrics.get("exposure", 0)))

            report = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ RISK METRICS REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 Timestamp: {self.format_timestamp(timestamp)}

💼 Portfolio:
  • Value: {self.format_currency(portfolio_value)}
  • Leverage: {self.format_decimal(leverage, 2)}x
  • Exposure: {self.format_currency(exposure)}

📊 Value at Risk (VaR):
  • 95% VaR: {self.format_currency(var_95)}
  • 99% VaR: {self.format_currency(var_99)}

📉 Drawdown:
  • Current: {self.format_percentage(current_drawdown)}
  • Maximum: {self.format_percentage(max_drawdown)}
"""

            # Add position limits if available
            if metrics.get("position_limits"):
                limits = metrics["position_limits"]
                report += "\n📏 Position Limits:\n"
                for symbol, limit_data in limits.items():
                    current = Decimal(str(limit_data.get("current", 0)))
                    max_limit = Decimal(str(limit_data.get("max", 0)))
                    usage_pct = (current / max_limit * Decimal("100")) if max_limit > 0 else Decimal("0")
                    report += f"  • {symbol}: {self.format_decimal(usage_pct, 1)}% of limit\n"

            if self.config.get("include_footer", True):
                report += f"\n{self.config.get('footer_text', '')}\n"

            report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            return report.strip()

        except Exception as e:
            logger.error("failed_to_generate_risk_report", error=str(e))
            raise

    def system_health_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generate system health report.

        Args:
            metrics: System health metrics

        Returns:
            Formatted report string
        """
        try:
            timestamp = metrics.get("timestamp", datetime.utcnow())
            status = metrics.get("status", "UNKNOWN")
            uptime_seconds = metrics.get("uptime_seconds", 0)
            active_strategies = metrics.get("active_strategies", 0)
            active_exchanges = metrics.get("active_exchanges", 0)
            avg_latency_ms = Decimal(str(metrics.get("avg_latency_ms", 0)))

            # Convert uptime to readable format
            uptime_hours = uptime_seconds // 3600
            uptime_minutes = (uptime_seconds % 3600) // 60

            status_emoji = {
                "HEALTHY": "✅",
                "DEGRADED": "⚠️",
                "UNHEALTHY": "❌",
                "UNKNOWN": "❓"
            }.get(status, "❓")

            report = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{status_emoji} SYSTEM HEALTH REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 Timestamp: {self.format_timestamp(timestamp)}

🔍 Status: {status}
⏱️ Uptime: {uptime_hours}h {uptime_minutes}m

📊 Active Components:
  • Strategies: {active_strategies}
  • Exchanges: {active_exchanges}

⚡ Performance:
  • Avg Latency: {self.format_decimal(avg_latency_ms, 2)}ms
"""

            # Add component health if available
            if metrics.get("components"):
                report += "\n🔧 Components:\n"
                for component, health in metrics["components"].items():
                    health_emoji = "✅" if health == "HEALTHY" else "❌"
                    report += f"  {health_emoji} {component}: {health}\n"

            # Add error summary if available
            if metrics.get("recent_errors"):
                error_count = len(metrics["recent_errors"])
                report += f"\n⚠️ Recent Errors: {error_count}\n"

            if self.config.get("include_footer", True):
                report += f"\n{self.config.get('footer_text', '')}\n"

            report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            return report.strip()

        except Exception as e:
            logger.error("failed_to_generate_health_report", error=str(e))
            raise

    def portfolio_summary_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generate portfolio summary report.

        Args:
            metrics: Portfolio metrics

        Returns:
            Formatted report string
        """
        try:
            timestamp = metrics.get("timestamp", datetime.utcnow())
            total_value = Decimal(str(metrics.get("total_value", 0)))
            cash_balance = Decimal(str(metrics.get("cash_balance", 0)))
            positions_value = Decimal(str(metrics.get("positions_value", 0)))
            total_pnl = Decimal(str(metrics.get("total_pnl", 0)))
            total_pnl_pct = Decimal(str(metrics.get("total_pnl_percentage", 0)))

            pnl_sign = "+" if total_pnl >= 0 else ""

            report = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💼 PORTFOLIO SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 Timestamp: {self.format_timestamp(timestamp)}

💰 Total Value: {self.format_currency(total_value)}
💵 Cash: {self.format_currency(cash_balance)}
📊 Positions: {self.format_currency(positions_value)}

📈 Total P&L: {pnl_sign}{self.format_currency(total_pnl)} ({pnl_sign}{self.format_percentage(total_pnl_pct)})
"""

            # Add positions if available
            if metrics.get("positions"):
                report += "\n📊 Open Positions:\n"
                for position in metrics["positions"]:
                    symbol = position.get("symbol", "N/A")
                    size = Decimal(str(position.get("size", 0)))
                    entry_price = Decimal(str(position.get("entry_price", 0)))
                    current_price = Decimal(str(position.get("current_price", 0)))
                    pnl = Decimal(str(position.get("pnl", 0)))
                    pnl_sign_pos = "+" if pnl >= 0 else ""

                    report += f"  • {symbol}: {self.format_decimal(size, 4)} @ {self.format_currency(entry_price)}\n"
                    report += f"    Current: {self.format_currency(current_price)} | P&L: {pnl_sign_pos}{self.format_currency(pnl)}\n"

            if self.config.get("include_footer", True):
                report += f"\n{self.config.get('footer_text', '')}\n"

            report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            return report.strip()

        except Exception as e:
            logger.error("failed_to_generate_portfolio_report", error=str(e))
            raise
