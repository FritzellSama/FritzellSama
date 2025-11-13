"""
Notification templates module for Quantum Trader AI.

This submodule provides a template system for generating consistent, reusable notification
messages across all notification channels. Templates support variable interpolation,
conditional rendering, and multi-language support.

Template Types:
    - AlertTemplate: Templates for trading and price alerts
    - PortfolioTemplate: Templates for portfolio updates and changes
    - OrderTemplate: Templates for order confirmations and updates
    - MarketTemplate: Templates for market data and analysis
    - RiskTemplate: Templates for risk alerts and warnings
    - PerformanceTemplate: Templates for performance summaries
    - NewsTemplate: Templates for news and event notifications
    - ReportTemplate: Templates for periodic reports and digests

Features:
    - Jinja2-based template rendering
    - Support for conditional blocks
    - Localization and internationalization
    - Template inheritance and reuse
    - Variable validation
    - Custom filters and functions
    - Template caching for performance

Example:
    >>> from quantum_trader.notifications.templates import AlertTemplate
    >>> template = AlertTemplate()
    >>> message = template.render(
    ...     symbol="AAPL",
    ...     price=150.25,
    ...     change=2.5,
    ...     change_percent=1.69,
    ...     language="en"
    ... )
    >>> print(message)
    'AAPL Price Alert: $150.25 (+2.5, +1.69%)'
"""

__version__ = "1.0.0"
__all__ = [
    "AlertTemplate",
    "PortfolioTemplate",
    "OrderTemplate",
    "MarketTemplate",
    "RiskTemplate",
    "PerformanceTemplate",
    "NewsTemplate",
    "ReportTemplate",
    "BaseTemplate",
    "TemplateManager",
]

try:
    from .base import BaseTemplate, TemplateManager
    from .alert import AlertTemplate
    from .portfolio import PortfolioTemplate
    from .order import OrderTemplate
    from .market import MarketTemplate
    from .risk import RiskTemplate
    from .performance import PerformanceTemplate
    from .news import NewsTemplate
    from .report import ReportTemplate
except ImportError:
    # Graceful handling if optional dependencies are not installed
    pass
