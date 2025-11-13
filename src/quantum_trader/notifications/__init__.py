"""
Notifications module for Quantum Trader AI.

This module provides a comprehensive notification system for the Quantum Trader AI platform,
including support for multiple notification channels (email, SMS, webhooks, etc.), filtering
mechanisms, and customizable notification templates.

Key Components:
    - Notification channels for multi-channel delivery
    - Advanced filtering system for targeted notifications
    - Customizable templates for consistent messaging
    - Delivery tracking and retry logic
    - Integration with external services (email providers, SMS gateways)

Features:
    - Multi-channel notification delivery
    - Real-time and scheduled notifications
    - Template-based message composition
    - Recipient filtering and segmentation
    - Delivery status tracking
    - Retry mechanisms with exponential backoff
    - Rate limiting and throttling

Example:
    >>> from quantum_trader.notifications import NotificationManager
    >>> manager = NotificationManager()
    >>> manager.send_email(
    ...     recipient="user@example.com",
    ...     subject="Trading Alert",
    ...     template="alert_template",
    ...     context={"symbol": "AAPL", "price": 150.25}
    ... )
"""

__version__ = "1.0.0"
__author__ = "Quantum Trader AI Team"
__all__ = [
    "NotificationManager",
    "NotificationChannel",
    "NotificationFilter",
    "NotificationTemplate",
    "channels",
    "filters",
    "templates",
]

# Lazy imports to improve module load time
def __getattr__(name):
    """Lazy load main notification components."""
    if name == "NotificationManager":
        from .manager import NotificationManager
        return NotificationManager
    elif name == "NotificationChannel":
        from .channel import NotificationChannel
        return NotificationChannel
    elif name == "NotificationFilter":
        from .filter import NotificationFilter
        return NotificationFilter
    elif name == "NotificationTemplate":
        from .template import NotificationTemplate
        return NotificationTemplate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
