"""
Notification channels module for Quantum Trader AI.

This submodule implements various notification delivery channels for sending alerts
and information to users through different mediums.

Supported Channels:
    - Email: SMTP-based email delivery with HTML support
    - SMS: SMS delivery via external providers
    - Webhook: HTTP webhook calls for third-party integrations
    - Slack: Slack message integration
    - Discord: Discord webhook integration
    - PushNotification: Mobile push notifications
    - InApp: In-application notifications

Each channel implements a standardized interface for message delivery, error handling,
and delivery status tracking.

Example:
    >>> from quantum_trader.notifications.channels import EmailChannel, SlackChannel
    >>> email = EmailChannel()
    >>> slack = SlackChannel()
    >>>
    >>> # Send via multiple channels
    >>> email.send("user@example.com", "Alert", "Trading alert message")
    >>> slack.send("#alerts", "Trading alert message")
"""

__version__ = "1.0.0"
__all__ = [
    "EmailChannel",
    "SMSChannel",
    "WebhookChannel",
    "SlackChannel",
    "DiscordChannel",
    "PushNotificationChannel",
    "InAppChannel",
    "BaseChannel",
]

try:
    from .base import BaseChannel
    from .email import EmailChannel
    from .sms import SMSChannel
    from .webhook import WebhookChannel
    from .slack import SlackChannel
    from .discord import DiscordChannel
    from .push import PushNotificationChannel
    from .inapp import InAppChannel
except ImportError:
    # Graceful handling if optional dependencies are not installed
    pass
