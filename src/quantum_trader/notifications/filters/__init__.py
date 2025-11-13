"""
Notification filters module for Quantum Trader AI.

This submodule provides advanced filtering mechanisms to control which notifications
are sent, to whom, and under what conditions. Filters enable sophisticated notification
targeting based on user preferences, alert severity, market conditions, and time windows.

Filter Types:
    - UserPreferenceFilter: Respects individual user notification preferences
    - SeverityFilter: Routes notifications based on alert severity levels
    - TimeWindowFilter: Restricts notifications to specific time periods
    - RateLimit: Limits notification frequency per recipient/channel
    - ContentFilter: Filters based on message content patterns
    - ScheduleFilter: Allows notifications only on specific schedules
    - GeoFilter: Filters based on geographic location
    - PortfolioFilter: Targets users with specific portfolio characteristics

Filters can be chained together to create complex notification rules.

Example:
    >>> from quantum_trader.notifications.filters import (
    ...     UserPreferenceFilter,
    ...     SeverityFilter,
    ...     TimeWindowFilter
    ... )
    >>> filters = [
    ...     UserPreferenceFilter(),
    ...     SeverityFilter(min_severity="high"),
    ...     TimeWindowFilter(start_hour=6, end_hour=22)
    ... ]
    >>>
    >>> # Apply filters to determine if notification should be sent
    >>> notification = {"severity": "high", "user_id": 123}
    >>> should_send = all(f.apply(notification) for f in filters)
"""

__version__ = "1.0.0"
__all__ = [
    "UserPreferenceFilter",
    "SeverityFilter",
    "TimeWindowFilter",
    "RateLimitFilter",
    "ContentFilter",
    "ScheduleFilter",
    "GeoFilter",
    "PortfolioFilter",
    "BaseFilter",
    "FilterChain",
]

try:
    from .base import BaseFilter, FilterChain
    from .user_preference import UserPreferenceFilter
    from .severity import SeverityFilter
    from .time_window import TimeWindowFilter
    from .rate_limit import RateLimitFilter
    from .content import ContentFilter
    from .schedule import ScheduleFilter
    from .geo import GeoFilter
    from .portfolio import PortfolioFilter
except ImportError:
    # Graceful handling if optional dependencies are not installed
    pass
