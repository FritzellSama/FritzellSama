"""DateTime Utility Functions.

Production-ready datetime utilities for timezone handling, parsing,
formatting, and time-based calculations in trading systems.
"""

from decimal import Decimal
from typing import Optional, Union, List
from datetime import datetime, timezone, timedelta
import time
from structlog import get_logger

logger = get_logger(__name__)


def utc_now() -> datetime:
    """Get current UTC datetime.

    Returns:
        Current datetime in UTC timezone
    """
    return datetime.now(timezone.utc)


def to_utc(dt: datetime) -> datetime:
    """Convert datetime to UTC.

    Args:
        dt: Datetime to convert

    Returns:
        UTC datetime

    Raises:
        ValueError: If datetime invalid
    """
    try:
        if dt.tzinfo is None:
            # Naive datetime, assume UTC
            return dt.replace(tzinfo=timezone.utc)
        else:
            # Convert to UTC
            return dt.astimezone(timezone.utc)

    except Exception as e:
        logger.error("utc_conversion_failed", error=str(e))
        raise ValueError(f"Failed to convert to UTC: {e}")


def timestamp_to_datetime(
    timestamp: Union[int, float],
    unit: str = "s"
) -> datetime:
    """Convert timestamp to datetime.

    Args:
        timestamp: Unix timestamp
        unit: Time unit ("s", "ms", "us", "ns")

    Returns:
        UTC datetime

    Raises:
        ValueError: If timestamp or unit invalid
    """
    try:
        # Convert to seconds
        if unit == "s":
            ts_seconds = float(timestamp)
        elif unit == "ms":
            ts_seconds = float(timestamp) / 1000.0
        elif unit == "us":
            ts_seconds = float(timestamp) / 1000000.0
        elif unit == "ns":
            ts_seconds = float(timestamp) / 1000000000.0
        else:
            raise ValueError(f"Invalid unit: {unit}")

        dt = datetime.fromtimestamp(ts_seconds, tz=timezone.utc)
        return dt

    except Exception as e:
        logger.error(
            "timestamp_conversion_failed",
            timestamp=timestamp,
            unit=unit,
            error=str(e)
        )
        raise ValueError(f"Failed to convert timestamp: {e}")


def datetime_to_timestamp(
    dt: datetime,
    unit: str = "s"
) -> Union[int, float]:
    """Convert datetime to timestamp.

    Args:
        dt: Datetime to convert
        unit: Time unit ("s", "ms", "us", "ns")

    Returns:
        Timestamp in specified unit

    Raises:
        ValueError: If datetime or unit invalid
    """
    try:
        # Ensure UTC
        utc_dt = to_utc(dt)

        # Get timestamp in seconds
        ts_seconds = utc_dt.timestamp()

        # Convert to requested unit
        if unit == "s":
            return int(ts_seconds)
        elif unit == "ms":
            return int(ts_seconds * 1000)
        elif unit == "us":
            return int(ts_seconds * 1000000)
        elif unit == "ns":
            return int(ts_seconds * 1000000000)
        else:
            raise ValueError(f"Invalid unit: {unit}")

    except Exception as e:
        logger.error("datetime_to_timestamp_failed", error=str(e))
        raise ValueError(f"Failed to convert datetime: {e}")


def parse_datetime(
    dt_string: str,
    format: Optional[str] = None
) -> datetime:
    """Parse datetime string.

    Args:
        dt_string: Datetime string
        format: Optional format string (defaults to ISO format)

    Returns:
        Parsed datetime in UTC

    Raises:
        ValueError: If parsing fails
    """
    try:
        if format:
            dt = datetime.strptime(dt_string, format)
        else:
            # Try ISO format
            dt = datetime.fromisoformat(dt_string.replace('Z', '+00:00'))

        # Ensure UTC
        return to_utc(dt)

    except Exception as e:
        logger.error(
            "datetime_parse_failed",
            string=dt_string,
            format=format,
            error=str(e)
        )
        raise ValueError(f"Failed to parse datetime: {e}")


def format_datetime(
    dt: datetime,
    format: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """Format datetime as string.

    Args:
        dt: Datetime to format
        format: Format string

    Returns:
        Formatted string
    """
    try:
        # Ensure UTC
        utc_dt = to_utc(dt)
        return utc_dt.strftime(format)

    except Exception as e:
        logger.error("datetime_format_failed", format=format, error=str(e))
        return str(dt)


def iso_format(dt: datetime) -> str:
    """Format datetime as ISO string.

    Args:
        dt: Datetime to format

    Returns:
        ISO formatted string
    """
    try:
        utc_dt = to_utc(dt)
        return utc_dt.isoformat()

    except Exception as e:
        logger.error("iso_format_failed", error=str(e))
        return str(dt)


def get_start_of_day(dt: Optional[datetime] = None) -> datetime:
    """Get start of day (00:00:00) in UTC.

    Args:
        dt: Optional datetime (defaults to now)

    Returns:
        Start of day datetime
    """
    try:
        if dt is None:
            dt = utc_now()

        utc_dt = to_utc(dt)
        return utc_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    except Exception as e:
        logger.error("start_of_day_failed", error=str(e))
        raise ValueError(f"Failed to get start of day: {e}")


def get_end_of_day(dt: Optional[datetime] = None) -> datetime:
    """Get end of day (23:59:59.999999) in UTC.

    Args:
        dt: Optional datetime (defaults to now)

    Returns:
        End of day datetime
    """
    try:
        if dt is None:
            dt = utc_now()

        utc_dt = to_utc(dt)
        return utc_dt.replace(hour=23, minute=59, second=59, microsecond=999999)

    except Exception as e:
        logger.error("end_of_day_failed", error=str(e))
        raise ValueError(f"Failed to get end of day: {e}")


def add_time(
    dt: datetime,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0
) -> datetime:
    """Add time to datetime.

    Args:
        dt: Base datetime
        days: Days to add
        hours: Hours to add
        minutes: Minutes to add
        seconds: Seconds to add

    Returns:
        New datetime
    """
    try:
        delta = timedelta(
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds
        )
        return dt + delta

    except Exception as e:
        logger.error("add_time_failed", error=str(e))
        raise ValueError(f"Failed to add time: {e}")


def subtract_time(
    dt: datetime,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0
) -> datetime:
    """Subtract time from datetime.

    Args:
        dt: Base datetime
        days: Days to subtract
        hours: Hours to subtract
        minutes: Minutes to subtract
        seconds: Seconds to subtract

    Returns:
        New datetime
    """
    try:
        delta = timedelta(
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds
        )
        return dt - delta

    except Exception as e:
        logger.error("subtract_time_failed", error=str(e))
        raise ValueError(f"Failed to subtract time: {e}")


def time_diff_seconds(dt1: datetime, dt2: datetime) -> float:
    """Calculate time difference in seconds.

    Args:
        dt1: First datetime
        dt2: Second datetime

    Returns:
        Difference in seconds (dt1 - dt2)
    """
    try:
        utc_dt1 = to_utc(dt1)
        utc_dt2 = to_utc(dt2)

        delta = utc_dt1 - utc_dt2
        return delta.total_seconds()

    except Exception as e:
        logger.error("time_diff_failed", error=str(e))
        raise ValueError(f"Failed to calculate time difference: {e}")


def is_market_hours(
    dt: Optional[datetime] = None,
    market_open_hour: int = 9,
    market_close_hour: int = 17,
    weekdays_only: bool = True
) -> bool:
    """Check if datetime is during market hours.

    Args:
        dt: Datetime to check (defaults to now)
        market_open_hour: Market open hour (24h format)
        market_close_hour: Market close hour (24h format)
        weekdays_only: Only consider weekdays as market days

    Returns:
        True if during market hours
    """
    try:
        if dt is None:
            dt = utc_now()

        utc_dt = to_utc(dt)

        # Check weekday
        if weekdays_only and utc_dt.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Check hours
        current_hour = utc_dt.hour
        return market_open_hour <= current_hour < market_close_hour

    except Exception as e:
        logger.error("market_hours_check_failed", error=str(e))
        return False


def round_to_timeframe(
    dt: datetime,
    timeframe_seconds: int,
    round_up: bool = False
) -> datetime:
    """Round datetime to timeframe.

    Args:
        dt: Datetime to round
        timeframe_seconds: Timeframe in seconds
        round_up: Round up instead of down

    Returns:
        Rounded datetime

    Raises:
        ValueError: If timeframe invalid
    """
    try:
        if timeframe_seconds <= 0:
            raise ValueError("Timeframe must be positive")

        utc_dt = to_utc(dt)
        timestamp = utc_dt.timestamp()

        if round_up:
            rounded_ts = ((int(timestamp) // timeframe_seconds) + 1) * timeframe_seconds
        else:
            rounded_ts = (int(timestamp) // timeframe_seconds) * timeframe_seconds

        return datetime.fromtimestamp(rounded_ts, tz=timezone.utc)

    except Exception as e:
        logger.error("round_timeframe_failed", timeframe=timeframe_seconds, error=str(e))
        raise ValueError(f"Failed to round to timeframe: {e}")


def get_timeframe_range(
    start: datetime,
    end: datetime,
    timeframe_seconds: int
) -> List[datetime]:
    """Generate list of datetime points for timeframe range.

    Args:
        start: Start datetime
        end: End datetime
        timeframe_seconds: Timeframe in seconds

    Returns:
        List of datetimes

    Raises:
        ValueError: If inputs invalid
    """
    try:
        if timeframe_seconds <= 0:
            raise ValueError("Timeframe must be positive")

        if start >= end:
            raise ValueError("Start must be before end")

        utc_start = to_utc(start)
        utc_end = to_utc(end)

        # Round start to timeframe
        rounded_start = round_to_timeframe(utc_start, timeframe_seconds, round_up=False)

        result = []
        current = rounded_start

        while current <= utc_end:
            result.append(current)
            current = add_time(current, seconds=timeframe_seconds)

        return result

    except Exception as e:
        logger.error("timeframe_range_failed", error=str(e))
        raise ValueError(f"Failed to generate timeframe range: {e}")


def age_seconds(dt: datetime) -> float:
    """Get age of datetime in seconds.

    Args:
        dt: Datetime to check age

    Returns:
        Age in seconds
    """
    try:
        return time_diff_seconds(utc_now(), dt)

    except Exception as e:
        logger.error("age_calculation_failed", error=str(e))
        return 0.0


def is_expired(
    dt: datetime,
    ttl_seconds: float
) -> bool:
    """Check if datetime has expired based on TTL.

    Args:
        dt: Datetime to check
        ttl_seconds: Time-to-live in seconds

    Returns:
        True if expired
    """
    try:
        age = age_seconds(dt)
        return age > ttl_seconds

    except Exception as e:
        logger.error("expiry_check_failed", error=str(e))
        return True  # Assume expired on error


def get_trading_session(dt: Optional[datetime] = None) -> str:
    """Get trading session name for datetime.

    Args:
        dt: Datetime to check (defaults to now)

    Returns:
        Session name ("asian", "european", "american", "off")
    """
    try:
        if dt is None:
            dt = utc_now()

        utc_dt = to_utc(dt)
        hour = utc_dt.hour

        # UTC hours for major sessions
        if 0 <= hour < 8:
            return "asian"
        elif 8 <= hour < 16:
            return "european"
        elif 16 <= hour < 24:
            return "american"
        else:
            return "off"

    except Exception as e:
        logger.error("trading_session_failed", error=str(e))
        return "unknown"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2h 15m 30s")
    """
    try:
        if seconds < 60:
            return f"{seconds:.1f}s"

        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)

        if minutes < 60:
            return f"{minutes}m {remaining_seconds}s"

        hours = minutes // 60
        remaining_minutes = minutes % 60

        if hours < 24:
            return f"{hours}h {remaining_minutes}m"

        days = hours // 24
        remaining_hours = hours % 24

        return f"{days}d {remaining_hours}h"

    except Exception as e:
        logger.error("duration_format_failed", seconds=seconds, error=str(e))
        return f"{seconds}s"


def get_next_timeframe(
    dt: datetime,
    timeframe_seconds: int
) -> datetime:
    """Get next timeframe boundary.

    Args:
        dt: Current datetime
        timeframe_seconds: Timeframe in seconds

    Returns:
        Next timeframe datetime
    """
    try:
        rounded = round_to_timeframe(dt, timeframe_seconds, round_up=True)
        return rounded

    except Exception as e:
        logger.error("next_timeframe_failed", error=str(e))
        raise ValueError(f"Failed to get next timeframe: {e}")


def sleep_until(target_dt: datetime) -> float:
    """Calculate seconds to sleep until target datetime.

    Args:
        target_dt: Target datetime

    Returns:
        Seconds to sleep (0 if target in past)
    """
    try:
        diff = time_diff_seconds(target_dt, utc_now())
        return max(0.0, diff)

    except Exception as e:
        logger.error("sleep_calculation_failed", error=str(e))
        return 0.0
