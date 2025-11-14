"""Helper Utility Functions.

General-purpose helper functions for common operations,
data validation, and system utilities.
"""

import hashlib
import uuid
import os
import sys
import platform
from decimal import Decimal
from typing import Optional, Dict, List, Any, Union, TypeVar, Callable
from datetime import datetime, timezone
import random
import string
from structlog import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


def generate_id(prefix: str = "", length: int = 12) -> str:
    """Generate unique identifier.

    Args:
        prefix: Optional prefix
        length: Length of random part

    Returns:
        Unique identifier string
    """
    try:
        random_part = ''.join(
            random.choices(string.ascii_letters + string.digits, k=length)
        )

        if prefix:
            return f"{prefix}_{random_part}"
        return random_part

    except Exception as e:
        logger.error("id_generation_failed", error=str(e))
        return str(uuid.uuid4())[:length]


def generate_uuid() -> str:
    """Generate UUID v4.

    Returns:
        UUID string
    """
    return str(uuid.uuid4())


def hash_string(
    value: str,
    algorithm: str = "sha256",
    truncate: Optional[int] = None
) -> str:
    """Hash string using specified algorithm.

    Args:
        value: String to hash
        algorithm: Hash algorithm (md5, sha256, sha512)
        truncate: Optional truncation length

    Returns:
        Hex digest string

    Raises:
        ValueError: If algorithm invalid
    """
    try:
        if algorithm == "md5":
            hasher = hashlib.md5()
        elif algorithm == "sha256":
            hasher = hashlib.sha256()
        elif algorithm == "sha512":
            hasher = hashlib.sha512()
        else:
            raise ValueError(f"Invalid hash algorithm: {algorithm}")

        hasher.update(value.encode('utf-8'))
        digest = hasher.hexdigest()

        if truncate:
            return digest[:truncate]

        return digest

    except Exception as e:
        logger.error("hash_failed", algorithm=algorithm, error=str(e))
        raise ValueError(f"Failed to hash string: {e}")


def safe_divide(
    numerator: Union[Decimal, float, int],
    denominator: Union[Decimal, float, int],
    default: Union[Decimal, float, int] = 0
) -> Union[Decimal, float]:
    """Safely divide with zero check.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if denominator is zero

    Returns:
        Division result or default
    """
    try:
        if denominator == 0:
            return default

        if isinstance(numerator, Decimal) or isinstance(denominator, Decimal):
            return Decimal(str(numerator)) / Decimal(str(denominator))
        else:
            return numerator / denominator

    except Exception as e:
        logger.error("division_failed", error=str(e))
        return default


def clamp(
    value: Union[int, float, Decimal],
    min_value: Union[int, float, Decimal],
    max_value: Union[int, float, Decimal]
) -> Union[int, float, Decimal]:
    """Clamp value between min and max.

    Args:
        value: Value to clamp
        min_value: Minimum value
        max_value: Maximum value

    Returns:
        Clamped value
    """
    try:
        return max(min_value, min(value, max_value))

    except Exception as e:
        logger.error("clamp_failed", error=str(e))
        return value


def chunk_list(
    items: List[T],
    chunk_size: int
) -> List[List[T]]:
    """Split list into chunks.

    Args:
        items: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks

    Raises:
        ValueError: If chunk_size invalid
    """
    try:
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")

        return [
            items[i:i + chunk_size]
            for i in range(0, len(items), chunk_size)
        ]

    except Exception as e:
        logger.error("chunk_failed", chunk_size=chunk_size, error=str(e))
        raise


def flatten_dict(
    data: Dict,
    parent_key: str = '',
    separator: str = '.'
) -> Dict:
    """Flatten nested dictionary.

    Args:
        data: Dictionary to flatten
        parent_key: Parent key prefix
        separator: Key separator

    Returns:
        Flattened dictionary
    """
    try:
        items: List[tuple] = []

        for key, value in data.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key

            if isinstance(value, dict):
                items.extend(
                    flatten_dict(value, new_key, separator).items()
                )
            else:
                items.append((new_key, value))

        return dict(items)

    except Exception as e:
        logger.error("flatten_failed", error=str(e))
        return data


def unflatten_dict(
    data: Dict,
    separator: str = '.'
) -> Dict:
    """Unflatten dictionary with dotted keys.

    Args:
        data: Flattened dictionary
        separator: Key separator

    Returns:
        Nested dictionary
    """
    try:
        result: Dict = {}

        for key, value in data.items():
            parts = key.split(separator)
            current = result

            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            current[parts[-1]] = value

        return result

    except Exception as e:
        logger.error("unflatten_failed", error=str(e))
        return data


def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries.

    Later dicts override earlier ones.

    Args:
        *dicts: Dictionaries to merge

    Returns:
        Merged dictionary
    """
    try:
        result = {}
        for d in dicts:
            result.update(d)
        return result

    except Exception as e:
        logger.error("merge_failed", error=str(e))
        return {}


def get_nested_value(
    data: Dict,
    path: str,
    default: Any = None,
    separator: str = '.'
) -> Any:
    """Get value from nested dictionary using path.

    Args:
        data: Dictionary to search
        path: Dot-separated path (e.g., "user.profile.name")
        default: Default value if not found
        separator: Path separator

    Returns:
        Value or default
    """
    try:
        keys = path.split(separator)
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    except Exception as e:
        logger.error("nested_get_failed", path=path, error=str(e))
        return default


def set_nested_value(
    data: Dict,
    path: str,
    value: Any,
    separator: str = '.'
) -> None:
    """Set value in nested dictionary using path.

    Args:
        data: Dictionary to modify
        path: Dot-separated path
        value: Value to set
        separator: Path separator
    """
    try:
        keys = path.split(separator)
        current = data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    except Exception as e:
        logger.error("nested_set_failed", path=path, error=str(e))


def remove_none_values(data: Dict) -> Dict:
    """Remove None values from dictionary.

    Args:
        data: Dictionary to clean

    Returns:
        Dictionary without None values
    """
    try:
        return {k: v for k, v in data.items() if v is not None}

    except Exception as e:
        logger.error("remove_none_failed", error=str(e))
        return data


def retry_on_exception(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    exceptions: tuple = (Exception,),
    **kwargs: Any
) -> Optional[T]:
    """Retry function on specific exceptions.

    Args:
        func: Function to retry
        *args: Positional arguments
        max_retries: Maximum retry attempts
        exceptions: Tuple of exceptions to catch
        **kwargs: Keyword arguments

    Returns:
        Function result or None if all retries failed
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            if attempt == max_retries - 1:
                logger.error(
                    "retry_exhausted",
                    func=func.__name__,
                    attempts=max_retries,
                    error=str(e)
                )
                return None
            logger.warning(
                "retry_attempt",
                func=func.__name__,
                attempt=attempt + 1,
                error=str(e)
            )
    return None


def get_system_info() -> Dict[str, str]:
    """Get system information.

    Returns:
        Dictionary with system details
    """
    try:
        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "hostname": platform.node()
        }

    except Exception as e:
        logger.error("system_info_failed", error=str(e))
        return {"error": str(e)}


def get_env_variable(
    var_name: str,
    default: Optional[str] = None,
    required: bool = False
) -> Optional[str]:
    """Get environment variable with validation.

    Args:
        var_name: Variable name
        default: Default value if not found
        required: Raise error if not found and no default

    Returns:
        Variable value or default

    Raises:
        ValueError: If required and not found
    """
    try:
        value = os.environ.get(var_name, default)

        if required and value is None:
            raise ValueError(f"Required environment variable not set: {var_name}")

        return value

    except Exception as e:
        logger.error("env_var_failed", var=var_name, error=str(e))
        if required:
            raise
        return default


def is_valid_email(email: str) -> bool:
    """Validate email address format.

    Args:
        email: Email address

    Returns:
        True if valid format
    """
    try:
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    except Exception as e:
        logger.error("email_validation_failed", error=str(e))
        return False


def is_valid_url(url: str) -> bool:
    """Validate URL format.

    Args:
        url: URL string

    Returns:
        True if valid format
    """
    try:
        import re
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return bool(re.match(pattern, url, re.IGNORECASE))

    except Exception as e:
        logger.error("url_validation_failed", error=str(e))
        return False


def truncate_string(
    text: str,
    max_length: int,
    suffix: str = "..."
) -> str:
    """Truncate string to maximum length.

    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix for truncated strings

    Returns:
        Truncated string
    """
    try:
        if len(text) <= max_length:
            return text

        truncated_length = max_length - len(suffix)
        return text[:truncated_length] + suffix

    except Exception as e:
        logger.error("truncate_failed", error=str(e))
        return text


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    try:
        import re
        # Remove invalid filename characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove leading/trailing whitespace and dots
        sanitized = sanitized.strip('. ')
        return sanitized

    except Exception as e:
        logger.error("filename_sanitize_failed", error=str(e))
        return filename


def calculate_percentage_change(
    old_value: Union[Decimal, float],
    new_value: Union[Decimal, float]
) -> Decimal:
    """Calculate percentage change between values.

    Args:
        old_value: Original value
        new_value: New value

    Returns:
        Percentage change as decimal

    Raises:
        ValueError: If old_value is zero
    """
    try:
        if old_value == 0:
            raise ValueError("Cannot calculate percentage change from zero")

        old_dec = Decimal(str(old_value))
        new_dec = Decimal(str(new_value))

        change = (new_dec - old_dec) / old_dec

        return change

    except Exception as e:
        logger.error("percentage_change_failed", error=str(e))
        raise


def moving_average(
    values: List[Union[int, float, Decimal]],
    window: int
) -> List[Decimal]:
    """Calculate simple moving average.

    Args:
        values: List of values
        window: Window size

    Returns:
        List of moving averages

    Raises:
        ValueError: If window invalid
    """
    try:
        if window <= 0:
            raise ValueError("Window size must be positive")

        if len(values) < window:
            raise ValueError("Not enough values for window size")

        result = []

        for i in range(len(values) - window + 1):
            window_values = values[i:i + window]
            avg = sum(Decimal(str(v)) for v in window_values) / Decimal(str(window))
            result.append(avg)

        return result

    except Exception as e:
        logger.error("moving_average_failed", window=window, error=str(e))
        raise


def deduplicate_list(
    items: List[T],
    key: Optional[Callable[[T], Any]] = None
) -> List[T]:
    """Remove duplicates from list while preserving order.

    Args:
        items: List to deduplicate
        key: Optional key function for comparison

    Returns:
        List without duplicates
    """
    try:
        if key is None:
            seen = set()
            result = []
            for item in items:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return result
        else:
            seen = set()
            result = []
            for item in items:
                k = key(item)
                if k not in seen:
                    seen.add(k)
                    result.append(item)
            return result

    except Exception as e:
        logger.error("deduplicate_failed", error=str(e))
        return items
