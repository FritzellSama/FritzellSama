"""String Utility Functions.

Production-ready string manipulation, validation, and formatting utilities
for text processing in trading systems.
"""

import re
from typing import Optional, List, Dict, Any
from structlog import get_logger

logger = get_logger(__name__)


def to_snake_case(text: str) -> str:
    """Convert string to snake_case.

    Args:
        text: Input string

    Returns:
        snake_case string
    """
    try:
        # Replace spaces and hyphens with underscores
        text = text.replace(' ', '_').replace('-', '_')

        # Insert underscore before uppercase letters
        text = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
        text = re.sub('([a-z0-9])([A-Z])', r'\1_\2', text)

        # Convert to lowercase
        return text.lower()

    except Exception as e:
        logger.error("snake_case_failed", text=text, error=str(e))
        return text


def to_camel_case(text: str) -> str:
    """Convert string to camelCase.

    Args:
        text: Input string

    Returns:
        camelCase string
    """
    try:
        # Split on spaces, hyphens, and underscores
        words = re.split(r'[\s_-]+', text)

        if not words:
            return text

        # First word lowercase, rest title case
        result = words[0].lower()
        for word in words[1:]:
            result += word.capitalize()

        return result

    except Exception as e:
        logger.error("camel_case_failed", text=text, error=str(e))
        return text


def to_pascal_case(text: str) -> str:
    """Convert string to PascalCase.

    Args:
        text: Input string

    Returns:
        PascalCase string
    """
    try:
        # Split on spaces, hyphens, and underscores
        words = re.split(r'[\s_-]+', text)

        # Capitalize all words
        return ''.join(word.capitalize() for word in words)

    except Exception as e:
        logger.error("pascal_case_failed", text=text, error=str(e))
        return text


def to_kebab_case(text: str) -> str:
    """Convert string to kebab-case.

    Args:
        text: Input string

    Returns:
        kebab-case string
    """
    try:
        # Replace spaces and underscores with hyphens
        text = text.replace(' ', '-').replace('_', '-')

        # Insert hyphen before uppercase letters
        text = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', text)
        text = re.sub('([a-z0-9])([A-Z])', r'\1-\2', text)

        # Convert to lowercase
        return text.lower()

    except Exception as e:
        logger.error("kebab_case_failed", text=text, error=str(e))
        return text


def truncate(
    text: str,
    max_length: int,
    suffix: str = "...",
    word_boundary: bool = False
) -> str:
    """Truncate string to maximum length.

    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix for truncated strings
        word_boundary: Truncate at word boundary

    Returns:
        Truncated string
    """
    try:
        if len(text) <= max_length:
            return text

        truncated_length = max_length - len(suffix)

        if word_boundary:
            # Find last space before limit
            truncated = text[:truncated_length]
            last_space = truncated.rfind(' ')
            if last_space > 0:
                truncated = truncated[:last_space]
            return truncated + suffix
        else:
            return text[:truncated_length] + suffix

    except Exception as e:
        logger.error("truncate_failed", max_length=max_length, error=str(e))
        return text


def strip_whitespace(text: str, internal: bool = False) -> str:
    """Strip whitespace from string.

    Args:
        text: Input string
        internal: Also collapse internal whitespace

    Returns:
        Cleaned string
    """
    try:
        # Strip leading/trailing
        cleaned = text.strip()

        if internal:
            # Collapse internal whitespace
            cleaned = re.sub(r'\s+', ' ', cleaned)

        return cleaned

    except Exception as e:
        logger.error("strip_whitespace_failed", error=str(e))
        return text


def remove_special_chars(
    text: str,
    keep: str = "",
    replace_with: str = ""
) -> str:
    """Remove special characters from string.

    Args:
        text: Input string
        keep: Characters to keep (in addition to alphanumeric)
        replace_with: Replacement for removed characters

    Returns:
        Cleaned string
    """
    try:
        # Build pattern: keep alphanumeric and specified chars
        pattern = f"[^a-zA-Z0-9{re.escape(keep)}]"

        return re.sub(pattern, replace_with, text)

    except Exception as e:
        logger.error("remove_special_failed", error=str(e))
        return text


def pad_left(text: str, length: int, char: str = " ") -> str:
    """Pad string on left to specified length.

    Args:
        text: Input string
        length: Target length
        char: Padding character

    Returns:
        Padded string
    """
    try:
        if len(text) >= length:
            return text

        padding = char * (length - len(text))
        return padding + text

    except Exception as e:
        logger.error("pad_left_failed", error=str(e))
        return text


def pad_right(text: str, length: int, char: str = " ") -> str:
    """Pad string on right to specified length.

    Args:
        text: Input string
        length: Target length
        char: Padding character

    Returns:
        Padded string
    """
    try:
        if len(text) >= length:
            return text

        padding = char * (length - len(text))
        return text + padding

    except Exception as e:
        logger.error("pad_right_failed", error=str(e))
        return text


def extract_numbers(text: str) -> List[str]:
    """Extract all numbers from string.

    Args:
        text: Input string

    Returns:
        List of number strings
    """
    try:
        # Match integers and decimals
        pattern = r'-?\d+\.?\d*'
        return re.findall(pattern, text)

    except Exception as e:
        logger.error("extract_numbers_failed", error=str(e))
        return []


def extract_words(text: str, min_length: int = 1) -> List[str]:
    """Extract words from string.

    Args:
        text: Input string
        min_length: Minimum word length

    Returns:
        List of words
    """
    try:
        # Extract alphabetic words
        words = re.findall(r'\b[a-zA-Z]+\b', text)

        # Filter by length
        return [w for w in words if len(w) >= min_length]

    except Exception as e:
        logger.error("extract_words_failed", error=str(e))
        return []


def contains_any(text: str, substrings: List[str], case_sensitive: bool = True) -> bool:
    """Check if string contains any of the substrings.

    Args:
        text: String to search
        substrings: List of substrings to find
        case_sensitive: Case-sensitive search

    Returns:
        True if any substring found
    """
    try:
        if not case_sensitive:
            text = text.lower()
            substrings = [s.lower() for s in substrings]

        return any(sub in text for sub in substrings)

    except Exception as e:
        logger.error("contains_any_failed", error=str(e))
        return False


def contains_all(text: str, substrings: List[str], case_sensitive: bool = True) -> bool:
    """Check if string contains all of the substrings.

    Args:
        text: String to search
        substrings: List of substrings to find
        case_sensitive: Case-sensitive search

    Returns:
        True if all substrings found
    """
    try:
        if not case_sensitive:
            text = text.lower()
            substrings = [s.lower() for s in substrings]

        return all(sub in text for sub in substrings)

    except Exception as e:
        logger.error("contains_all_failed", error=str(e))
        return False


def replace_multiple(
    text: str,
    replacements: Dict[str, str],
    case_sensitive: bool = True
) -> str:
    """Replace multiple substrings.

    Args:
        text: Input string
        replacements: Dictionary of {old: new} replacements
        case_sensitive: Case-sensitive replacement

    Returns:
        String with replacements applied
    """
    try:
        result = text

        for old, new in replacements.items():
            if case_sensitive:
                result = result.replace(old, new)
            else:
                # Case-insensitive replacement
                pattern = re.compile(re.escape(old), re.IGNORECASE)
                result = pattern.sub(new, result)

        return result

    except Exception as e:
        logger.error("replace_multiple_failed", error=str(e))
        return text


def split_preserve_quotes(text: str, delimiter: str = ',') -> List[str]:
    """Split string preserving quoted sections.

    Args:
        text: Input string
        delimiter: Delimiter character

    Returns:
        List of split parts

    Example:
        >>> split_preserve_quotes('a,b,"c,d",e')
        ['a', 'b', '"c,d"', 'e']
    """
    try:
        # Split on delimiter unless inside quotes
        pattern = f'{delimiter}(?=(?:[^"]*"[^"]*")*[^"]*$)'
        parts = re.split(pattern, text)

        # Strip whitespace
        return [p.strip() for p in parts]

    except Exception as e:
        logger.error("split_quotes_failed", error=str(e))
        return [text]


def mask_sensitive(
    text: str,
    visible_start: int = 4,
    visible_end: int = 4,
    mask_char: str = "*"
) -> str:
    """Mask sensitive information in string.

    Args:
        text: Input string
        visible_start: Number of visible characters at start
        visible_end: Number of visible characters at end
        mask_char: Masking character

    Returns:
        Masked string
    """
    try:
        length = len(text)

        if length <= visible_start + visible_end:
            # String too short to mask
            return mask_char * length

        start = text[:visible_start]
        end = text[-visible_end:] if visible_end > 0 else ""
        masked_length = length - visible_start - visible_end

        return f"{start}{mask_char * masked_length}{end}"

    except Exception as e:
        logger.error("mask_sensitive_failed", error=str(e))
        return mask_char * len(text)


def is_numeric(text: str) -> bool:
    """Check if string represents a number.

    Args:
        text: String to check

    Returns:
        True if numeric
    """
    try:
        # Try to convert to float
        float(text)
        return True
    except (ValueError, TypeError):
        return False


def is_alphanumeric(text: str) -> bool:
    """Check if string contains only alphanumeric characters.

    Args:
        text: String to check

    Returns:
        True if alphanumeric
    """
    try:
        return text.isalnum()
    except Exception as e:
        logger.error("alphanumeric_check_failed", error=str(e))
        return False


def reverse(text: str) -> str:
    """Reverse string.

    Args:
        text: Input string

    Returns:
        Reversed string
    """
    try:
        return text[::-1]
    except Exception as e:
        logger.error("reverse_failed", error=str(e))
        return text


def count_words(text: str) -> int:
    """Count words in string.

    Args:
        text: Input string

    Returns:
        Word count
    """
    try:
        words = text.split()
        return len(words)
    except Exception as e:
        logger.error("count_words_failed", error=str(e))
        return 0


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance
    """
    try:
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)

        for i, c1 in enumerate(s1):
            current_row = [i + 1]

            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)

                current_row.append(min(insertions, deletions, substitutions))

            previous_row = current_row

        return previous_row[-1]

    except Exception as e:
        logger.error("levenshtein_failed", error=str(e))
        return max(len(s1), len(s2))


def normalize_whitespace(text: str) -> str:
    """Normalize all whitespace to single spaces.

    Args:
        text: Input string

    Returns:
        Normalized string
    """
    try:
        # Replace all whitespace with single space
        return ' '.join(text.split())

    except Exception as e:
        logger.error("normalize_whitespace_failed", error=str(e))
        return text


def remove_duplicates(text: str, delimiter: str = ',') -> str:
    """Remove duplicate items from delimited string.

    Args:
        text: Delimited string
        delimiter: Delimiter character

    Returns:
        String without duplicates
    """
    try:
        items = [item.strip() for item in text.split(delimiter)]

        # Remove duplicates while preserving order
        seen = set()
        unique = []

        for item in items:
            if item not in seen:
                seen.add(item)
                unique.append(item)

        return delimiter.join(unique)

    except Exception as e:
        logger.error("remove_duplicates_failed", error=str(e))
        return text


def escape_html(text: str) -> str:
    """Escape HTML special characters.

    Args:
        text: Input string

    Returns:
        HTML-escaped string
    """
    try:
        replacements = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;'
        }

        return replace_multiple(text, replacements)

    except Exception as e:
        logger.error("escape_html_failed", error=str(e))
        return text


def unescape_html(text: str) -> str:
    """Unescape HTML entities.

    Args:
        text: HTML-escaped string

    Returns:
        Unescaped string
    """
    try:
        replacements = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#x27;': "'"
        }

        return replace_multiple(text, replacements)

    except Exception as e:
        logger.error("unescape_html_failed", error=str(e))
        return text
