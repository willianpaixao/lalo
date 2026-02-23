"""
Text normalization utilities for Lalo.

This module provides utilities to normalize text for natural speech synthesis,
particularly useful when converting ebooks to audiobooks. Currently handles:
- URL normalization (http/https, www, domain names)

Future extensions may include:
- Email address normalization
- Number-to-words conversion
- Date and time formatting
- Special character handling
"""

import re
from enum import Enum
from typing import Callable, Optional, Union

# Comprehensive URL matching pattern
# Uses negative lookahead to exclude trailing punctuation
URL_PATTERN = re.compile(
    r'https?://[^\s<>"{}|\\^`\[\]()]+'  # http(s)://...
    r"(?<![.,;:!?])"  # Don't capture trailing punctuation
    r'|www\.[^\s<>"{}|\\^`\[\]()]+'  # www....
    r"(?<![.,;:!?])"  # Don't capture trailing punctuation
    r"|\b(?:[\w-]+\.)+(?:com|org|net|edu|gov|io|ai|co|uk|de|fr|jp|cn|br)\b"  # domains
    r'(?:/[^\s<>"{}|\\^`\[\]()]*)?'  # optional path
    r"(?<![.,;:!?])",  # Don't capture trailing punctuation
    flags=re.IGNORECASE,
)


class URLReplacementStrategy(Enum):
    """Strategy for replacing URLs in text for natural speech synthesis.

    Attributes:
        GENERIC: Replace URLs with customizable link text (default: "a link")
        DOMAIN: Extract and speak the domain name (e.g., "example dot com")
        REMOVE: Remove URLs entirely from text
        CUSTOM: Use a custom string or callback function for replacement
    """

    GENERIC = "generic"
    DOMAIN = "domain"
    REMOVE = "remove"
    CUSTOM = "custom"


def _extract_domain(url: str) -> str:
    """
    Extract readable domain from URL for speech synthesis.

    Args:
        url: Full URL string

    Returns:
        Domain formatted for natural speech (e.g., "example dot com")

    Examples:
        >>> _extract_domain("https://www.example.com/path")
        "example dot com"
        >>> _extract_domain("http://github.com:8080/repo")
        "github dot com"
        >>> _extract_domain("www.python.org")
        "python dot org"
    """
    # Remove protocol
    domain = re.sub(r"^https?://", "", url)
    # Remove www prefix
    domain = re.sub(r"^www\.", "", domain)
    # Get just domain (before first / or ?)
    domain = re.split(r"[/?]", domain)[0]
    # Remove port number
    domain = domain.split(":")[0]
    # Replace dots with "dot" for speech
    domain = domain.replace(".", " dot ")
    return domain.strip()


def _replace_url_generic(url: str, link_text: str) -> str:
    """Replace URL with generic link text.

    Args:
        url: URL to replace
        link_text: Text to use as replacement

    Returns:
        The link text
    """
    return link_text


def _replace_url_domain(url: str) -> str:
    """Replace URL with spoken domain name.

    Args:
        url: URL to replace

    Returns:
        Domain formatted for speech
    """
    return _extract_domain(url)


def _replace_url_remove(url: str) -> str:
    """Remove URL entirely.

    Args:
        url: URL to remove

    Returns:
        Empty string
    """
    return ""


def _replace_url_custom(url: str, replacement: Union[str, Callable[[str], str]]) -> str:
    """Replace URL using custom logic.

    Args:
        url: URL to replace
        replacement: Custom string or callable that takes URL and returns replacement

    Returns:
        Replacement text

    Raises:
        TypeError: If replacement is not a string or callable
    """
    if callable(replacement):
        return replacement(url)
    elif isinstance(replacement, str):
        return replacement
    else:
        raise TypeError(
            f"custom_replacement must be str or callable, got {type(replacement).__name__}"
        )


def normalize_urls(
    text: str,
    strategy: Union[str, URLReplacementStrategy] = URLReplacementStrategy.GENERIC,
    custom_replacement: Optional[Union[str, Callable[[str], str]]] = None,
    link_text: str = "a link",
) -> str:
    """
    Normalize URLs in text for natural speech synthesis.

    This function identifies URLs in text and replaces them according to the
    specified strategy, making the text more suitable for text-to-speech conversion.

    Args:
        text: Input text containing URLs
        strategy: Replacement strategy (GENERIC, DOMAIN, REMOVE, or CUSTOM).
            Can be a URLReplacementStrategy enum or string. Defaults to GENERIC.
        custom_replacement: Custom string or callback for CUSTOM strategy.
            If callable, receives URL and returns replacement text.
        link_text: Text to use for GENERIC strategy. Defaults to "a link".

    Returns:
        Text with URLs normalized according to the strategy

    Raises:
        ValueError: If strategy is invalid or CUSTOM strategy used without custom_replacement
        TypeError: If custom_replacement is not a string or callable

    Examples:
        >>> normalize_urls("Check https://example.com for info")
        'Check a link for info'

        >>> normalize_urls("See https://example.com", strategy="domain")
        'See example dot com'

        >>> normalize_urls("Visit https://example.com", strategy="remove")
        'Visit'

        >>> normalize_urls("Go to https://github.com", link_text="the website")
        'Go to the website'

        >>> custom_fn = lambda url: "our repo" if "github" in url else "a link"
        >>> normalize_urls("See https://github.com/user", strategy="custom",
        ...                custom_replacement=custom_fn)
        'See our repo'
    """
    # Convert string strategy to enum
    if isinstance(strategy, str):
        try:
            strategy = URLReplacementStrategy(strategy.lower())
        except ValueError as e:
            valid = ", ".join([s.value for s in URLReplacementStrategy])
            raise ValueError(f"Invalid strategy '{strategy}'. Must be one of: {valid}") from e

    # Validate CUSTOM strategy has replacement
    if strategy == URLReplacementStrategy.CUSTOM and custom_replacement is None:
        raise ValueError("CUSTOM strategy requires custom_replacement parameter")

    # Define replacement function based on strategy
    def replace_match(match: re.Match) -> str:
        url = match.group(0)

        if strategy == URLReplacementStrategy.GENERIC:
            return _replace_url_generic(url, link_text)
        elif strategy == URLReplacementStrategy.DOMAIN:
            return _replace_url_domain(url)
        elif strategy == URLReplacementStrategy.REMOVE:
            return _replace_url_remove(url)
        elif strategy == URLReplacementStrategy.CUSTOM:
            # custom_replacement is guaranteed to be non-None here due to validation above
            assert custom_replacement is not None  # nosec - validated above
            return _replace_url_custom(url, custom_replacement)
        else:
            # This should never happen, but for type safety
            return url

    # Replace all URLs
    result = URL_PATTERN.sub(replace_match, text)

    # Clean up extra whitespace that may result from URL removal
    # But preserve single spaces between words
    result = re.sub(r" {2,}", " ", result)  # Only collapse 2+ spaces
    result = result.strip()

    return result


def normalize_text_for_speech(
    text: str,
    url_strategy: Union[str, URLReplacementStrategy] = URLReplacementStrategy.GENERIC,
    custom_url_replacement: Optional[Union[str, Callable[[str], str]]] = None,
    link_text: str = "a link",
) -> str:
    """
    Normalize text for natural speech synthesis by handling URLs and other patterns.

    This is the main entry point for text normalization. Currently handles:
    - URLs (http/https, www., domains)

    Future extensions may include:
    - Email addresses
    - Numbers and dates
    - Special characters

    Args:
        text: Input text to normalize
        url_strategy: How to handle URLs (GENERIC, DOMAIN, REMOVE, or CUSTOM).
            Defaults to GENERIC ("a link").
        custom_url_replacement: Custom replacement for URLs when using CUSTOM strategy.
            Can be a string or callable that takes URL and returns replacement.
        link_text: Text for GENERIC URL replacement. Defaults to "a link".

    Returns:
        Normalized text ready for TTS synthesis

    Raises:
        ValueError: If url_strategy is invalid or CUSTOM strategy used without replacement
        TypeError: If custom_url_replacement is not a string or callable

    Examples:
        >>> from lalo import normalize_text_for_speech, URLReplacementStrategy
        >>>
        >>> text = "Visit https://example.com for details"
        >>> normalize_text_for_speech(text)
        'Visit a link for details'
        >>>
        >>> normalize_text_for_speech(text, url_strategy=URLReplacementStrategy.DOMAIN)
        'Visit example dot com for details'
        >>>
        >>> normalize_text_for_speech(text, url_strategy="remove")
        'Visit for details'
        >>>
        >>> def handler(url):
        ...     return "the website" if "example" in url else "a link"
        >>> normalize_text_for_speech(text, url_strategy="custom",
        ...                          custom_url_replacement=handler)
        'Visit the website for details'
    """
    # Currently only handles URLs, but designed for easy extension
    return normalize_urls(
        text,
        strategy=url_strategy,
        custom_replacement=custom_url_replacement,
        link_text=link_text,
    )
