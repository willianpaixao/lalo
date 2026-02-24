"""
Text normalization utilities for Lalo.

This module provides utilities to normalize text for natural speech synthesis,
particularly useful when converting ebooks to audiobooks.  Handles:

- URL normalization (http/https, www, domain names)
- Email address normalization (user@host → "user at host dot com")
- Date expansion (ISO, written month, standalone years)
- Currency expansion (£2,500 → "two thousand five hundred pounds")
- Number-to-words conversion (1,234 → "one thousand two hundred thirty-four")
- Abbreviation expansion (Dr., e.g., etc.)
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Callable, Optional, Union

import num2words

from lalo.config import CURRENCY_MAP, NUM2WORDS_LANGUAGE_MAP

logger = logging.getLogger(__name__)

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


# Email normalization

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
)


def normalize_emails(text: str) -> str:
    """
    Normalize email addresses for natural speech.

    Converts characters that are unpronounceable as-is into spoken
    equivalents so the TTS model produces natural output.

    Args:
        text: Input text potentially containing email addresses

    Returns:
        Text with email addresses expanded for speech

    Examples:
        >>> normalize_emails("Contact info@example.com today")
        'Contact info at example dot com today'
        >>> normalize_emails("Mail john.doe+tag@company-name.co.uk")
        'Mail john dot doe plus tag at company dash name dot co dot uk'
    """

    def _expand_email(match: re.Match) -> str:
        email = match.group(0)
        local, domain = email.rsplit("@", 1)

        # Expand local part
        local = local.replace(".", " dot ")
        local = local.replace("_", " underscore ")
        local = local.replace("-", " dash ")
        local = local.replace("+", " plus ")

        # Expand domain part
        domain = domain.replace("-", " dash ")
        domain = domain.replace(".", " dot ")

        return f"{local} at {domain}"

    return EMAIL_PATTERN.sub(_expand_email, text)


# ISO date normalization

# Matches YYYY-MM-DD (with optional leading zeros)
ISO_DATE_PATTERN = re.compile(
    r"\b(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b",
)

_MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

_MONTH_NAMES_PATTERN = "|".join(_MONTH_NAMES)

# "July 4, 1776" or "February 23 2026"
WRITTEN_DATE_MDY_PATTERN = re.compile(
    rf"\b({_MONTH_NAMES_PATTERN})\s+(\d{{1,2}}),?\s+(\d{{4}})\b",
)

# "23 February 2026"
WRITTEN_DATE_DMY_PATTERN = re.compile(
    rf"\b(\d{{1,2}})\s+({_MONTH_NAMES_PATTERN})\s+(\d{{4}})\b",
)

# "in 1984", "since 2020", "from 1776" etc.
STANDALONE_YEAR_PATTERN = re.compile(
    r"\b(in|since|from|until|by|around|circa|after|before|during)\s+"
    r"(\d{4})\b",
    re.IGNORECASE,
)


def _num2words_safe(
    value: int | float,
    lang: str = "en",
    **kwargs: str,
) -> str:
    """Call num2words with a fallback to English on unsupported languages."""
    try:
        return str(num2words.num2words(value, lang=lang, **kwargs))
    except (NotImplementedError, OverflowError):
        return str(num2words.num2words(value, lang="en", **kwargs))


def normalize_dates(text: str, language: str = "English") -> str:
    """
    Normalize dates for natural speech.

    Handles the following formats:

    * **ISO** — ``2026-02-23``
    * **Written month** — ``February 23, 2026`` or ``23 February 2026``
    * **Standalone year** — ``in 1984``, ``since 2020`` (preceded by a
      preposition)

    US-slash (``12/31/2025``) and European-slash (``31/12/2025``) formats
    are deliberately **not** handled because they are ambiguous (is
    ``01/02/2025`` January 2nd or February 1st?) and the risk of
    mis-reading outweighs the benefit.

    Args:
        text: Input text potentially containing dates
        language: Language for ordinal day / year words (default: English)

    Returns:
        Text with dates expanded for speech

    Examples:
        >>> normalize_dates("Published on 2026-02-23.")
        'Published on February twenty-third, twenty twenty-six.'
        >>> normalize_dates("On July 4, 1776, it was signed.")
        'On July fourth, seventeen seventy-six, it was signed.'
        >>> normalize_dates("Written in 1984 by Orwell.")
        'Written in nineteen eighty-four by Orwell.'
    """
    lang_code = NUM2WORDS_LANGUAGE_MAP.get(language, "en")

    # Pass 1: ISO dates (YYYY-MM-DD)
    def _expand_iso(match: re.Match) -> str:
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        month_name = _MONTH_NAMES[month - 1]
        day_word = _num2words_safe(day, lang_code, to="ordinal")
        year_word = _num2words_safe(year, lang_code, to="year")
        return f"{month_name} {day_word}, {year_word}"

    text = ISO_DATE_PATTERN.sub(_expand_iso, text)

    # Pass 2: Written dates — "Month DD, YYYY" or "Month DD YYYY"
    # e.g. "July 4, 1776" or "February 23 2026"
    def _expand_written_mdy(match: re.Match) -> str:
        month_name = match.group(1)
        day = int(match.group(2))
        year = int(match.group(3))
        day_word = _num2words_safe(day, lang_code, to="ordinal")
        year_word = _num2words_safe(year, lang_code, to="year")
        return f"{month_name} {day_word}, {year_word}"

    text = WRITTEN_DATE_MDY_PATTERN.sub(_expand_written_mdy, text)

    # Pass 3: Written dates — "DD Month YYYY"
    # e.g. "23 February 2026"
    def _expand_written_dmy(match: re.Match) -> str:
        day = int(match.group(1))
        month_name = match.group(2)
        year = int(match.group(3))
        day_word = _num2words_safe(day, lang_code, to="ordinal")
        year_word = _num2words_safe(year, lang_code, to="year")
        return f"{day_word} of {month_name}, {year_word}"

    text = WRITTEN_DATE_DMY_PATTERN.sub(_expand_written_dmy, text)

    # Pass 4: Standalone years preceded by a preposition
    # e.g. "in 1984", "since 2020", "from 1776", "until 2030", "by 2050"
    def _expand_year(match: re.Match) -> str:
        prep = match.group(1)
        year = int(match.group(2))
        year_word = _num2words_safe(year, lang_code, to="year")
        return f"{prep} {year_word}"

    text = STANDALONE_YEAR_PATTERN.sub(_expand_year, text)

    return text


# Currency normalization

# Matches a currency symbol followed by an amount.
# Handles: $42  €1,234.56  £2,500,000  ¥100  ₩5000  R$42.50
# The R$ prefix is matched first (longest match).
CURRENCY_PATTERN = re.compile(
    r"(R\$|[$€£¥₩₽])\s?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)",
)


def normalize_currency(text: str, language: str = "English") -> str:
    """
    Normalize currency amounts for natural speech.

    Expands currency symbols and amounts into spoken words using
    ``num2words``.  Falls back to a simple expansion if the currency
    code is not recognised by ``num2words``.

    Args:
        text: Input text potentially containing currency amounts
        language: Language for number words (default: English)

    Returns:
        Text with currency amounts expanded for speech

    Examples:
        >>> normalize_currency("Costs £42.")
        'Costs forty-two pounds.'
        >>> normalize_currency("Price: $1,299.99")
        'Price: one thousand, two hundred and ninety-nine dollars, ninety-nine cents'
    """
    lang_code = NUM2WORDS_LANGUAGE_MAP.get(language, "en")

    def _expand_currency(match: re.Match) -> str:
        symbol = match.group(1)
        amount_str = match.group(2).replace(",", "")
        amount = float(amount_str)

        currency_info = CURRENCY_MAP.get(symbol)
        if currency_info is None:
            # Unknown symbol — just convert the number
            try:
                return str(num2words.num2words(amount, lang=lang_code))
            except (NotImplementedError, OverflowError):
                return str(num2words.num2words(amount, lang="en"))

        currency_code, currency_name = currency_info

        # Try num2words currency mode first
        try:
            return str(
                num2words.num2words(amount, to="currency", lang=lang_code, currency=currency_code)
            )
        except (NotImplementedError, OverflowError, ValueError):
            pass

        # Fallback: number + currency name
        try:
            number_words = num2words.num2words(amount, lang=lang_code)
        except (NotImplementedError, OverflowError):
            number_words = num2words.num2words(amount, lang="en")

        return f"{number_words} {currency_name}"

    return CURRENCY_PATTERN.sub(_expand_currency, text)


# Abbreviation normalization

# Honorific / title abbreviations.
# Order matters: longer matches first (Mrs. before Mr.).
_TITLE_ABBREVIATIONS: dict[str, str] = {
    "Mrs.": "Misses",
    "Mr.": "Mister",
    "Ms.": "Miss",
    "Dr.": "Doctor",
    "Prof.": "Professor",
    "Jr.": "Junior",
    "Sr.": "Senior",
    "Rev.": "Reverend",
    "Gen.": "General",
    "Sgt.": "Sergeant",
    "Capt.": "Captain",
    "Col.": "Colonel",
    "Lt.": "Lieutenant",
    "Gov.": "Governor",
    "Pres.": "President",
    "Sec.": "Secretary",
}

# Latin and general abbreviations.
_GENERAL_ABBREVIATIONS: dict[str, str] = {
    "e.g.": "for example",
    "i.e.": "that is",
    "etc.": "et cetera",
    "vs.": "versus",
    "approx.": "approximately",
    "dept.": "department",
    "govt.": "government",
    "inc.": "incorporated",
    "corp.": "corporation",
    "ltd.": "limited",
    "vol.": "volume",
    "ch.": "chapter",
    "pg.": "page",
    "no.": "number",
    "fig.": "figure",
    "ref.": "reference",
    "est.": "established",
}

# "St." is ambiguous: "Saint" vs "Street".
# Heuristic: preceded by a number → Street; otherwise → Saint.
_ST_PRECEDED_BY_NUMBER = re.compile(r"\d+\w*\s+St\.", re.IGNORECASE)
_ST_STANDALONE = re.compile(r"\bSt\.")


def _build_abbreviation_pattern(
    abbrevs: dict[str, str],
) -> re.Pattern:
    """Build a compiled regex matching any of the given abbreviations.

    Keys are sorted by length descending so "Mrs." matches before "Mr.".
    Each key is escaped and anchored on a word boundary at the start.
    """
    escaped = [re.escape(k) for k in sorted(abbrevs, key=len, reverse=True)]
    pattern = r"\b(?:" + "|".join(escaped) + ")"
    return re.compile(pattern, re.IGNORECASE)


_TITLE_PATTERN = _build_abbreviation_pattern(_TITLE_ABBREVIATIONS)
_GENERAL_PATTERN = _build_abbreviation_pattern(_GENERAL_ABBREVIATIONS)


def normalize_abbreviations(text: str) -> str:
    """
    Expand common abbreviations for natural speech.

    Handles two categories:

    1.  **Titles and honorifics** — ``Dr.``, ``Mr.``, ``Mrs.``, ``Prof.``, etc.
    2.  **Latin and general** — ``e.g.``, ``i.e.``, ``etc.``, ``vs.``, etc.

    The ambiguous ``St.`` is handled with a heuristic: if preceded by a
    number it becomes "Street", otherwise "Saint".

    Args:
        text: Input text potentially containing abbreviations

    Returns:
        Text with abbreviations expanded

    Examples:
        >>> normalize_abbreviations("Dr. Smith met Prof. Jones")
        'Doctor Smith met Professor Jones'
        >>> normalize_abbreviations("Several species, e.g. wolves")
        'Several species, for example wolves'
        >>> normalize_abbreviations("5th St. near St. Patrick")
        '5th Street near Saint Patrick'
    """
    # Build case-insensitive lookup (keys are stored capitalised as written)
    title_lookup = {k.lower(): v for k, v in _TITLE_ABBREVIATIONS.items()}
    general_lookup = {k.lower(): v for k, v in _GENERAL_ABBREVIATIONS.items()}

    def _replace_title(match: re.Match) -> str:
        original: str = match.group(0)
        key = original.lower()
        replacement = title_lookup.get(key)
        if replacement is None:
            return original
        # Preserve leading capitalisation from the original
        if original[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement

    def _replace_general(match: re.Match) -> str:
        original = match.group(0)
        key = original.lower()
        result = general_lookup.get(key)
        return result if result is not None else original

    # Handle "St." ambiguity first
    text = _ST_PRECEDED_BY_NUMBER.sub(lambda m: m.group(0).replace("St.", "Street"), text)
    # Remaining "St." → "Saint"
    text = _ST_STANDALONE.sub("Saint", text)

    # Titles
    text = _TITLE_PATTERN.sub(_replace_title, text)
    # General / Latin
    text = _GENERAL_PATTERN.sub(_replace_general, text)

    return text


# Number-to-words normalization

# Ordinal suffixes: 1st, 2nd, 3rd, 21st, 42nd, 103rd, etc.
ORDINAL_PATTERN = re.compile(
    r"\b(\d+)(?:st|nd|rd|th)\b",
    re.IGNORECASE,
)

# Percentage: 42%, 3.5%
PERCENTAGE_PATTERN = re.compile(
    r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s?%",
)

# Generic numbers: integers with optional commas and decimals.
# Must NOT match numbers that are part of a time (14:30), version
# strings (v2.3.1), or already-normalised text.
# Phone-number-like sequences (7+ digits with no commas) are skipped
# because digit-by-digit reading is rarely better than what the TTS does.
NUMBER_PATTERN = re.compile(
    r"(?<![:\w])"  # not preceded by colon (time) or word char (version)
    r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?"  # the number itself
    r"(?![:\d])"  # not followed by colon or more digits
    r"\b",
)


def normalize_numbers(text: str, language: str = "English") -> str:
    """
    Convert standalone numbers to words for natural speech.

    Handles:
    * **Ordinals** — ``1st`` → "first", ``23rd`` → "twenty-third"
    * **Percentages** — ``42%`` → "forty-two percent"
    * **Integers** — ``1,234`` → "one thousand, two hundred and thirty-four"
    * **Decimals** — ``3.14`` → "three point one four"
    * **Negatives** — ``-15`` → "minus fifteen"

    Sequences of 7+ unformatted digits (likely phone numbers, IDs) are
    left untouched because digit-by-digit expansion is rarely an
    improvement over what the TTS model produces natively.

    Args:
        text: Input text potentially containing numbers
        language: Language for number words (default: English)

    Returns:
        Text with numbers expanded to words

    Examples:
        >>> normalize_numbers("Population: 1,234,567 people.")
        'Population: one million, two hundred and thirty-four thousand, five hundred and sixty-seven people.'
        >>> normalize_numbers("She came in 1st place.")
        'She came in first place.'
        >>> normalize_numbers("Prices rose 42%.")
        'Prices rose forty-two percent.'
    """
    lang_code = NUM2WORDS_LANGUAGE_MAP.get(language, "en")

    # Pass 1: ordinals (1st, 2nd, 3rd, ...)
    def _expand_ordinal(match: re.Match) -> str:
        n = int(match.group(1))
        return _num2words_safe(n, lang_code, to="ordinal")

    text = ORDINAL_PATTERN.sub(_expand_ordinal, text)

    # Pass 2: percentages (42%)
    def _expand_percentage(match: re.Match) -> str:
        amount_str = match.group(1).replace(",", "")
        amount = float(amount_str) if "." in amount_str else int(amount_str)
        return f"{_num2words_safe(amount, lang_code)} percent"

    text = PERCENTAGE_PATTERN.sub(_expand_percentage, text)

    # Pass 3: generic numbers
    def _expand_number(match: re.Match) -> str:
        raw: str = match.group(0)
        cleaned = raw.replace(",", "")

        # Skip long digit-only sequences (phone numbers, IDs)
        digits_only = cleaned.lstrip("-").replace(".", "", 1)
        if len(digits_only) >= 7 and "," not in raw:
            return raw

        value = float(cleaned) if "." in cleaned else int(cleaned)
        return _num2words_safe(value, lang_code)

    text = NUMBER_PATTERN.sub(_expand_number, text)

    return text


# URL normalization


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
    language: str = "English",
    url_strategy: Union[str, URLReplacementStrategy] = URLReplacementStrategy.GENERIC,
    custom_url_replacement: Optional[Union[str, Callable[[str], str]]] = None,
    link_text: str = "a link",
) -> str:
    """
    Normalize text for natural speech synthesis.

    This is the main entry point for text normalization.  It chains
    multiple normalizers in an order chosen to avoid interference:

    1. **Emails** — must run before URLs so the URL regex does not
       consume the domain part of email addresses
    2. **URLs** — after emails, so remaining bare URLs are handled
    3. **Currency** — before generic numbers so "$42" keeps its
       currency context
    4. **Dates** — before generic numbers so "2026-02-23" isn't split
    5. **Numbers** — generic number-to-words, runs after currency and
       dates have consumed their numeric patterns
    6. **Abbreviations** — last, purely textual substitution

    Args:
        text: Input text to normalize
        language: Language for number/date words (default: English).
            Used to select the correct ``num2words`` locale.
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
        >>> text2 = "Dr. Chen published on 2026-02-23 for £42."
        >>> normalize_text_for_speech(text2)
        'Doctor Chen published on February twenty-third, twenty twenty-six for forty-two pounds.'
    """
    # 1. Emails — must run BEFORE URLs so the URL regex does not
    #    consume the domain part of email addresses (e.g. example.com
    #    inside info@example.com).
    text = normalize_emails(text)

    # 2. URLs
    text = normalize_urls(
        text,
        strategy=url_strategy,
        custom_replacement=custom_url_replacement,
        link_text=link_text,
    )

    # 3. Currency (before generic number expansion)
    text = normalize_currency(text, language)

    # 4. Dates — ISO, written, and standalone years (before generic numbers)
    text = normalize_dates(text, language)

    # 5. Numbers — ordinals, percentages, integers, decimals
    text = normalize_numbers(text, language)

    # 6. Abbreviations
    text = normalize_abbreviations(text)

    return text
