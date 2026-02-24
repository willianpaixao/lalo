"""
Tests for text normalization module.
"""

import pytest

from lalo.text_normalization import (
    URL_PATTERN,
    URLReplacementStrategy,
    _extract_domain,
    _replace_url_custom,
    _replace_url_domain,
    _replace_url_generic,
    _replace_url_remove,
    normalize_abbreviations,
    normalize_currency,
    normalize_dates,
    normalize_emails,
    normalize_numbers,
    normalize_text_for_speech,
    normalize_urls,
)


class TestURLReplacementStrategy:
    """Tests for URLReplacementStrategy enum."""

    def test_enum_values(self):
        """Test that all expected strategies exist."""
        assert URLReplacementStrategy.GENERIC.value == "generic"
        assert URLReplacementStrategy.DOMAIN.value == "domain"
        assert URLReplacementStrategy.REMOVE.value == "remove"
        assert URLReplacementStrategy.CUSTOM.value == "custom"

    def test_string_to_enum_conversion(self):
        """Test converting string to enum."""
        assert URLReplacementStrategy("generic") == URLReplacementStrategy.GENERIC
        assert URLReplacementStrategy("domain") == URLReplacementStrategy.DOMAIN
        assert URLReplacementStrategy("remove") == URLReplacementStrategy.REMOVE
        assert URLReplacementStrategy("custom") == URLReplacementStrategy.CUSTOM

    def test_invalid_strategy_raises_error(self):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError):
            URLReplacementStrategy("invalid")


class TestExtractDomain:
    """Tests for domain extraction helper."""

    def test_extract_from_full_url(self):
        """Test extracting domain from full URL."""
        assert _extract_domain("https://example.com") == "example dot com"
        assert _extract_domain("http://github.com") == "github dot com"

    def test_extract_from_www_url(self):
        """Test extracting domain from www URL."""
        assert _extract_domain("www.example.com") == "example dot com"
        assert _extract_domain("https://www.github.com") == "github dot com"

    def test_extract_with_port(self):
        """Test extracting domain with port number."""
        assert _extract_domain("http://example.com:8080") == "example dot com"
        assert _extract_domain("https://localhost:3000") == "localhost"

    def test_extract_with_path(self):
        """Test extracting domain with path."""
        assert _extract_domain("https://example.com/path/to/page") == "example dot com"
        assert _extract_domain("www.github.com/user/repo") == "github dot com"

    def test_extract_with_query_params(self):
        """Test extracting domain with query parameters."""
        assert _extract_domain("https://example.com?q=test") == "example dot com"
        assert _extract_domain("www.example.com?page=1&id=2") == "example dot com"

    def test_extract_subdomain(self):
        """Test extracting domain with subdomain."""
        assert _extract_domain("https://api.github.com") == "api dot github dot com"
        assert _extract_domain("docs.python.org") == "docs dot python dot org"


class TestReplaceUrlHelpers:
    """Tests for URL replacement helper functions."""

    def test_replace_url_generic(self):
        """Test generic URL replacement."""
        assert _replace_url_generic("https://example.com", "a link") == "a link"
        assert _replace_url_generic("http://test.com", "the website") == "the website"

    def test_replace_url_domain(self):
        """Test domain-based replacement."""
        assert _replace_url_domain("https://example.com") == "example dot com"
        assert _replace_url_domain("www.github.com") == "github dot com"

    def test_replace_url_remove(self):
        """Test URL removal."""
        assert _replace_url_remove("https://example.com") == ""
        assert _replace_url_remove("any-url") == ""

    def test_replace_url_custom_with_string(self):
        """Test custom replacement with string."""
        assert _replace_url_custom("https://example.com", "custom text") == "custom text"

    def test_replace_url_custom_with_callable(self):
        """Test custom replacement with callable."""

        def handler(url):
            return "github repo" if "github" in url else "a link"

        assert _replace_url_custom("https://github.com", handler) == "github repo"
        assert _replace_url_custom("https://example.com", handler) == "a link"

    def test_replace_url_custom_invalid_type(self):
        """Test custom replacement with invalid type raises error."""
        with pytest.raises(TypeError, match="must be str or callable"):
            _replace_url_custom("https://example.com", 123)


class TestNormalizeUrls:
    """Tests for URL normalization with different strategies."""

    # GENERIC strategy tests
    def test_generic_strategy_basic(self):
        """Test basic GENERIC strategy."""
        text = "Visit https://example.com for info"
        result = normalize_urls(text)
        assert result == "Visit a link for info"

    def test_generic_strategy_custom_link_text(self):
        """Test GENERIC strategy with custom link text."""
        text = "Check https://docs.python.org for details"
        result = normalize_urls(text, link_text="the documentation")
        assert result == "Check the documentation for details"

    def test_generic_multiple_urls(self):
        """Test GENERIC strategy with multiple URLs."""
        text = "See https://a.com and https://b.com for info"
        result = normalize_urls(text)
        assert result == "See a link and a link for info"

    def test_generic_with_www(self):
        """Test GENERIC strategy with www URLs."""
        text = "Visit www.example.com for more"
        result = normalize_urls(text)
        assert result == "Visit a link for more"

    # DOMAIN strategy tests
    def test_domain_strategy_basic(self):
        """Test basic DOMAIN strategy."""
        text = "Check https://github.com for code"
        result = normalize_urls(text, strategy=URLReplacementStrategy.DOMAIN)
        assert result == "Check github dot com for code"

    def test_domain_strategy_preserves_context(self):
        """Test DOMAIN strategy preserves surrounding text."""
        text = "The repository is at https://github.com/user/repo and contains code"
        result = normalize_urls(text, strategy="domain")
        assert result == "The repository is at github dot com and contains code"

    def test_domain_strategy_multiple_urls(self):
        """Test DOMAIN strategy with multiple URLs."""
        text = "See https://example.com and www.github.com"
        result = normalize_urls(text, strategy=URLReplacementStrategy.DOMAIN)
        assert result == "See example dot com and github dot com"

    # REMOVE strategy tests
    def test_remove_strategy_basic(self):
        """Test basic REMOVE strategy."""
        text = "Visit https://example.com for information"
        result = normalize_urls(text, strategy=URLReplacementStrategy.REMOVE)
        assert result == "Visit for information"

    def test_remove_strategy_cleans_extra_spaces(self):
        """Test REMOVE strategy cleans up extra whitespace."""
        text = "Check   https://example.com   for info"
        result = normalize_urls(text, strategy="remove")
        assert result == "Check for info"

    def test_remove_strategy_multiple_urls(self):
        """Test REMOVE strategy with multiple URLs."""
        text = "Links: https://a.com, https://b.com, and https://c.com here"
        result = normalize_urls(text, strategy=URLReplacementStrategy.REMOVE)
        assert result == "Links: , , and here"

    # CUSTOM strategy tests
    def test_custom_strategy_with_string(self):
        """Test CUSTOM strategy with static string."""
        text = "See https://example.com for details"
        result = normalize_urls(
            text, strategy=URLReplacementStrategy.CUSTOM, custom_replacement="our website"
        )
        assert result == "See our website for details"

    def test_custom_strategy_with_callable(self):
        """Test CUSTOM strategy with callable."""

        def handler(url):
            if "github" in url:
                return "our GitHub repository"
            elif "docs" in url:
                return "the documentation"
            return "a link"

        text = "Visit https://github.com/user for code"
        result = normalize_urls(
            text, strategy=URLReplacementStrategy.CUSTOM, custom_replacement=handler
        )
        assert result == "Visit our GitHub repository for code"

    def test_custom_strategy_conditional_replacement(self):
        """Test CUSTOM strategy with conditional logic."""

        def conditional(url):
            return "the source code" if "github" in url else "the website"

        text1 = "See https://github.com for code"
        text2 = "See https://example.com for info"

        result1 = normalize_urls(text1, strategy="custom", custom_replacement=conditional)
        result2 = normalize_urls(text2, strategy="custom", custom_replacement=conditional)

        assert result1 == "See the source code for code"
        assert result2 == "See the website for info"

    def test_custom_strategy_without_replacement_raises_error(self):
        """Test CUSTOM strategy without replacement raises ValueError."""
        text = "Visit https://example.com"
        with pytest.raises(ValueError, match="CUSTOM strategy requires"):
            normalize_urls(text, strategy=URLReplacementStrategy.CUSTOM)

    # Strategy validation tests
    def test_invalid_strategy_string_raises_error(self):
        """Test invalid strategy string raises ValueError."""
        text = "Visit https://example.com"
        with pytest.raises(ValueError, match="Invalid strategy"):
            normalize_urls(text, strategy="invalid_strategy")

    def test_strategy_case_insensitive(self):
        """Test strategy strings are case-insensitive."""
        text = "Visit https://example.com for info"
        result1 = normalize_urls(text, strategy="GENERIC")
        result2 = normalize_urls(text, strategy="Generic")
        result3 = normalize_urls(text, strategy="generic")
        assert result1 == result2 == result3 == "Visit a link for info"


class TestNormalizeTextForSpeech:
    """Tests for main entry point."""

    def test_default_behavior(self):
        """Test default behavior uses GENERIC strategy."""
        text = "Visit https://example.com for info"
        result = normalize_text_for_speech(text)
        assert result == "Visit a link for info"

    def test_with_different_strategies(self):
        """Test main function with different strategies."""
        text = "Check https://github.com for code"

        result1 = normalize_text_for_speech(text, url_strategy="generic")
        result2 = normalize_text_for_speech(text, url_strategy=URLReplacementStrategy.DOMAIN)
        result3 = normalize_text_for_speech(text, url_strategy="remove")

        assert result1 == "Check a link for code"
        assert result2 == "Check github dot com for code"
        assert result3 == "Check for code"

    def test_preserves_non_url_content(self):
        """Test that non-URL content is preserved exactly."""
        text = "This is a test sentence without any links."
        result = normalize_text_for_speech(text)
        assert result == text

    def test_custom_link_text_parameter(self):
        """Test custom link text parameter works."""
        text = "See https://example.com for details"
        result = normalize_text_for_speech(text, link_text="the reference")
        assert result == "See the reference for details"

    def test_custom_url_replacement_parameter(self):
        """Test custom URL replacement parameter works."""

        def handler(url):
            return "the repository"

        text = "Visit https://github.com for code"
        result = normalize_text_for_speech(
            text, url_strategy="custom", custom_url_replacement=handler
        )
        assert result == "Visit the repository for code"


class TestEdgeCases:
    """Tests for complex real-world scenarios."""

    def test_multiple_urls_in_sentence(self):
        """Test handling multiple URLs in one sentence."""
        text = "Check https://a.com, https://b.com, and https://c.com for info"
        result = normalize_urls(text)
        assert result == "Check a link, a link, and a link for info"

    def test_url_at_sentence_start(self):
        """Test URL at the beginning of sentence."""
        text = "https://example.com is a great resource"
        result = normalize_urls(text)
        assert result == "a link is a great resource"

    def test_url_at_sentence_end(self):
        """Test URL at the end of sentence."""
        text = "For more info, visit https://example.com"
        result = normalize_urls(text)
        assert result == "For more info, visit a link"

    def test_url_in_parentheses(self):
        """Test URL inside parentheses."""
        text = "See documentation (https://docs.example.com) for details"
        result = normalize_urls(text)
        assert result == "See documentation (a link) for details"

    def test_url_with_query_params(self):
        """Test URL with query parameters."""
        text = "Search at https://example.com/search?q=test&lang=en for results"
        result = normalize_urls(text)
        assert result == "Search at a link for results"

    def test_url_with_fragment(self):
        """Test URL with fragment identifier."""
        text = "Jump to https://example.com/page#section for info"
        result = normalize_urls(text)
        assert result == "Jump to a link for info"

    def test_mixed_url_formats(self):
        """Test mixed URL formats in same text."""
        text = "Visit https://example.com, www.github.com, and python.org for resources"
        result = normalize_urls(text)
        assert result == "Visit a link, a link, and a link for resources"

    def test_url_with_path_and_extension(self):
        """Test URL with path and file extension."""
        text = "Download from https://example.com/files/document.pdf today"
        result = normalize_urls(text)
        assert result == "Download from a link today"

    def test_no_false_positives_on_decimals(self):
        """Test that decimal numbers are not treated as URLs."""
        text = "The value is 3.14 and the version is 2.0"
        result = normalize_urls(text)
        assert result == text  # Should remain unchanged

    def test_no_false_positives_on_version_numbers(self):
        """Test that version numbers are not treated as URLs."""
        text = "Python 3.12 and Node.js 18.0 are required"
        result = normalize_urls(text)
        assert result == text  # Should remain unchanged

    def test_empty_text(self):
        """Test handling of empty text."""
        assert normalize_urls("") == ""
        assert normalize_urls("   ") == ""

    def test_text_without_urls(self):
        """Test text without any URLs."""
        text = "This is a normal sentence with no links at all."
        result = normalize_urls(text)
        assert result == text

    def test_url_pattern_matches_common_formats(self):
        """Test that URL pattern matches common formats."""
        urls = [
            "https://example.com",
            "http://example.com",
            "www.example.com",
            "example.com",
            "github.io",
            "test.co.uk",
            "https://example.com/path",
            "https://example.com/path?query=1",
            "https://example.com:8080",
        ]

        for url in urls:
            assert URL_PATTERN.search(url) is not None, f"Failed to match: {url}"

    def test_url_consecutive_urls(self):
        """Test consecutive URLs separated by spaces."""
        text = "Links: https://a.com https://b.com https://c.com"
        result = normalize_urls(text)
        assert result == "Links: a link a link a link"

    def test_url_with_special_tld(self):
        """Test URLs with special TLDs."""
        text = "Visit site.ai and platform.io for AI tools"
        result = normalize_urls(text)
        assert result == "Visit a link and a link for AI tools"

    def test_mixed_strategies_in_real_world_text(self):
        """Test realistic ebook text with multiple URLs."""
        text = """
        Chapter 5: Resources

        For tutorials, visit https://www.coursera.org/learn/machine-learning
        or check the documentation at https://scikit-learn.org/stable/.
        Additional resources: https://arxiv.org/list/cs.LG/recent
        """

        result = normalize_urls(text, strategy="generic")
        assert "https://" not in result
        assert "www." not in result
        assert "a link" in result

        result_domain = normalize_urls(text, strategy="domain")
        assert "coursera dot org" in result_domain
        assert "scikit-learn dot org" in result_domain

        result_remove = normalize_urls(text, strategy="remove")
        assert "https://" not in result_remove
        assert "coursera" not in result_remove


class TestNormalizeEmails:
    """Tests for email normalization."""

    def test_simple_email(self):
        """Test basic email expansion."""
        result = normalize_emails("Contact info@example.com today")
        assert result == "Contact info at example dot com today"

    def test_email_with_dots_in_local(self):
        """Test email with dots in the local part."""
        result = normalize_emails("Mail john.doe@example.com")
        assert result == "Mail john dot doe at example dot com"

    def test_email_with_plus(self):
        """Test email with plus sign."""
        result = normalize_emails("Send to user+tag@example.com")
        assert result == "Send to user plus tag at example dot com"

    def test_email_with_underscore(self):
        """Test email with underscore."""
        result = normalize_emails("Write to first_last@example.com")
        assert result == "Write to first underscore last at example dot com"

    def test_email_with_dash_in_domain(self):
        """Test email with dash in domain."""
        result = normalize_emails("Mail user@company-name.com")
        assert result == "Mail user at company dash name dot com"

    def test_complex_email(self):
        """Test a complex email with multiple special chars."""
        result = normalize_emails("john.doe+careers@company-name.co.uk")
        assert result == "john dot doe plus careers at company dash name dot co dot uk"

    def test_multiple_emails(self):
        """Test text with multiple emails."""
        result = normalize_emails("Send to a@b.com or c@d.org for info")
        assert "a at b dot com" in result
        assert "c at d dot org" in result

    def test_no_emails(self):
        """Test text without emails is unchanged."""
        text = "This has no email addresses."
        assert normalize_emails(text) == text

    def test_email_not_confused_with_at_sign(self):
        """Test that bare @ signs are not matched."""
        text = "We met @ the cafe"
        assert normalize_emails(text) == text


class TestNormalizeDates:
    """Tests for ISO date normalization."""

    def test_iso_date_basic(self):
        """Test basic ISO date expansion."""
        result = normalize_dates("Published on 2026-02-23.")
        assert "February" in result
        assert "twenty-third" in result
        assert "twenty twenty-six" in result

    def test_iso_date_january_first(self):
        """Test January 1st."""
        result = normalize_dates("Date: 1984-01-01")
        assert "January" in result
        assert "first" in result
        assert "nineteen eighty-four" in result

    def test_iso_date_december(self):
        """Test December date."""
        result = normalize_dates("Due 2025-12-31")
        assert "December" in result
        assert "thirty-first" in result

    def test_iso_date_preserves_surrounding(self):
        """Test that surrounding text is preserved."""
        result = normalize_dates("Before 2026-06-15 and after")
        assert result.startswith("Before ")
        assert result.endswith(" and after")

    def test_no_dates(self):
        """Test text without dates is unchanged."""
        text = "Nothing to see here."
        assert normalize_dates(text) == text

    def test_partial_date_not_matched(self):
        """Test that incomplete ISO date patterns are not matched."""
        text = "Version 2026-13-01 is invalid"
        # Month 13 shouldn't match
        assert normalize_dates(text) == text

    def test_iso_date_german(self):
        """Test date normalization in German."""
        result = normalize_dates("Am 2026-03-15", language="German")
        assert "March" in result  # Month name stays English (it's the standard name)
        # Day should be in German ordinal
        assert "fünfzehnte" in result or "fifteenth" in result

    def test_multiple_dates(self):
        """Test text with multiple ISO dates."""
        result = normalize_dates("From 2020-01-01 to 2026-12-31")
        assert "January" in result
        assert "December" in result


class TestNormalizeCurrency:
    """Tests for currency normalization."""

    def test_gbp_simple(self):
        """Test simple GBP amount."""
        result = normalize_currency("Costs £42.")
        assert "forty-two" in result
        assert "pound" in result.lower()
        assert "£" not in result

    def test_usd_with_cents(self):
        """Test USD with cents."""
        result = normalize_currency("Price: $4.50")
        assert "$" not in result
        assert "four" in result

    def test_usd_large_amount(self):
        """Test large USD amount with commas."""
        result = normalize_currency("Budget: $1,299.99")
        assert "$" not in result
        assert "1,299" not in result

    def test_euro(self):
        """Test EUR symbol."""
        result = normalize_currency("Flight: €89")
        assert "€" not in result
        assert "eighty-nine" in result

    def test_gbp_large(self):
        """Test large GBP amount."""
        result = normalize_currency("Inherited £2,500,000")
        assert "£" not in result
        assert "2,500,000" not in result
        assert "pound" in result.lower()

    def test_no_currency(self):
        """Test text without currency is unchanged."""
        text = "The value is high."
        assert normalize_currency(text) == text

    def test_currency_symbol_without_number_not_matched(self):
        """Test that bare currency symbols are not matched."""
        text = "The $ sign is used in many contexts"
        # No number follows $ so it shouldn't match
        assert normalize_currency(text) == text

    def test_yen(self):
        """Test JPY symbol."""
        result = normalize_currency("Price: ¥100")
        assert "¥" not in result
        assert "hundred" in result

    def test_currency_german_language(self):
        """Test currency with German language."""
        result = normalize_currency("Kosten: €42", language="German")
        assert "€" not in result
        assert "42" not in result


class TestNormalizeAbbreviations:
    """Tests for abbreviation expansion."""

    # Title abbreviations
    def test_doctor(self):
        """Test Dr. expansion."""
        assert "Doctor" in normalize_abbreviations("Dr. Smith")

    def test_mister(self):
        """Test Mr. expansion."""
        assert "Mister" in normalize_abbreviations("Mr. Jones")

    def test_misses(self):
        """Test Mrs. expansion."""
        assert "Misses" in normalize_abbreviations("Mrs. Adams")

    def test_professor(self):
        """Test Prof. expansion."""
        assert "Professor" in normalize_abbreviations("Prof. Johnson")

    def test_multiple_titles(self):
        """Test multiple title abbreviations in one text."""
        result = normalize_abbreviations("Dr. Smith and Prof. Jones discussed with Mr. Adams.")
        assert "Doctor" in result
        assert "Professor" in result
        assert "Mister" in result

    # Latin abbreviations
    def test_eg(self):
        """Test e.g. expansion."""
        result = normalize_abbreviations("Several species, e.g. wolves")
        assert "for example" in result

    def test_ie(self):
        """Test i.e. expansion."""
        result = normalize_abbreviations("large predators, i.e. bears")
        assert "that is" in result

    def test_etc(self):
        """Test etc. expansion."""
        result = normalize_abbreviations("wolves, bears, etc.")
        assert "et cetera" in result

    def test_vs(self):
        """Test vs. expansion."""
        result = normalize_abbreviations("cats vs. dogs")
        assert "versus" in result

    # St. ambiguity
    def test_st_as_saint(self):
        """Test St. as Saint when not preceded by number."""
        result = normalize_abbreviations("the church of St. Patrick")
        assert "Saint" in result

    def test_st_as_street(self):
        """Test St. as Street when preceded by a number."""
        result = normalize_abbreviations("They met at 5th St.")
        assert "Street" in result

    def test_st_mixed(self):
        """Test mixed St. usage in same text."""
        result = normalize_abbreviations("5th St. near St. Patrick")
        assert "Street" in result
        assert "Saint" in result

    # General abbreviations
    def test_approx(self):
        """Test approx. expansion."""
        result = normalize_abbreviations("approx. 450 units")
        assert "approximately" in result

    def test_no_abbreviations(self):
        """Test text without abbreviations is unchanged."""
        text = "Nothing to expand here."
        assert normalize_abbreviations(text) == text

    def test_case_insensitive_titles(self):
        """Test that title abbreviations are case-insensitive."""
        result = normalize_abbreviations("dr. smith")
        # Should still expand even if lowercase
        assert "octor" in result.lower()


class TestNormalizeDatesWritten:
    """Tests for written and standalone year date normalization."""

    def test_written_mdy(self):
        """Test 'Month DD, YYYY' format."""
        result = normalize_dates("On July 4, 1776, it was signed.")
        assert "July" in result
        assert "fourth" in result
        assert "seventeen seventy-six" in result
        assert "1776" not in result

    def test_written_mdy_no_comma(self):
        """Test 'Month DD YYYY' without comma."""
        result = normalize_dates("Date: February 23 2026")
        assert "February" in result
        assert "twenty-third" in result
        assert "twenty twenty-six" in result

    def test_written_dmy(self):
        """Test 'DD Month YYYY' format."""
        result = normalize_dates("On 23 February 2026 we met.")
        assert "twenty-third" in result
        assert "February" in result
        assert "twenty twenty-six" in result
        assert "of" in result  # "twenty-third of February"

    def test_standalone_year_in(self):
        """Test standalone year with 'in'."""
        result = normalize_dates("Written in 1984 by Orwell.")
        assert "nineteen eighty-four" in result
        assert "1984" not in result

    def test_standalone_year_since(self):
        """Test standalone year with 'since'."""
        result = normalize_dates("Active since 2020.")
        assert "twenty twenty" in result
        assert "2020" not in result

    def test_standalone_year_from_until(self):
        """Test standalone year with 'from' and 'until'."""
        result = normalize_dates("From 1776 until 1800.")
        assert "seventeen seventy-six" in result
        assert "eighteen hundred" in result

    def test_standalone_year_case_insensitive(self):
        """Test that preposition matching is case-insensitive."""
        result = normalize_dates("Since 1999 things changed.")
        assert "1999" not in result

    def test_year_without_preposition_not_matched(self):
        """Test that bare years without prepositions are not matched."""
        text = "The code 1984 is a reference."
        result = normalize_dates(text)
        assert "1984" in result  # should remain as-is

    def test_written_date_german(self):
        """Test written date with German language."""
        result = normalize_dates("On July 4, 1776, signed.", language="German")
        assert "July" in result
        assert "1776" not in result

    def test_multiple_written_dates(self):
        """Test text with multiple written dates."""
        result = normalize_dates("Born January 1, 1900 and died December 31, 1999.")
        assert "January" in result
        assert "December" in result
        assert "1900" not in result
        assert "1999" not in result

    def test_written_date_single_digit_day(self):
        """Test written date with single-digit day."""
        result = normalize_dates("March 5, 2020")
        assert "fifth" in result
        assert "twenty twenty" in result


class TestNormalizeNumbers:
    """Tests for number-to-words normalization."""

    # Ordinals
    def test_ordinal_1st(self):
        """Test 1st ordinal."""
        result = normalize_numbers("She came in 1st place.")
        assert "first" in result
        assert "1st" not in result

    def test_ordinal_23rd(self):
        """Test 23rd ordinal."""
        result = normalize_numbers("He finished 23rd.")
        assert "twenty-third" in result

    def test_ordinal_2nd(self):
        """Test 2nd ordinal."""
        result = normalize_numbers("The 2nd attempt.")
        assert "second" in result

    # Percentages
    def test_percentage_integer(self):
        """Test integer percentage."""
        result = normalize_numbers("Prices rose 42%.")
        assert "forty-two percent" in result
        assert "42%" not in result

    def test_percentage_decimal(self):
        """Test decimal percentage."""
        result = normalize_numbers("Growth of 3.5%.")
        assert "percent" in result
        assert "3.5%" not in result

    # Integers
    def test_simple_integer(self):
        """Test simple integer."""
        result = normalize_numbers("There are 42 items.")
        assert "forty-two" in result
        assert "42" not in result

    def test_large_integer_with_commas(self):
        """Test large number with comma separators."""
        result = normalize_numbers("Population: 1,234,567 people.")
        assert "1,234,567" not in result
        assert "million" in result

    def test_negative_number(self):
        """Test negative number."""
        result = normalize_numbers("Temperature: -15 degrees.")
        assert "minus" in result
        assert "fifteen" in result

    # Decimals
    def test_decimal(self):
        """Test decimal number."""
        result = normalize_numbers("The value is 3.14 approximately.")
        assert "three" in result
        assert "3.14" not in result

    # Edge cases
    def test_no_numbers(self):
        """Test text without numbers is unchanged."""
        text = "Nothing to convert here."
        assert normalize_numbers(text) == text

    def test_phone_number_skipped(self):
        """Test that long digit sequences are left alone."""
        text = "Call 5551234567 for info."
        result = normalize_numbers(text)
        assert "5551234567" in result

    def test_time_not_matched(self):
        """Test that time-like patterns are not matched."""
        text = "The meeting is at 14:30."
        result = normalize_numbers(text)
        assert "14" in result  # should remain part of the time

    def test_number_german(self):
        """Test number normalization in German."""
        result = normalize_numbers("Es gibt 42 Artikel.", language="German")
        assert "42" not in result
        assert "zweiundvierzig" in result

    def test_single_digit(self):
        """Test single digit."""
        result = normalize_numbers("There are 5 apples.")
        assert "five" in result

    def test_zero(self):
        """Test zero."""
        result = normalize_numbers("Score: 0 points.")
        assert "zero" in result


class TestNormalizePipeline:
    """Tests for the full normalize_text_for_speech pipeline with new normalizers."""

    def test_pipeline_preserves_plain_text(self):
        """Plain text without any patterns should pass through unchanged."""
        text = "The quick brown fox jumps over the lazy dog."
        assert normalize_text_for_speech(text) == text

    def test_pipeline_handles_email(self):
        """Pipeline should normalize emails."""
        result = normalize_text_for_speech("Contact info@example.com")
        assert "info at example dot com" in result

    def test_pipeline_handles_date(self):
        """Pipeline should normalize ISO dates."""
        result = normalize_text_for_speech("Published 2026-02-23")
        assert "February" in result
        assert "2026-02-23" not in result

    def test_pipeline_handles_currency(self):
        """Pipeline should normalize currency."""
        result = normalize_text_for_speech("Costs £42")
        assert "£" not in result

    def test_pipeline_handles_abbreviations(self):
        """Pipeline should normalize abbreviations."""
        result = normalize_text_for_speech("Dr. Smith said")
        assert "Doctor" in result

    def test_pipeline_mixed_content(self):
        """Test pipeline with text containing multiple pattern types."""
        text = (
            "On 2024-03-15, Dr. Chen published for £42. Contact research@university.edu for info."
        )
        result = normalize_text_for_speech(text)
        assert "2024-03-15" not in result
        assert "March" in result
        assert "Doctor" in result
        assert "£" not in result
        assert "research at university dot edu" in result

    def test_pipeline_with_language(self):
        """Test pipeline respects language parameter."""
        result = normalize_text_for_speech("Costs £42", language="German")
        assert "£" not in result

    def test_pipeline_url_and_email_coexist(self):
        """Test that URLs and emails are both normalized."""
        text = "Visit https://example.com or mail info@example.com"
        result = normalize_text_for_speech(text)
        assert "a link" in result
        assert "info at example dot com" in result

    def test_pipeline_handles_numbers(self):
        """Pipeline should normalize numbers."""
        result = normalize_text_for_speech("There are 42 items.")
        assert "forty-two" in result
        assert "42" not in result

    def test_pipeline_handles_written_date(self):
        """Pipeline should normalize written dates."""
        result = normalize_text_for_speech("On July 4, 1776, signed.")
        assert "fourth" in result
        assert "1776" not in result

    def test_pipeline_handles_standalone_year(self):
        """Pipeline should normalize standalone years."""
        result = normalize_text_for_speech("Written in 1984 by Orwell.")
        assert "nineteen eighty-four" in result

    def test_pipeline_currency_before_numbers(self):
        """Currency should be expanded before generic numbers run."""
        result = normalize_text_for_speech("It costs $42.")
        assert "dollar" in result.lower()
        assert "$" not in result

    def test_pipeline_backward_compatible(self):
        """Test that existing URL normalization still works."""
        text = "Visit https://example.com for info"
        result = normalize_text_for_speech(text)
        assert result == "Visit a link for info"
