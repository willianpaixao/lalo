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
