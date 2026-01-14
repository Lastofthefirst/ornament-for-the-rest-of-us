"""Tests for text_cleaner module."""

import pytest
from bookpipeline.text_cleaner import (
    dehyphenate_word,
    clean_text_block,
    process_page_json,
    detect_repeated_headers_footers,
    remove_repeated_headers_footers,
    clean_book_pages,
    get_dictionary,
)


class TestDehyphenation:
    """Tests for word dehyphenation."""

    def test_dehyphenate_valid_word(self):
        """Words that exist in dictionary should be joined."""
        # "currently" should be a valid word
        result = dehyphenate_word("cur", "rently")
        assert result == "currently"

    def test_dehyphenate_compound_word(self):
        """Compound words should keep the hyphen."""
        # "self-aware" is a compound, not a broken word
        result = dehyphenate_word("self", "aware")
        assert result == "self-aware"

    def test_dehyphenate_unknown_word(self):
        """Unknown combinations should keep the hyphen."""
        result = dehyphenate_word("xyz", "abc")
        assert result == "xyz-abc"


class TestCleanTextBlock:
    """Tests for text block cleaning."""

    def test_clean_simple_text(self):
        """Simple text should remain unchanged."""
        text = "This is a simple sentence."
        result = clean_text_block(text)
        assert result == "This is a simple sentence."

    def test_unwrap_single_newlines(self):
        """Single newlines within paragraphs should become spaces."""
        text = "This is a line\nthat continues here."
        result = clean_text_block(text)
        assert result == "This is a line that continues here."

    def test_dehyphenate_broken_word(self):
        """Hyphenated line breaks should be dehyphenated."""
        text = "The quick brown fox cur-\nrently lives here."
        result = clean_text_block(text)
        assert "currently" in result

    def test_clean_double_spaces(self):
        """Double spaces should be collapsed."""
        text = "Too  many   spaces here."
        result = clean_text_block(text)
        assert "  " not in result

    def test_empty_text(self):
        """Empty text should return empty."""
        assert clean_text_block("") == ""
        assert clean_text_block(None) is None


class TestProcessPageJson:
    """Tests for JSON page processing."""

    def test_process_text_blocks(self):
        """Text blocks should be extracted and joined."""
        json_data = [
            {"category": "Text", "text": "First paragraph."},
            {"category": "Text", "text": "Second paragraph."},
        ]
        result = process_page_json(json_data)
        assert "First paragraph." in result
        assert "Second paragraph." in result
        assert "\n\n" in result  # Paragraphs separated

    def test_exclude_headers(self):
        """Page headers should be excluded."""
        json_data = [
            {"category": "Page-header", "text": "CHAPTER ONE"},
            {"category": "Text", "text": "The content."},
        ]
        result = process_page_json(json_data)
        assert "CHAPTER ONE" not in result
        assert "The content." in result

    def test_exclude_footers(self):
        """Page footers should be excluded."""
        json_data = [
            {"category": "Text", "text": "The content."},
            {"category": "Page-footer", "text": "Footer text"},
        ]
        result = process_page_json(json_data)
        assert "Footer text" not in result
        assert "The content." in result

    def test_exclude_page_numbers(self):
        """Page numbers should be excluded."""
        json_data = [
            {"category": "Text", "text": "The content."},
            {"category": "Page-number", "text": "42"},
        ]
        result = process_page_json(json_data)
        assert "42" not in result

    def test_empty_json(self):
        """Empty JSON should return empty string."""
        assert process_page_json([]) == ""
        assert process_page_json(None) == ""


class TestRepeatedHeaderDetection:
    """Tests for repeated header/footer detection."""

    def test_detect_repeated_header(self):
        """Headers repeated many times should be detected."""
        pages = ["THE BOOK TITLE\n\nContent here."] * 15
        headers, footers = detect_repeated_headers_footers(pages, min_occurrences=10)
        assert "THE BOOK TITLE" in headers

    def test_detect_repeated_footer(self):
        """Footers repeated many times should be detected."""
        pages = ["Content here.\n\nPublished 2024"] * 15
        headers, footers = detect_repeated_headers_footers(pages, min_occurrences=10)
        assert "Published 2024" in footers

    def test_ignore_infrequent_phrases(self):
        """Phrases appearing less than threshold should be ignored."""
        pages = ["Rare header\n\nContent."] * 5
        headers, footers = detect_repeated_headers_footers(pages, min_occurrences=10)
        assert "Rare header" not in headers

    def test_ignore_long_lines(self):
        """Long lines (>80 chars) should not be detected as headers."""
        long_header = "A" * 100
        pages = [f"{long_header}\n\nContent."] * 15
        headers, footers = detect_repeated_headers_footers(pages, min_occurrences=10)
        assert long_header not in headers


class TestRemoveRepeatedHeadersFooters:
    """Tests for header/footer removal."""

    def test_remove_header(self):
        """Detected headers should be removed from page starts."""
        pages = ["HEADER\n\nContent here."]
        headers = {"HEADER"}
        result = remove_repeated_headers_footers(pages, headers, set())
        assert "HEADER" not in result[0]
        assert "Content here." in result[0]

    def test_remove_footer(self):
        """Detected footers should be removed from page ends."""
        pages = ["Content here.\n\nFOOTER"]
        footers = {"FOOTER"}
        result = remove_repeated_headers_footers(pages, set(), footers)
        assert "FOOTER" not in result[0]
        assert "Content here." in result[0]

    def test_no_headers_footers(self):
        """When no headers/footers, pages should be unchanged."""
        pages = ["Content here."]
        result = remove_repeated_headers_footers(pages, set(), set())
        assert result == pages


class TestCleanBookPages:
    """Tests for full book page cleaning."""

    def test_clean_pages_removes_repeated_headers(self):
        """Full cleaning should remove repeated headers."""
        # Each page has the same header but different content
        pages = [f"BOOK TITLE\n\nPage {i} content with unique text." for i in range(15)]
        result = clean_book_pages(pages, min_header_occurrences=10)
        # All pages should have header removed but content preserved
        for i, page in enumerate(result):
            assert "BOOK TITLE" not in page
            assert f"Page {i} content" in page

    def test_clean_pages_preserves_unique_content(self):
        """Unique first lines should not be removed."""
        pages = [
            "Unique intro\n\nContent one.",
            "Different intro\n\nContent two.",
        ]
        result = clean_book_pages(pages, min_header_occurrences=10)
        assert "Unique intro" in result[0]
        assert "Different intro" in result[1]


class TestDictionaryLoading:
    """Tests for dictionary singleton."""

    def test_dictionary_loads(self):
        """Dictionary should load without error."""
        dictionary = get_dictionary()
        assert dictionary is not None

    def test_dictionary_has_common_words(self):
        """Dictionary should contain common English words."""
        dictionary = get_dictionary()
        # These should be in any English dictionary
        assert dictionary.is_word("the")
        assert dictionary.is_word("hello")
        assert dictionary.is_word("world")

    def test_dictionary_rejects_nonsense(self):
        """Dictionary should reject nonsense words."""
        dictionary = get_dictionary()
        assert not dictionary.is_word("xyzabc123")

    def test_dictionary_case_insensitive(self):
        """Dictionary lookup should be case-insensitive."""
        dictionary = get_dictionary()
        assert dictionary.is_word("Hello")
        assert dictionary.is_word("HELLO")
        assert dictionary.is_word("hello")
