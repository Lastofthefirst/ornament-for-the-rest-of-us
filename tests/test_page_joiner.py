"""Tests for page_joiner module."""

import pytest
from bookpipeline.page_joiner import (
    PageJoiner,
    JoinDecision,
    PageBoundary,
    join_pages_simple,
)


class TestPageJoinerBasic:
    """Basic tests for page joining."""

    def test_empty_pages(self):
        """Empty list should return empty string."""
        joiner = PageJoiner()
        result, boundaries = joiner.join_pages([])
        assert result == ""
        assert boundaries == []

    def test_single_page(self):
        """Single page should be returned as-is."""
        joiner = PageJoiner()
        result, boundaries = joiner.join_pages(["Page one content."])
        assert result == "Page one content."
        assert boundaries == []

    def test_two_pages_simple(self):
        """Two pages should be joined with boundary info."""
        joiner = PageJoiner(add_page_markers=False)
        pages = [
            "First page content.",
            "Second page content."
        ]
        result, boundaries = joiner.join_pages(pages)
        assert "First page content." in result
        assert "Second page content." in result
        assert len(boundaries) == 1


class TestSentenceContinuation:
    """Tests for detecting sentence continuation across pages."""

    def test_lowercase_continuation(self):
        """Lowercase start should indicate continuation."""
        joiner = PageJoiner(add_page_markers=False)
        pages = [
            "The quick brown fox",
            "jumped over the lazy dog."
        ]
        result, boundaries = joiner.join_pages(pages)
        assert boundaries[0].decision == JoinDecision.DIRECT
        # Should be joined with single space
        assert "fox jumped" in result

    def test_uppercase_after_sentence(self):
        """Uppercase start after sentence should be new paragraph."""
        joiner = PageJoiner(add_page_markers=False)
        pages = [
            "First sentence ends here.",
            "Second sentence starts fresh."
        ]
        result, boundaries = joiner.join_pages(pages)
        assert boundaries[0].decision == JoinDecision.PARAGRAPH

    def test_lowercase_after_sentence(self):
        """Lowercase after complete sentence suggests continuation."""
        joiner = PageJoiner(add_page_markers=False)
        pages = [
            "The sentence ends here.",
            "but the thought continues."
        ]
        result, boundaries = joiner.join_pages(pages)
        # This is a special case - direct join despite sentence end
        assert boundaries[0].decision == JoinDecision.DIRECT


class TestSectionDetection:
    """Tests for detecting new sections/chapters."""

    def test_chapter_marker(self):
        """Chapter heading should trigger section break."""
        joiner = PageJoiner(add_page_markers=False)
        pages = [
            "End of previous chapter.",
            "Chapter 2\n\nNew chapter begins."
        ]
        result, boundaries = joiner.join_pages(pages)
        assert boundaries[0].decision == JoinDecision.SECTION

    def test_markdown_header(self):
        """Markdown headers should trigger section break."""
        joiner = PageJoiner(add_page_markers=False)
        pages = [
            "End of section.",
            "# New Section\n\nContent here."
        ]
        result, boundaries = joiner.join_pages(pages)
        assert boundaries[0].decision == JoinDecision.SECTION

    def test_numbered_section(self):
        """Numbered sections should be detected."""
        joiner = PageJoiner(add_page_markers=False)
        pages = [
            "Previous section.",
            "1. Introduction\n\nContent here."
        ]
        result, boundaries = joiner.join_pages(pages)
        assert boundaries[0].decision == JoinDecision.SECTION


class TestHyphenatedWords:
    """Tests for hyphenated word detection."""

    def test_hyphenated_word_at_page_end(self):
        """Hyphenated word break should trigger direct join."""
        joiner = PageJoiner(add_page_markers=False)
        pages = [
            "This is a word break with hyph-",
            "enated text continuing."
        ]
        result, boundaries = joiner.join_pages(pages)
        assert boundaries[0].decision == JoinDecision.DIRECT


class TestPageMarkers:
    """Tests for page marker insertion."""

    def test_markers_enabled(self):
        """Page markers should be inserted when enabled."""
        joiner = PageJoiner(add_page_markers=True)
        pages = ["First page.", "Second page."]
        result, _ = joiner.join_pages(pages)
        assert "<!-- page 1 -->" in result

    def test_markers_disabled(self):
        """Page markers should not appear when disabled."""
        joiner = PageJoiner(add_page_markers=False)
        pages = ["First page.", "Second page."]
        result, _ = joiner.join_pages(pages)
        assert "<!-- page" not in result


class TestEmptyPages:
    """Tests for handling empty pages."""

    def test_empty_page_in_middle(self):
        """Empty pages should be handled gracefully."""
        joiner = PageJoiner(add_page_markers=False)
        pages = ["First page.", "", "Third page."]
        result, boundaries = joiner.join_pages(pages)
        assert "First page." in result
        assert "Third page." in result

    def test_whitespace_only_page(self):
        """Whitespace-only pages should be treated as empty."""
        joiner = PageJoiner(add_page_markers=False)
        pages = ["First page.", "   \n\n   ", "Third page."]
        result, boundaries = joiner.join_pages(pages)
        assert "First page." in result
        assert "Third page." in result


class TestAbbreviations:
    """Tests for handling abbreviations."""

    def test_mr_abbreviation(self):
        """Mr. should not end a sentence."""
        joiner = PageJoiner()
        # The abbreviation logic is internal, but affects joining decisions
        assert not joiner._ends_with_complete_sentence("said Mr.")

    def test_dr_abbreviation(self):
        """Dr. should not end a sentence."""
        joiner = PageJoiner()
        assert not joiner._ends_with_complete_sentence("According to Dr.")

    def test_etc_abbreviation(self):
        """etc. should not end a sentence (in context)."""
        joiner = PageJoiner()
        assert not joiner._ends_with_complete_sentence("apples, oranges, etc.")


class TestConvenienceFunction:
    """Tests for the simple join function."""

    def test_join_pages_simple(self):
        """Simple function should work without markers."""
        pages = ["First.", "Second."]
        result = join_pages_simple(pages)
        assert "First." in result
        assert "Second." in result
        assert "<!-- page" not in result


class TestBoundaryInfo:
    """Tests for boundary information."""

    def test_boundary_contains_page_numbers(self):
        """Boundaries should track page numbers."""
        joiner = PageJoiner()
        pages = ["Page 0.", "Page 1.", "Page 2."]
        _, boundaries = joiner.join_pages(pages)

        assert len(boundaries) == 2
        assert boundaries[0].page_before == 0
        assert boundaries[0].page_after == 1
        assert boundaries[1].page_before == 1
        assert boundaries[1].page_after == 2

    def test_boundary_contains_reason(self):
        """Boundaries should explain the decision."""
        joiner = PageJoiner()
        pages = ["First sentence.", "second word"]
        _, boundaries = joiner.join_pages(pages)

        assert boundaries[0].reason is not None
        assert len(boundaries[0].reason) > 0
