"""
Intelligent page joining with proper sentence boundary detection.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class JoinDecision(Enum):
    """How to join two consecutive pages."""

    DIRECT = "direct"  # Join with single space (sentence continues)
    PARAGRAPH = "paragraph"  # Join with double newline (new paragraph)
    SECTION = "section"  # Keep page break marker (new section/chapter)


@dataclass
class PageBoundary:
    """Information about a page boundary."""

    page_before: int
    page_after: int
    decision: JoinDecision
    reason: str


class PageJoiner:
    """Intelligently joins OCR pages handling broken sentences."""

    # Sentence-ending punctuation (including CJK)
    SENTENCE_END = re.compile(r'[.!?;:]["\')\]]*\s*$|[。！？；：」』）】]\s*$')

    # Likely sentence continuation indicators
    CONTINUATION_START = re.compile(r'^\s*[a-z]')  # Starts with lowercase
    CONTINUATION_CHARS = re.compile(r'^[,;:\-–—]')  # Starts with continuation punctuation

    # Structural markers that indicate new sections
    SECTION_MARKERS = re.compile(
        r'^\s*(?:'
        r'chapter\s*\d*|'  # Chapter X
        r'part\s*\d*|'  # Part X
        r'section\s*\d*|'  # Section X
        r'#{1,6}\s+|'  # Markdown headers
        r'\d+\.\s+[A-Z]|'  # Numbered sections like "1. Introduction"
        r'[IVXLCDM]+\.\s*[A-Z]'  # Roman numeral sections
        r')',
        re.IGNORECASE
    )

    # Paragraph indicators at end of page
    PARAGRAPH_END = re.compile(
        r'(?:'
        r'[.!?]["\')\]]*\s*$|'  # Sentence ends
        r':\s*$|'  # Ends with colon (before list/quote)
        r'[。！？」』）】]\s*$'  # CJK sentence ends
        r')'
    )

    # Common abbreviations that end with period but don't end sentences
    ABBREVIATIONS = {
        'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'vs', 'etc', 'al',
        'inc', 'ltd', 'corp', 'eg', 'ie', 'viz', 'cf', 'fig', 'no', 'vol',
        'pp', 'ed', 'eds', 'trans', 'rev', 'st', 'ave', 'blvd',
    }

    def __init__(self, add_page_markers: bool = True) -> None:
        """Initialize page joiner.

        Args:
            add_page_markers: Whether to add HTML comment markers at page boundaries
        """
        self.add_page_markers = add_page_markers

    def join_pages(self, pages: list[str]) -> tuple[str, list[PageBoundary]]:
        """Join multiple pages into a single document.

        Args:
            pages: List of page texts in order

        Returns:
            Tuple of (joined text, list of boundary decisions)
        """
        if not pages:
            return "", []

        if len(pages) == 1:
            return pages[0], []

        boundaries: list[PageBoundary] = []
        result_parts: list[str] = []

        # Start with first page
        result_parts.append(pages[0].strip())

        for i in range(1, len(pages)):
            prev_text = pages[i - 1].strip()
            curr_text = pages[i].strip()

            # Handle empty pages
            if not prev_text or not curr_text:
                boundaries.append(PageBoundary(
                    page_before=i - 1,
                    page_after=i,
                    decision=JoinDecision.PARAGRAPH,
                    reason="Empty page"
                ))
                if curr_text:
                    result_parts.append(self._make_separator(i, JoinDecision.PARAGRAPH))
                    result_parts.append(curr_text)
                continue

            # Analyze the boundary
            decision, reason = self._analyze_boundary(prev_text, curr_text)

            boundaries.append(PageBoundary(
                page_before=i - 1,
                page_after=i,
                decision=decision,
                reason=reason
            ))

            # Add appropriate separator
            result_parts.append(self._make_separator(i, decision))
            result_parts.append(curr_text)

        joined = "".join(result_parts)

        # Log summary
        direct_joins = sum(1 for b in boundaries if b.decision == JoinDecision.DIRECT)
        logger.info(
            f"Joined {len(pages)} pages: "
            f"{direct_joins} sentence continuations, "
            f"{len(boundaries) - direct_joins} paragraph/section breaks"
        )

        return joined, boundaries

    def _analyze_boundary(self, prev_text: str, curr_text: str) -> tuple[JoinDecision, str]:
        """Analyze how two pages should be joined.

        Args:
            prev_text: Text of previous page
            curr_text: Text of current page

        Returns:
            Tuple of (JoinDecision, reason string)
        """
        # Check if current page starts a new section
        if self.SECTION_MARKERS.match(curr_text):
            return JoinDecision.SECTION, "New section/chapter detected"

        # Get the end of previous page for analysis
        prev_end = prev_text[-200:] if len(prev_text) > 200 else prev_text
        curr_start = curr_text[:100] if len(curr_text) > 100 else curr_text

        # Check if previous page ends with clear sentence boundary
        if self._ends_with_complete_sentence(prev_end):
            # But check if next page continues (lowercase start usually means continuation)
            if self._starts_as_continuation(curr_start):
                return JoinDecision.DIRECT, "Lowercase continuation after sentence"
            return JoinDecision.PARAGRAPH, "Complete sentence at page boundary"

        # Previous page ends mid-sentence
        if self._starts_as_continuation(curr_start):
            return JoinDecision.DIRECT, "Sentence continues across page"

        # Ambiguous case - check for other signals
        if self._ends_with_hyphen(prev_end):
            return JoinDecision.DIRECT, "Hyphenated word across page"

        # Default to paragraph break for safety
        return JoinDecision.PARAGRAPH, "Ambiguous boundary, defaulting to paragraph"

    def _ends_with_complete_sentence(self, text: str) -> bool:
        """Check if text ends with a complete sentence."""
        text = text.rstrip()

        if not text:
            return True

        # Check for sentence-ending punctuation
        if self.SENTENCE_END.search(text):
            # Make sure it's not an abbreviation
            words = text.split()
            if words:
                last_word = words[-1].rstrip('."\')]').lower()
                if last_word in self.ABBREVIATIONS:
                    return False
            return True

        return False

    def _starts_as_continuation(self, text: str) -> bool:
        """Check if text likely continues a previous sentence."""
        text = text.lstrip()

        if not text:
            return False

        # Starts with lowercase letter
        if self.CONTINUATION_START.match(text):
            return True

        # Starts with continuation punctuation
        if self.CONTINUATION_CHARS.match(text):
            return True

        return False

    def _ends_with_hyphen(self, text: str) -> bool:
        """Check if text ends with a hyphenated word break."""
        text = text.rstrip()
        # Word broken with hyphen at end of line
        return bool(re.search(r'\w-\s*$', text))

    def _make_separator(self, page_num: int, decision: JoinDecision) -> str:
        """Create the appropriate separator between pages."""
        marker = f"<!-- page {page_num} -->" if self.add_page_markers else ""

        if decision == JoinDecision.DIRECT:
            # Single space for sentence continuation
            return f" {marker}" if marker else " "

        elif decision == JoinDecision.PARAGRAPH:
            # Double newline for paragraph break
            return f"\n\n{marker}\n" if marker else "\n\n"

        else:  # SECTION
            # More prominent break for sections
            return f"\n\n{marker}\n\n" if marker else "\n\n---\n\n"


def join_pages_simple(pages: list[str]) -> str:
    """Simple convenience function for joining pages.

    Args:
        pages: List of page texts

    Returns:
        Joined text
    """
    joiner = PageJoiner(add_page_markers=False)
    text, _ = joiner.join_pages(pages)
    return text
