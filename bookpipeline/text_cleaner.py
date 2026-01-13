"""
Text cleaning for OCR output: dehyphenation and line unwrapping.

Uses JSON layout data for robust processing.
Uses dictionary validation for safe dehyphenation.
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Categories to exclude (headers, footers, page numbers)
EXCLUDED_CATEGORIES = {'Page-header', 'Page-footer', 'Page-number'}


class WordDictionary:
    """Simple dictionary for validating English words."""

    _instance = None
    _words: set[str] = set()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_dictionary()
        return cls._instance

    def _load_dictionary(self) -> None:
        """Load words from system dictionary."""
        dict_paths = [
            Path('/usr/share/dict/words'),
            Path('/usr/share/dict/american-english'),
            Path('/usr/share/dict/british-english'),
        ]

        for dict_path in dict_paths:
            if dict_path.exists():
                try:
                    with open(dict_path, 'r', encoding='utf-8', errors='ignore') as f:
                        self._words = {line.strip().lower() for line in f if line.strip()}
                    logger.debug(f"Loaded {len(self._words)} words from {dict_path}")
                    return
                except Exception as e:
                    logger.warning(f"Could not load dictionary from {dict_path}: {e}")

        logger.warning("No system dictionary found, dehyphenation will be conservative")

    def is_word(self, word: str) -> bool:
        """Check if a word exists in the dictionary."""
        if not self._words:
            return False
        return word.lower() in self._words


# Global dictionary instance
_dictionary = None


def get_dictionary() -> WordDictionary:
    """Get the singleton dictionary instance."""
    global _dictionary
    if _dictionary is None:
        _dictionary = WordDictionary()
    return _dictionary


def dehyphenate_word(word1: str, word2: str) -> str:
    """Attempt to dehyphenate a word broken across lines.

    Args:
        word1: First part of word (before hyphen)
        word2: Second part of word (after newline)

    Returns:
        Either joined word or hyphenated compound
    """
    dictionary = get_dictionary()

    # Try joining without hyphen
    joined = word1 + word2

    # Check if joined word exists in dictionary
    if dictionary.is_word(joined):
        return joined

    # Not a valid word when joined, keep as compound with hyphen
    return word1 + '-' + word2


def clean_text_block(text: str) -> str:
    """Clean a single text block: dehyphenate and unwrap lines.

    Args:
        text: Raw text from OCR with embedded newlines

    Returns:
        Cleaned text with proper spacing
    """
    if not text:
        return text

    # Step 1: Dehyphenate (join words broken across lines)
    # Pattern: word + hyphen + newline + lowercase word
    def dehyphenate_match(match):
        word1 = match.group(1)
        word2 = match.group(2)
        return dehyphenate_word(word1, word2)

    # Only match hyphen-breaks followed by lowercase (indicates potential word break)
    hyphen_pattern = re.compile(r'(\w+)-\n([a-z]\w*)')
    text = hyphen_pattern.sub(dehyphenate_match, text)

    # Step 2: Unwrap remaining single newlines (replace with space)
    # This handles line breaks within paragraphs
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Step 3: Clean up any double spaces
    text = re.sub(r'  +', ' ', text)

    return text.strip()


def process_page_json(json_data: list[dict]) -> str:
    """Process a page's JSON layout data into clean text.

    Args:
        json_data: List of layout blocks from OCR JSON output

    Returns:
        Clean text with headers/footers removed and text properly formatted
    """
    if not json_data:
        return ""

    text_blocks = []

    for block in json_data:
        category = block.get('category', 'Text')
        text = block.get('text', '')

        # Skip headers, footers, page numbers
        if category in EXCLUDED_CATEGORIES:
            continue

        # Clean the text block
        if text:
            cleaned = clean_text_block(text)
            if cleaned:
                text_blocks.append(cleaned)

    # Join blocks with paragraph breaks
    return '\n\n'.join(text_blocks)


def process_page_from_json_file(json_path: Path) -> str:
    """Process a page from its JSON file.

    Args:
        json_path: Path to the page's JSON file

    Returns:
        Clean text
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        return process_page_json(json_data)
    except Exception as e:
        logger.warning(f"Could not process JSON {json_path}: {e}")
        return ""


def clean_markdown_text(text: str) -> str:
    """Clean markdown text that wasn't processed via JSON.

    Fallback for when JSON isn't available.

    Args:
        text: Raw markdown text

    Returns:
        Cleaned text
    """
    # Split into paragraphs (double newline separated)
    paragraphs = re.split(r'\n\n+', text)

    cleaned_paragraphs = []
    for para in paragraphs:
        if para.strip():
            cleaned = clean_text_block(para)
            if cleaned:
                cleaned_paragraphs.append(cleaned)

    return '\n\n'.join(cleaned_paragraphs)


def detect_repeated_headers_footers(pages: list[str], min_occurrences: int = 10) -> tuple[set[str], set[str]]:
    """Detect phrases repeated at top/bottom of many pages (missed headers/footers).

    Secondary validation after JSON category filtering.

    Args:
        pages: List of page texts (already processed by JSON categories)
        min_occurrences: Minimum times a phrase must appear to be considered

    Returns:
        Tuple of (header phrases, footer phrases)
    """
    from collections import Counter

    first_lines = Counter()
    last_lines = Counter()

    for page in pages:
        if not page.strip():
            continue

        lines = [l.strip() for l in page.strip().split('\n') if l.strip()]
        if not lines:
            continue

        # Check first line - short lines only (headers are typically short)
        first = lines[0]
        if len(first) < 80:
            first_lines[first] += 1

        # Check last line
        last = lines[-1]
        if len(last) < 80:
            last_lines[last] += 1

    # Find phrases that repeat often
    headers = {phrase for phrase, count in first_lines.items() if count >= min_occurrences}
    footers = {phrase for phrase, count in last_lines.items() if count >= min_occurrences}

    if headers:
        logger.info(f"Detected {len(headers)} repeated headers: {headers}")
    if footers:
        logger.info(f"Detected {len(footers)} repeated footers: {footers}")

    return headers, footers


def remove_repeated_headers_footers(pages: list[str], headers: set[str], footers: set[str]) -> list[str]:
    """Remove detected repeated headers/footers from top/bottom of pages.

    Args:
        pages: List of page texts
        headers: Set of header phrases to remove from page starts
        footers: Set of footer phrases to remove from page ends

    Returns:
        Pages with headers/footers removed
    """
    if not headers and not footers:
        return pages

    cleaned_pages = []
    for page in pages:
        if not page.strip():
            cleaned_pages.append(page)
            continue

        lines = page.strip().split('\n')

        # Remove header from start if present
        while lines and lines[0].strip() in headers:
            lines = lines[1:]

        # Remove footer from end if present
        while lines and lines[-1].strip() in footers:
            lines = lines[:-1]

        cleaned_pages.append('\n'.join(lines).strip())

    return cleaned_pages


def clean_book_pages(pages: list[str], min_header_occurrences: int = 10) -> list[str]:
    """Secondary validation: remove repeated headers/footers missed by JSON categories.

    Args:
        pages: List of page texts (already processed by JSON categories)
        min_header_occurrences: Minimum times a phrase must repeat to be detected

    Returns:
        Cleaned pages
    """
    headers, footers = detect_repeated_headers_footers(pages, min_occurrences=min_header_occurrences)
    cleaned = remove_repeated_headers_footers(pages, headers, footers)
    return cleaned
