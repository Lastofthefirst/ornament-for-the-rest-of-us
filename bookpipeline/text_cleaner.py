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
    """Simple dictionary for validating English words.

    Uses lazy initialization - dictionary is loaded on first use.
    """

    def __init__(self):
        self._words: set[str] | None = None

    def _load_dictionary(self) -> set[str]:
        """Load words from system dictionary.

        Supports Linux, macOS, and falls back gracefully if no dictionary found.
        """
        import platform

        dict_paths = [
            # Linux paths
            Path('/usr/share/dict/words'),
            Path('/usr/share/dict/american-english'),
            Path('/usr/share/dict/british-english'),
            # macOS paths
            Path('/usr/share/dict/web2'),
            Path('/usr/share/dict/web2a'),
            # Homebrew on macOS
            Path('/opt/homebrew/share/dict/words'),
            Path('/usr/local/share/dict/words'),
        ]

        # Add platform-specific paths first for faster lookup
        if platform.system() == 'Darwin':  # macOS
            dict_paths = [
                Path('/usr/share/dict/web2'),
                Path('/usr/share/dict/web2a'),
            ] + dict_paths

        for dict_path in dict_paths:
            if dict_path.exists():
                try:
                    with open(dict_path, 'r', encoding='utf-8', errors='ignore') as f:
                        words = {line.strip().lower() for line in f if line.strip()}
                    logger.debug(f"Loaded {len(words)} words from {dict_path}")
                    return words
                except Exception as e:
                    logger.warning(f"Could not load dictionary from {dict_path}: {e}")

        logger.warning("No system dictionary found, dehyphenation will be conservative")
        return set()

    def is_word(self, word: str) -> bool:
        """Check if a word exists in the dictionary."""
        if self._words is None:
            self._words = self._load_dictionary()
        if not self._words:
            return False
        return word.lower() in self._words


# Module-level singleton instance (lazy-loaded)
_dictionary = WordDictionary()


def get_dictionary() -> WordDictionary:
    """Get the dictionary instance."""
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


def process_page_json(json_data: list[dict], check_hallucination: bool = True) -> str:
    """Process a page's JSON layout data into clean text.

    Args:
        json_data: List of layout blocks from OCR JSON output
        check_hallucination: Whether to check for and clean repetition hallucinations

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
            # Check for hallucination (repeated text)
            if check_hallucination:
                text, had_hallucination = clean_repetition_hallucination(text)

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


def extract_text_from_json_fragments(text: str) -> str | None:
    """Try to extract text fields from JSON-like content.

    Used when OCR returned JSON but it's malformed (e.g., due to truncation).
    Extracts "text": "..." values using regex.

    Args:
        text: String that might contain JSON fragments

    Returns:
        Extracted text if JSON-like content found, None otherwise
    """
    if not text or '"text":' not in text:
        return None

    # Pattern to match "text": "..." fields in JSON
    # Handles escaped quotes inside the text
    pattern = r'"text"\s*:\s*"((?:[^"\\]|\\.)*)(?:"|$)'
    matches = re.findall(pattern, text)

    if not matches:
        return None

    # Filter out short matches (likely headers/footers) and clean up escapes
    text_blocks = []
    for match in matches:
        # Unescape JSON escapes
        try:
            cleaned = match.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
        except Exception:
            cleaned = match

        # Skip very short text (likely page numbers, headers)
        if len(cleaned) > 50:
            text_blocks.append(cleaned)

    if text_blocks:
        return '\n\n'.join(text_blocks)

    return None


def clean_markdown_text(text: str) -> str:
    """Clean markdown text that wasn't processed via JSON.

    Fallback for when JSON isn't available.

    Args:
        text: Raw markdown text

    Returns:
        Cleaned text
    """
    # First, try to extract text from JSON fragments if this looks like JSON
    if text.strip().startswith('[{') or '"text":' in text:
        extracted = extract_text_from_json_fragments(text)
        if extracted:
            text = extracted

    # Split into paragraphs (double newline separated)
    paragraphs = re.split(r'\n\n+', text)

    cleaned_paragraphs = []
    for para in paragraphs:
        if para.strip():
            cleaned = clean_text_block(para)
            if cleaned:
                cleaned_paragraphs.append(cleaned)

    return '\n\n'.join(cleaned_paragraphs)


def _normalize_header(text: str) -> str:
    """Normalize header text for comparison (case-insensitive, stripped)."""
    return text.strip().upper()


def _header_interrupts_sentence(text_before: str, text_after: str) -> bool:
    """Check if a header interrupts a sentence using pysbd.

    If the text before and after, when joined, form a valid continuous
    sentence, then the header was interrupting that sentence.

    Args:
        text_before: Text immediately before the header
        text_after: Text immediately after the header

    Returns:
        True if the header interrupts a sentence
    """
    import pysbd

    # Clean up the text snippets - get last ~300 chars before, first ~300 after
    before = text_before.strip()[-300:] if text_before else ""
    after = text_after.strip()[:300] if text_after else ""

    if not before or not after:
        return False

    # Strip HTML comments (page markers like <!-- page 41 -->)
    before = re.sub(r'<!--.*?-->', '', before, flags=re.DOTALL).strip()
    after = re.sub(r'<!--.*?-->', '', after, flags=re.DOTALL).strip()

    # Strip chapter markers like *Chapter 6*, *Chapter* 7, etc. - these aren't sentences
    before = re.sub(r'\*?Chapter\*?\s*\d+\*?', '', before, flags=re.IGNORECASE).strip()

    # Check if "after" starts with something that looks like a new section:
    # - Bold text like **Introduction**
    # - Bullet points
    # - Numbered lists
    # - Blockquotes (epigraphs)
    # NOTE: Check BEFORE stripping blockquote markers so we can detect them
    after_first_line = after.split('\n')[0].strip()

    # Strip markdown blockquote formatting for sentence analysis
    before_clean = re.sub(r'^>\s*', '', before, flags=re.MULTILINE).strip()
    after_clean = re.sub(r'^>\s*', '', after, flags=re.MULTILINE).strip()

    if not before_clean or not after_clean:
        return False
    new_section_patterns = [
        r'^\*\*[^*]+\*\*\s*$',  # Bold text line (like **Introduction**)
        r'^\*\s+',              # Bullet point
        r'^\d+\.\s+',           # Numbered list
        r'^>\s*[A-Z"—]',        # Blockquote starting with uppercase/quote/em-dash (real epigraph)
        r'^—',                  # Attribution line
    ]
    for pattern in new_section_patterns:
        if re.match(pattern, after_first_line):
            return False  # After starts a new section, not a continuation

    seg = pysbd.Segmenter(language="en", clean=False)

    # Check sentences in the "before" text
    before_sentences = seg.segment(before_clean)
    if not before_sentences:
        return False

    # Get the last sentence fragment from "before"
    last_before = before_sentences[-1].strip()

    # Check sentences in the "after" text
    after_sentences = seg.segment(after_clean)
    if not after_sentences:
        return False

    # Get the first sentence fragment from "after"
    first_after = after_sentences[0].strip()

    # Key insight: if "before" ends with a complete sentence, pysbd will have
    # segmented it properly. If it doesn't end with terminal punctuation,
    # the last "sentence" is actually incomplete.

    # Check if the last part of "before" looks incomplete
    # (doesn't end with sentence-ending punctuation)
    terminal_punct = '.!?"'
    superscripts = '⁰¹²³⁴⁵⁶⁷⁸⁹'

    # Strip trailing footnotes/superscripts to find actual ending
    last_char_idx = len(last_before) - 1
    while last_char_idx >= 0 and (last_before[last_char_idx] in superscripts or
                                   last_before[last_char_idx].isdigit()):
        last_char_idx -= 1

    before_ends_sentence = (last_char_idx >= 0 and
                            last_before[last_char_idx] in terminal_punct)

    # If before ends with complete sentence, no interruption
    if before_ends_sentence:
        return False

    # Check if "after" starts with lowercase (strong continuation signal)
    after_starts_lowercase = (first_after and
                              first_after[0].islower())

    # If before doesn't end a sentence AND after starts lowercase = interruption
    if after_starts_lowercase:
        return True

    # Additional check: try joining and see if it makes one sentence
    # Only do this if the first_after looks like a real sentence continuation
    # (not a title, not a short fragment)
    if len(first_after) > 20 and not first_after.isupper():
        joined = last_before + " " + first_after
        joined_sentences = seg.segment(joined)
        # If joining creates exactly one sentence, it was one sentence
        if len(joined_sentences) == 1:
            return True

    return False


def remove_running_headers(text: str) -> str:
    """Remove running headers that interrupt sentences or duplicate previous headers.

    Uses two heuristics:
    1. Sentence interruption: If a header appears between text that doesn't end
       a sentence and text that continues it, the header is a running header.
       Uses pysbd for robust sentence boundary detection.
    2. Duplicate detection: If we've seen a header with the same text before
       (case-insensitive), subsequent occurrences are running headers.

    Args:
        text: Full document text with markdown headers

    Returns:
        Text with running headers removed
    """
    if not text:
        return text

    # Pattern to find markdown headers on their own line
    header_pattern = re.compile(
        r'^(#+\s+.+)$',
        re.MULTILINE
    )

    # Track headers we've seen (normalized for case-insensitive comparison)
    seen_headers: set[str] = set()

    # Find all headers and their positions
    headers_to_remove: list[tuple[int, int]] = []
    removed_count = 0

    for match in header_pattern.finditer(text):
        header_line = match.group(1)
        header_start = match.start()
        header_end = match.end()

        # Extract just the header text (without # prefix)
        header_text = re.sub(r'^#+\s+', '', header_line).strip()
        normalized = _normalize_header(header_text)

        # Get context: text before and after the header
        text_before = text[:header_start]
        text_after = text[header_end:]

        # Skip if this is at the very beginning (likely a real title)
        if not text_before.strip():
            seen_headers.add(normalized)
            continue

        # Heuristic 1: Check if header interrupts a sentence (using pysbd)
        interrupts_sentence = _header_interrupts_sentence(text_before, text_after)

        # Heuristic 2: Check if this is a duplicate header
        is_duplicate = normalized in seen_headers

        # Remove if either heuristic flags it
        should_remove = False

        if interrupts_sentence:
            logger.debug(f"Running header (interrupts sentence): '{header_text[:50]}'")
            should_remove = True
        elif is_duplicate:
            logger.debug(f"Running header (duplicate): '{header_text[:50]}'")
            should_remove = True

        if should_remove:
            headers_to_remove.append((header_start, header_end))
            removed_count += 1
        else:
            # Track this header as seen
            seen_headers.add(normalized)

    if removed_count > 0:
        logger.info(f"Removing {removed_count} running headers")

    # Remove headers in reverse order to preserve positions
    result = text
    for start, end in reversed(headers_to_remove):
        # Find the actual line boundaries
        line_start = result.rfind('\n', 0, start)
        line_start = line_start + 1 if line_start >= 0 else 0

        line_end = result.find('\n', end)
        line_end = line_end if line_end >= 0 else len(result)

        # Remove the line
        result = result[:line_start] + result[line_end:]

    # Clean up multiple consecutive blank lines
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()


def detect_repeated_section_headers(pages: list[str], min_occurrences: int = 3, window_size: int = 20) -> set[str]:
    """Detect section headers that repeat across multiple pages within a window.

    DEPRECATED: This function uses occurrence counting which is less accurate
    than the sentence-interruption heuristic. Use remove_running_headers() on
    joined text instead.

    These are running headers that OCR mistakenly categorized as Section-header
    instead of Page-header.

    Args:
        pages: List of page texts
        min_occurrences: Minimum times a header must appear within window
        window_size: Number of consecutive pages to check

    Returns:
        Set of repeated section header phrases to remove
    """
    from collections import Counter

    # Pattern to match markdown headers (# or ##) at start of line
    header_pattern = re.compile(r'^#+\s+(.+)$', re.MULTILINE)

    repeated_headers = set()

    # Use sliding window to find headers that repeat within short spans
    for start_idx in range(len(pages)):
        end_idx = min(start_idx + window_size, len(pages))
        window_pages = pages[start_idx:end_idx]

        header_counts = Counter()
        for page in window_pages:
            matches = header_pattern.findall(page)
            for match in matches:
                # Normalize: strip markdown, convert to uppercase for comparison
                normalized = match.strip().upper()
                # Only consider short headers (< 50 chars) - running headers are usually short
                if len(normalized) < 50:
                    header_counts[normalized] += 1

        # Headers appearing min_occurrences+ times in window are likely running headers
        for header, count in header_counts.items():
            if count >= min_occurrences:
                repeated_headers.add(header)

    if repeated_headers:
        logger.info(f"Detected {len(repeated_headers)} repeated section headers: {repeated_headers}")

    return repeated_headers


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


def remove_repeated_section_headers(pages: list[str], repeated_headers: set[str]) -> list[str]:
    """Remove repeated section headers from pages.

    Keeps the first occurrence of each header (the actual section title),
    removes subsequent occurrences (running headers).

    Args:
        pages: List of page texts
        repeated_headers: Set of normalized header text to remove (uppercase)

    Returns:
        Pages with repeated section headers removed
    """
    if not repeated_headers:
        return pages

    # Pattern to match markdown headers (# or ##) at start of line
    header_pattern = re.compile(r'^#+\s+(.+)$', re.MULTILINE)

    # Track first occurrence of each repeated header
    first_occurrence_seen = set()

    cleaned_pages = []
    for page in pages:
        if not page.strip():
            cleaned_pages.append(page)
            continue

        def should_remove_header(match):
            header_text = match.group(1).strip().upper()
            if header_text not in repeated_headers:
                return False  # Not a repeated header, keep it

            # Keep first occurrence (the actual section title)
            if header_text not in first_occurrence_seen:
                first_occurrence_seen.add(header_text)
                return False  # Keep first occurrence

            # Remove subsequent occurrences (running headers)
            return True

        # Remove lines that contain repeated section headers
        lines = []
        for line in page.split('\n'):
            match = header_pattern.match(line)
            if match and should_remove_header(match):
                # Skip this line - it's a repeated running header
                continue
            lines.append(line)

        cleaned_pages.append('\n'.join(lines))

    return cleaned_pages


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

    This includes both:
    1. Regular headers/footers at top/bottom of pages
    2. Section headers that repeat across multiple pages (running headers)

    Args:
        pages: List of page texts (already processed by JSON categories)
        min_header_occurrences: Minimum times a phrase must repeat to be detected

    Returns:
        Cleaned pages
    """
    # Step 1: Remove repeated section headers (running headers like "## INTRODUCTION")
    # These repeat within a short window (3+ times in 20 pages)
    repeated_section_headers = detect_repeated_section_headers(pages, min_occurrences=3, window_size=20)
    cleaned = remove_repeated_section_headers(pages, repeated_section_headers)

    # Step 2: Remove regular headers/footers at top/bottom of pages
    # These repeat across many pages (10+ times)
    headers, footers = detect_repeated_headers_footers(cleaned, min_occurrences=min_header_occurrences)
    cleaned = remove_repeated_headers_footers(cleaned, headers, footers)

    return cleaned


def detect_repetition_hallucination(text: str, min_repeats: int = 10, min_phrase_len: int = 15) -> tuple[str, str] | None:
    """Detect OCR hallucination where the same phrase repeats many times.

    This happens when the OCR model gets stuck in a loop, often due to
    interruptions like system sleep/wake or GPU issues.

    Uses an O(n) algorithm instead of O(n²) by checking fixed sample points.

    Args:
        text: Text to check
        min_repeats: Minimum number of consecutive repeats to flag (default 10)
        min_phrase_len: Minimum phrase length to consider (default 15 chars)

    Returns:
        Tuple of (raw phrase for regex, stripped phrase for display) if found, None otherwise
    """
    if not text or len(text) < min_phrase_len * min_repeats:
        return None

    text_len = len(text)
    max_phrase_len = min(100, text_len // min_repeats)

    # O(n) approach: sample at regular intervals and check for repetitions
    # Instead of checking every position, check at strategic points
    # If there's a repeating pattern, we'll find it by sampling

    # Strategy: for each phrase length, only check a limited number of start positions
    # distributed across the text. This is O(phrase_lengths * sample_count) = O(1) * O(n) = O(n)
    sample_count = min(50, text_len // min_phrase_len)  # Limit samples

    for phrase_len in range(min_phrase_len, max_phrase_len):
        # Sample start positions evenly distributed
        step = max(1, (text_len - phrase_len * min_repeats) // sample_count)

        for sample_idx in range(sample_count):
            start = sample_idx * step
            if start + phrase_len * min_repeats > text_len:
                break

            phrase = text[start:start + phrase_len]

            # Skip if phrase is mostly whitespace
            if len(phrase.strip()) < min_phrase_len // 2:
                continue

            # Count consecutive repetitions
            count = 1
            pos = start + phrase_len
            while pos + phrase_len <= text_len and text[pos:pos + phrase_len] == phrase:
                count += 1
                pos += phrase_len

            if count >= min_repeats:
                return (phrase, phrase.strip())

    return None


def clean_repetition_hallucination(text: str, min_repeats: int = 10) -> tuple[str, bool]:
    """Remove OCR hallucination repetitions from text.

    Args:
        text: Text to clean
        min_repeats: Minimum consecutive repeats to remove

    Returns:
        Tuple of (cleaned text, whether hallucination was found)
    """
    result = detect_repetition_hallucination(text, min_repeats=min_repeats)
    if result is None:
        return text, False

    raw_phrase, display_phrase = result

    # Remove all but one occurrence of the repeated phrase
    logger.warning(f"Detected OCR hallucination: '{display_phrase[:50]}...' repeated {min_repeats}+ times")

    # Find the repeated section and keep just one instance
    pattern = re.escape(raw_phrase)
    # Replace 2+ consecutive occurrences with just one
    cleaned = re.sub(f'({pattern}){{2,}}', raw_phrase, text)

    return cleaned, True
