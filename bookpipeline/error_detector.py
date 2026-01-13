"""
Error detection for OCR output quality.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for detected errors."""

    INFO = "info"  # Minor issue, probably fine
    WARNING = "warning"  # Should review manually
    ERROR = "error"  # Likely OCR failure, needs attention


@dataclass
class PageError:
    """An error detected on a specific page."""

    page_number: int
    severity: ErrorSeverity
    error_type: str
    message: str
    context: str = ""  # Snippet of problematic text


@dataclass
class ErrorReport:
    """Complete error report for a book."""

    total_pages: int
    successful_pages: int
    errors: list[PageError] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for e in self.errors if e.severity == ErrorSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for e in self.errors if e.severity == ErrorSeverity.WARNING)

    @property
    def has_critical_errors(self) -> bool:
        return self.error_count > 0

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"OCR Error Report",
            f"================",
            f"Total pages: {self.total_pages}",
            f"Successful: {self.successful_pages}",
            f"Errors: {self.error_count}",
            f"Warnings: {self.warning_count}",
            "",
        ]

        if self.errors:
            lines.append("Issues by page:")
            lines.append("-" * 40)

            for error in sorted(self.errors, key=lambda e: (e.page_number, e.severity.value)):
                severity_marker = {
                    ErrorSeverity.ERROR: "[ERROR]",
                    ErrorSeverity.WARNING: "[WARN] ",
                    ErrorSeverity.INFO: "[INFO] ",
                }[error.severity]

                lines.append(f"Page {error.page_number:4d} {severity_marker} {error.error_type}")
                lines.append(f"           {error.message}")
                if error.context:
                    # Truncate context for display
                    ctx = error.context[:100] + "..." if len(error.context) > 100 else error.context
                    lines.append(f"           Context: {ctx!r}")
                lines.append("")

        return "\n".join(lines)


class ErrorDetector:
    """Detects potential OCR errors and quality issues."""

    # Characters that shouldn't appear frequently in normal text
    GARBAGE_CHARS = re.compile(r'[□■◊◆●○►▼▲★☆♦♣♠♥※†‡¶§¤¢£¥€]')

    # Repeated characters that indicate OCR confusion
    REPEATED_CHARS = re.compile(r'(.)\1{4,}')  # Same char 5+ times

    # Common OCR confusion patterns
    OCR_CONFUSION = [
        (re.compile(r'[0O][0O][0O]'), "Possible 0/O confusion"),
        (re.compile(r'[1Il][1Il][1Il]'), "Possible 1/I/l confusion"),
        (re.compile(r'rn(?=\w)'), "Possible 'rn' vs 'm' confusion"),
        (re.compile(r'(?<!\w)vv(?=\w)'), "Possible 'vv' vs 'w' confusion"),
    ]

    # Brackets/quotes that should be balanced
    BRACKET_PAIRS = [
        ('(', ')'),
        ('[', ']'),
        ('{', '}'),
        ('"', '"'),
        ("'", "'"),
    ]

    def __init__(
        self,
        min_text_length: int = 50,
        max_error_ratio: float = 0.15,
        min_word_length_avg: float = 2.0,
    ) -> None:
        """Initialize error detector.

        Args:
            min_text_length: Pages with less text are flagged
            max_error_ratio: Maximum ratio of suspicious chars
            min_word_length_avg: Minimum average word length (below = gibberish)
        """
        self.min_text_length = min_text_length
        self.max_error_ratio = max_error_ratio
        self.min_word_length_avg = min_word_length_avg

    def analyze_page(self, page_number: int, text: str) -> list[PageError]:
        """Analyze a single page for errors.

        Args:
            page_number: Page index (0-based)
            text: OCR text for the page

        Returns:
            List of detected errors
        """
        errors: list[PageError] = []

        # Check for empty or very short pages
        if not text or len(text.strip()) < self.min_text_length:
            errors.append(PageError(
                page_number=page_number,
                severity=ErrorSeverity.ERROR if not text else ErrorSeverity.WARNING,
                error_type="SHORT_PAGE",
                message=f"Page has only {len(text.strip()) if text else 0} characters",
                context=text[:50] if text else "",
            ))

        if not text:
            return errors

        # Check for garbage characters
        garbage_matches = self.GARBAGE_CHARS.findall(text)
        if garbage_matches:
            ratio = len(garbage_matches) / len(text)
            if ratio > 0.01:  # More than 1% garbage
                errors.append(PageError(
                    page_number=page_number,
                    severity=ErrorSeverity.WARNING,
                    error_type="GARBAGE_CHARS",
                    message=f"Found {len(garbage_matches)} suspicious characters ({ratio:.1%})",
                    context="".join(set(garbage_matches)),
                ))

        # Check for repeated characters
        repeated = self.REPEATED_CHARS.findall(text)
        if repeated:
            errors.append(PageError(
                page_number=page_number,
                severity=ErrorSeverity.WARNING,
                error_type="REPEATED_CHARS",
                message=f"Found {len(repeated)} instances of repeated characters",
                context=str(repeated[:5]),
            ))

        # Check for OCR confusion patterns
        for pattern, description in self.OCR_CONFUSION:
            matches = pattern.findall(text)
            if len(matches) > 3:  # Allow a few occurrences
                errors.append(PageError(
                    page_number=page_number,
                    severity=ErrorSeverity.INFO,
                    error_type="OCR_CONFUSION",
                    message=f"{description}: found {len(matches)} instances",
                ))

        # Check average word length
        words = text.split()
        if words:
            avg_len = sum(len(w) for w in words) / len(words)
            if avg_len < self.min_word_length_avg:
                errors.append(PageError(
                    page_number=page_number,
                    severity=ErrorSeverity.WARNING,
                    error_type="SHORT_WORDS",
                    message=f"Average word length is {avg_len:.1f} (expected >{self.min_word_length_avg})",
                ))

        # Check for unbalanced brackets
        for open_char, close_char in self.BRACKET_PAIRS:
            if open_char == close_char:
                # For quotes, count should be even
                count = text.count(open_char)
                if count % 2 != 0 and count > 2:
                    errors.append(PageError(
                        page_number=page_number,
                        severity=ErrorSeverity.INFO,
                        error_type="UNBALANCED_QUOTES",
                        message=f"Odd number of {open_char!r} characters: {count}",
                    ))
            else:
                opens = text.count(open_char)
                closes = text.count(close_char)
                if abs(opens - closes) > 2:  # Allow small imbalances
                    errors.append(PageError(
                        page_number=page_number,
                        severity=ErrorSeverity.INFO,
                        error_type="UNBALANCED_BRACKETS",
                        message=f"Unbalanced {open_char}{close_char}: {opens} open, {closes} close",
                    ))

        # Check for excessive special character ratio
        special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if text:
            special_ratio = special_count / len(text)
            if special_ratio > self.max_error_ratio:
                errors.append(PageError(
                    page_number=page_number,
                    severity=ErrorSeverity.WARNING,
                    error_type="HIGH_SPECIAL_CHAR_RATIO",
                    message=f"Special character ratio is {special_ratio:.1%} (threshold: {self.max_error_ratio:.1%})",
                ))

        return errors

    def analyze_book(
        self,
        pages: list[str],
        ocr_successes: list[bool] | None = None,
    ) -> ErrorReport:
        """Analyze all pages in a book.

        Args:
            pages: List of OCR text per page
            ocr_successes: Optional list of whether OCR succeeded per page

        Returns:
            Complete error report
        """
        all_errors: list[PageError] = []

        for i, text in enumerate(pages):
            # Add OCR failure error if we know it failed
            if ocr_successes and not ocr_successes[i]:
                all_errors.append(PageError(
                    page_number=i,
                    severity=ErrorSeverity.ERROR,
                    error_type="OCR_FAILED",
                    message="OCR processing failed for this page",
                ))

            # Analyze the text content
            page_errors = self.analyze_page(i, text)
            all_errors.extend(page_errors)

        successful = len(pages)
        if ocr_successes:
            successful = sum(1 for s in ocr_successes if s)

        report = ErrorReport(
            total_pages=len(pages),
            successful_pages=successful,
            errors=all_errors,
        )

        logger.info(
            f"Error analysis complete: {report.error_count} errors, "
            f"{report.warning_count} warnings across {len(pages)} pages"
        )

        return report
