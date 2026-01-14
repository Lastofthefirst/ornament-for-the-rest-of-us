"""
Configuration for the book processing pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class PipelineConfig:
    """Configuration for the book processing pipeline.

    Attributes:
        input_dir: Directory containing input images/PDFs
        output_dir: Directory for all outputs
        book_title: Title of the book (used in EPUB metadata)
        book_author: Author of the book (used in EPUB metadata)
        language: Book language code (e.g., 'en', 'es', 'zh')

        # Image preprocessing
        sort_by_timestamp: Sort images by EXIF capture time
        supported_extensions: Image extensions to process

        # OCR settings
        ocr_backend: Which backend to use ('hf' for HuggingFace, 'vllm' for vLLM server)
        ocr_threads: Number of threads for OCR processing
        dots_ocr_path: Path to dots.ocr installation

        # Page joining
        join_broken_sentences: Attempt to join sentences split across pages
        add_page_markers: Add HTML comments marking page boundaries

        # Error handling
        min_text_length: Minimum characters per page (below = likely OCR failure)
        max_error_ratio: Maximum ratio of special chars (above = likely OCR failure)

        # Output
        output_formats: Which formats to generate
    """

    # Required
    input_dir: Path
    output_dir: Path
    book_title: str

    # Optional metadata
    book_author: str = "Unknown"
    language: str = "en"

    # Image preprocessing
    sort_by_timestamp: bool = True
    supported_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    resize_target: int = 1150  # Target pixels for short edge (0 to disable resize)

    # OCR settings
    ocr_backend: Literal["hf", "vllm"] = "hf"
    ocr_threads: int = 8
    dots_ocr_path: Path | None = None

    # Page joining
    join_broken_sentences: bool = True
    add_page_markers: bool = True

    # Error detection thresholds
    min_text_length: int = 50  # Pages with less text are flagged
    max_error_ratio: float = 0.15  # More than 15% special chars = likely error

    # Output formats
    output_formats: tuple[str, ...] = ("md", "epub")

    def __post_init__(self) -> None:
        """Validate and convert paths."""
        # Convert paths
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)

        if self.dots_ocr_path:
            self.dots_ocr_path = Path(self.dots_ocr_path)

        # Validate required fields
        if not self.book_title or not self.book_title.strip():
            raise ValueError("book_title cannot be empty")

        # Validate numeric constraints
        if self.resize_target < 0:
            raise ValueError(f"resize_target must be >= 0, got {self.resize_target}")

        if self.ocr_threads < 1:
            raise ValueError(f"ocr_threads must be >= 1, got {self.ocr_threads}")

        if not 0 < self.max_error_ratio <= 1:
            raise ValueError(f"max_error_ratio must be in (0, 1], got {self.max_error_ratio}")

        if self.min_text_length < 0:
            raise ValueError(f"min_text_length must be >= 0, got {self.min_text_length}")

        # Validate output formats
        valid_formats = {"md", "epub"}
        invalid = set(self.output_formats) - valid_formats
        if invalid:
            raise ValueError(f"Invalid output formats: {invalid}. Valid: {valid_formats}")

    @property
    def ocr_output_dir(self) -> Path:
        """Directory for raw OCR output."""
        return self.output_dir / "ocr_pages"

    @property
    def processed_images_dir(self) -> Path:
        """Directory for preprocessed (sorted) images."""
        return self.output_dir / "processed_images"

    @property
    def markdown_output(self) -> Path:
        """Path to final markdown file."""
        return self.output_dir / f"{self._safe_filename}.md"

    @property
    def epub_output(self) -> Path:
        """Path to final EPUB file."""
        return self.output_dir / f"{self._safe_filename}.epub"

    @property
    def error_report_path(self) -> Path:
        """Path to error report file."""
        return self.output_dir / "error_report.txt"

    @property
    def _safe_filename(self) -> str:
        """Generate safe filename from book title."""
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in self.book_title)
        return safe.strip().replace(" ", "_")[:100]
