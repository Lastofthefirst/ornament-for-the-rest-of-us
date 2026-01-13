"""
Configuration for the book processing pipeline.
"""

from dataclasses import dataclass, field
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
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)

        if self.dots_ocr_path:
            self.dots_ocr_path = Path(self.dots_ocr_path)

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
