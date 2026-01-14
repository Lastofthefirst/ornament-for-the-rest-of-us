"""
Main pipeline orchestration for book processing.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig
from .error_detector import ErrorDetector, ErrorReport
from .epub_builder import EPUBBuilder, EPUBMetadata
from .ocr import OCRProcessor, OCRResult
from .page_joiner import PageJoiner
from .preprocessor import ImagePreprocessor
from .text_cleaner import (
    process_page_json,
    process_page_from_json_file,
    clean_markdown_text,
    clean_book_pages,
    clean_repetition_hallucination,
    remove_running_headers,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of running the full pipeline."""

    success: bool
    markdown_path: Path | None
    epub_path: Path | None
    error_report: ErrorReport
    message: str


class BookPipeline:
    """Main orchestrator for the book processing pipeline.

    Usage:
        config = PipelineConfig(
            input_dir="./photos",
            output_dir="./output",
            book_title="My Book",
        )
        pipeline = BookPipeline(config)
        result = pipeline.run()
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self._setup_logging()
        self._validate_config()

    def _setup_logging(self) -> None:
        """Ensure logging is configured.

        Only sets up a basic config if no handlers are configured,
        allowing the CLI to control logging setup.
        """
        if not logging.root.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )

    def _validate_config(self) -> None:
        """Validate configuration before running."""
        if not self.config.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.config.input_dir}")

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> PipelineResult:
        """Run the complete pipeline.

        Returns:
            PipelineResult with output paths and status
        """
        import sys
        import time

        def step(num: int, name: str) -> None:
            sys.stderr.write(f"\n[{num}/5] {name}\n")
            sys.stderr.flush()

        start_time = time.time()
        sys.stderr.write(f"Processing: {self.config.book_title}\n")
        sys.stderr.write(f"Input: {self.config.input_dir}\n")
        sys.stderr.write(f"Output: {self.config.output_dir}\n")
        sys.stderr.flush()

        try:
            # Step 1: Preprocess images
            step(1, "Preprocessing images")
            processed_images = self._preprocess_images()

            if not processed_images:
                return PipelineResult(
                    success=False,
                    markdown_path=None,
                    epub_path=None,
                    error_report=ErrorReport(0, 0),
                    message="No images found to process",
                )

            # Step 2: Run OCR
            step(2, "Running OCR")
            ocr_results = self._run_ocr(processed_images)

            # Step 3: Detect errors
            step(3, "Analyzing OCR quality")
            error_report = self._analyze_errors(ocr_results)
            self.config.error_report_path.write_text(error_report.summary())

            successful = sum(1 for r in ocr_results if r.success)
            if error_report.error_count > 0:
                sys.stderr.write(f"  {successful}/{len(ocr_results)} pages OK, {error_report.error_count} errors\n")
            else:
                sys.stderr.write(f"  {successful}/{len(ocr_results)} pages OK\n")
            sys.stderr.flush()

            # Step 4: Join pages
            step(4, "Joining pages")
            joined_text = self._join_pages(ocr_results)

            # Step 5: Generate outputs
            step(5, "Generating outputs")
            markdown_path, epub_path = self._generate_outputs(joined_text)

            # Summary
            elapsed = time.time() - start_time
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)

            sys.stderr.write(f"\n{'─' * 40}\n")
            sys.stderr.write(f"Complete in {mins}m {secs}s\n")
            sys.stderr.write(f"  Pages: {len(ocr_results)}\n")
            if markdown_path:
                sys.stderr.write(f"  Markdown: {markdown_path}\n")
            if epub_path:
                sys.stderr.write(f"  EPUB: {epub_path}\n")
            if error_report.error_count > 0:
                sys.stderr.write(f"  Errors: {error_report.error_count} (see error_report.txt)\n")
            sys.stderr.flush()

            return PipelineResult(
                success=True,
                markdown_path=markdown_path,
                epub_path=epub_path,
                error_report=error_report,
                message=f"Successfully processed {len(ocr_results)} pages",
            )

        except Exception as e:
            logger.exception("Pipeline failed")
            return PipelineResult(
                success=False,
                markdown_path=None,
                epub_path=None,
                error_report=ErrorReport(0, 0),
                message=f"Pipeline failed: {e}",
            )

    def _preprocess_images(self) -> list[Path]:
        """Preprocess images: discover, sort, fix rotation, and resize."""
        preprocessor = ImagePreprocessor(
            supported_extensions=self.config.supported_extensions,
            resize_target=self.config.resize_target,
        )

        # Discover images
        images = preprocessor.discover_images(self.config.input_dir)

        if not images:
            logger.warning("No images found in input directory")
            return []

        # Sort by timestamp or filename
        if self.config.sort_by_timestamp:
            sorted_infos = preprocessor.sort_by_timestamp(images)
        else:
            sorted_infos = preprocessor.sort_by_filename(images)

        # Process images: fix rotation, resize, save with sequential names
        processed = preprocessor.process_images(
            sorted_infos,
            self.config.processed_images_dir,
        )

        return processed

    def _run_ocr(self, image_paths: list[Path]) -> list[OCRResult]:
        """Run OCR on processed images."""
        ocr = OCRProcessor(
            dots_ocr_path=self.config.dots_ocr_path,
            backend=self.config.ocr_backend,
            num_threads=self.config.ocr_threads,
        )

        results = ocr.process_images(image_paths, self.config.ocr_output_dir)
        return results

    def _analyze_errors(self, ocr_results: list[OCRResult]) -> ErrorReport:
        """Analyze OCR results for errors."""
        detector = ErrorDetector(
            min_text_length=self.config.min_text_length,
            max_error_ratio=self.config.max_error_ratio,
        )

        pages = [r.text_nohf for r in ocr_results]
        successes = [r.success for r in ocr_results]

        return detector.analyze_book(pages, successes)

    def _join_pages(self, ocr_results: list[OCRResult]) -> str:
        """Join OCR pages into single document."""
        joiner = PageJoiner(add_page_markers=self.config.add_page_markers)

        # Step 1: Process each page using JSON data for robust cleaning
        pages = []
        for r in ocr_results:
            if r.json_data and isinstance(r.json_data, list):
                # Use JSON for proper header removal and text cleaning
                cleaned = process_page_json(r.json_data)
            else:
                # Fallback to markdown cleaning
                # Still check for hallucination (repetition) in fallback path
                text, had_hallucination = clean_repetition_hallucination(r.text_nohf)
                if had_hallucination:
                    logger.warning(f"Page {r.page_number}: cleaned hallucination in fallback path")
                cleaned = clean_markdown_text(text)
            pages.append(cleaned)

        # Step 2: Secondary validation - remove repeated headers/footers
        # that may have been missed by JSON category detection
        pages = clean_book_pages(pages, min_header_occurrences=10)

        joined, boundaries = joiner.join_pages(pages)

        # Step 3: Remove running headers that interrupt sentences or duplicate
        # previous headers (works best on joined text where context is available)
        joined = remove_running_headers(joined)

        return joined

    def _generate_outputs(self, content: str) -> tuple[Path | None, Path | None]:
        """Generate markdown and/or EPUB outputs."""
        markdown_path = None
        epub_path = None

        if "md" in self.config.output_formats:
            markdown_path = self.config.markdown_output
            markdown_path.write_text(content, encoding="utf-8")
            logger.info(f"Markdown saved: {markdown_path}")

        if "epub" in self.config.output_formats:
            metadata = EPUBMetadata(
                title=self.config.book_title,
                author=self.config.book_author,
                language=self.config.language,
            )
            builder = EPUBBuilder(metadata)
            epub_path = builder.build(content, self.config.epub_output)
            logger.info(f"EPUB saved: {epub_path}")

        return markdown_path, epub_path

    def run_ocr_only(self, image_paths: list[Path] | None = None) -> list[OCRResult]:
        """Run only the OCR step (useful for re-processing).

        Args:
            image_paths: Optional list of images. If None, uses preprocessed images.

        Returns:
            List of OCR results
        """
        if image_paths is None:
            # Use existing preprocessed images
            image_paths = sorted(self.config.processed_images_dir.glob("page_*.jpg"))
            image_paths.extend(sorted(self.config.processed_images_dir.glob("page_*.png")))

        return self._run_ocr(image_paths)

    def run_join_only(self) -> str:
        """Run only the page joining step (useful after manual edits).

        Reads from OCR output directory and joins pages.

        Returns:
            Joined text
        """
        # Prefer JSON files for robust processing
        json_files = sorted(self.config.ocr_output_dir.glob("**/page_*.json"))

        pages = []
        if json_files:
            for f in json_files:
                cleaned = process_page_from_json_file(f)
                pages.append(cleaned)
        else:
            # Fallback to markdown files
            nohf_files = sorted(self.config.ocr_output_dir.glob("**/*_nohf.md"))
            for f in nohf_files:
                text = f.read_text(encoding="utf-8")
                pages.append(clean_markdown_text(text))

        # Secondary validation - remove repeated headers/footers
        pages = clean_book_pages(pages)

        joiner = PageJoiner(add_page_markers=self.config.add_page_markers)
        joined, _ = joiner.join_pages(pages)

        return joined
