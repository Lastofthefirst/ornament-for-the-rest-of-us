#!/usr/bin/env python3
"""
Command-line interface for BookPipeline.

Usage:
    # Full pipeline: photos -> EPUB
    bookpipeline process ./photos --title "My Book" --author "Author Name"

    # Just preprocess images (sort and rotate)
    bookpipeline preprocess ./photos --output ./processed

    # Just run OCR on preprocessed images
    bookpipeline ocr ./processed --output ./ocr_output

    # Just join pages from OCR output
    bookpipeline join ./ocr_output --output ./book.md

    # Convert markdown to EPUB
    bookpipeline epub ./book.md --title "My Book" --output ./book.epub
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_process(args: argparse.Namespace) -> int:
    """Run the full pipeline."""
    from .config import PipelineConfig
    from .pipeline import BookPipeline

    config = PipelineConfig(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        book_title=args.title,
        book_author=args.author,
        language=args.language,
        sort_by_timestamp=not args.sort_by_name,
        resize_target=args.resize_target,
        ocr_backend=args.backend,
        ocr_threads=args.threads,
        dots_ocr_path=Path(args.dots_ocr) if args.dots_ocr else None,
        output_formats=tuple(args.formats),
    )

    pipeline = BookPipeline(config)
    result = pipeline.run()

    if result.success:
        print(f"\n✓ Success: {result.message}")
        if result.markdown_path:
            print(f"  Markdown: {result.markdown_path}")
        if result.epub_path:
            print(f"  EPUB: {result.epub_path}")
        if result.error_report.has_critical_errors:
            print(f"\n⚠ {result.error_report.error_count} errors detected - check error_report.txt")
        return 0
    else:
        print(f"\n✗ Failed: {result.message}", file=sys.stderr)
        return 1


def cmd_preprocess(args: argparse.Namespace) -> int:
    """Preprocess images only."""
    from .preprocessor import ImagePreprocessor

    preprocessor = ImagePreprocessor(resize_target=args.resize_target)
    images = preprocessor.discover_images(Path(args.input))

    if not images:
        print("No images found", file=sys.stderr)
        return 1

    if args.sort_by_name:
        sorted_infos = preprocessor.sort_by_filename(images)
    else:
        sorted_infos = preprocessor.sort_by_timestamp(images)

    output_dir = Path(args.output)
    processed = preprocessor.process_images(
        sorted_infos,
        output_dir,
    )

    print(f"✓ Processed {len(processed)} images to {output_dir}")
    return 0


def cmd_ocr(args: argparse.Namespace) -> int:
    """Run OCR only."""
    from .ocr import OCRProcessor

    ocr = OCRProcessor(
        dots_ocr_path=Path(args.dots_ocr) if args.dots_ocr else None,
        backend=args.backend,
        num_threads=args.threads,
    )

    input_dir = Path(args.input)
    images = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))

    if not images:
        print("No images found", file=sys.stderr)
        return 1

    output_dir = Path(args.output)
    results = ocr.process_images(images, output_dir)

    successful = sum(1 for r in results if r.success)
    print(f"✓ OCR complete: {successful}/{len(results)} pages successful")

    if successful < len(results):
        print(f"⚠ {len(results) - successful} pages failed")
        return 1

    return 0


def cmd_join(args: argparse.Namespace) -> int:
    """Join OCR output pages."""
    from .page_joiner import PageJoiner
    from .text_cleaner import (
        process_page_from_json_file,
        clean_markdown_text,
        clean_book_pages,
    )

    input_dir = Path(args.input)

    # Prefer JSON files for robust processing (like the full pipeline)
    json_files = sorted(input_dir.glob("**/page_*.json"))

    if json_files:
        pages = [process_page_from_json_file(f) for f in json_files]
    else:
        # Fallback to markdown files
        nohf_files = sorted(input_dir.glob("**/*_nohf.md"))

        if not nohf_files:
            # Try regular .md files
            nohf_files = sorted(input_dir.glob("**/*.md"))

        if not nohf_files:
            print("No markdown or JSON files found", file=sys.stderr)
            return 1

        pages = [clean_markdown_text(f.read_text(encoding="utf-8")) for f in nohf_files]

    # Secondary validation - remove repeated headers/footers
    pages = clean_book_pages(pages)

    joiner = PageJoiner(add_page_markers=not args.no_markers)
    joined, boundaries = joiner.join_pages(pages)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(joined, encoding="utf-8")

    direct_joins = sum(1 for b in boundaries if b.decision.value == "direct")
    print(f"✓ Joined {len(pages)} pages ({direct_joins} sentence continuations)")
    print(f"  Output: {output_path}")

    return 0


def cmd_epub(args: argparse.Namespace) -> int:
    """Convert markdown to EPUB."""
    from .epub_builder import EPUBBuilder, EPUBMetadata

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    content = input_path.read_text(encoding="utf-8")

    metadata = EPUBMetadata(
        title=args.title,
        author=args.author,
        language=args.language,
    )

    builder = EPUBBuilder(metadata)
    output_path = Path(args.output)
    builder.build(content, output_path, split_chapters=not args.no_split)

    print(f"✓ EPUB created: {output_path}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze OCR output for errors."""
    from .error_detector import ErrorDetector

    input_dir = Path(args.input)

    # Find markdown files
    md_files = sorted(input_dir.glob("**/*_nohf.md"))
    if not md_files:
        md_files = sorted(input_dir.glob("**/*.md"))

    if not md_files:
        print("No markdown files found", file=sys.stderr)
        return 1

    pages = [f.read_text(encoding="utf-8") for f in md_files]

    detector = ErrorDetector()
    report = detector.analyze_book(pages)

    print(report.summary())

    if args.output:
        Path(args.output).write_text(report.summary())
        print(f"\nReport saved to: {args.output}")

    return 0 if not report.has_critical_errors else 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="bookpipeline",
        description="Convert photographed book pages to EPUB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # process command (full pipeline)
    p_process = subparsers.add_parser(
        "process",
        help="Run full pipeline: images -> EPUB",
        description="Process photos of book pages into markdown and EPUB",
    )
    p_process.add_argument("input", help="Input directory with images")
    p_process.add_argument("-o", "--output", default="./output", help="Output directory")
    p_process.add_argument("-t", "--title", required=True, help="Book title")
    p_process.add_argument("-a", "--author", default="Unknown", help="Book author")
    p_process.add_argument("-l", "--language", default="en", help="Language code (en, es, zh, etc.)")
    p_process.add_argument("--sort-by-name", action="store_true", help="Sort by filename instead of timestamp")
    p_process.add_argument("--resize-target", type=int, default=1150,
                          help="Resize images so short edge = this value (0 to disable, default: 1150)")
    p_process.add_argument("--backend", choices=["hf", "vllm"], default="hf", help="OCR backend")
    p_process.add_argument("--threads", type=int, default=8, help="OCR threads")
    p_process.add_argument("--dots-ocr", help="Path to dots.ocr directory")
    p_process.add_argument(
        "--formats",
        nargs="+",
        default=["md", "epub"],
        choices=["md", "epub"],
        help="Output formats",
    )
    p_process.set_defaults(func=cmd_process)

    # preprocess command
    p_preprocess = subparsers.add_parser(
        "preprocess",
        help="Preprocess images (sort by timestamp)",
    )
    p_preprocess.add_argument("input", help="Input directory with images")
    p_preprocess.add_argument("-o", "--output", default="./processed", help="Output directory")
    p_preprocess.add_argument("--sort-by-name", action="store_true", help="Sort by filename instead of timestamp")
    p_preprocess.add_argument("--resize-target", type=int, default=1150,
                             help="Resize images so short edge = this value (0 to disable, default: 1150)")
    p_preprocess.set_defaults(func=cmd_preprocess)

    # ocr command
    p_ocr = subparsers.add_parser(
        "ocr",
        help="Run OCR on images",
    )
    p_ocr.add_argument("input", help="Directory with preprocessed images")
    p_ocr.add_argument("-o", "--output", default="./ocr_output", help="Output directory")
    p_ocr.add_argument("--backend", choices=["hf", "vllm"], default="hf", help="OCR backend")
    p_ocr.add_argument("--threads", type=int, default=8, help="Number of threads")
    p_ocr.add_argument("--dots-ocr", help="Path to dots.ocr directory")
    p_ocr.set_defaults(func=cmd_ocr)

    # join command
    p_join = subparsers.add_parser(
        "join",
        help="Join OCR pages into single document",
    )
    p_join.add_argument("input", help="Directory with OCR output")
    p_join.add_argument("-o", "--output", default="./book.md", help="Output markdown file")
    p_join.add_argument("--no-markers", action="store_true", help="Don't add page markers")
    p_join.set_defaults(func=cmd_join)

    # epub command
    p_epub = subparsers.add_parser(
        "epub",
        help="Convert markdown to EPUB",
    )
    p_epub.add_argument("input", help="Input markdown file")
    p_epub.add_argument("-o", "--output", default="./book.epub", help="Output EPUB file")
    p_epub.add_argument("-t", "--title", required=True, help="Book title")
    p_epub.add_argument("-a", "--author", default="Unknown", help="Book author")
    p_epub.add_argument("-l", "--language", default="en", help="Language code")
    p_epub.add_argument("--no-split", action="store_true", help="Don't split into chapters")
    p_epub.set_defaults(func=cmd_epub)

    # analyze command
    p_analyze = subparsers.add_parser(
        "analyze",
        help="Analyze OCR output for errors",
    )
    p_analyze.add_argument("input", help="Directory with OCR output")
    p_analyze.add_argument("-o", "--output", help="Save report to file")
    p_analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    setup_logging(args.verbose)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
