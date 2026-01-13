# BookPipeline

Convert photographed book pages to EPUB. A complete pipeline for digitizing physical books.

## Features

- **Photo Processing**: Sort photos by capture timestamp, auto-rotate misoriented pages
- **OCR**: Integration with [dots.ocr](https://github.com/rednote-hilab/dots.ocr) for high-quality text extraction
- **Error Detection**: Automatic detection of OCR failures, garbled text, and quality issues
- **Smart Page Joining**: Intelligently joins text broken across page boundaries
- **EPUB Generation**: Creates properly formatted EPUB with automatic chapter detection

## Quick Start

```bash
# Install with uv
uv pip install -e .

# Full pipeline: photos -> EPUB
bookpipeline process ./my_photos --title "Book Title" --author "Author Name"

# Output will be in ./output/
#   - Book_Title.md      (Markdown)
#   - Book_Title.epub    (EPUB)
#   - error_report.txt   (Quality report)
```

## Installation

### Prerequisites

1. Python 3.11+
2. [uv](https://github.com/astral-sh/uv) package manager
3. dots.ocr set up (see [dots.ocr setup](#dotsocr-setup))

### Install BookPipeline

```bash
cd /path/to/OCRSample
uv pip install -e .
```

### dots.ocr Setup

BookPipeline uses dots.ocr for OCR processing. Set it up first:

```bash
# Clone dots.ocr
git clone https://github.com/rednote-hilab/dots.ocr.git

# Install PyTorch (adjust for your CUDA version)
uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Install dots.ocr dependencies
cd dots.ocr
uv pip install transformers==4.51.3 qwen_vl_utils huggingface_hub PyMuPDF accelerate
uv pip install deepspeed==0.15.4
uv pip install -e . --no-build-isolation

# Download model weights
python3 tools/download_model.py
```

## Usage

### Full Pipeline

Process a folder of photos into an EPUB:

```bash
bookpipeline process ./photos \
    --title "My Book" \
    --author "Author Name" \
    --output ./output
```

Options:
- `--title` (required): Book title
- `--author`: Book author (default: "Unknown")
- `--output`: Output directory (default: ./output)
- `--language`: Language code like en, es, zh (default: en)
- `--no-rotate`: Skip auto-rotation correction
- `--sort-by-name`: Sort by filename instead of photo timestamp
- `--formats`: Output formats: md, epub, or both (default: both)
- `--threads`: OCR processing threads (default: 8)

### Individual Steps

Run each step separately for more control:

```bash
# 1. Preprocess images (sort by timestamp, fix rotation)
bookpipeline preprocess ./raw_photos --output ./sorted_photos

# 2. Run OCR
bookpipeline ocr ./sorted_photos --output ./ocr_output

# 3. Analyze for errors
bookpipeline analyze ./ocr_output

# 4. Join pages into single document
bookpipeline join ./ocr_output --output ./book.md

# 5. Convert to EPUB
bookpipeline epub ./book.md --title "My Book" --output ./book.epub
```

### Python API

```python
from bookpipeline import BookPipeline, PipelineConfig

config = PipelineConfig(
    input_dir="./photos",
    output_dir="./output",
    book_title="My Book",
    book_author="Author Name",
)

pipeline = BookPipeline(config)
result = pipeline.run()

if result.success:
    print(f"EPUB created: {result.epub_path}")
else:
    print(f"Failed: {result.message}")
```

## Workflow

```
┌─────────────────┐
│  Photo Pages    │  (JPG/PNG files from camera)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessor   │  Sort by timestamp, auto-rotate
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    dots.ocr     │  Extract text from images
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Error Detector  │  Flag OCR failures, quality issues
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Page Joiner    │  Fix sentences broken across pages
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ EPUB Generator  │  Create formatted ebook
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   .md + .epub   │  Final outputs
└─────────────────┘
```

## Project Structure

```
OCRSample/
├── bookpipeline/           # Main package
│   ├── __init__.py
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration dataclass
│   ├── pipeline.py         # Main orchestrator
│   ├── preprocessor.py     # Image sorting & rotation
│   ├── ocr.py              # dots.ocr wrapper
│   ├── page_joiner.py      # Smart page joining
│   ├── error_detector.py   # OCR quality analysis
│   └── epub_builder.py     # EPUB generation
├── dots.ocr/               # OCR engine (git clone)
├── inputs/                 # Your input photos/PDFs
├── outputs/                # Generated output files
├── pyproject.toml          # Package configuration
└── README.md
```

## Error Detection

The pipeline automatically detects common OCR issues:

| Error Type | Description |
|------------|-------------|
| `OCR_FAILED` | OCR processing completely failed |
| `SHORT_PAGE` | Page has suspiciously little text |
| `GARBAGE_CHARS` | High ratio of unusual characters |
| `REPEATED_CHARS` | Same character repeated many times |
| `OCR_CONFUSION` | Common OCR mistakes (0/O, 1/l/I, rn/m) |
| `SHORT_WORDS` | Average word length too short (gibberish) |

Check `error_report.txt` after processing to review flagged pages.

## Tips for Best Results

1. **Good lighting**: Photograph pages with even, diffuse lighting
2. **Flat pages**: Press book flat or use a book scanner
3. **Consistent angle**: Keep camera perpendicular to page
4. **High resolution**: Use at least 300 DPI equivalent
5. **Clean pages**: Avoid shadows, fingers, or reflections

## Troubleshooting

### "dots.ocr not found"

Specify the path explicitly:
```bash
bookpipeline process ./photos --title "Book" --dots-ocr /path/to/dots.ocr
```

### OCR is slow

- Use `--threads 16` for more parallelism (if you have the RAM)
- Consider using vLLM backend for faster inference: `--backend vllm`

### Many pages have errors

- Check image quality and lighting
- Review `error_report.txt` for specific issues
- Re-photograph problematic pages

### Wrong page order

- Use `--sort-by-name` if filenames are in correct order
- Check that camera's date/time was set correctly

## License

MIT License
