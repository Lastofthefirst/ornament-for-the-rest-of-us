# BookPipeline

Convert photographed book pages to EPUB with intelligent text processing. A complete pipeline for digitizing physical books using OCR with advanced text cleaning and error detection.

## Features

### Core Pipeline
- **Photo Processing**: Sort photos by capture timestamp (parsed from filename or EXIF), auto-rotate misoriented pages
- **High-Quality OCR**: Integration with [dots.ocr](https://github.com/rednote-hilab/dots.ocr) using HuggingFace or vLLM backend
- **Smart Text Cleaning**:
  - Dictionary-validated dehyphenation (joins words broken across lines only if valid)
  - Intelligent line unwrapping within paragraphs
  - OCR hallucination detection (removes 10+ consecutive repetitions)
  - Running header detection (removes section headers misclassified by OCR)
  - Repeated header/footer removal (catches what JSON categories miss)
- **Error Detection**: Automatic detection of OCR failures, garbled text, and quality issues
- **Smart Page Joining**: Intelligently joins sentences broken across page boundaries
- **EPUB Generation**: Creates properly formatted EPUB with automatic chapter detection
- **Resume Support**: Interrupted processing can be resumed without reprocessing completed pages

### Text Processing Intelligence

**Dehyphenation**: When a word is broken across lines with a hyphen, the pipeline checks if joining creates a valid English word using system dictionaries. Only valid words are joined.

```
Input:  "The quick brown fox jumps over the com-\nputer system"
Output: "The quick brown fox jumps over the computer system"
```

**Hallucination Cleaning**: Detects when OCR gets stuck repeating the same phrase 10+ times (often from system sleep/wake) and removes duplicates.

```
Input:  "The Washington Post Washington Post Washington Post [×15] reported"
Output: "The Washington Post reported"
```

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

1. **Python 3.11+**
2. **[uv](https://github.com/astral-sh/uv)** package manager
3. **dots.ocr** set up (see [dots.ocr setup](#dotsocr-setup))
4. **System dictionary** (optional, for dehyphenation):
   - Linux: Usually pre-installed at `/usr/share/dict/words`
   - macOS: Pre-installed at `/usr/share/dict/words`
   - Without dictionary, dehyphenation will be conservative

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

# For vLLM backend (recommended for speed)
uv pip install vllm
```

## Usage

### Full Pipeline

Process a folder of photos into an EPUB:

```bash
bookpipeline process ./photos \
    --title "My Book" \
    --author "Author Name" \
    --output ./output \
    --backend vllm
```

**Options:**
- `--title` (required): Book title
- `--author`: Book author (default: "Unknown")
- `--output`: Output directory (default: ./output)
- `--language`: Language code like en, es, zh (default: en)
- `--sort-by-name`: Sort by filename instead of photo timestamp
- `--formats`: Output formats: md, epub, or both (default: both)
- `--backend`: OCR backend: hf (HuggingFace) or vllm (default: hf)
- `--threads`: OCR processing threads (default: 8)
- `--dots-ocr`: Path to dots.ocr directory (auto-detected if in ./dots.ocr)

**Resume Support:**
If processing is interrupted, simply run the same command again. The pipeline will:
- Skip already preprocessed images
- Skip pages with existing OCR output
- Continue from where it left off

### Individual Steps

Run each step separately for more control:

```bash
# 1. Preprocess images (sort by timestamp, fix rotation)
bookpipeline preprocess ./raw_photos --output ./sorted_photos

# 2. Run OCR
bookpipeline ocr ./sorted_photos --output ./ocr_output --backend vllm

# 3. Analyze for errors
bookpipeline analyze ./ocr_output --output ./error_report.txt

# 4. Join pages into single document
bookpipeline join ./ocr_output --output ./book.md

# 5. Convert to EPUB
bookpipeline epub ./book.md --title "My Book" --output ./book.epub
```

### Python API

```python
from bookpipeline import BookPipeline, PipelineConfig
from pathlib import Path

config = PipelineConfig(
    input_dir=Path("./photos"),
    output_dir=Path("./output"),
    book_title="My Book",
    book_author="Author Name",
    ocr_backend="vllm",  # or "hf"
)

pipeline = BookPipeline(config)
result = pipeline.run()

if result.success:
    print(f"EPUB created: {result.epub_path}")
    print(f"Error report: {result.error_report.summary()}")
else:
    print(f"Failed: {result.message}")
```

## Workflow

```
┌─────────────────┐
│  Photo Pages    │  JPG/PNG from camera with timestamp in filename
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessor   │  Parse timestamps (IMG_20240115_143022.jpg)
│                 │  Sort chronologically, apply EXIF rotation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    dots.ocr     │  Extract text + layout (HF or vLLM backend)
│                 │  Produces JSON with categories (Text, Page-header, etc)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Cleaner   │  Dehyphenate with dictionary validation
│                 │  Unwrap lines, detect hallucinations (10+ reps)
│                 │  Remove headers/footers by category + repetition
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Error Detector  │  Flag OCR failures, garbled text, quality issues
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Page Joiner    │  Fix sentences broken across page boundaries
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ EPUB Generator  │  Create formatted ebook with chapters
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   .md + .epub   │  Final outputs + error_report.txt
└─────────────────┘
```

## Project Structure

```
OCRSample/
├── bookpipeline/              # Main package
│   ├── __init__.py
│   ├── cli.py                 # Command-line interface
│   ├── config.py              # Configuration dataclass
│   ├── pipeline.py            # Main orchestrator
│   ├── preprocessor.py        # Image sorting & rotation
│   ├── ocr.py                 # dots.ocr wrapper (HF/vLLM)
│   ├── text_cleaner.py        # Dehyphenation, hallucination detection
│   ├── page_joiner.py         # Smart page boundary joining
│   ├── error_detector.py      # OCR quality analysis
│   ├── epub_builder.py        # EPUB generation
│   └── progress.py            # Terminal progress reporting
├── dots.ocr/                  # OCR engine (git clone)
├── inputs/                    # Your input photos
├── outputs/                   # Generated output files
├── pyproject.toml             # Package configuration
└── README.md
```

## Error Detection

The pipeline automatically detects common OCR issues:

| Error Type | Description | Severity |
|------------|-------------|----------|
| `OCR_FAILED` | OCR processing completely failed | ERROR |
| `SHORT_PAGE` | Page has suspiciously little text (< 100 chars) | WARNING |
| `HIGH_GARBAGE_RATIO` | High ratio of unusual characters | WARNING |
| `HIGH_SPECIAL_CHAR_RATIO` | Too many special characters | WARNING |
| `REPEATED_CHARS` | Same character repeated 5+ times consecutively | INFO |
| `OCR_CONFUSION` | Common OCR mistakes (0/O, 1/l/I, rn/m, vv/w) | INFO |
| `SHORT_WORDS` | Average word length too short (gibberish) | WARNING |
| `UNBALANCED_QUOTES` | Odd number of quotes (missing pair) | INFO |
| `UNBALANCED_BRACKETS` | Unmatched brackets/parentheses | INFO |

Check `error_report.txt` after processing to review flagged pages.

**Note:** Some issues like repeated section headers (running headers misclassified by OCR) are automatically fixed during text cleaning and won't appear in the error report.

## Text Cleaning Features

### Dehyphenation

The pipeline uses dictionary validation to safely dehyphenate words broken across lines:

**Safe dehyphenation:**
```
"under-\nstanding" → "understanding"  (valid word)
"com-\nputer" → "computer"  (valid word)
```

**Preserved compound words:**
```
"self-\ndriving" → "self-driving"  (invalid when joined)
"mother-\nin-law" → "mother-in-law"  (compound word)
```

Dictionary sources (checked in order):
1. `/usr/share/dict/words`
2. `/usr/share/dict/american-english`
3. `/usr/share/dict/british-english`

### Hallucination Detection

Detects and removes OCR hallucinations where the same phrase repeats 10+ times consecutively:

```python
# Detects minimum 10 consecutive repetitions
# Minimum phrase length: 15 characters
# Keeps one copy, removes rest
```

This commonly occurs when:
- System goes to sleep during OCR processing
- GPU issues cause inference loops
- Page has unusual layout confusing the model

### Header/Footer Removal

Three-stage approach:
1. **JSON categories**: Removes blocks marked as `Page-header`, `Page-footer`, `Page-number` by OCR
2. **Repeated section headers**: Detects section headers (# or ##) that repeat within a 20-page window 3+ times - these are running headers that OCR misclassified. Keeps the first occurrence (actual section title), removes subsequent ones
3. **Repetition analysis**: Detects phrases appearing on 10+ pages at top/bottom

**Example:**
```
Page 5:  # Introduction          ← Kept (first occurrence, actual section title)
Page 7:  ## INTRODUCTION         ← Removed (running header)
Page 9:  ## INTRODUCTION         ← Removed (running header)
Page 11: ## INTRODUCTION         ← Removed (running header)
```

Keeps unique section headers like "Chapter 1", "Preface", etc. that only appear once.

## Photo Naming Best Practices

For best results with timestamp sorting, use photos with timestamps in filename:

**Supported patterns:**
- `IMG_YYYYMMDD_HHMMSS.jpg` (Android, most cameras)
- `YYYYMMDD_HHMMSS.jpg`
- `YYYY-MM-DD_HH-MM-SS.jpg`
- `PXL_YYYYMMDD_HHMMSS.jpg` (Google Pixel)

**Examples:**
- `IMG_20240115_143022.jpg` → parsed as Jan 15, 2024, 14:30:22
- `20240115_143022.jpg` → parsed as Jan 15, 2024, 14:30:22
- `PXL_20240115_143022.jpg` → parsed as Jan 15, 2024, 14:30:22

**Fallback order:**
1. EXIF `DateTimeOriginal` tag
2. Timestamp in filename (patterns above)
3. File modification time (unreliable if files were copied)

## Tips for Best Results

### Photography
1. **Good lighting**: Even, diffuse lighting without harsh shadows
2. **Flat pages**: Press book flat or use a book scanner stand
3. **Consistent angle**: Keep camera perpendicular to page
4. **High resolution**: Use at least 300 DPI equivalent (8-12 MP camera)
5. **Clean pages**: Avoid shadows, fingers, or reflections
6. **Page order**: Name photos with timestamps or ensure EXIF data is preserved

### Processing
1. **Use vLLM backend**: Faster inference with `--backend vllm`
2. **Adjust threads**: More threads for faster processing (if you have RAM)
3. **Check error report**: Review `error_report.txt` for pages needing attention
4. **Resume processing**: Interrupted runs can be resumed by running the same command

## Troubleshooting

### "dots.ocr not found"

Specify the path explicitly:
```bash
bookpipeline process ./photos --title "Book" --dots-ocr /path/to/dots.ocr
```

Or create a symlink:
```bash
ln -s /path/to/dots.ocr ./dots.ocr
```

### OCR is slow

- Use vLLM backend: `--backend vllm` (much faster than HuggingFace)
- Increase threads: `--threads 16` (if you have the RAM)
- Check GPU utilization with `nvidia-smi`

### Many pages have errors

- Check image quality and lighting
- Review `error_report.txt` for specific issues
- Re-photograph problematic pages
- Verify pages are in correct order (check timestamps)

### Wrong page order

Use `--sort-by-name` if filenames indicate correct order:
```bash
bookpipeline process ./photos --title "Book" --sort-by-name
```

Or rename files with proper timestamps:
```bash
# Rename files to include timestamps
for i in page_*.jpg; do
    num=$(echo $i | grep -oP '\d+')
    mv "$i" "IMG_20240115_$(printf '%06d' $num).jpg"
done
```

### Table of Contents in wrong place

This usually means photos were not in chronological order. The pipeline sorts by:
1. Filename timestamp (IMG_YYYYMMDD_HHMMSS pattern)
2. EXIF timestamp
3. File modification time (unreliable)

To fix:
1. Delete the `output/preprocessed/` directory
2. Ensure photos have correct timestamps in filenames
3. Re-run the pipeline

### Repeated section titles (running headers)

If you see section titles like "## INTRODUCTION" appearing multiple times throughout a chapter, these are running headers that OCR incorrectly categorized as section headers instead of page headers.

The pipeline automatically detects and removes these:
- Detects headers that repeat 3+ times within a 20-page window
- Keeps the first occurrence (the actual section title)
- Removes subsequent occurrences (running headers)

This happens automatically in the `clean_book_pages()` function. No manual intervention needed.

### Repeated text (hallucinations)

The pipeline automatically detects and removes repetitions of 10+ occurrences. If you see shorter repetitions, they're likely legitimate (e.g., poetry, structured text).

To adjust the threshold, modify `text_cleaner.py`:
```python
clean_repetition_hallucination(text, min_repeats=5)  # More aggressive
```

### Missing/broken hyphenated words

The dehyphenation uses a system dictionary. If many valid words aren't being joined:

1. Check that `/usr/share/dict/words` exists
2. Install a dictionary package:
   ```bash
   # Debian/Ubuntu
   sudo apt-get install wamerican

   # RHEL/CentOS
   sudo yum install words
   ```

## Advanced Usage

### Custom OCR Backend

```python
from bookpipeline.ocr import OCRProcessor

# Use vLLM with custom settings
processor = OCRProcessor(
    backend="vllm",
    num_threads=16,
    dots_ocr_path=Path("./dots.ocr")
)
```

### Custom Text Cleaning

```python
from bookpipeline.text_cleaner import (
    clean_text_block,
    clean_repetition_hallucination,
    detect_repeated_headers_footers,
    detect_repeated_section_headers,
    remove_repeated_section_headers
)

# Clean individual text block
cleaned = clean_text_block(raw_ocr_text)

# Detect hallucinations
cleaned, had_hallucination = clean_repetition_hallucination(
    text,
    min_repeats=10
)

# Find repeated headers/footers at top/bottom of pages
pages = ["page 1 text...", "page 2 text...", ...]
headers, footers = detect_repeated_headers_footers(
    pages,
    min_occurrences=10
)

# Find repeated section headers (running headers)
repeated_headers = detect_repeated_section_headers(
    pages,
    min_occurrences=3,
    window_size=20
)
pages = remove_repeated_section_headers(pages, repeated_headers)
```

### Custom Error Detection

```python
from bookpipeline.error_detector import ErrorDetector

detector = ErrorDetector()
report = detector.analyze_book(pages)

# Check specific pages
for error in report.errors:
    if error.page_number == 42:
        print(f"{error.error_type}: {error.message}")
```

## Performance

Approximate processing times on typical hardware (RTX 4090, 64GB RAM):

| Task | Pages | Backend | Time |
|------|-------|---------|------|
| Preprocessing | 250 | N/A | ~30 sec |
| OCR | 250 | HuggingFace | ~45 min |
| OCR | 250 | vLLM | ~15 min |
| Text Cleaning | 250 | N/A | ~5 sec |
| Page Joining | 250 | N/A | ~2 sec |
| EPUB Generation | 250 | N/A | ~1 sec |

**Total (vLLM)**: ~16 minutes for 250-page book

## License

MIT License
