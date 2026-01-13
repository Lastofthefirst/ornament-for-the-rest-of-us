"""
BookPipeline - Convert photographed book pages to EPUB

A complete pipeline for:
1. Processing photos of book pages (sorted by capture time)
2. Auto-rotating misoriented images
3. Running OCR via dots.ocr
4. Detecting and reporting OCR errors
5. Intelligently joining pages across boundaries
6. Generating Markdown and EPUB output
"""

__version__ = "1.0.0"
__author__ = "BookPipeline"

from .pipeline import BookPipeline
from .config import PipelineConfig

__all__ = ["BookPipeline", "PipelineConfig"]
