"""
Image preprocessing: sorting by timestamp.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps
from PIL.ExifTags import TAGS

from .progress import ProgressReporter

logger = logging.getLogger(__name__)

# Common filename patterns with embedded timestamps
FILENAME_TIMESTAMP_PATTERNS = [
    # IMG_YYYYMMDD_HHMMSS (Android/common)
    re.compile(r'IMG_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})'),
    # YYYYMMDD_HHMMSS
    re.compile(r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})'),
    # YYYY-MM-DD_HH-MM-SS
    re.compile(r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})'),
    # PXL_YYYYMMDD_HHMMSS (Pixel phones)
    re.compile(r'PXL_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})'),
]


@dataclass
class ImageInfo:
    """Information about an image file."""

    path: Path
    timestamp: datetime | None


class ImagePreprocessor:
    """Handles image sorting, rotation correction, and resizing."""

    def __init__(
        self,
        supported_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
        resize_target: int = 1150,
    ) -> None:
        self.supported_extensions = supported_extensions
        self.resize_target = resize_target  # Target pixels for short edge (0 to disable)

    def discover_images(self, input_dir: Path) -> list[Path]:
        """Find all supported images in directory.

        Args:
            input_dir: Directory to search

        Returns:
            List of image paths (unsorted)
        """
        images = []
        for ext in self.supported_extensions:
            images.extend(input_dir.glob(f"*{ext}"))
            images.extend(input_dir.glob(f"*{ext.upper()}"))

        logger.info(f"Found {len(images)} images in {input_dir}")
        return images

    def get_image_info(self, image_path: Path) -> ImageInfo:
        """Extract timestamp from image.

        Tries in order:
        1. EXIF DateTimeOriginal
        2. Timestamp embedded in filename (common phone patterns)
        3. File modification time (least reliable)

        Args:
            image_path: Path to image file

        Returns:
            ImageInfo with timestamp
        """
        timestamp = None

        # Try EXIF first (using public API)
        try:
            with Image.open(image_path) as img:
                exif = img.getexif()

                if exif:
                    # Check for DateTimeOriginal (tag 36867) directly
                    # or iterate through tags
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if tag == "DateTimeOriginal":
                            try:
                                timestamp = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                            except (ValueError, TypeError):
                                pass
                            break

                    # Also check EXIF IFD for DateTimeOriginal if not found
                    if timestamp is None and hasattr(exif, 'get_ifd'):
                        try:
                            exif_ifd = exif.get_ifd(0x8769)  # EXIF IFD pointer
                            if exif_ifd:
                                dto = exif_ifd.get(36867)  # DateTimeOriginal tag
                                if dto:
                                    timestamp = datetime.strptime(dto, "%Y:%m:%d %H:%M:%S")
                        except (ValueError, TypeError, KeyError):
                            pass

        except Exception as e:
            logger.debug(f"Could not read EXIF from {image_path}: {e}")

        # Try parsing timestamp from filename
        if timestamp is None:
            timestamp = self._parse_filename_timestamp(image_path.name)

        # Last resort: file modification time
        if timestamp is None:
            timestamp = datetime.fromtimestamp(image_path.stat().st_mtime)

        return ImageInfo(path=image_path, timestamp=timestamp)

    def _parse_filename_timestamp(self, filename: str) -> datetime | None:
        """Try to extract timestamp from filename.

        Supports common patterns like IMG_YYYYMMDD_HHMMSS.

        Args:
            filename: Image filename

        Returns:
            Parsed datetime or None
        """
        for pattern in FILENAME_TIMESTAMP_PATTERNS:
            match = pattern.search(filename)
            if match:
                try:
                    year, month, day, hour, minute, second = match.groups()
                    return datetime(
                        int(year), int(month), int(day),
                        int(hour), int(minute), int(second)
                    )
                except (ValueError, TypeError):
                    continue
        return None

    def sort_by_timestamp(self, images: list[Path]) -> list[ImageInfo]:
        """Sort images by their capture timestamp.

        Args:
            images: List of image paths

        Returns:
            List of ImageInfo sorted by timestamp
        """
        infos = [self.get_image_info(img) for img in images]
        sorted_infos = sorted(infos, key=lambda x: x.timestamp or datetime.min)

        logger.info(f"Sorted {len(sorted_infos)} images by timestamp")
        return sorted_infos

    def sort_by_filename(self, images: list[Path]) -> list[ImageInfo]:
        """Sort images by filename (alphabetically).

        Args:
            images: List of image paths

        Returns:
            List of ImageInfo sorted by filename
        """
        infos = [self.get_image_info(img) for img in images]
        sorted_infos = sorted(infos, key=lambda x: x.path.name)

        logger.info(f"Sorted {len(sorted_infos)} images by filename")
        return sorted_infos

    def _resize_to_target(self, img: Image.Image) -> Image.Image:
        """Resize image so short edge matches target.

        Only resizes down, never up. Maintains aspect ratio.

        Args:
            img: PIL Image to resize

        Returns:
            Resized image (or original if no resize needed)
        """
        if self.resize_target <= 0:
            return img

        width, height = img.size
        short_edge = min(width, height)

        if short_edge <= self.resize_target:
            return img

        scale = self.resize_target / short_edge
        new_width = int(width * scale)
        new_height = int(height * scale)

        return img.resize((new_width, new_height), Image.LANCZOS)

    def process_images(
        self,
        image_infos: list[ImageInfo],
        output_dir: Path,
        resume: bool = True,
    ) -> list[Path]:
        """Process images: apply EXIF transpose and save with sequential names.

        Args:
            image_infos: Sorted list of ImageInfo
            output_dir: Directory to save processed images
            resume: If True, skip images that already have output

        Returns:
            List of processed image paths in order
        """
        import sys

        output_dir.mkdir(parents=True, exist_ok=True)
        processed = []
        to_process = []

        # Check which images need processing
        for idx, info in enumerate(image_infos):
            output_path = output_dir / f"page_{idx:04d}.jpg"
            processed.append(output_path)

            if resume and output_path.exists():
                # Validate cache: check if source is newer than output
                try:
                    source_mtime = info.path.stat().st_mtime
                    output_mtime = output_path.stat().st_mtime
                    if source_mtime <= output_mtime:
                        continue  # Cache is valid, skip
                    logger.debug(f"Cache stale for {info.path.name}, reprocessing")
                except OSError:
                    pass  # If stat fails, reprocess
            to_process.append((idx, info, output_path))

        if not to_process:
            sys.stderr.write(f"All {len(image_infos)} images already preprocessed (resume mode)\n")
            sys.stderr.flush()
            return processed

        skipped = len(image_infos) - len(to_process)
        desc = f"Preprocessing ({skipped} cached)" if skipped > 0 else "Preprocessing"

        with ProgressReporter(len(to_process), desc=desc, unit="images") as progress:
            for idx, info, output_path in to_process:
                # Load, apply EXIF rotation, resize, and save
                with Image.open(info.path) as img:
                    # Rotate pixels to match EXIF orientation, then strip EXIF
                    img_corrected = ImageOps.exif_transpose(img)
                    # Resize to target (only if larger than target)
                    img_resized = self._resize_to_target(img_corrected)
                    img_resized.save(output_path, "JPEG", quality=95)

                progress.update(item_name=info.path.name)

        return processed
