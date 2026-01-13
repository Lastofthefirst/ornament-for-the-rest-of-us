"""
Image preprocessing: sorting by timestamp.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps
from PIL.ExifTags import TAGS

logger = logging.getLogger(__name__)


@dataclass
class ImageInfo:
    """Information about an image file."""

    path: Path
    timestamp: datetime | None


class ImagePreprocessor:
    """Handles image sorting by timestamp."""

    def __init__(
        self,
        supported_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
    ) -> None:
        self.supported_extensions = supported_extensions

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

        Args:
            image_path: Path to image file

        Returns:
            ImageInfo with timestamp
        """
        timestamp = None

        try:
            with Image.open(image_path) as img:
                exif = img._getexif()

                if exif:
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if tag == "DateTimeOriginal":
                            try:
                                timestamp = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                            except ValueError:
                                pass
                            break

        except Exception as e:
            logger.warning(f"Could not read EXIF from {image_path}: {e}")

        # If no EXIF timestamp, use file modification time
        if timestamp is None:
            timestamp = datetime.fromtimestamp(image_path.stat().st_mtime)

        return ImageInfo(path=image_path, timestamp=timestamp)

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

    def process_images(
        self,
        image_infos: list[ImageInfo],
        output_dir: Path,
    ) -> list[Path]:
        """Process images: apply EXIF transpose and save with sequential names.

        Args:
            image_infos: Sorted list of ImageInfo
            output_dir: Directory to save processed images

        Returns:
            List of processed image paths in order
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        processed = []

        for idx, info in enumerate(image_infos):
            # Sequential naming: page_0000.jpg, page_0001.jpg, etc.
            output_path = output_dir / f"page_{idx:04d}.jpg"

            # Load, apply EXIF rotation, and save
            with Image.open(info.path) as img:
                # Rotate pixels to match EXIF orientation, then strip EXIF
                img_corrected = ImageOps.exif_transpose(img)
                img_corrected.save(output_path, "JPEG", quality=95)

            processed.append(output_path)

        logger.info(f"Processed {len(processed)} images to {output_dir}")
        return processed
