"""
OCR wrapper for dots.ocr integration.

Supports both vLLM (recommended) and HuggingFace backends.
When using vLLM, automatically starts and stops the server.
"""

import atexit
import json
import logging
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import weakref
from dataclasses import dataclass
from pathlib import Path

from .progress import ProgressReporter

logger = logging.getLogger(__name__)

# Track active vLLM servers for cleanup on exit/signal
_active_servers: weakref.WeakSet["VLLMServer"] = weakref.WeakSet()
_original_sigterm = None
_original_sigint = None


def _cleanup_servers() -> None:
    """Clean up any active vLLM servers."""
    for server in list(_active_servers):
        try:
            server.stop()
        except Exception as e:
            logger.debug(f"Error stopping server during cleanup: {e}")


def _signal_handler(signum, frame):
    """Handle termination signals by cleaning up servers first."""
    logger.info(f"Received signal {signum}, cleaning up...")
    _cleanup_servers()
    # Call original handler if it exists
    if signum == signal.SIGTERM and _original_sigterm:
        _original_sigterm(signum, frame)
    elif signum == signal.SIGINT and _original_sigint:
        _original_sigint(signum, frame)
    else:
        sys.exit(128 + signum)


def _setup_cleanup_handlers() -> None:
    """Set up atexit and signal handlers for cleanup."""
    global _original_sigterm, _original_sigint

    # Register atexit handler
    atexit.register(_cleanup_servers)

    # Register signal handlers (preserve originals)
    try:
        _original_sigterm = signal.signal(signal.SIGTERM, _signal_handler)
        _original_sigint = signal.signal(signal.SIGINT, _signal_handler)
    except (ValueError, OSError):
        # Signal handling may fail in some contexts (e.g., non-main thread)
        pass


# Set up handlers on module load
_setup_cleanup_handlers()

# Constants
VLLM_STARTUP_TIMEOUT = 300  # seconds to wait for vLLM server
VLLM_HEALTH_CHECK_INTERVAL = 2  # seconds between health checks
VLLM_PROGRESS_LOG_INTERVAL = 30  # seconds between "still waiting" logs
INFERENCE_MAX_RETRIES = 3  # retries for failed inference calls
INFERENCE_RETRY_DELAY = 2  # seconds between retries


@dataclass
class OCRResult:
    """Result from OCR processing of a single page."""

    page_number: int
    image_path: Path
    text: str  # Full text with headers/footers
    text_nohf: str  # Text without headers/footers
    json_data: dict | None  # Raw JSON layout data
    success: bool
    error_message: str | None = None


class VLLMServer:
    """Manages the vLLM server lifecycle."""

    def __init__(self, model_path: str, port: int = 8000, gpu_memory: float = 0.9):
        self.model_path = model_path
        self.port = port
        self.gpu_memory = gpu_memory
        self.process: subprocess.Popen | None = None

    def start(self, timeout: int = VLLM_STARTUP_TIMEOUT) -> bool:
        """Start the vLLM server and wait for it to be ready.

        Args:
            timeout: Maximum seconds to wait for server to start

        Returns:
            True if server started successfully
        """
        if self.is_running():
            logger.info("vLLM server already running")
            return True

        # Register for cleanup on exit/signal
        _active_servers.add(self)

        logger.info(f"Starting vLLM server with model: {self.model_path}")

        cmd = [
            "vllm", "serve", self.model_path,
            "--trust-remote-code",
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory),
            "--served-model-name", "model",
        ]
        logger.info(f"vLLM command: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        start_time = time.time()
        last_log = start_time
        while time.time() - start_time < timeout:
            if self.is_running():
                logger.info(f"vLLM server ready on port {self.port}")
                return True

            if self.process.poll() is not None:
                output = self.process.stdout.read() if self.process.stdout else ""
                logger.error(f"vLLM server failed to start:\n{output[-2000:]}")
                return False

            if time.time() - last_log > VLLM_PROGRESS_LOG_INTERVAL:
                elapsed = int(time.time() - start_time)
                logger.info(f"Waiting for vLLM server... ({elapsed}s elapsed)")
                last_log = time.time()

            time.sleep(VLLM_HEALTH_CHECK_INTERVAL)

        logger.error(f"vLLM server failed to start within {timeout}s")
        self.stop()
        return False

    def stop(self) -> None:
        """Stop the vLLM server."""
        if self.process:
            logger.info("Stopping vLLM server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            logger.info("vLLM server stopped")

        # Remove from cleanup tracking
        _active_servers.discard(self)

    def is_running(self) -> bool:
        """Check if the vLLM server is responding."""
        try:
            url = f"http://localhost:{self.port}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class OCRProcessor:
    """Wrapper for dots.ocr processing.

    Supports vLLM (recommended) and HuggingFace backends.
    """

    def __init__(
        self,
        dots_ocr_path: Path | None = None,
        backend: str = "vllm",
        num_threads: int = 16,
        vllm_port: int = 8000,
    ) -> None:
        """Initialize OCR processor.

        Args:
            dots_ocr_path: Path to dots.ocr directory. If None, tries to find it.
            backend: 'vllm' (recommended) or 'hf' for HuggingFace
            num_threads: Number of processing threads (for vLLM)
            vllm_port: Port for vLLM server
        """
        self.dots_ocr_path = dots_ocr_path or self._find_dots_ocr()
        self.backend = backend
        self.num_threads = num_threads
        self.vllm_port = vllm_port
        self._server: VLLMServer | None = None

        if not self.dots_ocr_path or not self.dots_ocr_path.exists():
            raise RuntimeError(
                f"dots.ocr not found at {self.dots_ocr_path}. "
                "Please provide the correct path or install dots.ocr."
            )

        # Add dots.ocr to path so we can import it
        dots_ocr_str = str(self.dots_ocr_path)
        if dots_ocr_str not in sys.path:
            sys.path.insert(0, dots_ocr_str)

        self.weights_path = self.dots_ocr_path / "weights" / "DotsOCR"
        if not self.weights_path.exists():
            raise RuntimeError(f"Model weights not found at {self.weights_path}")

        logger.info(f"OCR processor initialized (backend={backend})")

    def _find_dots_ocr(self) -> Path | None:
        """Try to find dots.ocr in common locations."""
        candidates = [
            Path.cwd() / "dots.ocr",
            Path.cwd().parent / "dots.ocr",
            Path(__file__).parent.parent / "dots.ocr",
        ]

        for candidate in candidates:
            if candidate.exists() and (candidate / "dots_ocr" / "parser.py").exists():
                return candidate

        return None

    def _start_vllm_server(self) -> bool:
        """Start the vLLM server if not running."""
        self._server = VLLMServer(
            model_path=str(self.weights_path),
            port=self.vllm_port,
        )
        return self._server.start()

    def _stop_vllm_server(self) -> None:
        """Stop the vLLM server if we started it."""
        if self._server:
            self._server.stop()
            self._server = None

    def process_images(
        self,
        image_paths: list[Path],
        output_dir: Path,
        resume: bool = True,
    ) -> list[OCRResult]:
        """Process multiple images through OCR.

        Args:
            image_paths: List of image paths to process (in order)
            output_dir: Directory for OCR output files
            resume: If True, skip pages that already have OCR output

        Returns:
            List of OCRResult objects in same order as input
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Suppress noisy logging from libraries during processing
        logging.getLogger("dots_ocr").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("vllm").setLevel(logging.WARNING)

        if self.backend == "vllm":
            return self._process_with_vllm(image_paths, output_dir, resume)
        else:
            return self._process_with_hf(image_paths, output_dir, resume)

    def _process_with_vllm(
        self,
        image_paths: list[Path],
        output_dir: Path,
        resume: bool = True,
    ) -> list[OCRResult]:
        """Process images using vLLM backend."""
        # Check which pages need processing
        to_process, results_dict = self._check_existing_results(
            image_paths, output_dir, resume
        )

        if not to_process:
            import sys
            sys.stderr.write(f"All {len(image_paths)} pages already processed (resume mode)\n")
            sys.stderr.flush()
            return self._results_dict_to_list(results_dict, image_paths)

        if not self._start_vllm_server():
            error_msg = "Failed to start vLLM server"
            logger.error(error_msg)
            # Fill in errors for unprocessed pages
            for idx, path in to_process:
                results_dict[idx] = OCRResult(
                    page_number=idx,
                    image_path=path,
                    text="",
                    text_nohf="",
                    json_data=None,
                    success=False,
                    error_message=error_msg,
                )
            return self._results_dict_to_list(results_dict, image_paths)

        try:
            from dots_ocr.parser import DotsOCRParser

            parser = DotsOCRParser(
                port=self.vllm_port,
                num_thread=self.num_threads,
                use_hf=False,
            )

            skipped = len(image_paths) - len(to_process)
            desc = f"OCR ({skipped} cached)" if skipped > 0 else "OCR"

            with ProgressReporter(len(to_process), desc=desc, unit="pages") as progress:
                for idx, image_path in to_process:
                    result = self._process_single_image(parser, idx, image_path, output_dir)
                    results_dict[idx] = result
                    progress.update(success=result.success, item_name=image_path.name)

            return self._results_dict_to_list(results_dict, image_paths)

        finally:
            self._stop_vllm_server()

    def _process_with_hf(
        self,
        image_paths: list[Path],
        output_dir: Path,
        resume: bool = True,
    ) -> list[OCRResult]:
        """Process images using HuggingFace backend."""
        import sys

        # Check which pages need processing
        to_process, results_dict = self._check_existing_results(
            image_paths, output_dir, resume
        )

        if not to_process:
            sys.stderr.write(f"All {len(image_paths)} pages already processed (resume mode)\n")
            sys.stderr.flush()
            return self._results_dict_to_list(results_dict, image_paths)

        try:
            from dots_ocr.parser import DotsOCRParser

            sys.stderr.write("Loading OCR model (this may take a minute)...\n")
            sys.stderr.flush()
            parser = DotsOCRParser(use_hf=True, num_thread=1)
            sys.stderr.write("OCR model loaded\n")
            sys.stderr.flush()

            skipped = len(image_paths) - len(to_process)
            desc = f"OCR ({skipped} cached)" if skipped > 0 else "OCR"

            with ProgressReporter(len(to_process), desc=desc, unit="pages") as progress:
                for idx, image_path in to_process:
                    result = self._process_single_image(parser, idx, image_path, output_dir)
                    results_dict[idx] = result
                    progress.update(success=result.success, item_name=image_path.name)

            return self._results_dict_to_list(results_dict, image_paths)

        except Exception as e:
            error_msg = f"Failed to load OCR model: {e}"
            logger.error(error_msg)
            # Fill in errors for unprocessed pages
            for idx, path in to_process:
                results_dict[idx] = OCRResult(
                    page_number=idx,
                    image_path=path,
                    text="",
                    text_nohf="",
                    json_data=None,
                    success=False,
                    error_message=error_msg,
                )
            return self._results_dict_to_list(results_dict, image_paths)

    def _check_existing_results(
        self,
        image_paths: list[Path],
        output_dir: Path,
        resume: bool,
    ) -> tuple[list[tuple[int, Path]], dict[int, OCRResult]]:
        """Check which pages already have OCR results.

        Args:
            image_paths: All image paths
            output_dir: OCR output directory
            resume: Whether to load existing results

        Returns:
            Tuple of (pages to process, results dict with existing loaded)
        """
        results: dict[int, OCRResult] = {}
        to_process: list[tuple[int, Path]] = []

        for idx, image_path in enumerate(image_paths):
            base_name = f"page_{idx:04d}"
            json_path = output_dir / f"{base_name}.json"
            md_path = output_dir / f"{base_name}.md"
            md_nohf_path = output_dir / f"{base_name}_nohf.md"

            # Check if we can resume from existing output
            if resume and json_path.exists() and md_nohf_path.exists():
                try:
                    # Validate cache: check if source is newer than output
                    source_mtime = image_path.stat().st_mtime
                    output_mtime = json_path.stat().st_mtime
                    if source_mtime > output_mtime:
                        logger.debug(f"Cache stale for page {idx}, reprocessing")
                        raise ValueError("Cache stale")  # Force reprocess

                    json_data = json.loads(json_path.read_text(encoding="utf-8"))
                    text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
                    text_nohf = md_nohf_path.read_text(encoding="utf-8")

                    results[idx] = OCRResult(
                        page_number=idx,
                        image_path=image_path,
                        text=text,
                        text_nohf=text_nohf,
                        json_data=json_data,
                        success=True,
                    )
                    continue
                except Exception as e:
                    # Log cache load failures for debugging
                    if not isinstance(e, ValueError) or "Cache stale" not in str(e):
                        logger.debug(f"Cache load failed for page {idx}: {e}, will reprocess")

            to_process.append((idx, image_path))

        return to_process, results

    def _results_dict_to_list(
        self,
        results_dict: dict[int, OCRResult],
        image_paths: list[Path],
    ) -> list[OCRResult]:
        """Convert results dict to ordered list, filling gaps with error results."""
        results = []
        for idx, image_path in enumerate(image_paths):
            if idx in results_dict:
                results.append(results_dict[idx])
            else:
                # This shouldn't happen, but handle gracefully
                results.append(OCRResult(
                    page_number=idx,
                    image_path=image_path,
                    text="",
                    text_nohf="",
                    json_data=None,
                    success=False,
                    error_message="Result not found (internal error)",
                ))
        return results

    def _process_single_image(
        self,
        parser,
        page_number: int,
        image_path: Path,
        output_dir: Path,
    ) -> OCRResult:
        """Process a single image through OCR."""
        from dots_ocr.utils.image_utils import fetch_image
        from dots_ocr.utils.layout_utils import post_process_output, draw_layout_on_image
        from dots_ocr.utils.format_transformer import layoutjson2md

        try:
            # Load image using absolute path
            origin_image = fetch_image(str(image_path.absolute()))

            prompt_mode = "prompt_layout_all_en"
            prompt = parser.get_prompt(prompt_mode)

            # Resize for model
            image = fetch_image(origin_image, min_pixels=parser.min_pixels, max_pixels=parser.max_pixels)

            # Run inference with retry logic
            response = self._inference_with_retry(parser, image, prompt)

            if response is None:
                raise RuntimeError("Inference failed after retries")

            # Post-process
            cells, filtered = post_process_output(
                response,
                prompt_mode,
                origin_image,
                image,
                min_pixels=parser.min_pixels,
                max_pixels=parser.max_pixels,
            )

            # Generate markdown
            if filtered:
                text = response if isinstance(response, str) else str(response)
                text_nohf = text
                json_data = {"raw_response": response}
            else:
                text = layoutjson2md(origin_image, cells, text_key='text')
                text_nohf = layoutjson2md(origin_image, cells, text_key='text', no_page_hf=True)
                json_data = cells

            # Save outputs using absolute paths
            base_name = f"page_{page_number:04d}"
            abs_output_dir = output_dir.absolute()

            md_path = abs_output_dir / f"{base_name}.md"
            md_path.write_text(text, encoding="utf-8")

            md_nohf_path = abs_output_dir / f"{base_name}_nohf.md"
            md_nohf_path.write_text(text_nohf, encoding="utf-8")

            json_path = abs_output_dir / f"{base_name}.json"
            json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")

            # Save annotated image
            try:
                if not filtered:
                    image_with_layout = draw_layout_on_image(origin_image, cells)
                else:
                    image_with_layout = origin_image
                img_path = abs_output_dir / f"{base_name}.jpg"
                image_with_layout.save(str(img_path))
            except Exception as e:
                logger.debug(f"Could not save annotated image: {e}")

            return OCRResult(
                page_number=page_number,
                image_path=image_path,
                text=text,
                text_nohf=text_nohf,
                json_data=json_data,
                success=True,
            )

        except Exception as e:
            import traceback
            error_msg = str(e)
            logger.error(f"OCR failed for page {page_number}: {error_msg}")
            logger.debug(traceback.format_exc())
            return OCRResult(
                page_number=page_number,
                image_path=image_path,
                text="",
                text_nohf="",
                json_data=None,
                success=False,
                error_message=error_msg,
            )

    def _inference_with_retry(self, parser, image, prompt) -> str | None:
        """Run inference with retry logic for transient failures."""
        last_error = None

        for attempt in range(INFERENCE_MAX_RETRIES):
            try:
                if parser.use_hf:
                    return parser._inference_with_hf(image, prompt)
                else:
                    response = parser._inference_with_vllm(image, prompt)
                    if response is not None:
                        return response
                    raise RuntimeError("vLLM returned None response")
            except Exception as e:
                last_error = e
                if attempt < INFERENCE_MAX_RETRIES - 1:
                    logger.warning(f"Inference attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(INFERENCE_RETRY_DELAY)
                else:
                    logger.error(f"Inference failed after {INFERENCE_MAX_RETRIES} attempts: {e}")

        return None
