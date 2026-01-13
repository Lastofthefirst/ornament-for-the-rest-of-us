"""
OCR wrapper for dots.ocr integration.

Supports both vLLM (recommended) and HuggingFace backends.
When using vLLM, automatically starts and stops the server.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


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

    def start(self, timeout: int = 300) -> bool:
        """Start the vLLM server and wait for it to be ready.

        Args:
            timeout: Maximum seconds to wait for server to start

        Returns:
            True if server started successfully
        """
        if self.is_running():
            logger.info("vLLM server already running")
            return True

        logger.info(f"Starting vLLM server with model: {self.model_path}")

        cmd = [
            "vllm", "serve", self.model_path,
            "--trust-remote-code",
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory),
            "--served-model-name", "model",  # Match the default in dots_ocr inference
        ]
        logger.info(f"vLLM command: {' '.join(cmd)}")

        # Start server process
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Wait for server to be ready
        start_time = time.time()
        last_log = start_time
        while time.time() - start_time < timeout:
            if self.is_running():
                logger.info(f"vLLM server ready on port {self.port}")
                return True

            # Check if process died
            if self.process.poll() is not None:
                # Process exited, read output for error
                output = self.process.stdout.read() if self.process.stdout else ""
                logger.error(f"vLLM server failed to start: {output[-2000:]}")
                return False

            # Log progress every 30 seconds
            if time.time() - last_log > 30:
                elapsed = int(time.time() - start_time)
                logger.info(f"Waiting for vLLM server... ({elapsed}s elapsed)")
                last_log = time.time()

            time.sleep(2)

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

    def is_running(self) -> bool:
        """Check if the vLLM server is responding."""
        import urllib.request
        import urllib.error

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
    ) -> list[OCRResult]:
        """Process multiple images through OCR.

        Args:
            image_paths: List of image paths to process (in order)
            output_dir: Directory for OCR output files

        Returns:
            List of OCRResult objects in same order as input
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.backend == "vllm":
            return self._process_with_vllm(image_paths, output_dir)
        else:
            return self._process_with_hf(image_paths, output_dir)

    def _process_with_vllm(
        self,
        image_paths: list[Path],
        output_dir: Path,
    ) -> list[OCRResult]:
        """Process images using vLLM backend."""
        # Start vLLM server
        if not self._start_vllm_server():
            logger.error("Failed to start vLLM server")
            return [
                OCRResult(
                    page_number=idx,
                    image_path=path,
                    text="",
                    text_nohf="",
                    json_data=None,
                    success=False,
                    error_message="Failed to start vLLM server",
                )
                for idx, path in enumerate(image_paths)
            ]

        try:
            # Use dots.ocr parser with vLLM
            original_cwd = os.getcwd()
            os.chdir(str(self.dots_ocr_path))

            try:
                from dots_ocr.parser import DotsOCRParser

                parser = DotsOCRParser(
                    port=self.vllm_port,
                    num_thread=self.num_threads,
                    use_hf=False,
                )

                results = []
                for idx, image_path in enumerate(image_paths):
                    logger.info(f"Processing page {idx + 1}/{len(image_paths)}: {image_path.name}")
                    result = self._process_single_image(parser, idx, image_path, output_dir)
                    results.append(result)

                successful = sum(1 for r in results if r.success)
                logger.info(f"OCR complete: {successful}/{len(results)} pages successful")
                return results

            finally:
                os.chdir(original_cwd)

        finally:
            self._stop_vllm_server()

    def _process_with_hf(
        self,
        image_paths: list[Path],
        output_dir: Path,
    ) -> list[OCRResult]:
        """Process images using HuggingFace backend."""
        original_cwd = os.getcwd()
        os.chdir(str(self.dots_ocr_path))

        try:
            from dots_ocr.parser import DotsOCRParser

            logger.info("Loading OCR model (this may take a minute)...")
            parser = DotsOCRParser(use_hf=True, num_thread=1)
            logger.info("OCR model loaded")

            results = []
            for idx, image_path in enumerate(image_paths):
                logger.info(f"Processing page {idx + 1}/{len(image_paths)}: {image_path.name}")
                result = self._process_single_image(parser, idx, image_path, output_dir)
                results.append(result)

            successful = sum(1 for r in results if r.success)
            logger.info(f"OCR complete: {successful}/{len(results)} pages successful")
            return results

        except Exception as e:
            logger.error(f"Failed to load OCR model: {e}")
            return [
                OCRResult(
                    page_number=idx,
                    image_path=path,
                    text="",
                    text_nohf="",
                    json_data=None,
                    success=False,
                    error_message=f"Failed to load OCR model: {e}",
                )
                for idx, path in enumerate(image_paths)
            ]
        finally:
            os.chdir(original_cwd)

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
            # Load image
            origin_image = fetch_image(str(image_path))

            # Get prompt
            prompt_mode = "prompt_layout_all_en"
            prompt = parser.get_prompt(prompt_mode)

            # Resize for model
            from dots_ocr.utils.image_utils import smart_resize
            image = fetch_image(origin_image, min_pixels=parser.min_pixels, max_pixels=parser.max_pixels)

            # Run inference
            if parser.use_hf:
                response = parser._inference_with_hf(image, prompt)
            else:
                response = parser._inference_with_vllm(image, prompt)

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

            # Save outputs
            base_name = f"page_{page_number:04d}"

            md_path = output_dir / f"{base_name}.md"
            md_path.write_text(text, encoding="utf-8")

            md_nohf_path = output_dir / f"{base_name}_nohf.md"
            md_nohf_path.write_text(text_nohf, encoding="utf-8")

            json_path = output_dir / f"{base_name}.json"
            json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")

            # Save annotated image
            try:
                if not filtered:
                    image_with_layout = draw_layout_on_image(origin_image, cells)
                else:
                    image_with_layout = origin_image
                img_path = output_dir / f"{base_name}.jpg"
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
            logger.error(f"OCR failed for page {page_number}: {e}")
            logger.debug(traceback.format_exc())
            return OCRResult(
                page_number=page_number,
                image_path=image_path,
                text="",
                text_nohf="",
                json_data=None,
                success=False,
                error_message=str(e),
            )
