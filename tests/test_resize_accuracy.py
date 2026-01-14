#!/usr/bin/env python3
"""Test OCR accuracy at different image resize levels.

Measures speed vs accuracy tradeoff for preprocessing optimization.
Keeps vLLM server running for efficiency.
"""

import sys
import time
import tempfile
import difflib
import json
from pathlib import Path
from dataclasses import dataclass, asdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image


@dataclass
class TestResult:
    """Result of OCR test at a specific scale."""
    scale: float
    width: int
    height: int
    ocr_time: float
    text: str
    char_count: int


def resize_image(image_path: Path, scale: float, output_path: Path) -> tuple[int, int]:
    """Resize image by scale factor and save."""
    img = Image.open(image_path)
    new_width = int(img.width * scale)
    new_height = int(img.height * scale)

    # Use LANCZOS for high quality downscaling
    resized = img.resize((new_width, new_height), Image.LANCZOS)
    resized.save(output_path, quality=95)

    return new_width, new_height


def run_single_ocr(parser, image_path: Path, output_dir: Path) -> tuple[float, str]:
    """Run OCR on a single image using existing parser."""
    from dots_ocr.utils.image_utils import fetch_image
    from dots_ocr.utils.layout_utils import post_process_output
    from dots_ocr.utils.format_transformer import layoutjson2md

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        # Load image
        origin_image = fetch_image(str(image_path.absolute()))

        prompt_mode = "prompt_layout_all_en"
        prompt = parser.get_prompt(prompt_mode)

        # Resize for model (uses parser's min/max pixels)
        image = fetch_image(origin_image, min_pixels=parser.min_pixels, max_pixels=parser.max_pixels)

        # Run inference
        if parser.use_hf:
            response = parser._inference_with_hf(image, prompt)
        else:
            response = parser._inference_with_vllm(image, prompt)

        if response is None:
            raise RuntimeError("Inference returned None")

        # Post-process
        cells, filtered = post_process_output(
            response,
            prompt_mode,
            origin_image,
            image,
            min_pixels=parser.min_pixels,
            max_pixels=parser.max_pixels,
        )

        # Generate markdown without headers/footers
        if filtered:
            text = response if isinstance(response, str) else str(response)
        else:
            text = layoutjson2md(origin_image, cells, text_key='text', no_page_hf=True)

        elapsed = time.time() - start_time
        return elapsed, text

    except Exception as e:
        elapsed = time.time() - start_time
        return elapsed, f"[OCR FAILED: {e}]"


def compare_texts(baseline: str, test: str) -> dict:
    """Compare two texts and return similarity metrics."""
    matcher = difflib.SequenceMatcher(None, baseline, test)
    similarity = matcher.ratio()

    char_diff = abs(len(baseline) - len(test))

    baseline_words = baseline.split()
    test_words = test.split()
    word_matcher = difflib.SequenceMatcher(None, baseline_words, test_words)
    word_similarity = word_matcher.ratio()

    return {
        "char_similarity": similarity,
        "char_diff": char_diff,
        "baseline_chars": len(baseline),
        "test_chars": len(test),
        "word_similarity": word_similarity,
        "baseline_words": len(baseline_words),
        "test_words": len(test_words),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test OCR accuracy at different image sizes")
    parser.add_argument("image_dir", type=Path, help="Directory containing test images")
    parser.add_argument("--num-images", type=int, default=3, help="Number of images to test")
    parser.add_argument("--scales", type=str, default="1.0,0.75,0.5,0.35,0.25",
                        help="Comma-separated scale factors to test")
    parser.add_argument("--output", type=Path, help="Output directory for results")

    args = parser.parse_args()

    # Parse scales
    scales = [float(s) for s in args.scales.split(",")]
    scales.sort(reverse=True)

    # Get image files
    image_files = sorted(args.image_dir.glob("*.jpg"))[:args.num_images]

    if not image_files:
        print(f"No JPG images found in {args.image_dir}")
        sys.exit(1)

    print(f"Testing {len(image_files)} images at scales: {scales}")
    print(f"Images: {[f.name for f in image_files]}")
    print()

    # Create work directory
    work_dir = args.output or Path(tempfile.mkdtemp(prefix="ocr_resize_test_"))
    work_dir.mkdir(exist_ok=True)
    print(f"Working directory: {work_dir}")
    print()

    # Start vLLM server once
    from bookpipeline.ocr import VLLMServer

    dots_ocr_path = Path(__file__).parent.parent / "dots.ocr"
    weights_path = dots_ocr_path / "weights" / "DotsOCR"

    print("Starting vLLM server (this may take a few minutes)...")
    server = VLLMServer(str(weights_path))
    if not server.start():
        print("ERROR: Failed to start vLLM server")
        sys.exit(1)
    print("vLLM server ready!")
    print()

    try:
        # Import after server is running
        dots_ocr_str = str(dots_ocr_path)
        if dots_ocr_str not in sys.path:
            sys.path.insert(0, dots_ocr_str)

        from dots_ocr.parser import DotsOCRParser
        ocr_parser = DotsOCRParser(port=8000, num_thread=16, use_hf=False)

        all_results = {}

        # Prepare all resized images first
        print("Preparing resized images...")
        resize_map = {}  # (image_name, scale) -> resized_path

        for img_path in image_files:
            img = Image.open(img_path)
            orig_size = img.size

            for scale in scales:
                if scale == 1.0:
                    resize_map[(img_path.name, scale)] = (img_path, orig_size)
                else:
                    resized_dir = work_dir / "resized" / f"scale_{int(scale * 100)}"
                    resized_dir.mkdir(parents=True, exist_ok=True)
                    resized_path = resized_dir / img_path.name
                    new_size = resize_image(img_path, scale, resized_path)
                    resize_map[(img_path.name, scale)] = (resized_path, new_size)

        print("Done preparing images.\n")

        # Process each image at each scale
        for img_path in image_files:
            print(f"Testing: {img_path.name}")
            all_results[img_path.name] = {}

            for scale in scales:
                resized_path, (w, h) = resize_map[(img_path.name, scale)]
                print(f"  Scale {scale:.0%}: {w}x{h}...", end=" ", flush=True)

                ocr_output = work_dir / img_path.stem / f"scale_{int(scale * 100)}"
                ocr_time, text = run_single_ocr(ocr_parser, resized_path, ocr_output)

                all_results[img_path.name][scale] = TestResult(
                    scale=scale,
                    width=w,
                    height=h,
                    ocr_time=ocr_time,
                    text=text,
                    char_count=len(text),
                )

                print(f"{ocr_time:.1f}s, {len(text)} chars")

            print()

    finally:
        print("Stopping vLLM server...")
        server.stop()

    # Save raw results
    results_file = work_dir / "results.json"
    json_results = {
        img: {str(s): asdict(r) for s, r in scales_dict.items()}
        for img, scales_dict in all_results.items()
    }
    results_file.write_text(json.dumps(json_results, indent=2))

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    baseline_scale = max(scales)

    for img_name, results in all_results.items():
        print(f"\n{img_name}:")
        baseline = results[baseline_scale]

        print(f"  {'Scale':<8} {'Size':<12} {'Time':>8} {'Chars':>8} {'Char Sim':>10} {'Word Sim':>10}")
        print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

        for scale in scales:
            r = results[scale]
            size_str = f"{r.width}x{r.height}"

            scale_str = f"{int(scale*100)}%"
            if scale == baseline_scale:
                print(f"  {scale_str:<8} {size_str:<12} {r.ocr_time:>7.1f}s {r.char_count:>8} {'baseline':>10} {'baseline':>10}")
            else:
                comparison = compare_texts(baseline.text, r.text)
                char_sim = f"{comparison['char_similarity']:.1%}"
                word_sim = f"{comparison['word_similarity']:.1%}"
                print(f"  {scale_str:<8} {size_str:<12} {r.ocr_time:>7.1f}s {r.char_count:>8} {char_sim:>10} {word_sim:>10}")

    # Aggregate stats
    print("\n" + "=" * 80)
    print("AGGREGATE STATS")
    print("=" * 80)

    print(f"\n{'Scale':<8} {'Avg Time':>10} {'Avg Char Sim':>12} {'Avg Word Sim':>12} {'Speedup':>10}")
    print(f"{'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")

    for scale in scales:
        times = [r[scale].ocr_time for r in all_results.values()]
        avg_time = sum(times) / len(times)
        scale_str = f"{int(scale*100)}%"

        if scale == baseline_scale:
            baseline_avg_time = avg_time
            print(f"{scale_str:<8} {avg_time:>9.1f}s {'baseline':>12} {'baseline':>12} {'1.0x':>10}")
        else:
            char_sims = []
            word_sims = []
            for results in all_results.values():
                comparison = compare_texts(results[baseline_scale].text, results[scale].text)
                char_sims.append(comparison['char_similarity'])
                word_sims.append(comparison['word_similarity'])

            avg_char_sim = sum(char_sims) / len(char_sims)
            avg_word_sim = sum(word_sims) / len(word_sims)
            speedup = baseline_avg_time / avg_time if avg_time > 0 else 0

            print(f"{scale_str:<8} {avg_time:>9.1f}s {avg_char_sim:>11.1%} {avg_word_sim:>11.1%} {speedup:>9.1f}x")

    print(f"\nResults saved to: {work_dir}")
    print(f"Raw data: {results_file}")


if __name__ == "__main__":
    main()
