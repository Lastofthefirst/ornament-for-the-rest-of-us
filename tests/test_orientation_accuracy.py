#!/usr/bin/env python3
"""Test OCR accuracy at different image orientations.

Tests whether the OCR model handles rotated images well or if
EXIF orientation correction is necessary for accuracy.
"""

import sys
import time
import tempfile
import difflib
import json
from pathlib import Path
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image


@dataclass
class TestResult:
    """Result of OCR test at a specific orientation."""
    rotation: int  # degrees clockwise
    ocr_time: float
    text: str
    char_count: int


def rotate_image(image_path: Path, degrees: int, output_path: Path) -> None:
    """Rotate image by degrees (clockwise) and save."""
    img = Image.open(image_path)
    # PIL rotates counter-clockwise, so negate for clockwise
    rotated = img.rotate(-degrees, expand=True)
    rotated.save(output_path, quality=95)


def run_single_ocr(parser, image_path: Path, output_dir: Path) -> tuple[float, str]:
    """Run OCR on a single image using existing parser."""
    from dots_ocr.utils.image_utils import fetch_image
    from dots_ocr.utils.layout_utils import post_process_output
    from dots_ocr.utils.format_transformer import layoutjson2md

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        origin_image = fetch_image(str(image_path.absolute()))
        prompt_mode = "prompt_layout_all_en"
        prompt = parser.get_prompt(prompt_mode)
        image = fetch_image(origin_image, min_pixels=parser.min_pixels, max_pixels=parser.max_pixels)

        if parser.use_hf:
            response = parser._inference_with_hf(image, prompt)
        else:
            response = parser._inference_with_vllm(image, prompt)

        if response is None:
            raise RuntimeError("Inference returned None")

        cells, filtered = post_process_output(
            response, prompt_mode, origin_image, image,
            min_pixels=parser.min_pixels, max_pixels=parser.max_pixels,
        )

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

    baseline_words = baseline.split()
    test_words = test.split()
    word_matcher = difflib.SequenceMatcher(None, baseline_words, test_words)
    word_similarity = word_matcher.ratio()

    return {
        "char_similarity": similarity,
        "char_diff": abs(len(baseline) - len(test)),
        "word_similarity": word_similarity,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test OCR accuracy at different orientations")
    parser.add_argument("image_dir", type=Path, help="Directory containing test images")
    parser.add_argument("--num-images", type=int, default=3, help="Number of images to test")
    parser.add_argument("--rotations", type=str, default="0,90,180,270",
                        help="Comma-separated rotation angles to test (degrees clockwise)")
    parser.add_argument("--output", type=Path, help="Output directory for results")

    args = parser.parse_args()

    rotations = [int(r) for r in args.rotations.split(",")]

    # Get image files
    image_files = sorted(args.image_dir.glob("*.jpg"))[:args.num_images]

    if not image_files:
        print(f"No JPG images found in {args.image_dir}")
        sys.exit(1)

    print(f"Testing {len(image_files)} images at rotations: {rotations}°")
    print(f"Images: {[f.name for f in image_files]}")
    print()

    work_dir = args.output or Path(tempfile.mkdtemp(prefix="ocr_orientation_test_"))
    work_dir.mkdir(exist_ok=True)
    print(f"Working directory: {work_dir}")
    print()

    # Start vLLM server
    from bookpipeline.ocr import VLLMServer

    dots_ocr_path = Path(__file__).parent.parent / "dots.ocr"
    weights_path = dots_ocr_path / "weights" / "DotsOCR"

    print("Starting vLLM server...")
    server = VLLMServer(str(weights_path))
    if not server.start():
        print("ERROR: Failed to start vLLM server")
        sys.exit(1)
    print("vLLM server ready!")
    print()

    try:
        dots_ocr_str = str(dots_ocr_path)
        if dots_ocr_str not in sys.path:
            sys.path.insert(0, dots_ocr_str)

        from dots_ocr.parser import DotsOCRParser
        ocr_parser = DotsOCRParser(port=8000, num_thread=16, use_hf=False)

        all_results = {}

        # Prepare rotated images
        print("Preparing rotated images...")
        rotate_map = {}

        for img_path in image_files:
            for rotation in rotations:
                if rotation == 0:
                    rotate_map[(img_path.name, rotation)] = img_path
                else:
                    rotated_dir = work_dir / "rotated" / f"rot_{rotation}"
                    rotated_dir.mkdir(parents=True, exist_ok=True)
                    rotated_path = rotated_dir / img_path.name
                    rotate_image(img_path, rotation, rotated_path)
                    rotate_map[(img_path.name, rotation)] = rotated_path

        print("Done preparing images.\n")

        # Process each image at each rotation
        for img_path in image_files:
            print(f"Testing: {img_path.name}")
            all_results[img_path.name] = {}

            for rotation in rotations:
                rotated_path = rotate_map[(img_path.name, rotation)]
                print(f"  Rotation {rotation}°...", end=" ", flush=True)

                ocr_output = work_dir / img_path.stem / f"rot_{rotation}"
                ocr_time, text = run_single_ocr(ocr_parser, rotated_path, ocr_output)

                all_results[img_path.name][rotation] = TestResult(
                    rotation=rotation,
                    ocr_time=ocr_time,
                    text=text,
                    char_count=len(text),
                )

                print(f"{ocr_time:.1f}s, {len(text)} chars")

            print()

    finally:
        print("Stopping vLLM server...")
        server.stop()

    # Save results
    results_file = work_dir / "results.json"
    json_results = {
        img: {str(r): asdict(res) for r, res in rots.items()}
        for img, rots in all_results.items()
    }
    results_file.write_text(json.dumps(json_results, indent=2))

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    baseline_rotation = 0

    for img_name, results in all_results.items():
        print(f"\n{img_name}:")
        baseline = results[baseline_rotation]

        print(f"  {'Rotation':<10} {'Time':>8} {'Chars':>8} {'Char Sim':>10} {'Word Sim':>10}")
        print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

        for rotation in rotations:
            r = results[rotation]

            if rotation == baseline_rotation:
                print(f"  {rotation}° (base) {r.ocr_time:>7.1f}s {r.char_count:>8} {'baseline':>10} {'baseline':>10}")
            else:
                comparison = compare_texts(baseline.text, r.text)
                char_sim = f"{comparison['char_similarity']:.1%}"
                word_sim = f"{comparison['word_similarity']:.1%}"
                print(f"  {rotation}°{' '*7} {r.ocr_time:>7.1f}s {r.char_count:>8} {char_sim:>10} {word_sim:>10}")

    # Aggregate
    print("\n" + "=" * 80)
    print("AGGREGATE STATS")
    print("=" * 80)

    print(f"\n{'Rotation':<10} {'Avg Time':>10} {'Avg Char Sim':>12} {'Avg Word Sim':>12}")
    print(f"{'-'*10} {'-'*10} {'-'*12} {'-'*12}")

    for rotation in rotations:
        times = [r[rotation].ocr_time for r in all_results.values()]
        avg_time = sum(times) / len(times)

        if rotation == baseline_rotation:
            print(f"{rotation}° (base) {avg_time:>9.1f}s {'baseline':>12} {'baseline':>12}")
        else:
            char_sims = []
            word_sims = []
            for results in all_results.values():
                comparison = compare_texts(results[baseline_rotation].text, results[rotation].text)
                char_sims.append(comparison['char_similarity'])
                word_sims.append(comparison['word_similarity'])

            avg_char_sim = sum(char_sims) / len(char_sims)
            avg_word_sim = sum(word_sims) / len(word_sims)

            print(f"{rotation}°{' '*7} {avg_time:>9.1f}s {avg_char_sim:>11.1%} {avg_word_sim:>11.1%}")

    print(f"\nResults saved to: {work_dir}")


if __name__ == "__main__":
    main()
