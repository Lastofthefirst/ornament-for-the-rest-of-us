#!/usr/bin/env python3
"""Test if VL model can detect image orientation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "dots.ocr"))

from PIL import Image
from bookpipeline.ocr import VLLMServer


def ask_about_orientation(image_path: str, port: int = 8000) -> str:
    """Ask the VL model about image orientation."""
    import base64
    import json
    import urllib.request

    # Load and encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Create prompt asking about orientation
    prompt = """Look at this image. Is the text in this image oriented correctly (right-side up), or is it rotated?

Please respond with one of:
- "correct" if the text is right-side up and readable normally
- "90_clockwise" if the image needs to be rotated 90° clockwise to be correct
- "90_counterclockwise" if the image needs to be rotated 90° counter-clockwise to be correct
- "upside_down" if the image is upside down (needs 180° rotation)

Just respond with one word."""

    # Build request for vLLM
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    request_data = {
        "model": "model",
        "messages": messages,
        "max_tokens": 50,
        "temperature": 0.1,
    }

    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=json.dumps(request_data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="Path to test image")
    parser.add_argument("--rotations", type=str, default="0,90,180,270", help="Rotations to test")
    args = parser.parse_args()

    image_path = Path(args.image)
    rotations = [int(r) for r in args.rotations.split(",")]

    # Start vLLM server
    dots_ocr_path = Path(__file__).parent.parent / "dots.ocr"
    weights_path = dots_ocr_path / "weights" / "DotsOCR"

    print("Starting vLLM server...")
    server = VLLMServer(str(weights_path))
    if not server.start():
        print("Failed to start vLLM server")
        sys.exit(1)

    try:
        print(f"\nTesting image: {image_path.name}")
        print(f"Rotations: {rotations}")
        print()

        # Load original image
        original = Image.open(image_path)

        for rotation in rotations:
            # Rotate image
            if rotation == 0:
                test_img = original
                test_path = str(image_path)
            else:
                test_img = original.rotate(-rotation, expand=True)
                test_path = f"/tmp/orientation_test_{rotation}.jpg"
                test_img.save(test_path, quality=95)

            print(f"Testing {rotation}° rotation ({test_img.size[0]}x{test_img.size[1]})...")

            try:
                response = ask_about_orientation(test_path)
                print(f"  Model says: {response}")

                # Check if model got it right
                expected = {
                    0: "correct",
                    90: "90_counterclockwise",  # Need to rotate back
                    180: "upside_down",
                    270: "90_clockwise",  # Need to rotate back
                }

                is_correct = expected.get(rotation, "").lower() in response.lower() or \
                            (rotation == 0 and "correct" in response.lower())
                print(f"  Expected: {expected.get(rotation, 'unknown')}")
                print(f"  Correct: {'✓' if is_correct else '✗'}")
            except Exception as e:
                print(f"  Error: {e}")

            print()

    finally:
        print("Stopping vLLM server...")
        server.stop()


if __name__ == "__main__":
    main()
