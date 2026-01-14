# Orientation Detection Research

**Date:** 2026-01-14
**Status:** Parked for future work

## Problem

When photographing book pages from above, the phone's gyroscope cannot reliably determine "up", so EXIF orientation may be incorrect. This leads to images being processed in the wrong orientation.

## OCR Behavior with Rotated Images

Tested DotsOCR with images at 0°, 90°, 180°, 270° rotations:

| Rotation | OCR Quality |
|----------|-------------|
| 0° | Good - baseline |
| 90° | Good - similar to baseline |
| 180° | Good - similar to baseline |
| 270° | **Poor** - ~40% accuracy, JSON errors, wrong language hallucinations |

**Conclusion:** DotsOCR handles most orientations well, but 270° rotation causes significant failures.

## VL Models Tested for Orientation Detection

Goal: Use a vision-language model to detect if an image needs rotation before OCR.

### SmolVLM2-2.2B-Instruct
- **Backend:** vLLM
- **Result:** Cannot detect orientation
- **Behavior:** Always answers "yes" regardless of actual orientation
- **Notes:** Required `num2words` package to be installed

### Qwen2-VL-2B-Instruct
- **Backend:** vLLM
- **Result:** Cannot detect orientation
- **Behavior:** Always gives same answer for all rotations
- **Notes:**
  - Requires images resized to ~600px short edge to fit 4096 token limit
  - Full-size images (2268x4032) produce ~11,697 tokens

### Prompts Tested
- "Is this text right-side up and readable? Answer yes or no"
- "Which direction is the text facing? Answer: up, down, left, or right"
- "Is this image rotated sideways? Answer: yes or no"
- "Describe the orientation of the text in this image"

All prompts produced identical responses regardless of actual image orientation.

## Conclusion

Small 2B parameter VL models cannot detect image orientation. They recognize text content but lack spatial reasoning to understand rotation.

## Future Options to Explore

1. **Larger VL models (7B+)** - May have better spatial understanding, but require more GPU memory
2. **Heuristic approach** - Run OCR at 0° and 90°, compare output quality, pick better result
3. **Specialized rotation detection** - Computer vision libraries (OpenCV, etc.) may have dedicated solutions
4. **EXIF-only** - Accept that some images will be wrong and rely on EXIF

## Relevant Files

- `tests/test_orientation_accuracy.py` - Tests OCR accuracy at different rotations
- `tests/test_orientation_detection.py` - Tests VL model orientation detection
- `/tmp/ocr_orientation_test/results.json` - Raw OCR results at different rotations
