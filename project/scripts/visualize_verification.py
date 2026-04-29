#!/usr/bin/env python3
"""
Visualization helper for Task 3b verification results.

Creates side-by-side comparisons of:
- Original detections (from Task 3a)
- Verified detections (after false positive filtering)
- Marked false positives (that were caught)

Usage:
    python scripts/visualize_verification.py \
        --query-image data/coco/val2017/000000123456.jpg \
        --original-detections results/task3_detections.json \
        --verified-detections results/task3b_verified_detections.json \
        --output viz/verification_comparison.jpg
"""

import json
import logging
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def draw_bboxes_on_image(
    image: Image.Image,
    bboxes: list[list[float]],
    color: str = "red",
    line_width: int = 2,
    label: str = "",
) -> Image.Image:
    """
    Draw bounding boxes on image.

    Args:
        image: PIL Image
        bboxes: List of bboxes in COCO format [x, y, width, height]
        color: Bbox color
        line_width: Line width
        label: Text label to add to image

    Returns:
        Image with bboxes drawn
    """
    result = image.copy()
    draw = ImageDraw.Draw(result)

    for idx, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        x2, y2 = x + w, y + h

        # Draw bbox
        draw.rectangle([x, y, x2, y2], outline=color, width=line_width)

        # Draw number
        draw.text((x, y - 10), str(idx + 1), fill=color)

    # Draw label
    if label:
        try:
            draw.text((5, 5), label, fill="white")
        except Exception:
            pass

    return result


def visualize_verification(
    query_image_path: str | Path,
    original_bboxes: list[list[float]],
    verified_bboxes: list[list[float]],
    caught_fp_indices: Optional[list[int]] = None,
    output_path: Optional[str | Path] = None,
) -> Image.Image:
    """
    Create side-by-side comparison visualization.

    Args:
        query_image_path: Path to query image
        original_bboxes: Detections from Task 3a (COCO format)
        verified_bboxes: Detections after verification
        caught_fp_indices: Indices of false positives that were caught
        output_path: Optional path to save visualization

    Returns:
        PIL Image with comparison
    """
    # Load query image
    query_image = Image.open(query_image_path).convert("RGB")
    img_w, img_h = query_image.size

    # Draw original detections (red)
    original_viz = draw_bboxes_on_image(
        query_image,
        original_bboxes,
        color="red",
        line_width=3,
        label=f"Original: {len(original_bboxes)} detections",
    )

    # Draw verified detections (green)
    verified_viz = draw_bboxes_on_image(
        query_image,
        verified_bboxes,
        color="green",
        line_width=3,
        label=f"Verified: {len(verified_bboxes)} detections",
    )

    # Highlight false positives (orange)
    if caught_fp_indices:
        caught_bboxes = [original_bboxes[i] for i in caught_fp_indices if i < len(original_bboxes)]
        fp_viz = draw_bboxes_on_image(
            query_image,
            caught_bboxes,
            color="orange",
            line_width=3,
            label=f"Caught FPs: {len(caught_bboxes)}",
        )
    else:
        fp_viz = query_image.copy()

    # Combine visualizations side-by-side
    combined_width = img_w * 3
    combined_height = img_h
    combined = Image.new("RGB", (combined_width, combined_height))

    combined.paste(original_viz, (0, 0))
    combined.paste(verified_viz, (img_w, 0))
    combined.paste(fp_viz, (img_w * 2, 0))

    # Add labels
    draw = ImageDraw.Draw(combined)
    label_y = 10
    draw.text((10, label_y), "ORIGINAL (Red)", fill="red")
    draw.text((img_w + 10, label_y), "VERIFIED (Green)", fill="green")
    draw.text((img_w * 2 + 10, label_y), "CAUGHT FPs (Orange)", fill="orange")

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.save(output_path)
        logger.info(f"Visualization saved to {output_path}")

    return combined


def compare_detection_results(
    original_file: str | Path,
    verified_file: str | Path,
    image_id: int,
    img_dir: str = "data/coco/val2017",
    output_dir: str = "viz",
) -> dict:
    """
    Create comparison visualization for a specific image.

    Args:
        original_file: Path to original detections JSON
        verified_file: Path to verified detections JSON
        image_id: COCO image ID
        img_dir: Directory containing images
        output_dir: Directory to save visualizations

    Returns:
        Dict with statistics
    """
    # Load detections
    with open(original_file) as f:
        original_detections = json.load(f)
    with open(verified_file) as f:
        verified_detections = json.load(f)

    # Filter by image_id
    original_for_image = [d for d in original_detections if d["image_id"] == image_id]
    verified_for_image = [d for d in verified_detections if d["image_id"] == image_id]

    logger.info(f"Image {image_id}:")
    logger.info(f"  Original detections: {len(original_for_image)}")
    logger.info(f"  Verified detections: {len(verified_for_image)}")

    # Get image path
    from pycocotools.coco import COCO
    from config import settings

    coco = COCO(settings.ann_file)
    img_info = coco.loadImgs(image_id)[0]
    image_path = Path(img_dir) / img_info["file_name"]

    if not image_path.exists():
        logger.warning(f"Image not found: {image_path}")
        return {}

    # Extract bboxes
    original_bboxes = [d["bbox"] for d in original_for_image]
    verified_bboxes = [d["bbox"] for d in verified_for_image]

    # Find which indices were filtered out
    caught_fp_indices = []
    for idx, det in enumerate(original_for_image):
        # Check if this detection is in verified set
        found = False
        for v_det in verified_for_image:
            if v_det.get("category_id") == det.get("category_id"):
                # Approximate match (same bbox)
                if all(abs(a - b) < 1 for a, b in zip(det["bbox"], v_det["bbox"])):
                    found = True
                    break
        if not found:
            caught_fp_indices.append(idx)

    # Create visualization
    output_path = Path(output_dir) / f"verification_image_{image_id}.png"
    visualize_verification(
        query_image_path=image_path,
        original_bboxes=original_bboxes,
        verified_bboxes=verified_bboxes,
        caught_fp_indices=caught_fp_indices,
        output_path=output_path,
    )

    stats = {
        "image_id": image_id,
        "original_count": len(original_for_image),
        "verified_count": len(verified_for_image),
        "false_positives_caught": len(caught_fp_indices),
        "fp_rate": len(caught_fp_indices) / len(original_for_image)
        if original_for_image else 0,
    }

    return stats


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize Task 3b verification results"
    )
    parser.add_argument(
        "--original-detections",
        required=True,
        type=str,
        help="Path to original Task 3a detections JSON",
    )
    parser.add_argument(
        "--verified-detections",
        required=True,
        type=str,
        help="Path to verified Task 3b detections JSON",
    )
    parser.add_argument(
        "--image-id",
        type=int,
        required=True,
        help="COCO image ID to visualize",
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default="data/coco/val2017",
        help="Directory containing images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="viz",
        help="Output directory for visualizations",
    )

    args = parser.parse_args()

    logger.info("Task 3b Verification Visualization")
    logger.info("=" * 80)

    stats = compare_detection_results(
        original_file=args.original_detections,
        verified_file=args.verified_detections,
        image_id=args.image_id,
        img_dir=args.img_dir,
        output_dir=args.output_dir,
    )

    logger.info("=" * 80)
    if not stats:
        logger.warning("No statistics returned (image may not have been found).")
        return
    logger.info(f"Image {stats['image_id']}:")
    logger.info(f"  Original: {stats['original_count']} detections")
    logger.info(f"  Verified: {stats['verified_count']} detections")
    logger.info(f"  FPs caught: {stats['false_positives_caught']}")
    logger.info(f"  FP rate: {stats['fp_rate']:.2%}")


if __name__ == "__main__":
    main()
