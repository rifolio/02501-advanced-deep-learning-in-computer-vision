#!/usr/bin/env python3
"""
Helper script to convert Task 3a results to JSON format for Task 3b verification.

Task 3a logs detections to W&B, but we need them in COCO JSON format for Task 3b.
This script either:
1. Downloads results from W&B artifacts
2. Converts in-memory results to JSON
3. Reconstructs from cached results

Usage:
    python scripts/convert_task3_results.py --run-id epwh05hk
    python scripts/convert_task3_results.py --coco-results results/task3_raw.json
"""

import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional

import wandb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_from_wandb(run_id: str, project: str = "DLCV-Task3-VLM-DINO-Fusion") -> list:
    """
    Download detection results from W&B artifact.

    Args:
        run_id: W&B run ID (e.g., 'epwh05hk')
        project: W&B project name

    Returns:
        List of detections in COCO format
    """
    logger.info(f"Downloading results from W&B run {run_id}...")

    api = wandb.Api()
    run = api.run(f"user/{project}/{run_id}")

    # Look for detection artifacts
    detections = []
    for file in run.files():
        if "detection" in file.name.lower() or "result" in file.name.lower():
            logger.info(f"Found artifact: {file.name}")
            # Download and parse
            file.download(replace=True)
            with open(file.name) as f:
                if file.name.endswith(".json"):
                    detections.extend(json.load(f))

    if not detections:
        logger.warning(f"No detection artifacts found in run {run_id}")

    return detections


def convert_task3_results(
    input_file: str | Path,
    output_file: Optional[str | Path] = None,
) -> list:
    """
    Convert Task 3a raw results to COCO JSON format.

    Handles various input formats:
    - COCO format results (list of dicts with image_id, category_id, bbox, score)
    - Python list representation
    - W&B logged results

    Args:
        input_file: Path to input file
        output_file: Path to save converted results (default: derived from input)

    Returns:
        List of detections in COCO format
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Loading results from {input_path}...")

    # Load results
    with open(input_path) as f:
        content = f.read()

    # Try parsing as JSON
    try:
        results = json.loads(content)
    except json.JSONDecodeError:
        logger.error("Could not parse input file as JSON")
        raise

    # Validate format
    if isinstance(results, list) and len(results) > 0:
        first_item = results[0]
        required_keys = {"image_id", "category_id", "bbox"}

        if isinstance(first_item, dict):
            if required_keys.issubset(first_item.keys()):
                logger.info(f"Format OK: COCO format with {len(results)} detections")
                converted_results = results
            else:
                logger.warning(f"Unexpected format. Keys: {first_item.keys()}")
                converted_results = results
        else:
            logger.error(f"Unexpected item type: {type(first_item)}")
            converted_results = results
    else:
        logger.error("Input is not a non-empty list")
        converted_results = results

    # Determine output path
    if output_file is None:
        output_file = input_path.parent / f"{input_path.stem}_converted.json"

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    with open(output_path, "w") as f:
        json.dump(converted_results, f, indent=2)

    logger.info(f"Converted results saved to {output_path}")
    logger.info(f"Total detections: {len(converted_results)}")

    # Print statistics
    if isinstance(converted_results, list) and len(converted_results) > 0:
        images = set(det.get("image_id") for det in converted_results)
        categories = set(det.get("category_id") for det in converted_results)
        logger.info(f"Unique images: {len(images)}")
        logger.info(f"Unique categories: {len(categories)}")

        # Group by category
        by_cat = defaultdict(int)
        for det in converted_results:
            by_cat[det.get("category_id")] += 1
        logger.info("Detections by category:")
        for cat_id, count in sorted(by_cat.items()):
            logger.info(f"  Category {cat_id}: {count} detections")

    return converted_results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Task 3a results to Task 3b JSON format"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to input JSON file (Task 3a results)",
    )
    input_group.add_argument(
        "--wandb-run",
        type=str,
        help="W&B run ID to download from (e.g., 'epwh05hk')",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output path (default: derived from input)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="DLCV-Task3-VLM-DINO-Fusion",
        help="W&B project name (when using --wandb-run)",
    )

    args = parser.parse_args()

    logger.info("Task 3a → Task 3b Results Converter")
    logger.info("=" * 80)

    if args.wandb_run:
        logger.info(f"Downloading from W&B run: {args.wandb_run}")
        results = download_from_wandb(args.wandb_run, args.project)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
    else:
        convert_task3_results(args.input, args.output)

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
