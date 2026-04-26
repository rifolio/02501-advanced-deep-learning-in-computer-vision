#!/usr/bin/env python3
"""
Task 3b: VLM-as-Verifier - Filter False Positives from DINO Detections

Pipeline:
1. Load Task 3a results (Grounding DINO detections from VLM-generated prompts)
2. For each detection:
   - Crop bbox region from original image
   - Show cropped detection + K support examples to VLM
   - Ask: "Does this detection match the target object?"
   - Get confidence score from VLM
3. Filter detections based on confidence threshold
4. Evaluate and compare metrics: precision/recall before vs after verification
5. Log: how many false positives were caught

Usage:
    uv run python task3b_verify.py --detection-results results/task3_detections.json --k-shot 2
    uv run python task3b_verify.py --detection-results results/task3_detections.json --confidence-threshold 0.6
    
Environment variables:
    EVAL_SPLIT_PATH: Path to subset JSON (default: full COCO val2017)
    K_SHOT: Number of support examples (default: 2)
    CONFIDENCE_THRESHOLD: Verification confidence threshold (default: 0.5)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

import torch
import wandb
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval

from config import settings
from data.dataloaders import get_coco_few_shot_dataloader
from data.support_sampler import SupportSetSampler
from models.qwen import Qwen2_5_VL
from models.vlm_verifier import VLMVerifier, VerificationResponse
from prompts.verification_strategy import VerificationStrategy
from pipeline import FewShotExperiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Task3bVerificationExperiment:
    """
    Task 3b: VLM-as-Verifier for Filtering False Positives.

    Workflow:
    1. Load DINO detections from Task 3a results
    2. For each image/class:
       - Get support examples for that class
       - For each detection:
         * Crop bbox region
         * Ask VLM: "Does this match the target?"
         * Store verification confidence
    3. Filter by confidence threshold
    4. Evaluate: precision/recall before vs after
    """

    def __init__(
        self,
        project_name: str,
        config: dict,
    ):
        """
        Initialize Task 3b verification experiment.

        Args:
            project_name: W&B project name
            config: Configuration dict with:
                - test_loader: dataloader for query images
                - verification_model: VLM for verification (e.g., Qwen2.5-VL)
                - coco_gt: COCO ground truth for evaluation
                - confidence_threshold: threshold for filtering (default: 0.5)
                - k_shot: number of support examples
        """
        self.project_name = project_name
        self.config = config
        self.dataloader = config["test_loader"]
        self.verifier = config.get("verification_model")
        self.coco_gt = config.get("coco_gt")
        self.k_shot = config.get("k_shot", 2)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.verify_strategy = VerificationStrategy(show_bbox_numbers=True)

        # Initialize WandB
        wandb_cfg = {
            "task": "Task3b_VLM_Verification",
            "verifier_model": self.verifier.model_name if self.verifier else "None",
            "k_shot": self.k_shot,
            "confidence_threshold": self.confidence_threshold,
            "dataset": config.get("dataset", "coco_val2017"),
        }

        if settings.eval_split_path:
            wandb_cfg["eval_split_path"] = settings.eval_split_path

        wandb.init(
            project=self.project_name,
            name=f"Task3b_Verifier_th{self.confidence_threshold:.1f}",
            config=wandb_cfg,
        )

        logger.info("=" * 80)
        logger.info("TASK 3b: VLM-as-Verifier Pipeline")
        logger.info("=" * 80)
        logger.info(f"Verifier: {self.verifier.model_name if self.verifier else 'None'}")
        logger.info(f"K-shot: {self.k_shot}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        logger.info("=" * 80)

    def load_detections(
        self,
        detection_file: str | Path,
    ) -> dict[int, list[dict]]:
        """
        Load Task 3a detection results.

        Args:
            detection_file: Path to JSON file with detections

        Returns:
            Dict mapping image_id to list of detections:
            {
                image_id: [
                    {"category_id": 1, "bbox": [x,y,w,h], "score": 1.0},
                    ...
                ]
            }
        """
        detection_file = Path(detection_file)
        if not detection_file.exists():
            raise FileNotFoundError(f"Detection file not found: {detection_file}")

        with open(detection_file) as f:
            detections_list = json.load(f)

        # Reorganize: list → dict by image_id
        detections_by_image = defaultdict(list)
        for det in detections_list:
            image_id = det["image_id"]
            detections_by_image[image_id].append(det)

        logger.info(f"Loaded {len(detections_by_image)} images with detections "
                   f"({len(detections_list)} total detections)")
        return detections_by_image

    def run_verification(self, detections_by_image: dict[int, list[dict]]) -> tuple[list, dict]:
        """
        Run verification on all detections.

        Args:
            detections_by_image: Detection results from Task 3a

        Returns:
            Tuple of:
            - verified_results: Filtered detections in COCO format
            - verification_stats: Statistics on verification process
        """
        verified_results = []
        verification_stats = {
            "total_images": 0,
            "total_detections_pre_filter": 0,
            "total_detections_post_filter": 0,
            "false_positives_caught": 0,
            "total_verification_errors": 0,
            "verification_by_class": {},
        }

        logger.info("Starting verification process...")

        for batch_idx, batch in enumerate(tqdm(
            self.dataloader,
            desc="Task 3b: VLM Verification of Detections"
        )):
            for item in batch:
                image_id = item["image_id"]
                query_image = item["image"]
                img_w, img_h = item["width"], item["height"]
                support_by_cat = item.get("support_by_cat", {})

                # Skip if no detections for this image
                if image_id not in detections_by_image:
                    continue

                verification_stats["total_images"] += 1

                # Get detections for this image, grouped by class
                detections = detections_by_image[image_id]
                detections_by_class = defaultdict(list)
                for det in detections:
                    cat_id = det["category_id"]
                    detections_by_class[cat_id].append(det)

                # Verify detections for each class
                for cat_id, class_detections in detections_by_class.items():
                    # Get class name from dataloader
                    class_name = None
                    for target in item.get("targets", []):
                        if target["category_id"] == cat_id:
                            class_name = target["class_name"]
                            break

                    if not class_name:
                        logger.warning(f"Could not find class name for category {cat_id}")
                        continue

                    support_examples = support_by_cat.get(cat_id, [])
                    support_images = [ex.image for ex in support_examples]

                    # Verify this class's detections
                    filtered, stats = self.verifier.batch_verify_detections(
                        query_image=query_image,
                        detections=[det["bbox"] for det in class_detections],
                        support_images=support_images,
                        class_name=class_name,
                        confidence_threshold=self.confidence_threshold,
                    )

                    verification_stats["total_detections_pre_filter"] += len(class_detections)
                    verification_stats["total_detections_post_filter"] += len(filtered)
                    verification_stats["false_positives_caught"] += (
                        len(class_detections) - len(filtered)
                    )
                    verification_stats["total_verification_errors"] += stats.get(
                        "verification_errors", 0
                    )

                    # Track per-class stats
                    if class_name not in verification_stats["verification_by_class"]:
                        verification_stats["verification_by_class"][class_name] = {
                            "pre_filter": 0,
                            "post_filter": 0,
                            "false_positives": 0,
                        }

                    verification_stats["verification_by_class"][class_name][
                        "pre_filter"
                    ] += len(class_detections)
                    verification_stats["verification_by_class"][class_name][
                        "post_filter"
                    ] += len(filtered)
                    verification_stats["verification_by_class"][class_name][
                        "false_positives"
                    ] += len(class_detections) - len(filtered)

                    # Add verified detections to results
                    for filtered_det in filtered:
                        verified_results.append({
                            "image_id": image_id,
                            "category_id": cat_id,
                            "bbox": filtered_det["bbox"],
                            "score": filtered_det.get("confidence", 0.8),
                        })

        logger.info("=" * 80)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total images verified: {verification_stats['total_images']}")
        logger.info(f"Pre-filter detections: {verification_stats['total_detections_pre_filter']}")
        logger.info(f"Post-filter detections: {verification_stats['total_detections_post_filter']}")
        logger.info(f"False positives caught: {verification_stats['false_positives_caught']}")
        logger.info(f"Verification errors: {verification_stats['total_verification_errors']}")

        if verification_stats["total_detections_pre_filter"] > 0:
            fp_rate = (
                verification_stats["false_positives_caught"]
                / verification_stats["total_detections_pre_filter"]
            )
            logger.info(f"False positive rate: {fp_rate:.2%}")

        logger.info("\nPer-class breakdown:")
        for class_name, stats in verification_stats["verification_by_class"].items():
            logger.info(
                f"  {class_name}: {stats['pre_filter']} → {stats['post_filter']} "
                f"({stats['false_positives']} caught)"
            )
        logger.info("=" * 80)

        # Log to W&B
        wandb.log({
            "total_detections_pre_filter": verification_stats["total_detections_pre_filter"],
            "total_detections_post_filter": verification_stats["total_detections_post_filter"],
            "false_positives_caught": verification_stats["false_positives_caught"],
            "verification_errors": verification_stats["total_verification_errors"],
        })

        return verified_results, verification_stats

    def evaluate_results(self, verified_results: list[dict]) -> dict:
        """
        Evaluate verified results against ground truth.

        Args:
            verified_results: Detections in COCO format after verification

        Returns:
            Dict with evaluation metrics (AP, AR, etc.)
        """
        if not self.coco_gt:
            logger.warning("No ground truth provided, skipping evaluation")
            return {}

        logger.info("Evaluating verified results...")

        # Create COCO results format
        coco_dt = self.coco_gt.loadRes(verified_results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        metrics = {
            "AP": coco_eval.stats[0],
            "AP50": coco_eval.stats[1],
            "AP75": coco_eval.stats[2],
            "APsmall": coco_eval.stats[3],
            "APmedium": coco_eval.stats[4],
            "APlarge": coco_eval.stats[5],
            "AR1": coco_eval.stats[6],
            "AR10": coco_eval.stats[7],
            "AR100": coco_eval.stats[8],
            "ARsmall": coco_eval.stats[9],
            "ARmedium": coco_eval.stats[10],
            "ARlarge": coco_eval.stats[11],
        }

        logger.info("Evaluation metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        # Log to W&B
        wandb.log({"verified_" + k: v for k, v in metrics.items()})

        return metrics


def main():
    """Main entry point for Task 3b verification pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Task 3b: VLM-as-Verifier - Filter false positives from DINO detections"
    )
    parser.add_argument(
        "--detection-results",
        type=str,
        required=True,
        help="Path to Task 3a detection results JSON",
    )
    parser.add_argument(
        "--k-shot",
        type=int,
        default=2,
        help="Number of support examples (default: 2)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Verification confidence threshold for filtering (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/task3b_verified_detections.json",
        help="Output path for verified results",
    )

    args = parser.parse_args()

    logger.info("Task 3b: VLM-as-Verifier Verification Pipeline")
    logger.info(f"Detection results: {args.detection_results}")
    logger.info(f"K-shot: {args.k_shot}")
    logger.info(f"Confidence threshold: {args.confidence_threshold}")

    # Initialize components
    logger.info("Initializing components...")
    from pycocotools.coco import COCO

    device = settings.device
    logger.info(f"Using device: {device}")

    # Load COCO ground truth
    coco_gt = COCO(settings.ann_file)

    # Get excluded image IDs
    eval_split_path = settings.resolved_eval_split_path()
    excluded_ids = set()
    if eval_split_path:
        with open(eval_split_path) as f:
            eval_split = json.load(f)
            excluded_ids = set(eval_split.get("image_ids", []))
    else:
        # Use all COCO val2017 images
        excluded_ids = set(coco_gt.getImgIds())

    # Initialize support sampler
    support_sampler = SupportSetSampler(
        ann_file=settings.ann_file,
        img_dir=settings.img_dir,
        excluded_image_ids=excluded_ids,
        seed=settings.few_shot_seed,
    )

    # Get dataloader
    test_loader = get_coco_few_shot_dataloader(
        ann_file=settings.ann_file,
        img_dir=settings.img_dir,
        support_sampler=support_sampler,
        k_shot=args.k_shot,
        batch_size=1,
        num_workers=0,
        eval_split_path=eval_split_path,
    )

    # Initialize VLM verifier
    logger.info("Initializing VLM for verification...")
    vlm = Qwen2_5_VL(device=device)
    verifier = VLMVerifier(vlm=vlm, device=device)

    # Create experiment config
    config = {
        "test_loader": test_loader,
        "verification_model": verifier,
        "coco_gt": coco_gt,
        "k_shot": args.k_shot,
        "confidence_threshold": args.confidence_threshold,
        "dataset": "coco_val2017_subset" if eval_split_path else "coco_val2017",
    }

    # Run experiment
    experiment = Task3bVerificationExperiment(
        project_name="DLCV-Task3b-Verification",
        config=config,
    )

    # Load detections from Task 3a
    logger.info(f"Loading Task 3a detections from {args.detection_results}...")
    detections_by_image = experiment.load_detections(args.detection_results)

    # Run verification
    logger.info("Running verification...")
    verified_results, verification_stats = experiment.run_verification(detections_by_image)

    # Evaluate results
    logger.info("Evaluating verified results...")
    eval_metrics = experiment.evaluate_results(verified_results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(verified_results, f, indent=2)

    logger.info(f"Verified results saved to {output_path}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TASK 3b SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Original detections: {verification_stats['total_detections_pre_filter']}")
    logger.info(f"Verified detections: {verification_stats['total_detections_post_filter']}")
    logger.info(f"False positives filtered: {verification_stats['false_positives_caught']}")

    if verification_stats["total_detections_pre_filter"] > 0:
        precision_improvement = (
            verification_stats["false_positives_caught"]
            / verification_stats["total_detections_pre_filter"]
        )
        logger.info(f"False positive rate: {precision_improvement:.2%}")

    logger.info("=" * 80)

    wandb.finish()


if __name__ == "__main__":
    main()
