#!/usr/bin/env python3
"""
Task 3 Evaluation Script: VLM + Detector Fusion Pipeline

Demonstrates the VLM-to-text pipeline:
1. VLM analyzes support examples from the dataset subset
2. VLM generates detailed text descriptions of objects
3. Grounding DINO uses these descriptions to detect objects in query images
4. Evaluates performance on COCO validation set

IMPORTANT: To use your created subset (val_pilot.json), set:
    export EVAL_SPLIT_PATH=data/splits/val_pilot.json

Run with:
    uv run python task3_eval.py
    
Or with custom config:
    EVAL_SPLIT_PATH=data/splits/val_pilot.json uv run python task3_eval.py --k-shot 2
    
The support sets are sampled from the full COCO dataset (not the subset).
The evaluation (query images) uses your specified subset.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import wandb
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval

from config import settings
from data.dataloaders import get_coco_few_shot_dataloader
from data.support_sampler import SupportSetSampler
from models.vlm_dino_fusion import VLMDINOFusion
from models.qwen import Qwen2_5_VL
from models.grounding_dino import GroundingDINO
from pipeline import FewShotExperiment
from prompts import get_prompt_strategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Task3Experiment(FewShotExperiment):
    """
    Task 3: VLM + Detector Fusion Pipeline Experiment.
    
    Extends FewShotExperiment to explicitly demonstrate VLM text generation
    for Grounding DINO prompting.
    """
    
    def __init__(self, project_name: str, config: dict):
        """Initialize Task 3 experiment with VLM-DINO fusion."""
        # Initialize parent with VLMDINOFusion model
        config["model"] = VLMDINOFusion(device=settings.device)
        super().__init__(project_name, config)
        
        # Update WandB config with Task 3 specifics
        wandb.config.update({
            "task": "Task3_VLM_DINO_Fusion",
            "vlm_model": config["model"].vlm.model_name,
            "detector_model": config["model"].dino.model_name,
        })
    
    def run_evaluation(self):
        """Run evaluation with enhanced logging for Task 3."""
        results = []
        evaluated_image_ids: list[int] = []
        generation_stats = {
            "total_descriptions_generated": 0,
            "total_descriptions_failed": 0,
            "total_dino_detections": 0,
            "total_dino_failures": 0,
        }
        
        counters = {
            "num_items_total": 0,
            "num_items_skipped_no_targets": 0,
            "num_class_queries": 0,
            "num_queries_zero_boxes": 0,
            "num_predictions_written": 0,
            "parser_fallback_used": 0,
        }

        logger.info("=" * 80)
        logger.info("TASK 3: VLM + Detector Fusion Pipeline")
        logger.info("=" * 80)
        logger.info(f"VLM: {self.model.vlm.model_name}")
        logger.info(f"Detector: {self.model.dino.model_name}")
        logger.info(f"K-shot: {self.k_shot}")
        logger.info(f"Subset: {len(self.dataloader.dataset)} images")
        logger.info("=" * 80)

        for batch in tqdm(self.dataloader, desc="Task 3: VLM→Text→DINO"):
            for item in batch:
                counters["num_items_total"] += 1
                if not item["targets"]:
                    counters["num_items_skipped_no_targets"] += 1
                    continue

                image_id = item["image_id"]
                evaluated_image_ids.append(int(image_id))
                query_image = item["image"]
                img_w, img_h = item["width"], item["height"]
                support_by_cat = item.get("support_by_cat", {})

                unique_classes = {
                    target["category_id"]: target["class_name"]
                    for target in item["targets"]
                }
                
                for cat_id, class_name in unique_classes.items():
                    counters["num_class_queries"] += 1
                    
                    # Get support examples for this category
                    support_examples = support_by_cat.get(cat_id, [])
                    support_images = [ex.image for ex in support_examples]
                    
                    logger.debug(
                        f"[Task3] Image {image_id}, Class {class_name}, "
                        f"Support examples: {len(support_images)}"
                    )
                    
                    # Build prompt (includes VLM text generation)
                    prompt_bundle = self.prompt_strategy.build_prompt(
                        query_image=query_image,
                        support_by_cat=support_by_cat,
                        class_name=class_name,
                        cat_id=cat_id,
                    )
                    
                    support_images = prompt_bundle["images"][:-1]
                    prompt_text = prompt_bundle["text"]
                    query_for_model = prompt_bundle["images"][-1]
                    
                    # VLMDINOFusion.predict_few_shot will:
                    # 1. Use VLM to generate description from support_images
                    # 2. Use Grounding DINO with generated description
                    try:
                        boxes = self.model.predict_few_shot(
                            query_for_model,
                            support_images,
                            prompt_text,
                            img_w,
                            img_h,
                        )
                        
                        # Track model stats
                        runtime_stats = self.model.pop_runtime_stats()
                        if "vlm_descriptions_generated" in runtime_stats:
                            generation_stats["total_descriptions_generated"] += 1
                        if "vlm_generation_failed" in runtime_stats:
                            generation_stats["total_descriptions_failed"] += 1
                        if "dino_predictions_made" in runtime_stats:
                            generation_stats["total_dino_detections"] += 1
                        if "dino_detection_failed" in runtime_stats:
                            generation_stats["total_dino_failures"] += 1
                        
                        counters["parser_fallback_used"] += runtime_stats.get(
                            "parser_fallback_used", 0
                        )
                        
                    except NotImplementedError as e:
                        logger.error(f"Model error: {e}")
                        boxes = []
                    
                    if not boxes:
                        counters["num_queries_zero_boxes"] += 1
                        logger.info(
                            f"[Task3_Zero_Detections] image_id={image_id}, "
                            f"class={class_name}, support_images={len(support_images)}"
                        )

                    for box in boxes:
                        results.append({
                            "image_id": image_id,
                            "category_id": cat_id,
                            "bbox": box,
                            "score": 1.0,
                        })
                        counters["num_predictions_written"] += 1

        # Logging and evaluation
        logger.info("=" * 80)
        logger.info("TASK 3 EVALUATION SUMMARY")
        logger.info("=" * 80)
        for key, value in counters.items():
            logger.info(f"  {key}: {value}")
        logger.info("\nGeneration Statistics:")
        for key, value in generation_stats.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 80)

        # Save results and evaluate
        self._evaluate_and_log(results, evaluated_image_ids, counters)


def main():
    """Main entry point for Task 3 evaluation."""
    # Validate configuration
    eval_split = settings.resolved_eval_split_path()
    
    if eval_split:
        logger.info(f"✓ Using subset for EVALUATION: {eval_split}")
    else:
        logger.warning(
            "✗ No eval split specified. Using full COCO validation set. "
            "To use your subset, set: EVAL_SPLIT_PATH=data/splits/val_pilot.json"
        )
    
    logger.info(f"✓ Using full COCO dataset for SUPPORT SAMPLING")
    logger.info(f"  (Support examples will be sampled from full dataset, excluding eval images)")
    
    logger.info(f"Device: {settings.device}")
    logger.info(f"Model: vlm_dino_fusion (VLM: Qwen2.5-VL-7B + Detector: Grounding DINO)")
    logger.info(f"K-shot: {settings.k_shot}")
    logger.info(f"Experiment mode: few_shot")
    
    # Get dataloader - uses eval_split_path if set
    test_loader = get_coco_few_shot_dataloader(k_shot=settings.k_shot)
    
    # Configure experiment
    experiment_config = {
        "test_loader": test_loader,
        "model": None,  # Will be set in Task3Experiment.__init__
        "dataset": "coco_val2017",
    }
    
    # Run Task 3 evaluation
    experiment = Task3Experiment(
        project_name="Task3_VLM_DINO_Fusion",
        config=experiment_config
    )
    experiment.run_evaluation()
    
    logger.info("\n✓ Task 3 evaluation complete!")
    logger.info(f"  Results logged to WandB: {experiment.project_name}")


if __name__ == "__main__":
    main()
