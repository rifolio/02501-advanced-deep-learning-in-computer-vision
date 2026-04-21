import json
import logging
from pathlib import Path
import wandb
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval

from config import settings
from prompts import get_prompt_strategy

logger = logging.getLogger(__name__)

class Experiment:
    def __init__(self, project_name: str, config: dict):
        self.project_name = project_name
        self.config = config
        self.dataloader = config["test_loader"]
        self.model = config["model"]
        self.dataset_name = config["dataset"]

        wandb_cfg = {
            "model": self.model.model_name,
            "dataset": self.dataset_name,
            "settings_model_name": settings.model_name,
            "experiment_mode": settings.experiment_mode,
        }
        if settings.eval_split_path:
            wandb_cfg["eval_split_path"] = settings.eval_split_path
        ds = self.dataloader.dataset
        man = getattr(ds, "manifest", None)
        if man is not None:
            wandb_cfg["eval_split_n_images"] = len(man.image_ids)
            wandb_cfg["eval_split_seed"] = man.seed
            wandb_cfg["eval_cat_ids"] = (
                man.eval_cat_ids if man.eval_cat_ids is not None else "all_80"
            )

        wandb.init(
            project=self.project_name,
            name=f"{self.model.model_name}_ZeroShot",
            config=wandb_cfg,
        )

    def run_evaluation(self):
        results = []
        counters = {
            "num_items_total": 0,
            "num_items_skipped_no_targets": 0,
            "num_class_queries": 0,
            "num_queries_zero_boxes": 0,
            "num_predictions_written": 0,
            "parser_fallback_used": 0,
        }

        for batch in tqdm(self.dataloader, desc=f"Evaluating {self.model.model_name}"):
            for item in batch:
                counters["num_items_total"] += 1
                if not item["targets"]:
                    counters["num_items_skipped_no_targets"] += 1
                    continue
                image_id = item["image_id"]
                image = item["image"]
                img_w, img_h = item["width"], item["height"]

                unique_classes = {
                    target["category_id"]: target["class_name"]
                    for target in item["targets"]
                }

                for cat_id, class_name in unique_classes.items():
                    counters["num_class_queries"] += 1
                    boxes = self.model.predict(image, class_name, img_w, img_h)
                    counters["parser_fallback_used"] += self._consume_runtime_stat("parser_fallback_used")
                    if not boxes:
                        counters["num_queries_zero_boxes"] += 1
                        logger.info(
                            (
                                "[query_debug] model=%s mode=zero_shot image_id=%s category_id=%s "
                                "class_name=%r parsed_boxes=0"
                            ),
                            self.model.model_name,
                            image_id,
                            cat_id,
                            class_name,
                        )
                    for box in boxes:
                        results.append(
                            {
                                "image_id": image_id,
                                "category_id": cat_id,
                                "bbox": box,
                                "score": 1.0,
                            }
                        )
                        counters["num_predictions_written"] += 1

        # Save and Evaluate
        result_file = f"{self.model.model_name}_results.json"
        with open(result_file, "w") as f:
            json.dump(results, f)
        self._log_run_diagnostics(result_file, counters)
            
        self._calculate_and_log_metrics(result_file)
        wandb.finish()

    def _consume_runtime_stat(self, key: str) -> int:
        if not hasattr(self.model, "pop_runtime_stats"):
            return 0
        stats = self.model.pop_runtime_stats()
        return int(stats.get(key, 0))

    def _log_run_diagnostics(self, result_file: str, counters: dict) -> None:
        logger.info(
            (
                "Run diagnostics for %s: total_items=%s skipped_no_targets=%s class_queries=%s "
                "zero_box_queries=%s predictions_written=%s parser_fallback_used=%s"
            ),
            result_file,
            counters["num_items_total"],
            counters["num_items_skipped_no_targets"],
            counters["num_class_queries"],
            counters["num_queries_zero_boxes"],
            counters["num_predictions_written"],
            counters["parser_fallback_used"],
        )
        wandb.log(counters)
        diagnostics_file = Path(result_file).with_name(Path(result_file).stem + "_diagnostics.json")
        with open(diagnostics_file, "w", encoding="utf-8") as f:
            json.dump(counters, f, indent=2)
            f.write("\n")

    def _calculate_and_log_metrics(self, result_file: str):
        with open(result_file) as f:
            preds = json.load(f)

        if not preds:
            logger.warning(
                "No predictions found in %s. Skipping COCOeval and logging zero metrics.",
                result_file,
            )
            wandb.log(
                {
                    "mAP_0.5:0.95": 0.0,
                    "mAP_0.5": 0.0,
                    "mAP_0.75": 0.0,
                    "mAP_small": 0.0,
                    "mAP_medium": 0.0,
                    "mAP_large": 0.0,
                    "num_predictions": 0,
                    "eval_skipped_no_predictions": 1,
                }
            )
            return

        cocoGt = self.dataloader.dataset.coco
        cocoDt = cocoGt.loadRes(result_file)

        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        man = getattr(self.dataloader.dataset, "manifest", None)
        if man is not None:
            cocoEval.params.imgIds = sorted(man.image_ids)
            if man.eval_cat_ids is not None:
                cocoEval.params.catIds = sorted(man.eval_cat_ids)

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        # Log to WandB
        metrics = {
            "mAP_0.5:0.95": cocoEval.stats[0],
            "mAP_0.5": cocoEval.stats[1],
            "mAP_0.75": cocoEval.stats[2],
            "mAP_small": cocoEval.stats[3],
            "mAP_medium": cocoEval.stats[4],
            "mAP_large": cocoEval.stats[5],
            "num_predictions": len(preds),
            "eval_skipped_no_predictions": 0,
        }
        wandb.log(metrics)


class FewShotExperiment(Experiment):
    def __init__(self, project_name: str, config: dict):
        super().__init__(project_name, config)
        self.k_shot = settings.k_shot
        self.prompt_strategy_name = settings.prompt_strategy
        self.prompt_strategy = get_prompt_strategy(self.prompt_strategy_name)
        if wandb.run is not None:
            wandb.run.name = f"{self.model.model_name}_FewShot_{self.prompt_strategy_name}_{self.k_shot}shot"
        wandb.config.update(
            {
                "k_shot": self.k_shot,
                "prompt_strategy": self.prompt_strategy_name,
                "experiment_mode": settings.experiment_mode,
                "settings_model_name": settings.model_name,
            }
        )

    def run_evaluation(self):
        results = []
        fallback_to_zero_shot_count = 0
        fallback_reasons: set[str] = set()
        counters = {
            "num_items_total": 0,
            "num_items_skipped_no_targets": 0,
            "num_class_queries": 0,
            "num_queries_zero_boxes": 0,
            "num_predictions_written": 0,
            "parser_fallback_used": 0,
        }

        for batch in tqdm(self.dataloader, desc=f"Evaluating {self.model.model_name} (few-shot)"):
            for item in batch:
                counters["num_items_total"] += 1
                if not item["targets"]:
                    counters["num_items_skipped_no_targets"] += 1
                    continue

                image_id = item["image_id"]
                query_image = item["image"]
                img_w, img_h = item["width"], item["height"]
                support_by_cat = item.get("support_by_cat", {})

                unique_classes = {
                    target["category_id"]: target["class_name"]
                    for target in item["targets"]
                }
                for cat_id, class_name in unique_classes.items():
                    counters["num_class_queries"] += 1
                    prompt_bundle = self.prompt_strategy.build_prompt(
                        query_image=query_image,
                        support_by_cat=support_by_cat,
                        class_name=class_name,
                        cat_id=cat_id,
                    )
                    support_images = prompt_bundle["images"][:-1]
                    prompt_text = prompt_bundle["text"]
                    query_for_model = prompt_bundle["images"][-1]

                    try:
                        boxes = self.model.predict_few_shot(
                            query_for_model,
                            support_images,
                            prompt_text,
                            img_w,
                            img_h,
                        )
                    except NotImplementedError as e:
                        fallback_to_zero_shot_count += 1
                        fallback_reasons.add(str(e) or "predict_few_shot not implemented")
                        boxes = self.model.predict(query_for_model, class_name, img_w, img_h)
                    counters["parser_fallback_used"] += self._consume_runtime_stat("parser_fallback_used")
                    if not boxes:
                        counters["num_queries_zero_boxes"] += 1
                        logger.info(
                            (
                                "[query_debug] model=%s mode=few_shot image_id=%s category_id=%s "
                                "class_name=%r support_images=%s parsed_boxes=0"
                            ),
                            self.model.model_name,
                            image_id,
                            cat_id,
                            class_name,
                            len(support_images),
                        )

                    for box in boxes:
                        results.append(
                            {
                                "image_id": image_id,
                                "category_id": cat_id,
                                "bbox": box,
                                "score": 1.0,
                            }
                        )
                        counters["num_predictions_written"] += 1

        result_file = f"{self.model.model_name}_few_shot_results.json"
        with open(result_file, "w") as f:
            json.dump(results, f)
        self._log_run_diagnostics(result_file, counters)

        if fallback_to_zero_shot_count:
            reason_txt = "; ".join(sorted(fallback_reasons))
            logger.warning(
                "Few-shot fallback used %s times (predict_few_shot -> predict). Reasons: %s",
                fallback_to_zero_shot_count,
                reason_txt,
            )
            wandb.log(
                {
                    "few_shot_fallback_to_zero_shot_count": fallback_to_zero_shot_count,
                    "few_shot_fallback_used": 1,
                }
            )
        else:
            wandb.log({"few_shot_fallback_to_zero_shot_count": 0, "few_shot_fallback_used": 0})

        self._calculate_and_log_metrics(result_file)
        wandb.finish()