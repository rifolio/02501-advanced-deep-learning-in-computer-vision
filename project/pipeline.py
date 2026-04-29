import json
import logging
from pathlib import Path
from collections import defaultdict
import torch
import wandb
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from PIL import Image, ImageDraw

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
        evaluated_image_ids: list[int] = []
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
                image_id = item["image_id"]
                evaluated_image_ids.append(int(image_id))
                image = item["image"]
                img_w, img_h = item["width"], item["height"]

                query_targets = item.get("query_targets", [])
                for target in query_targets:
                    cat_id = target["category_id"]
                    class_name = target["class_name"]
                    counters["num_class_queries"] += 1
                    scored_predictions = self.model.predict_with_scores(image, class_name, img_w, img_h)
                    counters["parser_fallback_used"] += self._consume_runtime_stat("parser_fallback_used")
                    if not scored_predictions:
                        counters["num_queries_zero_boxes"] += 1
                        logger.info(
                            (
                                "[query_debug] model=%s mode=zero_shot image_id=%s category_id=%s "
                                "class_name=%r num_predictions=0"
                            ),
                            self.model.model_name,
                            image_id,
                            cat_id,
                            class_name,
                        )
                    for prediction in scored_predictions:
                        results.append(
                            {
                                "image_id": image_id,
                                "category_id": cat_id,
                                "bbox": prediction["bbox"],
                                "score": float(prediction["score"]),
                            }
                        )
                        counters["num_predictions_written"] += 1

        # Save and Evaluate
        result_file = f"{self.model.model_name}_results.json"
        with open(result_file, "w") as f:
            json.dump(results, f)
        self._log_run_diagnostics(result_file, counters)
            
        self._calculate_and_log_metrics(result_file)
        self._maybe_log_viz_artifact(result_file, evaluated_image_ids)
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

    def _maybe_log_viz_artifact(self, result_file: str, evaluated_image_ids: list[int]) -> None:
        if not settings.log_viz_artifact:
            return
        run = wandb.run
        if run is None:
            logger.warning("Skipping viz artifact logging because wandb.run is None.")
            return

        with open(result_file, encoding="utf-8") as f:
            preds = json.load(f)
        if not preds:
            logger.info("Skipping viz artifact logging because %s has no predictions.", result_file)
            return

        dataset = self.dataloader.dataset
        coco = getattr(dataset, "coco", None)
        img_dir = getattr(dataset, "img_dir", None)
        if coco is None or img_dir is None:
            logger.warning("Skipping viz artifact logging because dataset is missing coco/img_dir.")
            return

        target_cat_id = None
        if settings.viz_target_category:
            cat_ids = coco.getCatIds(catNms=[settings.viz_target_category])
            if not cat_ids:
                logger.warning(
                    "viz_target_category=%r not found in COCO categories; using all predictions.",
                    settings.viz_target_category,
                )
            else:
                target_cat_id = cat_ids[0]

        preds_by_img: dict[int, list[dict]] = defaultdict(list)
        for pred in preds:
            if target_cat_id is not None and pred["category_id"] != target_cat_id:
                continue
            preds_by_img[int(pred["image_id"])].append(pred)

        stem = Path(result_file).stem
        out_dir = Path(settings.viz_output_dir) / f"{stem}_side_by_side"
        out_dir.mkdir(parents=True, exist_ok=True)

        cat_id_to_name = {
            cat["id"]: cat["name"]
            for cat in coco.loadCats(coco.getCatIds())
        }

        unique_eval_ids = list(dict.fromkeys(int(x) for x in evaluated_image_ids))
        if not unique_eval_ids:
            logger.info("No evaluated image IDs captured for visualization; skipping viz artifact.")
            return

        rendered_paths: list[Path] = []
        for idx, image_id in enumerate(unique_eval_ids):
            if idx >= settings.viz_max_images:
                break
            image_preds = preds_by_img.get(image_id, [])

            img_info = coco.loadImgs(image_id)[0]
            img_path = Path(img_dir) / img_info["file_name"]
            if not img_path.exists():
                logger.warning("Image not found for visualization: %s", img_path)
                continue

            base_img = Image.open(img_path).convert("RGB")
            gt_panel = base_img.copy()
            pred_panel = base_img.copy()
            draw_gt = ImageDraw.Draw(gt_panel)
            draw_pred = ImageDraw.Draw(pred_panel)

            # Draw GT boxes for target category (if set) or all categories in the image.
            # pycocotools treats catIds=None as [None] (matches nothing);
            # an empty list [] means "no category filter" (matches all).
            if target_cat_id is not None:
                gt_cat_ids = [target_cat_id]
            else:
                gt_cat_ids = []
            ann_ids = coco.getAnnIds(imgIds=[image_id], catIds=gt_cat_ids, iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                x, y, w, h = ann["bbox"]
                cat_name = cat_id_to_name.get(int(ann["category_id"]), str(ann["category_id"]))
                draw_gt.rectangle([x, y, x + w, y + h], outline="lime", width=3)
                draw_gt.text((x, max(0, y - 14)), f"GT:{cat_name}", fill="lime")

            for pred_item in image_preds:
                x, y, w, h = pred_item["bbox"]
                cat_name = cat_id_to_name.get(int(pred_item["category_id"]), str(pred_item["category_id"]))
                draw_pred.rectangle([x, y, x + w, y + h], outline="red", width=3)
                draw_pred.text((x, max(0, y - 14)), f"P:{cat_name}", fill="red")

            w, h = base_img.size
            header_h = 30
            canvas = Image.new("RGB", (w * 2, h + header_h), color="white")
            canvas.paste(gt_panel, (0, header_h))
            canvas.paste(pred_panel, (w, header_h))
            draw_canvas = ImageDraw.Draw(canvas)
            draw_canvas.text((10, 8), "Ground Truth (green)", fill="black")
            draw_canvas.text((w + 10, 8), "Prediction (red)", fill="black")

            out_path = out_dir / f"{idx + 1:03d}_{image_id}.jpg"
            canvas.save(out_path, quality=90)
            rendered_paths.append(out_path)

        if not rendered_paths:
            logger.info("No visualization files were rendered; skipping artifact upload.")
            return

        preview_count = max(0, settings.viz_preview_count)
        if preview_count:
            preview_images = [wandb.Image(str(p)) for p in rendered_paths[:preview_count]]
            wandb.log({"viz_side_by_side_preview": preview_images})

        artifact = wandb.Artifact(
            name=f"{run.id}-side-by-side-viz",
            type="evaluation-viz",
            description=f"GT vs prediction overlays for {stem}",
            metadata={
                "model": self.model.model_name,
                "dataset": self.dataset_name,
                "result_file": result_file,
                "num_rendered_images": len(rendered_paths),
                "viz_max_images": settings.viz_max_images,
                "viz_target_category": settings.viz_target_category or "all",
            },
        )
        artifact.add_dir(str(out_dir))
        run.log_artifact(artifact)
        logger.info(
            "Logged side-by-side visualization artifact from %s with %s images.",
            out_dir,
            len(rendered_paths),
        )


class FewShotExperiment(Experiment):
    def __init__(self, project_name: str, config: dict):
        super().__init__(project_name, config)
        self.k_shot = settings.k_shot
        self.prompt_strategy_name = settings.prompt_strategy
        if self.prompt_strategy_name.lower() == "verification":
            raise ValueError(
                "Invalid configuration: prompt_strategy='verification' cannot run in the bbox "
                "detection few-shot pipeline. Run the dedicated verification pipeline instead."
            )
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
        evaluated_image_ids: list[int] = []
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

                image_id = item["image_id"]
                evaluated_image_ids.append(int(image_id))
                query_image = item["image"]
                img_w, img_h = item["width"], item["height"]
                support_by_cat = item.get("support_by_cat", {})

                query_targets = item.get("query_targets", [])
                for target in query_targets:
                    cat_id = target["category_id"]
                    class_name = target["class_name"]
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
                    logger.info(
                        (
                            "[few_shot_input_debug] model=%s prompt_strategy=%s image_id=%s "
                            "category_id=%s class_name=%r support_images=%s"
                        ),
                        self.model.model_name,
                        self.prompt_strategy_name,
                        image_id,
                        cat_id,
                        class_name,
                        len(support_images),
                    )
                    #auxiliary VRAM checking
                    #vram_used = torch.cuda.memory_allocated() / (1024**3)
                    #logger.info(f"VRAM before predicting {class_name}: {vram_used:.2f} GB")
                    try:
                        scored_predictions = self.model.predict_few_shot_with_scores(
                            query_for_model,
                            support_images,
                            prompt_text,
                            img_w,
                            img_h,
                            class_name=class_name,
                        )
                    except NotImplementedError as e:
                        fallback_to_zero_shot_count += 1
                        fallback_reasons.add(str(e) or "predict_few_shot not implemented")
                        scored_predictions = self.model.predict_with_scores(
                            query_for_model,
                            class_name,
                            img_w,
                            img_h,
                        )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    counters["parser_fallback_used"] += self._consume_runtime_stat("parser_fallback_used")
                    if not scored_predictions:
                        counters["num_queries_zero_boxes"] += 1
                        logger.info(
                            (
                                "[query_debug] model=%s mode=few_shot image_id=%s category_id=%s "
                                "class_name=%r support_images=%s num_predictions=0"
                            ),
                            self.model.model_name,
                            image_id,
                            cat_id,
                            class_name,
                            len(support_images),
                        )

                    for prediction in scored_predictions:
                        results.append(
                            {
                                "image_id": image_id,
                                "category_id": cat_id,
                                "bbox": prediction["bbox"],
                                "score": float(prediction["score"]),
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
        self._maybe_log_viz_artifact(result_file, evaluated_image_ids)
        wandb.finish()