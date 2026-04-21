import json
import wandb
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval

from config import settings
from prompts import get_prompt_strategy


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
        }
        ds = self.dataloader.dataset
        man = getattr(ds, "manifest", None)
        if man is not None:
            wandb_cfg["eval_split_n_images"] = len(man.image_ids)
            wandb_cfg["eval_split_seed"] = man.seed
            wandb_cfg["eval_cat_ids"] = (
                man.eval_cat_ids if man.eval_cat_ids is not None else "all_80"
            )
            if settings.eval_split_path:
                wandb_cfg["eval_split_path"] = settings.eval_split_path

        wandb.init(
            project=self.project_name,
            name=f"{self.model.model_name}_ZeroShot",
            config=wandb_cfg,
        )

    def run_evaluation(self):
        results = []

        for batch in tqdm(self.dataloader, desc=f"Evaluating {self.model.model_name}"):
            for item in batch:
                if not item["targets"]:
                    continue
                image_id = item["image_id"]
                image = item["image"]
                img_w, img_h = item["width"], item["height"]

                unique_classes = {
                    target["category_id"]: target["class_name"]
                    for target in item["targets"]
                }

                for cat_id, class_name in unique_classes.items():
                    boxes = self.model.predict(image, class_name, img_w, img_h)
                    for box in boxes:
                        results.append(
                            {
                                "image_id": image_id,
                                "category_id": cat_id,
                                "bbox": box,
                                "score": 1.0,
                            }
                        )

        # Save and Evaluate
        result_file = f"{self.model.model_name}_results.json"
        with open(result_file, "w") as f:
            json.dump(results, f)
            
        self._calculate_and_log_metrics(result_file)
        wandb.finish()

    def _calculate_and_log_metrics(self, result_file: str):
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
                "experiment_mode": "few_shot",
            }
        )

    def run_evaluation(self):
        results = []

        for batch in tqdm(self.dataloader, desc=f"Evaluating {self.model.model_name} (few-shot)"):
            for item in batch:
                if not item["targets"]:
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
                    except NotImplementedError:
                        boxes = self.model.predict(query_for_model, class_name, img_w, img_h)

                    for box in boxes:
                        results.append(
                            {
                                "image_id": image_id,
                                "category_id": cat_id,
                                "bbox": box,
                                "score": 1.0,
                            }
                        )

        result_file = f"{self.model.model_name}_few_shot_results.json"
        with open(result_file, "w") as f:
            json.dump(results, f)

        self._calculate_and_log_metrics(result_file)
        wandb.finish()