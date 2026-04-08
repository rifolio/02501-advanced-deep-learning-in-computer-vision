import json
import wandb
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval

class Experiment:
    def __init__(self, project_name: str, config: dict):
        self.project_name = project_name
        self.config = config
        self.dataloader = config['test_loader']
        self.model = config['model']
        self.dataset_name = config['dataset']
        
        wandb.init(
            project=self.project_name,
            name=f"{self.model.model_name}_ZeroShot",
            config={
                "model": self.model.model_name,
                "dataset": self.dataset_name
            }
        )

    def run_evaluation(self):
        results = []
        
        for batch in tqdm(self.dataloader, desc=f"Evaluating {self.model.model_name}"):
            for item in batch:
                image_id = item['image_id']
                image = item['image']
                img_w, img_h = item['width'], item['height']
                
                for target in item['targets']:
                    cat_id = target['category_id']
                    class_name = target['class_name']
                    
                    # Call the plug-and-play model
                    boxes = self.model.predict(image, class_name, img_w, img_h)
                    
                    for box in boxes:
                        results.append({
                            "image_id": image_id,
                            "category_id": cat_id,
                            "bbox": box,
                            "score": 1.0
                        })

        # Save and Evaluate
        result_file = f"{self.model.model_name}_results.json"
        with open(result_file, "w") as f:
            json.dump(results, f)
            
        self._calculate_and_log_metrics(result_file)
        wandb.finish()

    def _calculate_and_log_metrics(self, result_file: str):
        cocoGt = self.dataloader.dataset.coco
        cocoDt = cocoGt.loadRes(result_file)
        
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
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