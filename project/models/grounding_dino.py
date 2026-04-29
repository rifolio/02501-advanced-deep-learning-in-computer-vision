import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Assuming base_vlm.py is in the same directory. Adjust import if needed.
from .base_vlm import BaseVLM 

class GroundingDINO(BaseVLM):
    def __init__(self, device: str):
        super().__init__(device)
        self.model_name = "Grounding-DINO-Tiny"
        self.model_id = "IDEA-Research/grounding-dino-tiny"
        
        # Initialize processor and model
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)

    def _run_detection(self, image, target_class: str, img_width: int, img_height: int):
        # 1. Image Preprocessing
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif getattr(image, 'mode', None) != 'RGB':
            image = image.convert('RGB')

        # 2. Text Formatting — processor expects a single string (or List[str] for batch).
        # Ensure the prompt ends with a period; Grounding DINO uses "." as a phrase separator.
        text_prompt = target_class.strip()
        if not text_prompt.endswith("."):
            text_prompt += "."

        # 3. Model Inference
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 4. Post-Processing
        # target_sizes expects (height, width)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.4,  # Note: HF updated 'threshold' to 'box_threshold' in recent versions
            text_threshold=0.3,
            target_sizes=[(img_height, img_width)] 
        )
        return results[0]

    def predict_with_scores(
        self,
        image,
        target_class: str,
        img_width: int,
        img_height: int,
    ) -> list[dict]:
        result = self._run_detection(image, target_class, img_width, img_height)
        scored_predictions = []
        boxes = result["boxes"]
        scores = result.get("scores")

        for idx, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box.tolist()
            width = xmax - xmin
            height = ymax - ymin
            score = float(scores[idx].item()) if scores is not None else 0.0
            scored_predictions.append(
                {
                    "bbox": [xmin, ymin, width, height],
                    "score": score,
                    "score_source": "model",
                    "score_policy": "grounding_dino_postprocess",
                }
            )
        return scored_predictions

    def predict(self, image, target_class: str, img_width: int, img_height: int) -> list:
        coco_boxes = []
        for prediction in self.predict_with_scores(image, target_class, img_width, img_height):
            coco_boxes.append(prediction["bbox"])
        return coco_boxes