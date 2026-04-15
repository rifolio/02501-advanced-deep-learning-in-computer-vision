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

    def predict(self, image, target_class: str, img_width: int, img_height: int) -> list:
        # 1. Image Preprocessing
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif getattr(image, 'mode', None) != 'RGB':
            image = image.convert('RGB')

        # 2. Text Formatting (DINO expects a list of lists for batched inputs)
        text_labels = [[target_class]]

        # 3. Model Inference
        inputs = self.processor(images=image, text=text_labels, return_tensors="pt").to(self.device)
        
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

        result = results[0]
        coco_boxes = []
        
        # 5. Convert [xmin, ymin, xmax, ymax] to [x, y, width, height]
        for box in result["boxes"]:
            xmin, ymin, xmax, ymax = box.tolist()
            
            width = xmax - xmin
            height = ymax - ymin
            
            # Append in COCO format
            coco_boxes.append([xmin, ymin, width, height])
            
        return coco_boxes