import re
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from .base_vlm import BaseVLM

class Qwen2_5_VL(BaseVLM):
    def __init__(self, device: str):
        super().__init__(device)
        self.model_name = "Qwen2.5-VL-7B"
        self.model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def _parse_boxes(self, text: str, img_width: int, img_height: int) -> list:
        boxes = []
        # changed matches to the following way for QWEN2.5: absolute coordinates in (xmin, ymin), (xmax, ymax) format
        # Matches <|box_start|>(x1, y1), (x2, y2)<|box_end|> with optional spaces
        pattern = r"<\|box_start\|>\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)<\|box_end\|>"
        matches = re.findall(pattern, text)

        #Calculate the dimensions the processor resized the image to
        resized_width = round(img_width / 28.0) * 28
        resized_height = round(img_height / 28.0) * 28
        
        #Calculate the ratio to scale back to the original
        width_ratio = img_width / resized_width if resized_width > 0 else 1
        height_ratio = img_height / resized_height if resized_height > 0 else 1

        for match in matches:
        #changed this as QWEN2.5 outputs absolute scales! there's no need to 
        #worry about normalized or scaled coordinates AFAIK

            xmin, ymin, xmax, ymax = map(int, match)
            
            # Calculate width and height directly using absolute coordinates
            width = xmax - xmin
            height = ymax - ymin

            # Append in COCO format: [x, y, width, height]
            boxes.append([abs_xmin, abs_ymin, width, height])
        return boxes

    def predict(self, image, target_class: str, img_width: int, img_height: int) -> list:
        prompt_text = f"Detect all {target_class} in this image."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]
        
        return self._parse_boxes(output_text, img_width, img_height)