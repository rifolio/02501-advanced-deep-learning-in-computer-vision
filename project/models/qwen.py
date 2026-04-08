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
        pattern = r"<\|box_start\|>\((\d+),(\d+),(\d+),(\d+)\)<\|box_end\|>"
        matches = re.findall(pattern, text)
        
        for match in matches:
            ymin, xmin, ymax, xmax = map(int, match)
            abs_xmin = (xmin / 1000.0) * img_width
            abs_ymin = (ymin / 1000.0) * img_height
            abs_xmax = (xmax / 1000.0) * img_width
            abs_ymax = (ymax / 1000.0) * img_height
            
            width = abs_xmax - abs_xmin
            height = abs_ymax - abs_ymin
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