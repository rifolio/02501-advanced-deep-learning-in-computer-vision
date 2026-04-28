import re
import logging
import hashlib
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from .base_vlm import BaseVLM

logger = logging.getLogger(__name__)

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

    def _parse_boxes(self, text: str, img_width: int, img_height: int) -> tuple[list, bool]:
        boxes = []
        # Qwen2.5 grounding output format:
        # <|box_start|>(x1, y1), (x2, y2)<|box_end|>
        pattern = r"<\|box_start\|>\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)<\|box_end\|>"
        matches = re.findall(pattern, text)
        parser_fallback_used = False

        if not matches:
            # Fallback: tolerate decimals and optional inner brackets.
            fallback_pattern = (
                r"<\|box_start\|>\(\s*\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?\s*\)\s*,\s*"
                r"\(\s*\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?\s*\)<\|box_end\|>"
            )
            matches = re.findall(fallback_pattern, text)
            parser_fallback_used = bool(matches)

        if not matches:
            # Secondary fallback for strict list output:
            # [[x1,y1,x2,y2], ...]
            list_pattern = (
                r"\[\s*\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
                r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?\s*\]"
            )
            matches = re.findall(list_pattern, text)
            parser_fallback_used = bool(matches)

        # Processor resizes image sides to multiples of 28.
        resized_width = round(img_width / 28.0) * 28
        resized_height = round(img_height / 28.0) * 28

        # Scale model output back to original dimensions.
        width_ratio = img_width / resized_width if resized_width > 0 else 1
        height_ratio = img_height / resized_height if resized_height > 0 else 1

        for match in matches:
            xmin, ymin, xmax, ymax = map(float, match)

            abs_xmin = xmin * width_ratio
            abs_ymin = ymin * height_ratio
            abs_xmax = xmax * width_ratio
            abs_ymax = ymax * height_ratio
            width = abs_xmax - abs_xmin
            height = abs_ymax - abs_ymin
            boxes.append([abs_xmin, abs_ymin, width, height])
        return boxes, parser_fallback_used

    def _provisional_score_policy(self) -> str:
        """
        Qwen grounding text output does not include calibrated confidence.
        """
        return "qwen_box_only_rank_decay_v1"

    def _log_inference_debug(
        self,
        prompt_text: str,
        output_text: str,
        parsed_box_count: int,
        parser_fallback_used: bool,
        support_image_count: int,
    ) -> None:
        prompt_hash = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:12]
        prompt_snippet = prompt_text.strip().replace("\n", " ")[:120]
        output_snippet = output_text.strip().replace("\n", " ")[:180]
        level = logging.INFO if parsed_box_count == 0 else logging.DEBUG
        logger.log(
            level,
            (
                "[inference_debug] model=%s support_images=%s prompt_hash=%s prompt_snippet=%r "
                "parsed_boxes=%s parser_fallback_used=%s output_snippet=%r"
            ),
            self.model_name,
            support_image_count,
            prompt_hash,
            prompt_snippet,
            parsed_box_count,
            parser_fallback_used,
            output_snippet,
        )

    @staticmethod
    def _strict_output_tail() -> str:
        return (
            "\nOutput requirements:\n"
            "Return ONLY a JSON array of boxes in this exact format: [[x1,y1,x2,y2], ...]\n"
            "Coordinates must be integers in [0,1000].\n"
            "Use x1 < x2 and y1 < y2.\n"
            "Do not return words, labels, markdown, or explanations.\n"
            "If no instance is present, return [] exactly."
        )

    def _run_messages(self, content: list, img_width: int, img_height: int) -> list:
        messages = [
            {
                "role": "user",
                "content": content,
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
        parsed_boxes, parser_fallback_used = self._parse_boxes(output_text, img_width, img_height)
        if parser_fallback_used:
            self._bump_runtime_stat("parser_fallback_used")
        if not parsed_boxes:
            self._bump_runtime_stat("queries_with_zero_boxes")

        text_blocks = [item.get("text", "") for item in content if item.get("type") == "text"]
        prompt_text = text_blocks[-1] if text_blocks else ""
        support_image_count = max(0, sum(1 for item in content if item.get("type") == "image") - 1)
        self._log_inference_debug(
            prompt_text=prompt_text,
            output_text=output_text,
            parsed_box_count=len(parsed_boxes),
            parser_fallback_used=parser_fallback_used,
            support_image_count=support_image_count,
        )
        return parsed_boxes

    def predict(self, image, target_class: str, img_width: int, img_height: int) -> list:
        prompt_text = (
            f"Detect all {target_class} in this image."
            f"{self._strict_output_tail()}"
        )
        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]
        return self._run_messages(content, img_width, img_height)

    def predict_few_shot(
        self,
        query_image,
        support_images: list,
        prompt_text: str,
        img_width: int,
        img_height: int,
    ) -> list:
        strict_prompt_text = f"{prompt_text.rstrip()}{self._strict_output_tail()}"
        content = [{"type": "image", "image": img} for img in support_images]
        content.append({"type": "image", "image": query_image})
        content.append({"type": "text", "text": strict_prompt_text})
        return self._run_messages(content, img_width, img_height)