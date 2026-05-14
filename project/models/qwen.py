import re
import ast
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
        parser_fallback_used = False

        # --- Pattern 1: native <|box_start|>(x1, y1), (x2, y2)<|box_end|> (integer) ---
        native_pattern = r"<\|box_start\|>\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)<\|box_end\|>"
        matches = re.findall(native_pattern, text)
        matched_format = "native"

        if not matches:
            # --- Pattern 2: native with decimals/brackets ---
            fallback_native_pattern = (
                r"<\|box_start\|>\(\s*\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?\s*\)\s*,\s*"
                r"\(\s*\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?\s*\)<\|box_end\|>"
            )
            matches = re.findall(fallback_native_pattern, text)
            if matches:
                matched_format = "fallback_native"
                parser_fallback_used = True

        if not matches:
            matches = self._parse_absolute_quads(text)
            if matches:
                matched_format = "list_absolute"
                parser_fallback_used = True

        if matched_format in ("native", "fallback_native"):
            # Coords are in the resized pixel space (multiples of 28).
            resized_width = round(img_width / 28.0) * 28
            resized_height = round(img_height / 28.0) * 28
            width_ratio = img_width / resized_width if resized_width > 0 else 1
            height_ratio = img_height / resized_height if resized_height > 0 else 1

            for match in matches:
                xmin, ymin, xmax, ymax = map(float, match)
                abs_xmin = xmin * width_ratio
                abs_ymin = ymin * height_ratio
                abs_xmax = xmax * width_ratio
                abs_ymax = ymax * height_ratio
                if abs_xmax <= abs_xmin or abs_ymax <= abs_ymin:
                    continue
                boxes.append([abs_xmin, abs_ymin, abs_xmax - abs_xmin, abs_ymax - abs_ymin])
        else:
            # Qwen2.5-VL grounding outputs absolute coordinates on the image scale.
            x_bound = max(0.0, float(img_width))
            y_bound = max(0.0, float(img_height))
            for match in matches:
                xmin, ymin, xmax, ymax = map(float, match)
                abs_xmin = max(0.0, min(xmin, x_bound))
                abs_ymin = max(0.0, min(ymin, y_bound))
                abs_xmax = max(0.0, min(xmax, x_bound))
                abs_ymax = max(0.0, min(ymax, y_bound))
                if abs_xmax <= abs_xmin or abs_ymax <= abs_ymin:
                    continue
                boxes.append([abs_xmin, abs_ymin, abs_xmax - abs_xmin, abs_ymax - abs_ymin])

        return boxes, parser_fallback_used

    @staticmethod
    def _parse_absolute_quads(text: str) -> list[tuple[float, float, float, float]]:
        """
        Extract [x1, y1, x2, y2] quads from list-style or JSON grounding replies.
        """
        candidates = []
        for match in re.finditer(r"\[[\s\S]*\]", text):
            candidates.append(match.group(0))
        candidates.sort(key=len, reverse=True)

        for candidate in candidates:
            try:
                parsed = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue

            quads: list[tuple[float, float, float, float]] = []
            seen: set[tuple[float, float, float, float]] = set()
            self_ref_stack = [parsed]
            while self_ref_stack:
                item = self_ref_stack.pop()
                if isinstance(item, dict):
                    bbox = item.get("bbox_2d")
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        quad = tuple(float(v) for v in bbox)
                        if quad not in seen:
                            seen.add(quad)
                            quads.append(quad)
                    self_ref_stack.extend(item.values())
                    continue
                if isinstance(item, (list, tuple)):
                    if len(item) == 4 and all(isinstance(v, (int, float)) for v in item):
                        quad = tuple(float(v) for v in item)
                        if quad not in seen:
                            seen.add(quad)
                            quads.append(quad)
                    else:
                        self_ref_stack.extend(item)
            if quads:
                return quads

        quad_pattern = re.compile(
            r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
        )
        return [tuple(map(float, match)) for match in quad_pattern.findall(text)]

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
        level = logging.DEBUG
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
            "Coordinates must be absolute image pixels, using the original image size.\n"
            "Use x1 < x2 and y1 < y2.\n"
            "Do not return words, labels, markdown, or explanations.\n"
            "If no instance is present, return [] exactly."
        )

    @classmethod
    def _rewrite_prompt_for_absolute_coords(cls, prompt_text: str) -> str:
        text = prompt_text.rstrip()
        text = re.sub(
            r"coordinates normalized to \[0\s*,\s*1000\]",
            "coordinates in absolute image pixels",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"Coordinates must be integers in \[0\s*,\s*1000\]\.?",
            "Coordinates must be absolute image pixels in the original image.",
            text,
            flags=re.IGNORECASE,
        )
        return text

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
        **kwargs,
    ) -> list:
        base_prompt_text = self._rewrite_prompt_for_absolute_coords(prompt_text)
        strict_prompt_text = f"{base_prompt_text}{self._strict_output_tail()}"
        content = [{"type": "image", "image": img} for img in support_images]
        content.append({"type": "image", "image": query_image})
        content.append({"type": "text", "text": strict_prompt_text})
        return self._run_messages(content, img_width, img_height)
