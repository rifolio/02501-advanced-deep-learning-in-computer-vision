import re
import logging
import hashlib
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel

from .base_vlm import BaseVLM

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True): #lowered max_num to speed up training
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
        
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
        
    return processed_images


class InternVL2_5_8B(BaseVLM):
    def __init__(self, device: str):
        super().__init__(device)
        self.model_name = "InternVL2.5-8B"
        self.model_id = "OpenGVLab/InternVL2_5-8B"

        # Flash Attention is generally unsupported on MPS or CPU backends.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            trust_remote_code=True, 
            use_fast=False
        )
        
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            #load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True, #since we mostly use volta's we keep it false; if we use a100 we can set it to true and compile
            #device_map="auto",           
            trust_remote_code=True
        ).eval().to(self.device)
        # REMOVED: .to(self.device) because device_map="auto" and load_in_8bit handle it; otherwise we delete them and add it
        logger.info(
            "Initialized model=%s model_id=%s device=%s",
            self.model_name,
            self.model_id,
            self.device,
        )

    def _parse_boxes(self, text: str, img_width: int, img_height: int) -> tuple[list, bool]:
        boxes = []
        
        # InternVL outputs grounding coordinates in a [0, 1000] normalized range
        # formatted similarly to: <ref>class</ref><box>[[x1, y1, x2, y2]]</box>
        pattern = r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
        matches = re.findall(pattern, text)
        parser_fallback_used = False

        if not matches:
            # Fallback for decimal coordinates and nested brackets.
            fallback_pattern = r"\[\s*\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?\s*\]"
            matches = re.findall(fallback_pattern, text)
            parser_fallback_used = bool(matches)
        
        for match in matches:
            xmin, ymin, xmax, ymax = map(float, match)
            
            # Map the normalized [0, 1000] space back to absolute pixel dimensions
            abs_xmin = (xmin / 1000.0) * img_width
            abs_ymin = (ymin / 1000.0) * img_height
            abs_xmax = (xmax / 1000.0) * img_width
            abs_ymax = (ymax / 1000.0) * img_height
            
            width = abs_xmax - abs_xmin
            height = abs_ymax - abs_ymin
            
            boxes.append([abs_xmin, abs_ymin, width, height])
            
        return boxes, parser_fallback_used

    def _log_inference_debug(
        self,
        question: str,
        output_text: str,
        parsed_box_count: int,
        parser_fallback_used: bool,
        image_count: int,
    ) -> None:
        prompt_hash = hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
        prompt_snippet = question.strip().replace("\n", " ")[:120]
        output_snippet = output_text.strip().replace("\n", " ")[:180]
        level = logging.INFO if parsed_box_count == 0 else logging.DEBUG
        logger.log(
            level,
            (
                "[inference_debug] model=%s image_count=%s prompt_hash=%s prompt_snippet=%r "
                "parsed_boxes=%s parser_fallback_used=%s output_snippet=%r"
            ),
            self.model_name,
            image_count,
            prompt_hash,
            prompt_snippet,
            parsed_box_count,
            parser_fallback_used,
            output_snippet,
        )

    def predict(self, image, target_class: str, img_width: int, img_height: int) -> list:
        structured_inputs = [
            {
                "image": image,
                "text": (
                    "Please provide the bounding box coordinates of the region this sentence "
                    f"describes: Detect all {target_class}."
                ),
            }
        ]
        return self._run_structured_inputs(structured_inputs, img_width, img_height)

    def _prepare_multi_image_pixels(
        self,
        images: list[Image.Image],
        max_num: int = 6,
    ) -> tuple[torch.Tensor, list[int]]:
        transform = build_transform(input_size=448)
        all_tiles = []
        num_patches_list = []

        for image in images:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif getattr(image, "mode", None) != "RGB":
                image = image.convert("RGB")

            processed = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=max_num)
            num_patches_list.append(len(processed))
            all_tiles.extend(processed)

        pixel_values = [transform(img) for img in all_tiles]
        pixel_values = torch.stack(pixel_values).to(torch.bfloat16).to(self.device)
        return pixel_values, num_patches_list

    def _run_with_images(self, images: list[Image.Image], question: str, img_width: int, img_height: int) -> list:
        pixel_values, num_patches_list = self._prepare_multi_image_pixels(images)

        generation_config = dict(max_new_tokens=256, do_sample=False)

        with torch.no_grad():
            output_text = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                return_history=False,
            )

        parsed_boxes, parser_fallback_used = self._parse_boxes(output_text, img_width, img_height)
        if parser_fallback_used:
            self._bump_runtime_stat("parser_fallback_used")
        if not parsed_boxes:
            self._bump_runtime_stat("queries_with_zero_boxes")
        self._log_inference_debug(
            question=question,
            output_text=output_text,
            parsed_box_count=len(parsed_boxes),
            parser_fallback_used=parser_fallback_used,
            image_count=len(images),
        )
        return parsed_boxes

    def _run_structured_inputs(
        self,
        structured_inputs: list[dict],
        img_width: int,
        img_height: int,
    ) -> list:
        images = [item["image"] for item in structured_inputs]
        prompt_parts = []
        for i, item in enumerate(structured_inputs, start=1):
            prompt_parts.append(f"Image-{i}: <image>\n{item['text']}")
        question = "\n".join(prompt_parts)
        return self._run_with_images(images, question, img_width, img_height)

    def predict_few_shot(
        self,
        query_image,
        support_images: list,
        prompt_text: str,
        img_width: int,
        img_height: int,
    ) -> list:
        strict_output_tail = (
            "\nOutput requirements:\n"
            "Return ONLY a Python-style list in this exact format: [[x1, y1, x2, y2], ...]\n"
            "Coordinates must be integers in [0, 1000].\n"
            "Use [x1, y1, x2, y2] with x1 < x2 and y1 < y2.\n"
            "Do not return words, labels, markdown, or explanations.\n"
            "If no instance is present, return [] exactly."
        )
        strict_prompt_text = f"{prompt_text.rstrip()}{strict_output_tail}"
        structured_inputs = [
            {"image": support_image, "text": "Reference support example."}
            for support_image in support_images
        ]
        structured_inputs.append({"image": query_image, "text": strict_prompt_text})
        return self._run_structured_inputs(structured_inputs, img_width, img_height)
