import re
import ast
import logging
import hashlib
import math
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
from config import settings
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

class InternVL(BaseVLM):
    def __init__(
        self,
        device: str,
        model_id: str = "OpenGVLab/InternVL2_5-1B",
        model_name: str | None = None,
    ):
        super().__init__(device)
        self.model_id = model_id
        self.model_name = model_name or model_id.split("/")[-1]

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            trust_remote_code=True, 
            use_fast=False
        )
        
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            use_flash_attn=False,
            trust_remote_code=True,
        ).eval().to(self.device)
        logger.info(
            "Initialized model=%s model_id=%s device=%s",
            self.model_name,
            self.model_id,
            self.device,
        )

    def _parse_boxes(self, text: str, img_width: int, img_height: int) -> tuple[list, bool]:
        boxes = []
        parser_fallback_used = False

        # Our prompt explicitly asks for [[x1,y1,x2,y2], ...] Python-list output,
        # so we parse that first with ast.literal_eval (handles multi-box, flat,
        # and nested formats robustly).
        # If the model instead used InternVL's native <box>[...]</box> grounding
        # format, we fall back to regex extraction.
        matches = self._parse_boxes_ast(text)
        used_box_tags = False

        if not matches:
            matches = self._parse_boxes_regex_quads(text)

        if not matches:
            box_payloads = re.findall(r"<box>(.*?)</box>", text, flags=re.DOTALL)
            if box_payloads:
                int_pattern = r"\[\s*\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?\s*\]"
                float_pattern = (
                    r"\[\s*\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
                    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?\s*\]"
                )
                for payload in box_payloads:
                    matches.extend(re.findall(int_pattern, payload))
                if not matches:
                    for payload in box_payloads:
                        matches.extend(re.findall(float_pattern, payload))
                used_box_tags = bool(matches)

        return self._scale_matches_to_boxes(matches, img_width, img_height), used_box_tags

    @staticmethod
    def _parse_boxes_ast(text: str) -> list[tuple]:
        """Extract (xmin, ymin, xmax, ymax) tuples from raw text using ast.literal_eval."""
        # Find the longest bracket-delimited substring to cover [[...],[...]] lists
        best_candidates: list[str] = []
        for m in re.finditer(r"\[[\s\S]*\]", text):
            best_candidates.append(m.group(0))
        # Sort by length descending so we try the largest first
        best_candidates.sort(key=len, reverse=True)

        for candidate in best_candidates:
            try:
                parsed = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue

            tuples: list[tuple] = []
            if isinstance(parsed, list) and parsed:
                if isinstance(parsed[0], (list, tuple)):
                    for item in parsed:
                        if isinstance(item, (list, tuple)) and len(item) == 4:
                            tuples.append(tuple(item))
                elif len(parsed) == 4 and all(isinstance(v, (int, float)) for v in parsed):
                    tuples.append(tuple(parsed))
            if tuples:
                return tuples
        return []

    @staticmethod
    def _parse_boxes_regex_quads(text: str) -> list[tuple]:
        """
        Find every [x1, y1, x2, y2] int quad in raw text (handles prose + comma-separated lists
        where greedy ast.literal_eval on the full reply fails).
        """
        pat = re.compile(
            r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"
        )
        out: list[tuple] = []
        seen: set[tuple] = set()
        for m in pat.finditer(text):
            x1, y1, x2, y2 = (int(g) for g in m.groups())
            # COCO-style normalized quads are in [0, 1000]; drop junk like metadata lists.
            if not all(0 <= v <= 1000 for v in (x1, y1, x2, y2)):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            wn, hn = x2 - x1, y2 - y1
            if wn * hn < 100:  # ignore tiny unrelated quads (e.g. [1,2,3,4])
                continue
            t = (x1, y1, x2, y2)
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out

    def _scale_matches_to_boxes(
        self, matches: list, img_width: int, img_height: int,
    ) -> list:
        """Convert [0,1000]-normalized coordinate tuples to pixel-space COCO xywh."""
        boxes = []
        x_bound = max(0.0, float(img_width))
        y_bound = max(0.0, float(img_height))
        for match in matches:
            xmin, ymin, xmax, ymax = map(float, match)
            if not all(math.isfinite(v) for v in (xmin, ymin, xmax, ymax)):
                continue

            abs_xmin = max(0.0, min((xmin / 1000.0) * img_width, x_bound))
            abs_ymin = max(0.0, min((ymin / 1000.0) * img_height, y_bound))
            abs_xmax = max(0.0, min((xmax / 1000.0) * img_width, x_bound))
            abs_ymax = max(0.0, min((ymax / 1000.0) * img_height, y_bound))

            if abs_xmax <= abs_xmin or abs_ymax <= abs_ymin:
                continue

            width = abs_xmax - abs_xmin
            height = abs_ymax - abs_ymin
            boxes.append([abs_xmin, abs_ymin, width, height])

        return boxes

    def _provisional_score_policy(self) -> str:
        """
        InternVL currently returns boxes without confidence logits.
        """
        return "internvl_box_only_rank_decay_v1"

    def _log_inference_debug(
        self,
        question: str,
        output_text: str,
        parsed_box_count: int,
        parser_fallback_used: bool,
        image_count: int,
    ) -> None:
        prompt_hash = hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
        ######
        full_text = settings.vlm_log_full_text        
        prompt_clean = question.strip().replace("\n", " ")
        output_clean = output_text.strip().replace("\n", " ")
        prompt_snippet = prompt_clean if full_text else prompt_clean[:120]
        output_snippet = output_clean if full_text else output_clean[:180]
        ######
        level = logging.INFO if parsed_box_count == 0 else logging.DEBUG
        logger.log(
            level,
            (
                "[inference_debug] model=%s image_count=%s prompt_hash=%s prompt_snippet=%r "
                "parsed_boxes=%s used_box_tags(fallback)=%s output_snippet=%r"
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
                    f"describes: Detect all the instances of the following object: {target_class}.\n"
                    "Return ONLY a Python-style list in this exact format: [[x1, y1, x2, y2], ...]\n"
                    "Coordinates must be integers in [0, 1000].\n"
                    "Use [x1, y1, x2, y2] with x1 < x2 and y1 < y2.\n"
                    "Do not return words, labels, markdown, or explanations.\n"
                    "If no instance is present, return [] exactly."
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

        for i, image in enumerate(images):
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif getattr(image, "mode", None) != "RGB":
                image = image.convert("RGB")

            # If this is not the LAST image in the list, it is a support crop.
            is_query_image = (i == len(images) - 1)
            w, h = image.size
            
            # Restrict to 1 patch if it's a crop, OR if the original image is just naturally tiny
            if not is_query_image or (w * h <= 448 * 448):
                current_max_num = 1
            else:
                current_max_num = max_num

            processed = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=current_max_num)
            num_patches_list.append(len(processed))
            all_tiles.extend(processed)

        pixel_values = [transform(img) for img in all_tiles]
        pixel_values = torch.stack(pixel_values).to(torch.bfloat16).to(self.device)
        return pixel_values, num_patches_list

    def _run_with_images(self, images: list[Image.Image], question: str, img_width: int, img_height: int) -> list:
        pixel_values, num_patches_list = self._prepare_multi_image_pixels(images)

        generation_config = dict(
            max_new_tokens=int(settings.internvl_max_new_tokens),
            do_sample=False,
        )

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
            # "fallback" here means the model used <box>...</box> tags instead of
            # the plain [[x,y,x,y]] list our prompt requests — uncommon but handled.
            self._bump_runtime_stat("parser_fallback_used")
            logger.info(
                "[parser_debug] model=%s used_box_tags=true parsed_boxes=%s",
                self.model_name,
                len(parsed_boxes),
            )
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
        prompt = False
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
        **kwargs,
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
            {
                "image": support_image,
                "text": (
                    f"Support example {idx}: use this as visual grounding context for the task below.\n"
                    f"{prompt_text.rstrip()}"
                ),
            }
            for idx, support_image in enumerate(support_images, start=1)
        ]
        structured_inputs.append({"image": query_image, "text": strict_prompt_text})
        return self._run_structured_inputs(structured_inputs, img_width, img_height)


InternVL2_5_8B = InternVL  # backward-compat alias
