import logging

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from config import settings

from .base_vlm import BaseVLM

logger = logging.getLogger(__name__)

# Match Hugging Face `GroundingDinoProcessor.post_process_grounded_object_detection` defaults
# (see transformers.models.grounding_dino.processing_grounding_dino). Tiny benefits from not
# using the stricter 0.4/0.3 demo values on every COCO image.
_DEFAULT_BOX_THRESHOLD = 0.25
_DEFAULT_TEXT_THRESHOLD = 0.25


def _format_grounding_prompt(target_class: str) -> str:
    """HF-style short noun phrases: lowercased, article prefix, dot-terminated."""
    phrase = target_class.strip().lower()
    if not phrase:
        logger.warning("[grounding_dino] empty target_class; using fallback phrase 'object'")
        phrase = "object"
    if not (phrase.startswith("a ") or phrase.startswith("an ")):
        article = "an " if phrase[:1] in "aeiou" else "a "
        phrase = f"{article}{phrase}"
    if not phrase.endswith("."):
        phrase += "."
    return phrase


class GroundingDINO(BaseVLM):
    def __init__(self, device: str):
        super().__init__(device)
        self.model_name = "Grounding-DINO-Tiny"
        self.model_id = "IDEA-Research/grounding-dino-tiny"

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)

        self._box_threshold = _DEFAULT_BOX_THRESHOLD
        self._text_threshold = _DEFAULT_TEXT_THRESHOLD

    def _run_detection(self, image, target_class: str, img_width: int, img_height: int):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif getattr(image, "mode", None) != "RGB":
            image = image.convert("RGB")

        text_prompt = _format_grounding_prompt(target_class)

        pil_w, pil_h = image.size
        if (pil_w, pil_h) != (img_width, img_height):
            logger.warning(
                (
                    "[grounding_dino] PIL size (w,h)=(%s,%s) != dataset (w,h)=(%s,%s); "
                    "boxes are rescaled with dataset dimensions — check annotations vs files."
                ),
                pil_w,
                pil_h,
                img_width,
                img_height,
            )

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probs = torch.sigmoid(logits)
        max_per_query, _ = probs.max(dim=-1)
        global_max = float(max_per_query.max().item())
        num_queries_above = int((max_per_query > self._box_threshold).sum().item())
        # Full (query × vocab) logits can contain non-finite entries while per-query max scores stay finite.
        logits_nonfinite_n = int((~torch.isfinite(logits)).sum().item())
        query_scores_finite = bool(torch.isfinite(max_per_query).all().item())

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[(img_height, img_width)],
        )
        result = results[0]
        n_boxes = len(result["boxes"])

        if settings.grounding_dino_debug:
            logger.info(
                (
                    "[grounding_dino] class_name=%r prompt=%r max_sigmoid=%.4f "
                    "queries_above_box_thr=%d/%d box_threshold=%.2f text_threshold=%.2f "
                    "postprocess_boxes=%d logits_nonfinite_n=%d query_scores_finite=%s"
                ),
                target_class,
                text_prompt,
                global_max,
                num_queries_above,
                max_per_query.numel(),
                self._box_threshold,
                self._text_threshold,
                n_boxes,
                logits_nonfinite_n,
                query_scores_finite,
            )
        elif logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                (
                    "[grounding_dino] class_name=%r prompt=%r max_sigmoid=%.4f "
                    "queries_above_box_thr=%d/%d box_threshold=%.2f text_threshold=%.2f "
                    "postprocess_boxes=%d logits_nonfinite_n=%d query_scores_finite=%s"
                ),
                target_class,
                text_prompt,
                global_max,
                num_queries_above,
                max_per_query.numel(),
                self._box_threshold,
                self._text_threshold,
                n_boxes,
                logits_nonfinite_n,
                query_scores_finite,
            )

        return result

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
