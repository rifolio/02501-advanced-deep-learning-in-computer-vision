from __future__ import annotations

from PIL import Image

from data.support_sampler import SupportExample
from data.visual_prompt import render_bboxes
from .base_strategy import PromptStrategy


class OracleShotStrategy(PromptStrategy):
    def build_prompt(
        self,
        query_image: Image.Image,
        support_by_cat: dict[int, list[SupportExample]],
        class_name: str,
        cat_id: int,
    ) -> dict:
        supports = support_by_cat.get(cat_id, [])
        annotated_supports = [
            render_bboxes(ex.image, ex.boxes, color="green", line_width=3)
            for ex in supports
        ]
        images = annotated_supports + [query_image]
        text = (
            f"The first image shows the exact ground-truth annotations for all {class_name} instances, "
            f"marked with green bounding boxes. "
            f"Reproduce those exact boxes for the last image.\n"
            "Return ONLY a Python-style list in this exact format: [[x1, y1, x2, y2], ...]\n"
            "Rules:\n"
            "- Coordinates must be integers in [0, 1000].\n"
            "- Use [x1, y1, x2, y2] with x1 < x2 and y1 < y2.\n"
            "- Do not return words, labels, markdown, or explanations.\n"
            "- If no instance is present, return [] exactly."
        )
        return {"images": images, "text": text}
