from __future__ import annotations

from PIL import Image

from data.support_sampler import SupportExample
from data.visual_prompt import render_bboxes
from .base_strategy import PromptStrategy


class TextFromVisionStrategy(PromptStrategy):
    def build_prompt(
        self,
        query_image: Image.Image,
        support_by_cat: dict[int, list[SupportExample]],
        class_name: str,
        cat_id: int,
    ) -> dict:
        supports = support_by_cat.get(cat_id, [])
        annotated_supports = [
            render_bboxes(ex.image, ex.boxes, color="red", line_width=3, numbered_labels=True)
            for ex in supports
        ]
        images = annotated_supports + [query_image]
        text = (
            f"From the marked support images, infer visual properties of {class_name}. "
            f"Then detect all {class_name} in the last image and return ONLY a JSON array of boxes "
            f"formatted as [[x1,y1,x2,y2], ...], with coordinates normalized to [0,1000]. "
            f"Use x1 < x2 and y1 < y2. Return [] if none are present."
        )
        return {"images": images, "text": text}
