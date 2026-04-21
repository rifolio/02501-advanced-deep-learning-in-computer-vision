from __future__ import annotations

from PIL import Image

from data.support_sampler import SupportExample
from data.visual_prompt import render_bboxes
from .base_strategy import PromptStrategy


class SideBySideStrategy(PromptStrategy):
    def __init__(self, numbered_labels: bool = True):
        self.numbered_labels = numbered_labels

    def build_prompt(
        self,
        query_image: Image.Image,
        support_by_cat: dict[int, list[SupportExample]],
        class_name: str,
        cat_id: int,
    ) -> dict:
        supports = support_by_cat.get(cat_id, [])
        annotated_supports = [
            render_bboxes(
                ex.image,
                ex.boxes,
                color="red",
                line_width=3,
                numbered_labels=self.numbered_labels,
            )
            for ex in supports
        ]
        images = annotated_supports + [query_image]
        text = (
            f"Examples show {class_name} with red boxes. "
            f"Detect all instances of {class_name} in the last image only."
        )
        return {"images": images, "text": text}
