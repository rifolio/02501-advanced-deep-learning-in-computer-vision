from __future__ import annotations

from PIL import Image

from data.support_sampler import SupportExample
from data.visual_prompt import render_bboxes
from .base_strategy import PromptStrategy


class SetOfMarkStrategy(PromptStrategy):
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
            f"Support images show {class_name} with red marks. "
            f"Use those references to detect {class_name} in the final image.\n"
            "Return ONLY a JSON array of boxes in this exact format: [[x1,y1,x2,y2], ...]\n"
            "Rules:\n"
            "- Coordinates must be integers in [0,1000].\n"
            "- Use x1 < x2 and y1 < y2.\n"
            "- Do not return words, markdown, labels, or explanations.\n"
            "- If no instance is present, return [] exactly."
        )
        return {"images": images, "text": text}
