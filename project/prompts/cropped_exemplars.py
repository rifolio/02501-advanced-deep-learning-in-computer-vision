from __future__ import annotations

from PIL import Image

from data.support_sampler import SupportExample
from .base_strategy import PromptStrategy


class CroppedExemplarsStrategy(PromptStrategy):
    def build_prompt(
        self,
        query_image: Image.Image,
        support_by_cat: dict[int, list[SupportExample]],
        class_name: str,
        cat_id: int,
    ) -> dict:
        supports = support_by_cat.get(cat_id, [])
        images: list[Image.Image] = []

        for ex in supports:
            for box in ex.boxes:
                x, y, w, h = box
                crop = ex.image.crop((x, y, x + w, y + h))
                images.append(crop)

        images.append(query_image)
        text = (
            f"The first images are object exemplars of {class_name}. "
            f"Detect all {class_name} instances in the last image."
        )
        return {"images": images, "text": text}
