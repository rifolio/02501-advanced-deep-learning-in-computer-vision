from __future__ import annotations

from PIL import Image

from data.support_sampler import SupportExample
from .base_strategy import PromptStrategy


class CroppedExemplarsStrategy(PromptStrategy):

    def __init__(self, max_crops_total: int = 4):
        self.max_crops_total = max_crops_total

    def build_prompt(
        self,
        query_image: Image.Image,
        support_by_cat: dict[int, list[SupportExample]],
        class_name: str,
        cat_id: int,
    ) -> dict:
        supports = support_by_cat.get(cat_id, [])
        
        #check out candidate boxes with area 
        candidate_crops = []
        for ex in supports:
            for box in ex.boxes:
                x, y, w, h = box
                area = w * h
                candidate_crops.append({
                    "image": ex.image,
                    "box": box,
                    "area": area
                })

        #sort by descending area and keep only top K crops
        candidate_crops.sort(key=lambda item: item["area"], reverse=True)
        top_candidates = candidate_crops[:self.max_crops_total]

        images: list[Image.Image] = []
        for candidate in top_candidates:
                x,y,w,h = candidate["box"]
                crop = candidate["image"].crop((x, y, x + w, y + h))
                images.append(crop)


        images.append(query_image)
        crops_collected = len(top_candidates)
        text = (
            f"The first {crops_collected} images are object exemplars of {class_name}. "
            f"Detect all {class_name} instances in the last image only.\n"
            "Return ONLY a Python-style list in this exact format: [[x1, y1, x2, y2], ...]\n"
            "Rules:\n"
            "- Coordinates must be integers in [0, 1000].\n"
            "- Use [x1, y1, x2, y2] with x1 < x2 and y1 < y2.\n"
            "- Do not return words, labels, markdown, or explanations.\n"
            "- If no instance is present, return [] exactly."
        )
        return {"images": images, "text": text}
