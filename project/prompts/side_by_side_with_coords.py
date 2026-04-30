from __future__ import annotations

from PIL import Image

from data.support_sampler import SupportExample
from data.visual_prompt import render_bboxes
from .base_strategy import PromptStrategy


class SideBySideWithCoordsStrategy(PromptStrategy):
    """
    Few-shot prompt strategy that combines:
    1) visual grounding via red boxed support images, and
    2) textual grounding via normalized support box coordinates.
    """

    def __init__(self, numbered_labels: bool = True):
        self.numbered_labels = numbered_labels

    @staticmethod
    def _to_norm_xyxy_1000(box_xywh: list[float], image_w: int, image_h: int) -> list[int]:
        x, y, w, h = box_xywh
        x1 = max(0.0, min(float(x), float(image_w)))
        y1 = max(0.0, min(float(y), float(image_h)))
        x2 = max(0.0, min(float(x + w), float(image_w)))
        y2 = max(0.0, min(float(y + h), float(image_h)))

        if x2 <= x1 or y2 <= y1:
            return []

        nx1 = int(round((x1 / max(1.0, float(image_w))) * 1000.0))
        ny1 = int(round((y1 / max(1.0, float(image_h))) * 1000.0))
        nx2 = int(round((x2 / max(1.0, float(image_w))) * 1000.0))
        ny2 = int(round((y2 / max(1.0, float(image_h))) * 1000.0))
        return [nx1, ny1, nx2, ny2]

    def _build_support_coords_section(self, supports: list[SupportExample]) -> str:
        lines: list[str] = []
        for idx, ex in enumerate(supports, start=1):
            img_w, img_h = ex.image.size
            normalized_boxes: list[list[int]] = []
            for box in ex.boxes:
                normalized = self._to_norm_xyxy_1000(box, img_w, img_h)
                if normalized:
                    normalized_boxes.append(normalized)
            lines.append(f"S{idx}: {normalized_boxes}")
        return "\n".join(lines)

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
        support_coords_text = self._build_support_coords_section(supports)
        images = annotated_supports + [query_image]
        text = (
            f"Support images show {class_name} marked with red boxes.\n"
            f"Support box coordinates (xyxy in [0,1000]):\n{support_coords_text}\n"
            f"Detect all {class_name} in the LAST image only.\n"
            "Return ONLY a Python-style list: [[x1, y1, x2, y2], ...]\n"
            "Coordinates must be integers in [0, 1000], x1 < x2, y1 < y2.\n"
            "No words, labels, markdown, or explanations. Return [] if none."
        )
        return {"images": images, "text": text}
