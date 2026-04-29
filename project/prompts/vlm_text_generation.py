from __future__ import annotations

from PIL import Image

from data.support_sampler import SupportExample
from data.visual_prompt import render_bboxes
from models.vlm_text_generator import VLMTextGenerator
from .base_strategy import PromptStrategy


class VLMTextGenerationStrategy(PromptStrategy):
    """
    VLM-based prompt generation strategy for Task 3.

    Uses a VLM (Qwen/InternVL) to analyze support examples and generate
    a detailed text description of visual properties, which is then used
    as the prompt for Grounding DINO.

    Pipeline:
    1. Show support examples to VLM with visual prompts (bboxes marked)
    2. Ask VLM to describe visual attributes (appearance, context, parts, etc.)
    3. Extract generated text as prompt for Grounding DINO
    4. Return images and generated text for downstream model
    """

    def __init__(self, vlm_model=None):
        self.vlm_model = vlm_model
        self._text_generator = VLMTextGenerator(vlm_model) if vlm_model else None

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

        if self._text_generator and supports:
            vlm_output = self._text_generator.generate_class_description(
                support_images=annotated_supports,
                class_name=class_name,
                num_examples=len(annotated_supports),
            )
        else:
            vlm_output = VLMTextGenerator._template_description_static(class_name)

        return {"images": images, "text": vlm_output, "vlm_output": vlm_output}
