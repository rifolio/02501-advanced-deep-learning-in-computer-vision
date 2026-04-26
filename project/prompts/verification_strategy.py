"""
Verification Strategy: Builds prompts for VLM-as-Verifier.

Displays side-by-side: support examples + detected region
Asks VLM: "Does this detection match the target object?"
"""

from __future__ import annotations

from PIL import Image

from data.support_sampler import SupportExample
from data.visual_prompt import render_bboxes
from .base_strategy import PromptStrategy


class VerificationStrategy(PromptStrategy):
    """
    Prompt strategy for verification of individual detections.

    Combines support examples (with bboxes) and a single detected region,
    asks VLM if they match the target object.
    """

    def __init__(self, show_bbox_numbers: bool = True):
        """
        Initialize verification strategy.

        Args:
            show_bbox_numbers: Whether to display bbox numbers in support examples
        """
        self.show_bbox_numbers = show_bbox_numbers

    def build_prompt(
        self,
        query_image: Image.Image,
        support_by_cat: dict[int, list[SupportExample]],
        class_name: str,
        cat_id: int,
    ) -> dict:
        """
        Build verification prompt with annotated supports + query.

        Returns:
            Dict with 'images' (support + query) and 'text' (verification question)
        """
        supports = support_by_cat.get(cat_id, [])

        # Annotate support examples with bounding boxes
        annotated_supports = [
            render_bboxes(
                ex.image,
                ex.boxes,
                color="green",
                line_width=2,
                numbered_labels=self.show_bbox_numbers,
            )
            for ex in supports
        ]

        # Combine: annotated supports + query image
        images = annotated_supports + [query_image]

        num_supports = len(annotated_supports)
        text = (
            f"You are verifying object detections.\n\n"
            f"The first {num_supports} image(s) show support example(s) with bounding boxes "
            f"highlighting {class_name}.\n"
            f"The last image shows a detected region (cropped from a query image).\n\n"
            f"Question: Does the detected region (last image) contain a {class_name}?\n\n"
            f"Respond in this exact format:\n"
            f"ANSWER: YES or NO\n"
            f"CONFIDENCE: [0-100]\n"
            f"REASONING: [brief explanation of your decision]\n\n"
            f"Be strict and precise. Only answer YES if the region clearly matches a {class_name}."
        )

        return {"images": images, "text": text}

    def build_simple_verification_prompt(
        self,
        cropped_detection: Image.Image,
        support_images: list[Image.Image],
        support_bboxes: list[list[list[float]]] | None,
        class_name: str,
    ) -> dict:
        """
        Build verification prompt for a single cropped detection.

        Args:
            cropped_detection: Cropped bbox region (already extracted)
            support_images: List of support images
            support_bboxes: Optional bounding boxes for each support image
            class_name: Target class name

        Returns:
            Dict with 'images' and 'text' keys
        """
        # Annotate support images if bboxes provided
        annotated_supports = []
        for i, support_img in enumerate(support_images):
            if support_bboxes and i < len(support_bboxes):
                annotated = render_bboxes(
                    support_img,
                    support_bboxes[i],
                    color="green",
                    line_width=2,
                    numbered_labels=True,
                )
                annotated_supports.append(annotated)
            else:
                annotated_supports.append(support_img)

        # Combine: annotated supports + cropped detection
        images = annotated_supports + [cropped_detection]

        num_supports = len(annotated_supports)
        text = (
            f"You are verifying an object detection.\n\n"
            f"The first {num_supports} image(s) show example(s) of a {class_name} "
            f"(with bounding boxes if available).\n"
            f"The last image shows a region detected as a potential {class_name}.\n\n"
            f"Question: Is the detected region (last image) actually a {class_name}?\n\n"
            f"Respond in this EXACT format:\n"
            f"ANSWER: YES or NO\n"
            f"CONFIDENCE: [0-100]\n"
            f"REASONING: [1-2 sentence explanation]\n\n"
            f"Be strict: only YES if you are confident this is a {class_name}."
        )

        return {"images": images, "text": text}
