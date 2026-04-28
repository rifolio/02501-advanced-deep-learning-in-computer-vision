"""
VLM + Grounding DINO Fusion Model for Task 3.

Implements the complete pipeline:
1. VLM analyzes support examples → generates detailed text description
2. Grounding DINO uses that text to detect objects in query image
"""

from __future__ import annotations

import logging
from PIL import Image

from .base_vlm import BaseVLM
from .qwen import Qwen2_5_VL
from .grounding_dino import GroundingDINO
from .vlm_text_generator import VLMTextGenerator

logger = logging.getLogger(__name__)


class VLMDINOFusion(BaseVLM):
    """
    VLM + Detector Fusion Pipeline.
    
    Uses a Vision-Language Model (Qwen/InternVL) to analyze support examples
    and generate optimal text prompts for Grounding DINO detector.
    
    Task 3 Implementation:
    - Input: Support examples + query image
    - Process: VLM analyzes supports → generates text description
    - Output: Grounding DINO detections using generated prompt
    """
    
    def __init__(self, device: str, vlm_model: BaseVLM | None = None):
        """
        Initialize VLM + DINO fusion model.
        
        Args:
            device: Device to run models on ('cuda' or 'cpu')
            vlm_model: VLM instance for text generation (default: Qwen2_5_VL)
        """
        super().__init__(device)
        self.model_name = "VLM-DINO-Fusion"
        
        # Initialize VLM for text generation
        if vlm_model is None:
            logger.info("Initializing default VLM: Qwen2.5-VL")
            self.vlm = Qwen2_5_VL(device=device)
        else:
            self.vlm = vlm_model
        
        # Initialize Grounding DINO for detection
        logger.info("Initializing Grounding DINO detector")
        self.dino = GroundingDINO(device=device)
        
        # Initialize text generator
        self.text_generator = VLMTextGenerator(self.vlm)

    @staticmethod
    def _extract_bboxes(scored_predictions: list[dict]) -> list:
        return [prediction["bbox"] for prediction in scored_predictions]
    
    def predict(self, image, target_class: str, img_width: int, img_height: int) -> list:
        """
        Zero-shot detection fallback (no support examples).
        
        Args:
            image: Query image (PIL Image or path)
            target_class: Target class name
            img_width: Image width
            img_height: Image height
            
        Returns:
            List of bounding boxes in COCO format [x, y, width, height]
        """
        # Generate description without support examples
        description = self.text_generator._template_description(target_class)
        scored_predictions = self.dino.predict_with_scores(
            image,
            description,
            img_width,
            img_height,
        )
        return self._extract_bboxes(scored_predictions)

    def predict_with_scores(
        self,
        image,
        target_class: str,
        img_width: int,
        img_height: int,
    ) -> list[dict]:
        """
        Score-aware API that preserves detector confidences.
        """
        description = self.text_generator._template_description(target_class)
        return self.dino.predict_with_scores(image, description, img_width, img_height)
    
    def predict_few_shot(
        self,
        query_image,
        support_images: list,
        prompt_text: str,
        img_width: int,
        img_height: int,
    ) -> list:
        """
        Few-shot detection with VLM-generated prompts.
        
        Pipeline:
        1. Use VLM to analyze support_images
        2. Generate detailed text description
        3. Use Grounding DINO with generated description on query_image
        
        Args:
            query_image: Image to detect objects in
            support_images: List of support images showing target objects
            prompt_text: Optional class name or initial prompt (used as fallback)
            img_width: Query image width
            img_height: Query image height
            
        Returns:
            List of bounding boxes in COCO format [x, y, width, height]
        """
        scored_predictions = self.predict_few_shot_with_scores(
            query_image,
            support_images,
            prompt_text,
            img_width,
            img_height,
        )
        return self._extract_bboxes(scored_predictions)

    def predict_few_shot_with_scores(
        self,
        query_image,
        support_images: list,
        prompt_text: str,
        img_width: int,
        img_height: int,
    ) -> list[dict]:
        """
        Few-shot score-aware API that preserves detector confidences.
        """
        if not support_images:
            logger.info("No support images provided, using zero-shot detection")
            return self.predict_with_scores(query_image, prompt_text, img_width, img_height)
        
        # Extract class name from prompt (e.g., "Detect all dogs" → "dogs")
        class_name = self._extract_class_name(prompt_text)
        
        logger.info(
            f"VLM-DINO Fusion: analyzing {len(support_images)} support "
            f"images for class '{class_name}'"
        )
        
        try:
            # Step 1: VLM analyzes support examples and generates description
            generated_description = self.text_generator.generate_class_description(
                support_images=support_images,
                class_name=class_name,
                num_examples=len(support_images),
            )
            
            logger.info(f"Generated prompt: {generated_description[:150]}...")
            
            # Track the generated prompt for debugging
            self._bump_runtime_stat("vlm_descriptions_generated")
            
        except Exception as e:
            logger.error(f"VLM description generation failed: {e}, using fallback")
            generated_description = self.text_generator._template_description(class_name)
            self._bump_runtime_stat("vlm_generation_failed")
        
        # Step 2: Use Grounding DINO with generated description
        try:
            scored_predictions = self.dino.predict_with_scores(
                query_image,
                generated_description,
                img_width,
                img_height,
            )
            self._bump_runtime_stat("dino_predictions_made")
            return scored_predictions
            
        except Exception as e:
            logger.error(f"Grounding DINO detection failed: {e}")
            self._bump_runtime_stat("dino_detection_failed")
            return []
    
    def _extract_class_name(self, prompt_text: str) -> str:
        """
        Extract class name from prompt text.
        
        Examples:
            "Detect all dogs" → "dogs"
            "dogs" → "dogs"
            "A person walking" → "person walking"
        """
        if not prompt_text:
            return "object"
        
        # Simple extraction: lowercase, clean punctuation
        text = prompt_text.lower().strip()
        
        # Remove common prompt prefixes
        for prefix in ["detect all ", "find all ", "locate all ", "find the ", "detect the "]:
            if text.startswith(prefix):
                text = text[len(prefix):]
        
        # Remove trailing punctuation
        text = text.rstrip(".,!?")
        
        return text if text else "object"
