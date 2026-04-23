"""
VLM Text Generator - Generates descriptive text from support images using VLMs.
This enables the VLM-to-text pipeline for Task 3.
"""

from __future__ import annotations

import logging
from typing import Optional
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class VLMTextGenerator:
    """
    Generates descriptive text from support images using Vision-Language Models.
    
    This module enables the first part of Task 3's VLM + Detector Fusion:
    - Takes support examples with marked objects
    - Queries VLM to generate detailed text descriptions
    - Returns optimized text prompts for use in detectors like Grounding DINO
    """
    
    def __init__(self, vlm_model):
        """
        Initialize with a VLM model instance.
        
        Args:
            vlm_model: VLM instance (Qwen2_5_VL or InternVL2_5_8B)
        """
        self.vlm_model = vlm_model
        self.model_name = vlm_model.model_name
    
    def generate_class_description(
        self,
        support_images: list[Image.Image],
        class_name: str,
        num_examples: int,
    ) -> str:
        """
        Generate a detailed class description from support images.
        
        Args:
            support_images: List of PIL images with target objects
            class_name: Name of the target class (e.g., "dog", "person")
            num_examples: Number of support examples provided
            
        Returns:
            Detailed text description optimized for object detection
        """
        if not support_images:
            return self._template_description(class_name)
        
        # Create prompt for VLM
        prompt = self._create_description_prompt(class_name, num_examples)
        
        # Call VLM to generate description
        if self.model_name == "Qwen2.5-VL-7B":
            return self._generate_qwen_description(support_images, prompt)
        elif self.model_name == "InternVL2.5-8B":
            return self._generate_internvl_description(support_images, prompt)
        else:
            logger.warning(f"Unknown VLM: {self.model_name}, using template")
            return self._template_description(class_name)
    
    def _create_description_prompt(self, class_name: str, num_examples: int) -> str:
        """Create a structured prompt for VLM to analyze support examples."""
        return (
            f"You are analyzing {num_examples} visual examples of '{class_name}' objects. "
            f"Based on these images, generate a detailed and specific text description that would help "
            f"an object detector identify similar objects. Include:\n"
            f"1. Visual appearance: shape, distinctive features, colors, textures\n"
            f"2. Size and proportions relative to surrounding objects\n"
            f"3. Typical context: where and how this object usually appears\n"
            f"4. Distinguishing characteristics from similar objects\n"
            f"5. Key attributes and parts to look for\n"
            f"Be precise and descriptive. This description will be used as a detection prompt.\n"
            f"Return ONLY the description, no preamble or numbered list - use natural language."
        )
    
    def _generate_qwen_description(
        self,
        images: list[Image.Image],
        prompt: str
    ) -> str:
        """Generate description using Qwen2.5-VL-7B."""
        try:
            # Build content list for Qwen
            content = [{"type": "image", "image": img} for img in images]
            content.append({"type": "text", "text": prompt})
            
            # Use Qwen's message API directly
            messages = [{"role": "user", "content": content}]
            
            # Apply chat template
            text = self.vlm_model.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process images
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Prepare inputs
            inputs = self.vlm_model.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.vlm_model.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.vlm_model.model.generate(
                    **inputs,
                    max_new_tokens=256,  # Allow longer descriptions
                    temperature=0.7,
                    top_p=0.9,
                )
            
            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.vlm_model.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True
            )[0]
            
            logger.info(f"Qwen generated description: {output_text[:100]}...")
            return output_text.strip()
            
        except Exception as e:
            logger.error(f"Qwen description generation failed: {e}")
            return self._template_description("object")
    
    def _generate_internvl_description(
        self,
        images: list[Image.Image],
        prompt: str
    ) -> str:
        """Generate description using InternVL2.5-8B."""
        try:
            # InternVL requires different processing
            # Stack images and use generate method
            
            # For simplicity, process first image (can be extended to handle multiple)
            if not images:
                return self._template_description("object")
            
            # Use InternVL's generate interface (simplified approach)
            # This would need to be implemented based on InternVL's actual API
            logger.warning(
                "InternVL text generation not yet fully implemented, using template"
            )
            return self._template_description("object")
            
        except Exception as e:
            logger.error(f"InternVL description generation failed: {e}")
            return self._template_description("object")
    
    def _template_description(self, class_name: str) -> str:
        """Fallback template description."""
        return (
            f"A {class_name} with distinctive visual properties. "
            f"Look for typical appearance, characteristic features, and size. "
            f"Detect all visible {class_name} instances."
        )
