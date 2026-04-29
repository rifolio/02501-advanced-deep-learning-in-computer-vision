"""
VLM Text Generator - Generates descriptive text from support images using VLMs.
This enables the VLM-to-text pipeline for Task 3.
"""

from __future__ import annotations
import logging
from PIL import Image
import torch
from qwen_vl_utils import process_vision_info

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
            vlm_model: VLM instance (Qwen2_5_VL or InternVL)
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
        elif self.model_name.startswith("InternVL"):
            return self._generate_internvl_description(support_images, prompt)
        else:
            logger.warning(f"Unknown VLM: {self.model_name}, using template")
            return self._template_description(class_name)
    
    def _create_description_prompt(self, class_name: str, num_examples: int) -> str:
        """Create a prompt that yields a short noun-phrase for Grounding DINO.

        Grounding DINO works best with concise phrases (< 15 words),
        not paragraph-length descriptions.
        """
        return (
            f"You are looking at {num_examples} example image(s) of '{class_name}'. "
            f"Describe the object in one short phrase (under 15 words) that captures "
            f"its key visual features — shape, color, texture, and distinguishing details. "
            f"Output ONLY the phrase, nothing else."
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
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Prepare inputs
            inputs = self.vlm_model.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.vlm_model.device)
            
            with torch.no_grad():
                generated_ids = self.vlm_model.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.3,
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
            # For simplicity, process first image (can be extended to handle multiple)
            if not images:
                return self._template_description("object")
            
            pixel_values, num_patches_list = self.vlm_model._prepare_multi_image_pixels(images)
            prompt_parts = []

            for i in range(1, len(images)+1):
                prompt_parts.append(f"Image-{i}: <image>")
                #place a line-break in between prompt elements with the line-break as the glue/
                # or separator between each item
            prompt_parts.append(prompt)
            question = "\n".join(prompt_parts)
            generation_config = dict(max_new_tokens=50, do_sample=False)

            with torch.no_grad():
                output_text = self.vlm_model.model.chat(
                    self.vlm_model.tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    num_patches_list,
                    return_history=False,
                )

            
            logger.info(f"InternVL generated description: {output_text[:100]}...")
            return output_text.strip()
            
        except Exception as e:
            logger.error(f"InternVL description generation failed: {e}")
            return self._template_description("object")
    
    def _template_description(self, class_name: str) -> str:
        """Fallback template description."""
        return VLMTextGenerator._template_description_static(class_name)

    @staticmethod
    def _template_description_static(class_name: str) -> str:
        """Fallback template description (static, usable without a VLM instance)."""
        return (
            f"A {class_name} with distinctive visual properties. "
            f"Look for typical appearance, characteristic features, and size. "
            f"Detect all visible {class_name} instances."
        )
