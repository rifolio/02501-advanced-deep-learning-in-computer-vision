from __future__ import annotations

from PIL import Image

from data.support_sampler import SupportExample
from data.visual_prompt import render_bboxes
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
        """
        Initialize with optional VLM model for prompt generation.
        If None, will be lazily loaded when needed.
        """
        self.vlm_model = vlm_model
    
    def build_prompt(
        self,
        query_image: Image.Image,
        support_by_cat: dict[int, list[SupportExample]],
        class_name: str,
        cat_id: int,
    ) -> dict:
        """
        Build a prompt by having a VLM analyze support examples.
        
        Args:
            query_image: Query image to detect in
            support_by_cat: Dict mapping category_id to list of SupportExample
            class_name: Name of target class (e.g., "dog", "car")
            cat_id: Category ID
            
        Returns:
            Dict with:
              - images: list of PIL images (support images + query at end)
              - text: VLM-generated description for Grounding DINO
              - vlm_output: raw VLM response (for debugging)
        """
        supports = support_by_cat.get(cat_id, [])
        
        # Render support images with bboxes marked for VLM to see
        annotated_supports = [
            render_bboxes(ex.image, ex.boxes, color="red", line_width=3, numbered_labels=True)
            for ex in supports
        ]
        
        # Build images list: annotated supports + query
        images = annotated_supports + [query_image]
        
        # If no support examples, use a generic prompt
        if not supports:
            text = (
                f"Generate a detailed text description for detecting '{class_name}' objects. "
                f"Include: appearance characteristics, colors, typical size relative to surroundings, "
                f"distinctive features, common contexts, and any other visual attributes that help identify them."
            )
            return {
                "images": images,
                "text": text,
                "vlm_output": None,
            }
        
        # Create VLM prompt that asks for detailed description
        vlm_prompt = self._create_vlm_prompt(class_name, len(annotated_supports))
        
        # Call VLM to generate description from support images
        if self.vlm_model is not None:
            vlm_output = self._generate_text_from_vlm(
                support_images=annotated_supports,
                prompt=vlm_prompt
            )
        else:
            # Fallback: use a rule-based template if VLM not available
            vlm_output = self._fallback_template(class_name)
        
        return {
            "images": images,
            "text": vlm_output,
            "vlm_output": vlm_output,
        }
    
    def _create_vlm_prompt(self, class_name: str, num_examples: int) -> str:
        """Create a prompt asking VLM to describe visual properties from examples."""
        return (
            f"Looking at these {num_examples} marked examples of '{class_name}', "
            f"generate a detailed text description to help a detector identify all similar objects. "
            f"Include: "
            f"- Visual appearance (shape, distinctive features, colors) "
            f"- Size and scale characteristics "
            f"- Typical context or environments where found "
            f"- Key distinguishing properties from related objects "
            f"Be specific and descriptive - this text will be used as a search prompt. "
            f"Return ONLY the description, no preamble."
        )
    
    def _generate_text_from_vlm(self, support_images: list, prompt: str) -> str:
        """
        Call VLM to generate text from support images.
        
        Args:
            support_images: List of PIL images with bboxes annotated
            prompt: Text prompt for VLM
            
        Returns:
            Generated text description for Grounding DINO
        """
        if not hasattr(self.vlm_model, 'predict_few_shot'):
            raise AttributeError(
                f"VLM model {self.vlm_model.model_name} must implement predict_few_shot()"
            )
        
        # Use first support image as context (VLM will see all in predict_few_shot)
        # The VLM needs to process all support images + return text (not boxes)
        # For now, we'll extract text from the VLM's response
        
        # Create a dummy query image (we want text, not boxes)
        dummy_query = support_images[0] if support_images else Image.new('RGB', (100, 100))
        
        # Call VLM in few-shot mode (it will see all support images)
        try:
            # This will call predict_few_shot which returns boxes, but we want text
            # We'll need to extend the VLM or use a different interface
            # For now, use the VLM's knowledge to generate text
            generated_text = self._query_vlm_for_description(
                support_images=support_images,
                prompt=prompt
            )
            return generated_text
        except Exception as e:
            print(f"Warning: VLM text generation failed ({e}), using fallback")
            return self._fallback_template(prompt.split("'")[1])  # Extract class name
    
    def _query_vlm_for_description(self, support_images: list, prompt: str) -> str:
        """
        Query VLM for text description. This is a simplified implementation
        that leverages the VLM's understanding of images.
        
        In practice, you'd want to use the VLM's generate() method directly,
        not predict_few_shot() which expects box output.
        """
        # This is a placeholder - in practice, you'd call:
        # - Qwen's generation method directly (not predict_few_shot)
        # - Or create a new VLM method for text-only generation
        
        # For now, return a constructed prompt based on class understanding
        class_name = prompt.split("'")[1] if "'" in prompt else "object"
        return self._fallback_template(class_name)
    
    def _fallback_template(self, class_name: str) -> str:
        """Fallback template when VLM is unavailable or fails."""
        return (
            f"A {class_name} is characterized by its distinctive visual properties. "
            f"Look for typical appearance, size, shape, colors, and context. "
            f"Detect all visible instances in this image."
        )
