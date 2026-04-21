from abc import ABC, abstractmethod
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict, Any
#support_examples: list[dict])
class BaseICLStrategy(ABC):
    @abstractmethod
    def apply(self, query_image: Image.Image, target_class: str, support_examples: List[Dict[str, Any]]) -> Tuple[List[Image.Image], str]:
        """
        Applies the In-Context Learning strategy.
        
        Args:
            query_image: The target PIL Image to run inference on.
            target_class: The string class name to detect.
            support_examples: List of dicts, e.g., {"image": PIL.Image, "boxes": [[x, y, w, h], ...]}
            
        Returns:
            A tuple of (List of images to pass to the VLM, The formatted text prompt)
        """
        pass

class SideBySideStrategy(BaseICLStrategy):
    def apply(self, query_image: Image.Image, target_class: str, support_examples: list[dict]) -> list[dict]:
        structured_inputs = []
        
        # 1. Format the Support Examples
        for i, example in enumerate(support_examples):
            # Pass the bounding boxes in the format your specific VLM expects. 
            # Note: You may need to normalize these to [0, 1000] for InternVL 
            # or keep them absolute depending on how you fine-tune/prompt.
            box_text = f"Example {i+1}: This image contains {target_class} at bounding boxes: {example['boxes']}."
            
            structured_inputs.append({
                "image": example["image"],
                "text": box_text
            })
            
        # 2. Format the Query Image
        query_text = f"Now, look at this final image. Please provide the bounding box coordinates of the region this sentence describes: Detect all {target_class}."
        structured_inputs.append({
            "image": query_image,
            "text": query_text
        })
        
        return structured_inputs

class TextFromVisionStrategy(BaseICLStrategy):
    """
    (c) Text-from-vision: Passes only the query image visually. Support examples are 
    passed purely as mathematical coordinates in the text prompt.
    """
    def apply(self, query_image: Image.Image, target_class: str, support_examples: List[Dict[str, Any]]) -> Tuple[List[Image.Image], str]:
        prompt = f"You are an expert object detection system. Here are examples of bounding boxes for '{target_class}' extracted from other images:\n"
        
        for i, ex in enumerate(support_examples):
            img_w, img_h = ex["image"].width, ex["image"].height
            prompt += f"- Reference Image {i+1} (Dimensions: {img_w}x{img_h}): {ex['boxes']}\n"
            
        prompt += f"\n<image>\nBased on the coordinates above, provide the bounding box coordinates [x, y, w, h] for all instances of '{target_class}' in this new image."
        
        return [query_image], prompt

class CroppedExemplarsStrategy(BaseICLStrategy):
    """
    (b) Cropped exemplars: Crops the specific target objects out of the support images 
    and passes them as individual visual tokens preceding the query image.
    """
    def apply(self, query_image: Image.Image, target_class: str, support_examples: List[Dict[str, Any]]) -> Tuple[List[Image.Image], str]:
        images_to_return = []
        prompt = f"You are tasked with detecting '{target_class}'. Here are visual crops demonstrating what '{target_class}' looks like:\n"
        
        crop_idx = 1
        for ex in support_examples:
            img = ex["image"]
            for box in ex["boxes"]:
                x, y, w, h = box
                # Crop requires tuple: (left, upper, right, lower)
                crop = img.crop((x, y, x + w, y + h))
                images_to_return.append(crop)
                prompt += f"Crop {crop_idx}: <image>\n"
                crop_idx += 1
                
        images_to_return.append(query_image)
        prompt += f"\nNow analyze the full query image: <image>\nProvide the bounding box coordinates [x, y, w, h] for all instances of '{target_class}' in this query image."
        
        return images_to_return, prompt

class SetOfMarkStrategy(BaseICLStrategy):
    """
    (d) Set-of-Mark + visual examples: Draws the ground truth bounding boxes directly 
    onto the support images and passes them as visual tokens.
    """
    def __init__(self, box_color: str = "red", line_width: int = 3):
        self.box_color = box_color
        self.line_width = line_width

    def apply(self, query_image: Image.Image, target_class: str, support_examples: List[Dict[str, Any]]) -> Tuple[List[Image.Image], str]:
        images_to_return = []
        prompt = f"You are tasked with detecting '{target_class}'. Here are reference images where '{target_class}' is explicitly marked with {self.box_color} bounding boxes:\n"
        
        for i, ex in enumerate(support_examples):
            # Operate on a copy to prevent mutating the original dataset images
            marked_img = ex["image"].copy()
            draw = ImageDraw.Draw(marked_img)
            
            for box in ex["boxes"]:
                x, y, w, h = box
                # Draw rectangle [x0, y0, x1, y1]
                draw.rectangle([x, y, x + w, y + h], outline=self.box_color, width=self.line_width)
                
            images_to_return.append(marked_img)
            prompt += f"Marked Reference {i+1}: <image>\n"
            
        images_to_return.append(query_image)
        prompt += f"\nNow analyze the unmarked query image: <image>\nProvide the bounding box coordinates [x, y, w, h] for all instances of '{target_class}' in this query image."
        
        return images_to_return, prompt