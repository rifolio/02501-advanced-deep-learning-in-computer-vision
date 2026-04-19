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
    """
    (a) Side-by-side layout: Spatially concatenates support images and the query image 
    into a single canvas horizontally.
    """
    def apply(self, query_image: Image.Image, target_class: str, support_examples: List[Dict[str, Any]]) -> Tuple[List[Image.Image], str]:
        # Resize support images to match the query image height for uniform concatenation
        target_h = query_image.height
        images_to_stitch = [ex["image"] for ex in support_examples] + [query_image]
        
        resized_images = []
        for img in images_to_stitch:
            if img.height != target_h:
                aspect_ratio = img.width / img.height
                new_w = int(target_h * aspect_ratio)
                resized_images.append(img.resize((new_w, target_h)))
            else:
                resized_images.append(img)
        
        total_w = sum(img.width for img in resized_images)
        stitched_image = Image.new('RGB', (total_w, target_h))
        
        prompt = f"Detect all instances of {target_class}. Here are support examples alongside the query image.\n"
        x_offset = 0
        
        for i, img in enumerate(resized_images):
            stitched_image.paste(img, (x_offset, 0))
            
            # If it is a support example, map its ground truth boxes to the new stitched coordinate space
            if i < len(support_examples):
                orig_w, orig_h = support_examples[i]["image"].width, support_examples[i]["image"].height
                scale_x = img.width / orig_w
                scale_y = img.height / orig_h
                
                adjusted_boxes = []
                for box in support_examples[i]["boxes"]:
                    x, y, w, h = box
                    new_x = (x * scale_x) + x_offset
                    new_y = (y * scale_y)
                    new_w = w * scale_x
                    new_h = h * scale_y
                    adjusted_boxes.append([round(new_x, 1), round(new_y, 1), round(new_w, 1), round(new_h, 1)])
                
                prompt += f"Image section {i+1} boxes: {adjusted_boxes}\n"
            
            x_offset += img.width
            
        prompt += f"\n<image>\nProvide the bounding box coordinates [x, y, w, h] for {target_class} in the final (rightmost) section of this composite image."
        
        return [stitched_image], prompt

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