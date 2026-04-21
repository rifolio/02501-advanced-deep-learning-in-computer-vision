import json
import os

def build_class_support_set(
    annotations_path: str,
    image_dir: str,
    target_class: str,
    k: int = 5
) -> list[dict]:
    """
    Builds a support set of k images for a specific novel class.
    """
    # Direct mapping for only the 20 PASCAL VOC novel classes
    novel_class_to_id = {
        "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, 
        "airplane": 4, "bus": 5, "train": 6, "boat": 8, 
        "bird": 14, "cat": 15, "dog": 16, "horse": 17, 
        "sheep": 18, "cow": 19, "bottle": 39, "chair": 56, 
        "couch": 57, "potted plant": 58, "dining table": 60, "tv": 62
    }
    
    if target_class not in novel_class_to_id:
        raise ValueError(f"Class '{target_class}' is not in the novel classes subset.")
        
    target_id = novel_class_to_id[target_class]
    
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
        
    support_set = []
    
    for ann in annotations:
        if len(support_set) >= k:
            break
            
        objects = ann["objects"]
        
        # Find all bounding box indices where the category matches our target ID
        target_indices = [
            i for i, cat_id in enumerate(objects["category"]) 
            if cat_id == target_id
        ]
        
        if not target_indices:
            continue
            
        class_bboxes = [objects["bbox"][i] for i in target_indices]
        
        support_set.append({
            "image_path": os.path.join(image_dir, ann["file_name"]),
            "class_name": target_class,
            "bboxes": class_bboxes
        })
        
    return support_set

# --- Usage Example ---
if __name__ == "__main__":
    ANNOTATIONS_FILE = "./data/coco_novel_10_shot/hf_subset_annotations.json"
    IMAGE_DIRECTORY = "./data/coco_novel_10_shot"
    
    # Get a 3-shot support set for the "dog" class
    dog_support_set = build_class_support_set(
        annotations_path=ANNOTATIONS_FILE,
        image_dir=IMAGE_DIRECTORY,
        target_class="dog",
        k=3
    )
    
    # Print the output beautifully
    print(json.dumps(dog_support_set, indent=2))