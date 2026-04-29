import os
import json

"""
COCO FSOD-style split: 20 PASCAL-VOC-overlapping classes as 'novel', remaining 60 as 'base'.

Used by Kang et al. style protocols (Few-shot Object Detection via Feature Reweighting).
Names match MSCOCO `instances_*.json` category `name` fields.
"""

# 20 COCO categories that overlap with PASCAL VOC 20 (canonical names in COCO)
COCO_NOVEL_CLASS_NAMES: tuple[str, ...] = (
    "airplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorcycle",
    "person",
    "potted plant",
    "sheep",
    "couch",
    "train",
    "tv",
)


def novel_cat_ids_from_coco(coco) -> list[int]:
    """Resolve novel class names to sorted COCO category ids using a loaded COCO object."""
    name_to_id = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    missing = [n for n in COCO_NOVEL_CLASS_NAMES if n not in name_to_id]
    if missing:
        raise KeyError(f"Unknown COCO category names: {missing}")
    return sorted(name_to_id[n] for n in COCO_NOVEL_CLASS_NAMES)


def base_cat_ids_from_coco(coco) -> list[int]:
    """All COCO category ids not in the VOC-overlap novel set."""
    novel = set(novel_cat_ids_from_coco(coco))
    all_ids = set(coco.getCatIds())
    return sorted(all_ids - novel)

def download_hf_coco_subset(
    class_names: tuple[str, ...],
    export_dir: str,
    samples_per_class: int = 10
) -> None:
    """
    Streams the COCO dataset from Hugging Face, extracting exactly the requested
    number of images for specific classes, completely bypassing database requirements.
    """
    from datasets import load_dataset  # optional; only for HF streaming helper

    os.makedirs(export_dir, exist_ok=True)
    
    print("Connecting to Hugging Face and streaming COCO...")
    # Streaming mode avoids downloading the full dataset archive
    dataset = load_dataset("detection-datasets/coco", split="train", streaming=True)
    
    # Dynamically extract the mapping of class names to integer IDs from the dataset features
    categories = dataset.features["objects"]["category"].feature.names
    name_to_id = {name: i for i, name in enumerate(categories)}
    
    # Filter target classes to get their exact dataset IDs
    target_ids = {name_to_id[name]: name for name in class_names if name in name_to_id}
    
    # Track how many images we've saved per class
    class_counts = {cat_id: 0 for cat_id in target_ids.keys()}
    
    saved_images = 0
    annotations = []
    
    print(f"Hunting for {samples_per_class} images across {len(class_names)} classes...")
    
    # Iterate sequentially through the stream
    for sample in dataset:
        # Stop if we've fulfilled the quota for all requested classes
        if all(count >= samples_per_class for count in class_counts.values()):
            break
            
        objects = sample["objects"]
        present_categories = set(objects["category"])
        
        # Check if the image contains any of the target classes we still need
        useful_classes = present_categories.intersection(target_ids.keys())
        
        is_useful = False
        for cat_id in useful_classes:
            if class_counts[cat_id] < samples_per_class:
                class_counts[cat_id] += 1
                is_useful = True
                
        # If the image contains a needed class, save it to disk
        if is_useful:
            image_id = sample["image_id"]
            image = sample["image"]
            
            # Save the raw PIL image 
            file_name = f"{image_id:012d}.jpg"
            image.save(os.path.join(export_dir, file_name))
            saved_images += 1
            
            # Store the annotation details for this specific image
            annotations.append({
                "image_id": image_id,
                "file_name": file_name,
                "objects": objects
            })

    # Save the custom annotations JSON so you have bounding box data
    with open(os.path.join(export_dir, "hf_subset_annotations.json"), "w") as f:
        json.dump(annotations, f)
        
    print(f"\nDone! Saved {saved_images} unique images to {export_dir}.")

# --- Usage Example ---
if __name__ == "__main__":
    
    TARGET_DIRECTORY = "./data/coco_novel_10_shot"
    
    download_hf_coco_subset(
        class_names=COCO_NOVEL_CLASS_NAMES,
        export_dir=TARGET_DIRECTORY,
        samples_per_class=10
    )