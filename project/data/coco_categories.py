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
