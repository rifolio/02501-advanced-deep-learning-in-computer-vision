import argparse
import json
import os
from collections.abc import Sequence

from data.coco_categories import COCO_NOVEL_CLASS_NAMES

# Mapping from novel class names to Hugging Face category indices used in
# detection-datasets/coco `objects["category"]`.
NOVEL_CLASS_TO_HF_ID = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "boat": 8,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "bottle": 39,
    "chair": 56,
    "couch": 57,
    "potted plant": 58,
    "dining table": 60,
    "tv": 62,
}

_ALLOWED_NOVEL_CLASSES = tuple(COCO_NOVEL_CLASS_NAMES)
_ALLOWED_NOVEL_CLASS_SET = set(_ALLOWED_NOVEL_CLASSES)


def _normalize_target_classes(target_classes: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(target_classes, str):
        raw = [part.strip() for part in target_classes.split(",")]
    else:
        raw = []
        for item in target_classes:
            raw.extend(part.strip() for part in item.split(","))

    classes = tuple(dict.fromkeys(name for name in raw if name))
    if not classes:
        raise ValueError("No target classes were provided.")

    invalid = [name for name in classes if name not in _ALLOWED_NOVEL_CLASS_SET]
    if invalid:
        allowed = ", ".join(_ALLOWED_NOVEL_CLASSES)
        invalid_joined = ", ".join(invalid)
        raise ValueError(
            f"Class(es) '{invalid_joined}' are not in the novel classes subset. "
            f"Allowed classes: {allowed}"
        )

    missing_from_hf_map = [name for name in classes if name not in NOVEL_CLASS_TO_HF_ID]
    if missing_from_hf_map:
        missing_joined = ", ".join(missing_from_hf_map)
        raise ValueError(f"Class(es) missing from HF category map: {missing_joined}")

    return classes


def build_class_support_sets(
    annotations_path: str,
    image_dir: str,
    target_classes: str | Sequence[str],
    k: int = 5,
) -> dict[str, list[dict]]:
    """
    Build per-class support sets of up to k images for one or more novel classes.
    """
    classes = _normalize_target_classes(target_classes)

    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    support_sets: dict[str, list[dict]] = {}
    for target_class in classes:
        target_id = NOVEL_CLASS_TO_HF_ID[target_class]
        support_set = []

        for ann in annotations:
            if len(support_set) >= k:
                break

            objects = ann["objects"]
            target_indices = [
                i for i, cat_id in enumerate(objects["category"]) if cat_id == target_id
            ]
            if not target_indices:
                continue

            class_bboxes = [objects["bbox"][i] for i in target_indices]
            support_set.append(
                {
                    "image_path": os.path.join(image_dir, ann["file_name"]),
                    "class_name": target_class,
                    "bboxes": class_bboxes,
                }
            )

        support_sets[target_class] = support_set

    return support_sets


def build_class_support_set(
    annotations_path: str,
    image_dir: str,
    target_class: str,
    k: int = 5,
) -> list[dict]:
    """
    Backward-compatible single-class wrapper around build_class_support_sets.
    """
    return build_class_support_sets(
        annotations_path=annotations_path,
        image_dir=image_dir,
        target_classes=target_class,
        k=k,
    )[target_class]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build support sets for one or more novel classes."
    )
    parser.add_argument(
        "--annotations-path",
        default="./data/coco_novel_10_shot/hf_subset_annotations.json",
        help="Path to hf_subset_annotations.json.",
    )
    parser.add_argument(
        "--image-dir",
        default="./data/coco_novel_10_shot",
        help="Directory containing support images.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["all"],
        #required=True,
        help='Classes to include, e.g. --classes car boat horse or --classes "car,boat".',
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Maximum number of support images per class.",
    )
    args = parser.parse_args()

    if "all" in args.classes:
        eval_classes = COCO_NOVEL_CLASS_NAMES
    else:
        eval_classes = args.classes

    support_sets = build_class_support_sets(
        annotations_path=args.annotations_path,
        image_dir=args.image_dir,
        target_classes=eval_classes,
        k=args.k,
    )
    print(json.dumps(support_sets, indent=2))
