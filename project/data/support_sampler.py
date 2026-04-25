from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional
import json
from PIL import Image
from pycocotools.coco import COCO


@dataclass(frozen=True)
class SupportExample:
    image_id: int
    image: Image.Image
    boxes: list[list[float]]
    category_id: int
    class_name: Optional[str] = None


class SupportSetSampler:
    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        excluded_image_ids: set[int] | None = None,
        seed: int = 42,
    ):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.seed = seed
        self.excluded_image_ids = excluded_image_ids or set()

        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_name = {cat["id"]: cat["name"] for cat in cats}
        self._image_pool_by_cat = self._build_image_pool()

    def _build_image_pool(self) -> dict[int, list[int]]:
        image_pool_by_cat: dict[int, list[int]] = {}

        for cat_id in self.coco.getCatIds():
            cat_img_ids = set(self.coco.getImgIds(catIds=[cat_id]))
            cat_img_ids.difference_update(self.excluded_image_ids)
            image_pool_by_cat[cat_id] = sorted(cat_img_ids)

        return image_pool_by_cat

    def _load_support_example(self, image_id: int, cat_id: int) -> SupportExample:
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[image_id], catIds=[cat_id])
        anns = self.coco.loadAnns(ann_ids)
        boxes = [ann["bbox"] for ann in anns]

        return SupportExample(
            image_id=image_id,
            image=image,
            boxes=boxes,
            category_id=cat_id,
            class_name=self.cat_id_to_name.get(cat_id),
        )

    def sample(self, cat_id: int, k: int) -> list[SupportExample]:
        if k <= 0:
            return []

        pool = self._image_pool_by_cat.get(cat_id, [])
        if not pool:
            return []

        rng = random.Random(self.seed + int(cat_id))
        if len(pool) >= k:
            sampled_img_ids = rng.sample(pool, k)
        else:
            sampled_img_ids = [rng.choice(pool) for _ in range(k)]

        return [self._load_support_example(img_id, cat_id) for img_id in sampled_img_ids]

class HFSupportSampler:
    def __init__(self, hf_ann_file: str, hf_img_dir: str, seed: int = 42):
        self.hf_img_dir = hf_img_dir
        self.seed = seed
        
        with open(hf_ann_file, "r", encoding="utf-8") as f:
            self.hf_annotations = json.load(f)

        # Map standard MSCOCO cat_ids to your HF category IDs
        self.coco_to_hf_map = {
            1: 0,   # person
            2: 1,   # bicycle
            3: 2,   # car
            4: 3,   # motorcycle
            5: 4,   # airplane
            6: 5,   # bus
            7: 6,   # train
            9: 8,   # boat
            16: 14, # bird
            17: 15, # cat
            18: 16, # dog
            19: 17, # horse
            20: 18, # sheep
            21: 19, # cow
            44: 39, # bottle
            62: 56, # chair
            63: 57, # couch
            64: 58, # potted plant
            67: 60, # dining table
            72: 62, # tv
        }

    def sample(self, cat_id: int, k: int) -> list[SupportExample]:
        if k <= 0 or cat_id not in self.coco_to_hf_map:
            return []

        hf_target_id = self.coco_to_hf_map[cat_id]
        valid_anns = []

        # Find all images containing this specific HF category
        for ann in self.hf_annotations:
            if hf_target_id in ann["objects"]["category"]:
                valid_anns.append(ann)

        if not valid_anns:
            return []

        rng = random.Random(self.seed + int(cat_id))
        
        # Handle cases where we have fewer images than k
        if len(valid_anns) >= k:
            sampled_anns = rng.sample(valid_anns, k)
        else:
            sampled_anns = [rng.choice(valid_anns) for _ in range(k)]

        support_examples = []
        for ann in sampled_anns:
            image_id = ann["image_id"]
            img_path = os.path.join(self.hf_img_dir, ann["file_name"])
            image = Image.open(img_path).convert("RGB")

            # Extract only the boxes for the target category
            objects = ann["objects"]
            boxes = [
                objects["bbox"][i]
                for i, c_id in enumerate(objects["category"])
                if c_id == hf_target_id
            ]

            support_examples.append(
                SupportExample(
                    image_id=image_id,
                    image=image,
                    boxes=boxes,
                    category_id=cat_id,
                )
            )

        return support_examples