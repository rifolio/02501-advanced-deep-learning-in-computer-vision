from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

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
