import os

from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from data.eval_split import EvalSplitManifest
from data.support_sampler import SupportSetSampler


class COCOZeroShotDataset(Dataset):
    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        image_ids: list[int] | None = None,
        eval_cat_ids: set[int] | None = None,
        manifest: EvalSplitManifest | None = None,
    ):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.manifest = manifest
        self.eval_cat_ids = eval_cat_ids

        if image_ids is not None:
            self.img_ids = list(image_ids)
        else:
            self.img_ids = self.coco.getImgIds()

        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_name = {cat["id"]: cat["name"] for cat in cats}
        if self.eval_cat_ids is not None:
            self.query_cat_ids = sorted(c for c in self.cat_id_to_name if c in self.eval_cat_ids)
        else:
            self.query_cat_ids = sorted(self.cat_id_to_name)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        present_cat_ids = list({ann["category_id"] for ann in anns})

        if self.eval_cat_ids is not None:
            present_cat_ids = [c for c in present_cat_ids if c in self.eval_cat_ids]

        targets = []
        for cat_id in present_cat_ids:
            targets.append(
                {
                    "category_id": cat_id,
                    "class_name": self.cat_id_to_name[cat_id],
                }
            )
        query_targets = [
            {"category_id": cat_id, "class_name": self.cat_id_to_name[cat_id]}
            for cat_id in self.query_cat_ids
        ]

        return {
            "image_id": img_id,
            "image": image,
            "width": img_info["width"],
            "height": img_info["height"],
            "targets": targets,
            "query_targets": query_targets,
        }


class COCOFewShotDataset(Dataset):
    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        support_sampler: SupportSetSampler,
        k_shot: int,
        image_ids: list[int] | None = None,
        eval_cat_ids: set[int] | None = None,
        manifest: EvalSplitManifest | None = None,
    ):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.manifest = manifest
        self.eval_cat_ids = eval_cat_ids
        self.support_sampler = support_sampler
        self.k_shot = k_shot

        if image_ids is not None:
            self.img_ids = list(image_ids)
        else:
            self.img_ids = self.coco.getImgIds()

        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_name = {cat["id"]: cat["name"] for cat in cats}
        if self.eval_cat_ids is not None:
            self.query_cat_ids = sorted(c for c in self.cat_id_to_name if c in self.eval_cat_ids)
        else:
            self.query_cat_ids = sorted(self.cat_id_to_name)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        present_cat_ids = list({ann["category_id"] for ann in anns})

        if self.eval_cat_ids is not None:
            present_cat_ids = [c for c in present_cat_ids if c in self.eval_cat_ids]

        targets = []
        support_by_cat = {}
        for cat_id in present_cat_ids:
            targets.append(
                {
                    "category_id": cat_id,
                    "class_name": self.cat_id_to_name[cat_id],
                }
            )
        for cat_id in self.query_cat_ids:
            support_by_cat[cat_id] = self.support_sampler.sample(cat_id, self.k_shot)
        query_targets = [
            {"category_id": cat_id, "class_name": self.cat_id_to_name[cat_id]}
            for cat_id in self.query_cat_ids
        ]

        return {
            "image_id": img_id,
            "image": image,
            "width": img_info["width"],
            "height": img_info["height"],
            "targets": targets,
            "query_targets": query_targets,
            "support_by_cat": support_by_cat,
        }
