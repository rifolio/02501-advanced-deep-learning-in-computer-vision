import os
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class COCOZeroShotDataset(Dataset):
    def __init__(self, ann_file: str, img_dir: str):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.img_ids = self.coco.getImgIds()
        
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in cats}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        image = Image.open(img_path).convert("RGB")
        
        # Get ground truth classes present in this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        present_cat_ids = list(set([ann['category_id'] for ann in anns]))
        
        targets = []
        for cat_id in present_cat_ids:
            targets.append({
                "category_id": cat_id,
                "class_name": self.cat_id_to_name[cat_id]
            })
            
        return {
            "image_id": img_id,
            "image": image,
            "width": img_info['width'],
            "height": img_info['height'],
            "targets": targets
        }