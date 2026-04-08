from torch.utils.data import DataLoader
from .datasets import COCOZeroShotDataset
from config import settings

def vlm_collate_fn(batch):
    # Returns a list of dictionaries to handle variable image sizes
    return batch

def get_coco_dataloader(batch_size=1, num_workers=4):
    dataset = COCOZeroShotDataset(
        ann_file=settings.ann_file, 
        img_dir=settings.img_dir
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=vlm_collate_fn
    )
    
    return dataloader

coco_val_loader = get_coco_dataloader()