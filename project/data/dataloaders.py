import logging
from pathlib import Path

from torch.utils.data import DataLoader

from config import settings
from data.eval_split import load_eval_split
from data.support_sampler import SupportSetSampler
from .datasets import COCOFewShotDataset, COCOZeroShotDataset

logger = logging.getLogger(__name__)


def vlm_collate_fn(batch):
    # Returns a list of dictionaries to handle variable image sizes
    return batch


def _load_manifest_for_dataset():
    path = settings.resolved_eval_split_path()
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"eval_split_path not found: {path}")
    manifest = load_eval_split(path)
    ann_resolved = Path(settings.ann_file)
    if not ann_resolved.is_absolute():
        ann_resolved = Path.cwd() / ann_resolved
    ann_resolved = ann_resolved.resolve()
    man_ann = Path(manifest.ann_file)
    if not man_ann.is_absolute():
        man_ann = Path.cwd() / man_ann
    man_ann = man_ann.resolve()
    if man_ann != ann_resolved:
        logger.warning(
            "Manifest ann_file (%s) does not match settings.ann_file (%s); using settings.ann_file for COCO().",
            man_ann,
            ann_resolved,
        )
    return manifest


def get_coco_dataloader(batch_size=1, num_workers=4):
    manifest = _load_manifest_for_dataset()
    image_ids = manifest.image_ids if manifest else None
    eval_cat_ids = set(manifest.eval_cat_ids) if manifest and manifest.eval_cat_ids else None

    dataset = COCOZeroShotDataset(
        ann_file=settings.ann_file,
        img_dir=settings.img_dir,
        image_ids=image_ids,
        eval_cat_ids=eval_cat_ids,
        manifest=manifest,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=vlm_collate_fn,
    )


def get_coco_few_shot_dataloader(batch_size=1, num_workers=4, k_shot: int = 1):
    manifest = _load_manifest_for_dataset()
    image_ids = manifest.image_ids if manifest else None
    eval_cat_ids = set(manifest.eval_cat_ids) if manifest and manifest.eval_cat_ids else None
    excluded_ids = set(image_ids) if image_ids else set()

    support_sampler = SupportSetSampler(
        ann_file=settings.ann_file,
        img_dir=settings.img_dir,
        excluded_image_ids=excluded_ids,
        seed=settings.few_shot_seed,
    )

    dataset = COCOFewShotDataset(
        ann_file=settings.ann_file,
        img_dir=settings.img_dir,
        support_sampler=support_sampler,
        k_shot=k_shot,
        image_ids=image_ids,
        eval_cat_ids=eval_cat_ids,
        manifest=manifest,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=vlm_collate_fn,
    )

coco_val_loader = get_coco_dataloader()