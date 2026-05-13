import logging
from pathlib import Path

from torch.utils.data import DataLoader
from pycocotools.coco import COCO

from config import settings
from data.eval_split import load_eval_split
from data.support_sampler import HFSupportSampler, SupportSetSampler
from .datasets import COCOFewShotDataset, COCOOracleShotDataset, COCOZeroShotDataset

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
    excluded_ids: set[int] = set(image_ids) if image_ids else set()
    excluded_filenames: set[str] = set()

    if excluded_ids:
        coco = COCO(settings.ann_file)
        img_infos = coco.loadImgs(list(excluded_ids))
        excluded_filenames = {str(info["file_name"]) for info in img_infos}

    use_hf = settings.few_shot_support_source.lower() == "hf"
    hf_ann = Path(settings.hf_support_ann_file)
    if not hf_ann.is_absolute():
        hf_ann = Path.cwd() / hf_ann
    hf_ann = hf_ann.resolve()

    if use_hf and hf_ann.is_file():
        support_sampler = HFSupportSampler(
            hf_ann_file=str(settings.hf_support_ann_file),
            hf_img_dir=str(settings.hf_support_img_dir),
            seed=settings.few_shot_seed,
            excluded_image_ids=excluded_ids,
            excluded_filenames=excluded_filenames,
        )
        logger.info("Few-shot support pool: HF subset at %s", hf_ann)
    else:
        if use_hf:
            logger.warning(
                "FEW_SHOT_SUPPORT_SOURCE=hf but annotations not found at %s; "
                "using COCO val support pool (same ann_file / img_dir as queries).",
                hf_ann,
            )
        support_sampler = SupportSetSampler(
            ann_file=settings.ann_file,
            img_dir=settings.img_dir,
            excluded_image_ids=excluded_ids,
            seed=settings.few_shot_seed,
        )
        logger.info("Few-shot support pool: COCO val (%s)", settings.ann_file)

    if isinstance(support_sampler, HFSupportSampler) and manifest:
        overlap = support_sampler.audit_exclusion_overlap()
        if overlap["overlapping_image_ids"] or overlap["overlapping_filenames"]:
            logger.warning(
                "Few-shot support overlap audit: found %d image_id overlap(s) and %d filename overlap(s) "
                "between eval split and HF support pool; excluded from support sampling.",
                overlap["overlapping_image_ids"],
                overlap["overlapping_filenames"],
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


def get_coco_oracle_shot_dataloader(batch_size=1, num_workers=4):
    manifest = _load_manifest_for_dataset()
    image_ids = manifest.image_ids if manifest else None
    eval_cat_ids = set(manifest.eval_cat_ids) if manifest and manifest.eval_cat_ids else None

    dataset = COCOOracleShotDataset(
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