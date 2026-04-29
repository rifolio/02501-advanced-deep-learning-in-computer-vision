#!/usr/bin/env python3
"""
Build a persistent COCO val eval split manifest (JSON).

Example:
  uv run python scripts/build_eval_split.py --out data/splits/val_pilot.json --seed 42 --max-images 700 --novel-only
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from pycocotools.coco import COCO

# Allow running as script from project root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.coco_categories import novel_cat_ids_from_coco  # noqa: E402
from data.eval_split import EvalSplitManifest, save_eval_split  # noqa: E402


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_ann_path(ann: str) -> Path:
    p = Path(ann)
    if not p.is_absolute():
        p = project_root() / p
    return p.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ann-file",
        default="data/coco/annotations/instances_val2017.json",
        help="Path to instances_val2017.json (relative to project root or absolute).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output manifest path (e.g. data/splits/val_pilot.json).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling image ids before capping.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images after filtering. Omit for no cap (all filtered images).",
    )
    parser.add_argument(
        "--novel-only",
        action="store_true",
        help="Keep only val images that contain at least one VOC-overlap novel class.",
    )
    parser.add_argument(
        "--eval-novel-categories-only",
        action="store_true",
        help="Set eval_cat_ids in manifest to the 20 novel class ids (for novel-only mAP).",
    )
    parser.add_argument(
        "--contain-cat-names",
        default="",
        help=(
            "Comma-separated COCO category names (e.g. horse). "
            "Only images with at least one annotation in these categories are kept."
        ),
    )
    parser.add_argument(
        "--contain-cat-ids",
        default="",
        help="Comma-separated COCO category ids (alternative to --contain-cat-names).",
    )
    parser.add_argument(
        "--eval-cat-names",
        default="",
        help=(
            "Comma-separated COCO names; sets manifest eval_cat_ids for scoring and dataloader "
            "(e.g. horse-only mAP). Overrides --eval-novel-categories-only if both are set."
        ),
    )
    parser.add_argument(
        "--eval-cat-ids",
        default="",
        help="Comma-separated category ids for eval_cat_ids (alternative to --eval-cat-names).",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Optional description string stored in the manifest.",
    )
    args = parser.parse_args()

    ann_abs = resolve_ann_path(args.ann_file)
    if not ann_abs.is_file():
        print(f"Annotation file not found: {ann_abs}", file=sys.stderr)
        return 1

    coco = COCO(str(ann_abs))
    all_img_ids = coco.getImgIds()
    all_img_set = set(all_img_ids)

    def _parse_int_list(raw: str) -> list[int]:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]

    contain_ids: list[int] = []
    if args.contain_cat_ids:
        contain_ids.extend(_parse_int_list(args.contain_cat_ids))
    if args.contain_cat_names:
        for name in [n.strip() for n in args.contain_cat_names.split(",") if n.strip()]:
            found = coco.getCatIds(catNms=[name])
            if not found:
                print(f"Unknown COCO category name: {name!r}", file=sys.stderr)
                return 1
            contain_ids.extend(int(x) for x in found)
    contain_ids = list(dict.fromkeys(contain_ids))

    if args.novel_only:
        novel_ids = set(novel_cat_ids_from_coco(coco))
        ann_ids = coco.getAnnIds(catIds=list(novel_ids))
        img_with_novel = {a["image_id"] for a in coco.loadAnns(ann_ids)}
        candidates = sorted(img_with_novel & all_img_set)
    else:
        candidates = sorted(all_img_set)

    if contain_ids:
        ann_ids = coco.getAnnIds(catIds=contain_ids)
        img_with_cat = {a["image_id"] for a in coco.loadAnns(ann_ids)}
        candidates = sorted(set(candidates) & img_with_cat)

    rng = random.Random(args.seed)
    order = list(candidates)
    rng.shuffle(order)

    if args.max_images is not None:
        order = order[: max(0, args.max_images)]

    rel_ann = str(Path(args.ann_file))
    if not Path(args.ann_file).is_absolute():
        rel_ann = str(Path(args.ann_file).as_posix())

    eval_cat_ids: list[int] | None = None
    if args.eval_novel_categories_only:
        eval_cat_ids = novel_cat_ids_from_coco(coco)
    if args.eval_cat_ids:
        eval_cat_ids = _parse_int_list(args.eval_cat_ids)
    if args.eval_cat_names:
        eids: list[int] = []
        for name in [n.strip() for n in args.eval_cat_names.split(",") if n.strip()]:
            found = coco.getCatIds(catNms=[name])
            if not found:
                print(f"Unknown eval category name: {name!r}", file=sys.stderr)
                return 1
            eids.extend(int(x) for x in found)
        eval_cat_ids = list(dict.fromkeys(eids))

    manifest = EvalSplitManifest(
        version=1,
        ann_file=rel_ann,
        image_ids=order,
        eval_cat_ids=eval_cat_ids,
        seed=args.seed,
        description=args.description or "",
    )

    out_path = args.out
    if not out_path.is_absolute():
        out_path = project_root() / out_path
    save_eval_split(manifest, out_path)

    print(f"Wrote {out_path}")
    print(f"  images: {len(manifest.image_ids)}")
    print(f"  eval_cat_ids: {eval_cat_ids if eval_cat_ids else 'null (all categories)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
