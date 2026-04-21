#!/usr/bin/env python3
"""
Create visual few-shot prompt previews from COCO data.

This script samples support images for a category, overlays GT boxes, and saves:
  - per-support annotated images
  - the query image
  - a side-by-side contact sheet (supports + query)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO

# Allow running as script from project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.eval_split import load_eval_split  # noqa: E402
from data.support_sampler import SupportSetSampler  # noqa: E402
from data.visual_prompt import render_bboxes  # noqa: E402


def resolve_project_path(raw_path: str) -> Path:
    p = Path(raw_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p.resolve()


def make_side_by_side(images: list[Image.Image], target_height: int = 512) -> Image.Image:
    resized: list[Image.Image] = []
    for img in images:
        w, h = img.size
        if h != target_height:
            new_w = max(1, int(w * (target_height / h)))
            resized.append(img.resize((new_w, target_height)))
        else:
            resized.append(img)

    total_w = sum(img.width for img in resized)
    canvas = Image.new("RGB", (total_w, target_height), color=(255, 255, 255))
    x = 0
    for img in resized:
        canvas.paste(img, (x, 0))
        x += img.width
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann-file", default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--img-dir", default="data/coco/val2017")
    parser.add_argument("--eval-split", default="data/splits/val_pilot.json")
    parser.add_argument("--cat-id", type=int, required=True, help="COCO category id to preview")
    parser.add_argument("--k-shot", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="outputs/few_shot_preview")
    args = parser.parse_args()

    ann_file = resolve_project_path(args.ann_file)
    img_dir = resolve_project_path(args.img_dir)
    split_path = resolve_project_path(args.eval_split)
    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ann_file.is_file():
        raise FileNotFoundError(f"Missing ann file: {ann_file}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing image dir: {img_dir}")
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing eval split: {split_path}")

    manifest = load_eval_split(split_path)
    excluded = set(manifest.image_ids)
    sampler = SupportSetSampler(
        ann_file=str(ann_file),
        img_dir=str(img_dir),
        excluded_image_ids=excluded,
        seed=args.seed,
    )

    supports = sampler.sample(args.cat_id, args.k_shot)
    if not supports:
        print(f"No support samples found for cat_id={args.cat_id}")
        return 1

    coco = COCO(str(ann_file))
    cat_name = sampler.cat_id_to_name.get(args.cat_id, f"cat_{args.cat_id}")

    # pick first eval image containing this category as query
    query_id = None
    for img_id in manifest.image_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[args.cat_id])
        if ann_ids:
            query_id = img_id
            break
    if query_id is None:
        print(f"No query image in eval split contains cat_id={args.cat_id}")
        return 1

    qinfo = coco.loadImgs(query_id)[0]
    query_img = Image.open(img_dir / qinfo["file_name"]).convert("RGB")

    support_annotated: list[Image.Image] = []
    for i, ex in enumerate(supports, start=1):
        img = render_bboxes(ex.image, ex.boxes, color="red", line_width=3, numbered_labels=True)
        support_annotated.append(img)
        out_file = out_dir / f"support_{i:02d}_{cat_name}_img{ex.image_id}.png"
        img.save(out_file)

    query_file = out_dir / f"query_{cat_name}_img{query_id}.png"
    query_img.save(query_file)

    sheet = make_side_by_side(support_annotated + [query_img], target_height=512)
    sheet_file = out_dir / f"side_by_side_{cat_name}_k{args.k_shot}_query{query_id}.png"
    sheet.save(sheet_file)

    print("Saved preview files:")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p}")
    print("\nOpen the side-by-side file to inspect the full prompt visual.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
