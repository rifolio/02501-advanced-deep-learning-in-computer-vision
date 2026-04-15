# COCO evaluation split manifests

Persistent JSON files describe **which validation `image_ids`** to iterate and optionally **which `eval_cat_ids`** to run detection on and to pass into COCOeval.

## Schema (`version`: 1)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | int | yes | Must be `1`. |
| `ann_file` | string | yes | Path relative to project root (or absolute) to the COCO instances JSON used when the split was built. |
| `image_ids` | int[] | yes | Ordered list of COCO `image_id` values for evaluation. |
| `eval_cat_ids` | int[] or null | no | If set, only these categories are evaluated (targets filtered; `COCOeval.params.catIds`). If omitted or `null`, all 80 categories are allowed on the selected images. |
| `seed` | int or null | no | RNG seed used to build the split (reproducibility). |
| `description` | string | no | Human-readable note. |

## Generate a split

From `project/`:

```bash
uv run python scripts/build_eval_split.py --help
```

Example (novel-only images, capped for a fast pilot):

```bash
uv run python scripts/build_eval_split.py \
  --out data/splits/val_pilot_novel.json \
  --seed 42 \
  --max-images 700 \
  --novel-only
```

Point `eval_split_path` in settings (or `.env`) at the generated file.

See also [`val_pilot.example.json`](val_pilot.example.json).
