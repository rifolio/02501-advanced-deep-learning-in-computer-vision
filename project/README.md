# 02501 Final Project

## UV

Navigate to `./project` and run:

```bash
uv sync
```

Run Python commands via `uv run ...` so the project environment is used consistently.

## COCO data (official download — no Google Drive)

Val2017 images and annotations are fetched from [cocodataset.org](https://cocodataset.org/#download) over HTTPS (no OAuth).

From `./project`:

```bash
uv run python scripts/download_coco_val.py
```

Downloads use **HTTP** by default (same URLs as on [cocodataset.org](https://cocodataset.org/#download)), which avoids TLS errors on some networks. Use `--https` for HTTPS, or `--https --insecure` only if you still see certificate errors behind a proxy.

This creates:

- `data/coco/val2017/` — validation images
- `data/coco/annotations/instances_val2017.json` — instance annotations

Zips are cached under `data/coco/.download_cache/` and reused on re-runs. Use `--force` to wipe extracted folders and re-download.

**Do not use `dvc pull` for COCO** — there is no default DVC remote anymore, so you will see `No remote provided and no default remote set` / “Everything is up to date” with nothing downloaded. Use the script above instead.

The `VIRTUAL_ENV` / `.venv` warning from `uv run` means your activated env path differs from the project’s `.venv`; either run commands from the repo without activating another venv, or use `uv run --active …` if you intend to use the active environment.

## Persistent eval split (optional)

To fix **which val images** (and optionally **which categories**) you score on—reproducibly across runs—generate a manifest JSON and point settings at it.

From `./project`, with COCO val annotations present:

```bash
uv run python scripts/build_eval_split.py \
  --out data/splits/val_pilot.json \
  --seed 42 \
  --max-images 700 \
  --novel-only \
  --eval-novel-categories-only
```

Alternative for no uv and other CUDA devices
```bash
CUDA_VISIBLE_DEVICES=1\
EXPERIMENT_MODE=few_shot \
MODEL_NAME=internvl \
K_SHOT=3 \
FEW_SHOT_SEED=42 \
PROMPT_STRATEGY=side_by_side \
EVAL_SPLIT_PATH=data/splits/val_novel20.json \
python main.py
```


Then either set in `.env` (or environment):

```bash
EVAL_SPLIT_PATH=data/splits/val_pilot.json
```

Or rely on the default `eval_split_path=None` in [`config.py`](config.py) for **full** val2017 evaluation.

Schema and details: [`data/splits/README.md`](data/splits/README.md).

## Running batch job

```bash
bsub < jobscript.sh
```

## Run evaluations (zero-shot / few-shot)

The entrypoint is `main.py`. Use `.env` (or exported env vars) to select mode/model.

Example `.env` values:

```bash
# Common
MODEL_NAME=qwen               # qwen | internvl | grounding_dino
EVAL_SPLIT_PATH=data/splits/val_pilot.json

# Mode switch
EXPERIMENT_MODE=zero_shot     # zero_shot | few_shot

# Few-shot only
K_SHOT=3
FEW_SHOT_SEED=42
PROMPT_STRATEGY=side_by_side  # side_by_side | cropped_exemplars | text_from_vision | set_of_mark

# Side-by-side GT vs prediction visualizations are enabled by default
# (set to false to disable)
LOG_VIZ_ARTIFACT=true
VIZ_MAX_IMAGES=50
VIZ_PREVIEW_COUNT=12
VIZ_TARGET_CATEGORY=car       # optional; leave unset to include all classes
VIZ_OUTPUT_DIR=viz
```

Run:

```bash
uv run python main.py
```

By default (`LOG_VIZ_ARTIFACT=true`), the pipeline generates side-by-side JPEGs
(`Ground Truth` on the left in green, `Prediction` on the right in red),
for the evaluated images from the chosen eval split/dataset, logs a preview panel
to the run, and uploads the full folder as a W&B artifact.

## Visual few-shot preview images

To inspect concrete support/query examples and bounding-box overlays, generate preview PNGs:

```bash
uv run python scripts/preview_few_shot_examples.py \
  --cat-id 17 \
  --k-shot 3 \
  --eval-split data/splits/val_pilot.json \
  --out-dir outputs/few_shot_preview
```

This saves:
- per-support annotated images (`support_*.png`)
- query image (`query_*.png`)
- stitched side-by-side prompt sheet (`side_by_side_*.png`)
