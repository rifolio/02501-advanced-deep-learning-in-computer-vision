# 02501 Final Project

## UV

Navigate to `./project` and run:

```bash
uv sync
```

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
