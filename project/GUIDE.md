# Guide: venv, interactive GPU node, and example runs

This document assumes you work on a **Linux cluster** with **CUDA** (e.g. an interactive GPU allocation). Adjust queue names and `module load` lines to your site.

---

## 1. Project layout and venv

Always run commands from the **`project/`** directory (where `main.py`, `config.py`, and `.venv` live).

```bash
cd /path/to/02501-advanced-deep-learning-in-computer-vision/project
```

### Create the venv (first time, or after changing `pyproject.toml`)

If you use **uv**:

```bash
uv venv --python 3.12 .venv
uv sync
```

If `uv sync` fails (disk space, lock issues), you can install into `.venv` with `uv pip install …` or `pip` after `source .venv/bin/activate`.

### Activate the venv (every session)

```bash
source .venv/bin/activate
```

You should see `(.venv)` in your shell prompt. Prefer **`python main.py`** after activation so you do not trigger an unintended dependency sync (see below).

### Deactivate

```bash
deactivate
```

---

## 2. Interactive node with GPU

### Request a GPU session

Use your cluster’s workflow (Slurm `salloc`, LSF `bsub -Is`, etc.) so your shell runs **on a node that has a GPU**. On a **login node** without a GPU job, `torch` may still import but **nothing will use the GPU** for your training/eval.

### Load CUDA (if your site requires it)

Example (version must match your PyTorch build):

```bash
module load cuda/12.4
```

### Check the GPU and PyTorch

```bash
nvidia-smi
python -c "import torch; print('cuda_available:', torch.cuda.is_available(), 'cuda:', torch.version.cuda)"
```

If `cuda_available` is **False** but `nvidia-smi` works, fix your **PyTorch install** (CUDA wheel) or module stack.

### Pin which GPU to use

```bash
export CUDA_VISIBLE_DEVICES=0   # or 1, etc.
export DEVICE=cuda              # optional; default is cuda if torch sees a GPU
```

---

## 3. Data and splits

### COCO val (if missing)

```bash
python scripts/download_coco_val.py
```

### Horse-only eval splits (examples)

```bash
# 50 images, horse category only (eval_cat_ids + images that contain horse)
python scripts/build_eval_split.py \
  --out data/splits/val_horse_50.json \
  --seed 42 \
  --max-images 50 \
  --contain-cat-names horse \
  --eval-cat-names horse \
  --description "horse-only, 50 images"
```

Change `--max-images` to `100`, `200`, etc. Outputs under `data/splits/`.

Point runs at a split with:

```bash
export EVAL_SPLIT_PATH=data/splits/val_horse_50.json
```

Omit `EVAL_SPLIT_PATH` for **full val2017** (slow).

---

## 4. Weights & Biases (W&B)

Credentials can come from **`API_KEY`** in `.env` or `wandb login`.

### Online (default)

Do **not** set offline mode:

```bash
unset WANDB_MODE
# or
export WANDB_MODE=online
```

### Offline (no upload; local only under `wandb/`)

```bash
export WANDB_MODE=offline
```

### Sync an offline run later

```bash
wandb sync wandb/offline-run-<id>
```

### Less media in the UI (defaults in `config`)

- `VIZ_MAX_IMAGES` — how many side-by-side JPEGs are written and added to the artifact (default **15**).
- `VIZ_PREVIEW_COUNT` — how many appear in the run’s media panel (default **15**).

Raise them when you need more visuals, e.g. `VIZ_MAX_IMAGES=50`.

---

## 5. Running evaluations

Always from **`project/`** with venv **activated**. Recommended pattern:

```bash
python main.py
```

### Why `python` and not `uv run`?

- **`uv run python main.py`** may **re-resolve** the lockfile and reinstall packages (e.g. switching `transformers` versions), which can break a working venv or fill disk cache.
- After **`source .venv/bin/activate`**, **`python main.py`** uses exactly that environment.

Use `uv run` only when you intend uv to manage installs.

---

## 6. Example commands

Replace paths and GPU index as needed.

### Zero-shot, InternVL, horse split

```bash
source .venv/bin/activate
cd /path/to/.../project

CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
EVAL_SPLIT_PATH=data/splits/val_horse_50.json \
MODEL_NAME=internvl \
EXPERIMENT_MODE=zero_shot \
python main.py
```

### Few-shot, K=1, horse split

Support images are sampled from **COCO val** (excluding query split images), unless you set `FEW_SHOT_SUPPORT_SOURCE=hf` and provide the HF subset files.

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
EVAL_SPLIT_PATH=data/splits/val_horse_50.json \
MODEL_NAME=internvl \
EXPERIMENT_MODE=few_shot \
K_SHOT=1 \
FEW_SHOT_SEED=42 \
PROMPT_STRATEGY=side_by_side \
python main.py
```

### Few-shot, K=3

```bash
K_SHOT=3
```

Same as above with `K_SHOT=3`.

### Other models

```bash
MODEL_NAME=qwen               # or grounding_dino, etc. (see main.py)
```

### Full raw VLM text in logs (debugging)

Already **on by default** in `config` (`VLM_LOG_FULL_TEXT`); to shorten logs:

```bash
export VLM_LOG_FULL_TEXT=false
```

### Longer generation for many boxes (InternVL)

```bash
export INTERNVL_MAX_NEW_TOKENS=2048
```

### Log run to a file

```bash
python main.py 2>&1 | tee logs/my_run.log
```

---

## 7. Checklist if something fails

| Symptom | What to check |
|--------|----------------|
| `device=cpu` but you have a GPU | GPU job, `nvidia-smi`, `torch.cuda.is_available()`, `DEVICE=cuda` |
| `FileNotFoundError` for split JSON | Build split or set `EVAL_SPLIT_PATH` to an existing file |
| Few-shot HF file missing | Use default COCO support pool, or run with `FEW_SHOT_SUPPORT_SOURCE=coco` (default) |
| No W&B online | `unset WANDB_MODE`, API key, network |
| `uv` / disk errors | Use `python` + activated `.venv`; point `UV_CACHE_DIR` to a large filesystem if needed |

---

## 8. See also

- [`README.md`](README.md) — UV, COCO download, batch jobs
- [`data/splits/README.md`](data/splits/README.md) — split manifest schema
