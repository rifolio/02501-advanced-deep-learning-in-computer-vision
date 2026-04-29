#!/bin/bash
# Short InternVL + car-only split debug job on DTU HPC (LSF).
# 1) Set PROJECT_ROOT below to your clone path on the cluster.
# 2) Copy .env from env.car_debug.example (or merge EVAL_SPLIT_PATH + K_SHOT into your .env).
# 3) bsub < jobscript_car_debug.sh

#BSUB -J VLM_InternVL_CAR_DEBUG
#BSUB -q gpua100  # Note: Verify if this should be the 02501 queue instead!
#BSUB -gpu "num=1:mode=exclusive_process"

# Email notifications
#BSUB -u s254355@dtu.dk
#BSUB -B
#BSUB -N

# CPU cores and memory
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"

# Wall clock (debug)
#BSUB -W 00:30

# Output files
#BSUB -o logs/output_car_debug_%J_%I.out
#BSUB -e logs/error_car_debug_%J_%I.err

# --- Edit this to your repo path on HPC ---
PROJECT_ROOT="/zhome/70/5/224711/02501-advanced-deep-learning-in-computer-vision/project"

echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Job Name: $LSB_JOBNAME"
echo "Start Time: $(date)"
echo "Running on host: $(hostname)"
echo "Working directory (before cd): $(pwd)"
echo "=========================================="

module load cuda/12.4

cd "$PROJECT_ROOT" || exit 1
mkdir -p logs

source "$PROJECT_ROOT/.venv/bin/activate"

echo "Project root: $PROJECT_ROOT"
echo "Working directory: $(pwd)"

# Optional: uncomment to override .env for this job only (export wins over .env in pydantic-settings)
# export MODEL_NAME=internvl
# export EXPERIMENT_MODE=few_shot
# export K_SHOT=1
# export PROMPT_STRATEGY=side_by_side
# export EVAL_SPLIT_PATH=data/splits/val_car_debug_50.json

nvidia-smi

echo "Python version:"
python --version

echo "UV version:"
uv --version

echo "Starting evaluation (config from project/.env unless exports above are set)..."
python main.py

EXIT_CODE=$?

echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
