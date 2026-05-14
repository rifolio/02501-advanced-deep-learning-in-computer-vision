#!/bin/bash
# Oracle-shot evaluation on val_test_700 split (~6h on A100 for Qwen 7B).
# Usage: bsub < jobscript_oracle_700.sh
#
# Change PROJECT_ROOT to your home path on the cluster before submitting.

#BSUB -J VLM_OracleShot_700
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s254355@student.dtu.dk
#BSUB -B
#BSUB -N
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 07:00
#BSUB -o logs/output_oracle_700_%J.out
#BSUB -e logs/error_oracle_700_%J.err

PROJECT_ROOT="/zhome/99/f/223556/02501-advanced-deep-learning-in-computer-vision/project"

echo "=========================================="
echo "Job ID: $LSB_JOBID  |  Start: $(date)  |  Host: $(hostname)"
echo "=========================================="

module load cuda/12.4

cd "$PROJECT_ROOT" || exit 1
mkdir -p logs

source "$PROJECT_ROOT/.venv/bin/activate"

export EXPERIMENT_MODE=oracle_shot
export PROMPT_STRATEGY=oracle_shot
export MODEL_NAME=qwen          # or internvl
export EVAL_SPLIT_PATH=data/splits/val_test_700.json
export LOG_LEVEL=INFO

nvidia-smi
python --version

echo "Starting oracle-shot evaluation (700 images, 20 classes = 14000 queries)..."
python main.py

EXIT_CODE=$?
echo "=========================================="
echo "Finished: $(date)  |  Exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
