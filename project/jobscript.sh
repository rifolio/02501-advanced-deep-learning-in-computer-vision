#!/bin/bash
#BSUB -J VLM_ZeroShot_COCO_qwen
#BSUB -q gpua100  # Note: Verify if this should be the 02501 queue instead!
#BSUB -gpu "num=1:mode=exclusive_process"

# Email notifications (add your email with #BSUB -u your_email@dtu.dk if desired)
#BSUB -u s253510@student.dtu.dk
#BSUB -B
#BSUB -N
# CPU cores and memory
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"  # Increased to 32GB to safely load 7B models

# Max wall clock time (6 hours is plenty for a 5k image dataset)
#BSUB -W 12:00

# Output files
#BSUB -o logs/output_%J_%I.out      # Standard output log (%J is JobID, %I is Array Index)
#BSUB -e logs/error_%J_%I.err

# Print job information
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Job Name: $LSB_JOBNAME"
echo "Start Time: $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Load necessary modules for DTU HPC (uncomment if required by the cluster)
# module load python/3.11
# module load cudnn/v9.13.0.50-prod-cuda-12.X
module load cuda/12.4

source /zhome/99/f/223556/02501-advanced-deep-learning-in-computer-vision/project/.venv/bin/activate


# Print GPU info to verify we got a card with enough VRAM (ideally 16GB+)
nvidia-smi

# Print Python environment info
echo "Python version:"
python --version

echo "UV version:"
uv --version

# Run the evaluation pipeline
echo "Starting VLM Zero-Shot Evaluation..."
python main.py

# Capture exit code
EXIT_CODE=$?

# Print completion info
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

# Exit with the same code as the main script
exit $EXIT_CODE