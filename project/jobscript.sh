#!/bin/bash
#BSUB -J VLM_ZeroShot_COCO
#BSUB -q c02516  # Note: Verify if this should be the 02501 queue instead!
#BSUB -gpu "num=1:mode=exclusive_process"

# Email notifications (add your email with #BSUB -u your_email@dtu.dk if desired)
#BSUB -N

# CPU cores and memory
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"  # Increased to 32GB to safely load 7B models

# Max wall clock time (6 hours is plenty for a 5k image dataset)
#BSUB -W 6:00

# Output files
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err

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
# module load cuda/12.1

# Print GPU info to verify we got a card with enough VRAM (ideally 16GB+)
nvidia-smi

# Print Python environment info
echo "Python version:"
python --version

echo "UV version:"
uv --version

# Run the evaluation pipeline
echo "Starting VLM Zero-Shot Evaluation..."
uv run main.py

# Capture exit code
EXIT_CODE=$?

# Print completion info
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

# Exit with the same code as the main script
exit $EXIT_CODE