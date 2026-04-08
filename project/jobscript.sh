#!/bin/bash
#BSUB -J DLCV-part-2-project
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"

# Email notifications
#BSUB -N

# CPU cores and memory
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"

# Max wall clock time (increased for training)
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

# Load any necessary modules (if needed)
# module load python/3.11
# module load cuda/11.8

# Print GPU info
nvidia-smi

# Print Python environment info
echo "Python version:"
python --version

echo "UV version:"
uv --version

# Run the main script with error handling
echo "Starting training..."
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