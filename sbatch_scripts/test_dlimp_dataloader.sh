#!/bin/bash
#SBATCH --job-name=test_dlimp_dataloader
#SBATCH --output=logs/test_dlimp_dataloader.out
#SBATCH --error=logs/test_dlimp_dataloader.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:L40S:1

# Change to the project directory
cd /home/saksham3/projects/AIRe/robocoin

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dataloader_new

# print diagnostics
echo "Running on host: $(hostname)"
echo "Current conda environment: $CONDA_DEFAULT_ENV"

# Run the test script
python survey_scripts/test_dlimp_dataloader.py