#!/bin/bash
#SBATCH --job-name=test_pytorch_dataloader
#SBATCH --output=logs/test_pytorch_dataloader.out
#SBATCH --error=logs/test_pytorch_dataloader.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:L40S:1

# Change to the project directory
cd /home/saksham3/projects/AIRe/robocoin

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate main

# print diagnostics
echo "Running on host: $(hostname)"
echo "Current conda environment: $CONDA_DEFAULT_ENV"

# Run the test script
python survey_scripts/test_pytorch_dataloader.py --use_multi --use_crop --batch_size 512 --num_workers 16
python survey_scripts/test_pytorch_dataloader.py --use_multi --batch_size 512 --num_workers 16