#!/bin/bash
#SBATCH --job-name=TFT_train
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --gres=shard:48
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=Dance
#SBATCH --mem=50G



echo "test"
.venv/bin/python train-tft.py
