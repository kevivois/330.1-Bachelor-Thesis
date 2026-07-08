#!/bin/bash
#SBATCH --job-name=TFT_Train  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --mem-per-cpu=25G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --output=logs/job%j.out
#SBATCH --error=logs/job%j.err
#SBATCH --partition=Dance
#SBATCH --time=24:00:00
#SBATCH --nodelist=disco
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1




echo "test"
.venv/bin/python train-k-fold-tft.py
