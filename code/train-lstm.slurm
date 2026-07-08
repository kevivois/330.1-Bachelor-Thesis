#!/bin/bash
#SBATCH --job-name=LSTM_train
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --gres=shard:16
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=Dance



echo "test"
.venv/bin/python train-lstm.py
