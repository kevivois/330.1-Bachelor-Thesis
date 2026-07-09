#!/bin/bash
#SBATCH --job-name=data-preprocessing  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=25G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --output=logs/job%j.out
#SBATCH --error=logs/job%j.err
#SBATCH --partition=Dance
#SBATCH --nodelist=disco


.venv/bin/python preprocessing.py