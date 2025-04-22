#!/bin/bash
#SBATCH -J data-preproc                 # Job name
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                     # server memory requested (per node)
#SBATCH -t 48:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=unkillable               # Request partition
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100l:1                  # Type/number of GPUs needed
#SBATCH -c 4

srun python -u src/train.py experiment=training/tarflow_up_to_8aa \