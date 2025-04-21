#!/bin/bash
#SBATCH -J chore                 # Job name
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=48G                     # server memory requested (per node)
#SBATCH -t 24:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=main-cpu              # Request partition
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --open-mode=append            # Do not overwrite logs

python -u scripts/charlie/data_moving.py