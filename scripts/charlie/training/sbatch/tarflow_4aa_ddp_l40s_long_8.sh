#!/bin/bash
#SBATCH -J train_nf                 # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 2                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=48G                     # server memory requested (per node)
#SBATCH -t 168:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=long               # Request partition
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:l40s:4                  # Type/number of GPUs needed
#SBATCH -c 4
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption
#SBATCH --signal=SIGUSR1@90

RUN_NAME=$1

srun python -u src/train.py \
experiment=training/tarflow_4aa_ddp_8 logger=wandb \
tags=[4aa,ddp] \
hydra.run.dir='${paths.log_dir}/${task_name}/runs/'${RUN_NAME} \
ckpt_path='${paths.log_dir}/${task_name}/runs/'${RUN_NAME}/checkpoints/last.ckpt \
logger.wandb.id=${RUN_NAME} \
$2
