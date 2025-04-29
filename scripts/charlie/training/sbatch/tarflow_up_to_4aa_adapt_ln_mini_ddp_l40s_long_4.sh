#!/bin/bash
#SBATCH -J train_nf                 # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=256G                     # server memory requested (per node)
#SBATCH -t 168:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=long               # Request partition
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:l40s:4                  # Type/number of GPUs needed
#SBATCH -c 4
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption
#SBATCH --signal=SIGUSR1@90

RUN_NAME="tarflow_up_to_4aa_adapt_ln_mini_v1"

srun python -u src/train.py \
experiment=training/tarflow_up_to_4aa logger=wandb \
trainer=ddp \
data.batch_size=256 \
trainer.limit_train_batches=2000 \
+trainer.accumulate_grad_batches=2 \
tags=[up_to_4aa,ddp,adapt_ln_mini] \
model.net.use_adapt_ln=True \
hydra.run.dir='${paths.log_dir}/${task_name}/runs/'${RUN_NAME} \
ckpt_path='${paths.log_dir}/${task_name}/runs/'${RUN_NAME}/checkpoints/last.ckpt \
logger.wandb.id=${RUN_NAME}
