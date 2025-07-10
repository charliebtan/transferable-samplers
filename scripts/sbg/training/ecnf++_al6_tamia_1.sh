#!/bin/bash
#SBATCH -J train_nf                 # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --mem=256G                     # server memory requested (per node)
#SBATCH -t 3:00:00                  # Time limit (hh:mm:ss)
#SBATCH --account=aip-necludov               
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4                  # Type/number of GPUs needed
#SBATCH -c 4
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption
#SBATCH --signal=SIGUSR1@90

env=tbg3
module purge
module load python/3.11 cuda/12.2
module load openmm/8.2.0
module load httpproxy/1.0
source $HOME/envs/$env/bin/activate
wandb online

RUN_NAME=encf++_al6_1_v4

srun python -u src/train.py \
trainer=ddp \
trainer.strategy='ddp_find_unused_parameters_true' \
experiment=training/ecnf++_al6 \
data.data_dir="/project/aip-necludov/tanc/sbg_data" \
data.batch_size=512 \
model.sampling_config.batch_size=64 \
model.sampling_config.num_proposal_samples=64 \
trainer.accumulate_grad_batches=1 \
tags=[sgb,al6,ecnf++_sweep_v4] \
model.optimizer.weight_decay=1e-4 \
hydra.run.dir='${paths.log_dir}/${task_name}/runs/'${RUN_NAME} \
ckpt_path='${paths.log_dir}/${task_name}/runs/'${RUN_NAME}/checkpoints/last.ckpt \
logger.wandb.id=${RUN_NAME} \
seed=1
