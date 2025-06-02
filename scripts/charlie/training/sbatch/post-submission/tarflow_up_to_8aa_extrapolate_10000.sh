#!/bin/bash
#SBATCH -J extrapolate_10000                 # Job name
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

echo $SLURM_NNODES
RUN_NAME="tarflow_up_to_8aa_extrapolate_10000_v1"

srun python -u src/train.py \
experiment=training/tarflow_up_to_8aa logger=wandb \
trainer=ddp \
data.data_dir='/project/aip-necludov/shared/self-consume-bg/data/new' \
data.batch_size=256 \
trainer.limit_train_batches=2000 \
trainer.accumulate_grad_batches=2 \
model.net.use_adapt_ln=True \
model.net.use_transition=True \
model.net.use_attn_pair_bias=False \
model.net.perm_type=globloc \
model.net.cond_embed.sinusoid_div_value=10000 \
trainer.num_nodes=$SLURM_NNODES \
tags=[up_to_8aa,ddp,full,extrapolate] \
hydra.run.dir='${paths.log_dir}/${task_name}/runs/'${RUN_NAME} \
ckpt_path='${paths.log_dir}/${task_name}/runs/'${RUN_NAME}/checkpoints/last.ckpt \
logger.wandb.id=${RUN_NAME}
