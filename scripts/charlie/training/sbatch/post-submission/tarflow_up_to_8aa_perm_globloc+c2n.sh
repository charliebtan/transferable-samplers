#!/bin/bash
#SBATCH -J perm_globloc+c2n                 # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 2                          # Total number of nodes requested
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
RUN_NAME="tarflow_up_to_8aa_perm_globloc+c2n_v1"

srun python -u src/train.py \
experiment=training/tarflow_up_to_8aa logger=wandb \
trainer=ddp \
data.data_dir='/project/aip-necludov/shared/self-consume-bg/data/new' \
data.batch_size=512 \
trainer.limit_train_batches=1000 \
model.net.use_adapt_ln=True \
model.net.use_transition=True \
model.net.use_attn_pair_bias=False \
model.net.perm_type="globloc+c2n" \
+model.net.cond_embed.sinusoid_div_value=1000 \
trainer.num_nodes=$SLURM_NNODES \
trainer.check_val_every_n_epoch=50 \
model.sampling_config.num_proposal_samples=10_000 \
model.sampling_config.clip_reweighting_logits=0.002 \
tags=[up_to_8aa,ddp,full,perms] \
hydra.run.dir='${paths.log_dir}/${task_name}/runs/'${RUN_NAME} \
ckpt_path='${paths.log_dir}/${task_name}/runs/'${RUN_NAME}/checkpoints/last.ckpt \
logger.wandb.id=${RUN_NAME}
