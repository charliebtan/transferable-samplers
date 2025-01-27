#!/bin/bash
#SBATCH -J train_nf                 # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=128G                     # server memory requested (per node)
#SBATCH -t 3:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=short-unkillable               # Request partition
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100l:4                  # Type/number of GPUs needed
#SBATCH -c 12
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption
#SBATCH --signal=SIGUSR1@90

RUN_NAME="al3_cfm_v2_v3"
srun python -u src/train.py experiment=aldp logger=wandb seed=42 data=al3 trainer=ddp \
model/net=egnn_dynamics_ad2_cat_v2 tags=[al,cnf,v11,egnn_v2] \
  trainer.check_val_every_n_epoch=200 data.batch_size=2048 \
  +data.repeat_factor=4 trainer.max_epochs=250 \
  model.sampling_config.batch_size=80 +trainer.num_sanity_val_steps=0 \
  model.sampling_config.num_proposal_samples=1000 model.net.hidden_nf=256 \
  trainer.strategy=ddp_find_unused_parameters_true data.num_workers=6 \
hydra.run.dir='${paths.log_dir}/${task_name}/runs/'${RUN_NAME} \
ckpt_path='${paths.log_dir}/${task_name}/runs/'${RUN_NAME}/checkpoints/last.ckpt \
logger.wandb.id=${RUN_NAME}
