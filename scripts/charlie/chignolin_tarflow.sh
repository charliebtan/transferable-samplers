#!/bin/bash
#SBATCH -J train_nf                 # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=975G                     # server memory requested (per node)
#SBATCH -t 3:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=short-unkillable               # Request partition
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100l:4                  # Type/number of GPUs needed
#SBATCH -c 48
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption
#SBATCH --signal=SIGUSR1@90

RUN_NAME="chignolin_tarflow_moonshot"

# python -u src/train.py \

srun python -u src/train.py \
model=normalizing_flow logger=wandb \
data=chignolin \
trainer=ddp trainer.max_epochs=1000 \
trainer.strategy=ddp_find_unused_parameters_true \
model.optimizer._target_=torch.optim.AdamW \
model.optimizer.weight_decay=4e-4 \
model.optimizer.lr=1e-4 \
model.sampling_config.batch_size=2048 \
model.sampling_config.num_proposal_samples=20_000 \
tags=[tarflow,mle,chignolin] \
model.net.num_blocks=8 \
model.net.layers_per_block=8 \
model.net.channels=512 \
trainer.check_val_every_n_epoch=100000 \
hydra.run.dir='${paths.log_dir}/${task_name}/runs/'${RUN_NAME} \
ckpt_path='${paths.log_dir}/${task_name}/runs/'${RUN_NAME}/checkpoints/last.ckpt \
logger.wandb.id=${RUN_NAME} \
+data.com_augmentation=1 \
+trainer.num_sanity_val_steps=0 \
data.batch_size=512 \
callbacks.model_checkpoint_time.every_n_epochs=5
# +trainer.limit_train_batches=10.0 \

