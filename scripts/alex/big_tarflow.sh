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

RUN_NAME=""

#python -u src/train.py \
srun python -u src/train.py \
model=normalizing_flow logger=wandb \
data=al4 \
trainer=ddp trainer.max_epochs=1000 \
model.optimizer._target_=torch.optim.AdamW \
model.optimizer.weight_decay=0.0001 \
model.optimizer.lr=0.0001 \
model.sampling_config.batch_size=2000 \
model.sampling_config.num_proposal_samples=10000 \
tags=[tarflow,mle,aldp,big,v2] \
model.net.num_blocks=6 \
model.net.layers_per_block=6 \
model.net.channels=1024 \
trainer.check_val_every_n_epoch=50 \
hydra.run.dir='${paths.log_dir}/${task_name}/runs/'${RUN_NAME} \
ckpt_path='${paths.log_dir}/${task_name}/runs/'${RUN_NAME}/checkpoints/last.ckpt \
logger.wandb.id=${RUN_NAME} \
+model.force_gaussian_loss=1 \
model.mean_free_prior=1 \
+data.com_augmentation=1 \
+trainer.num_sanity_val_steps=0 \
callbacks.model_checkpoint.monitor=null \
callbacks.model_checkpoint.save_top_k=-1 \
callbacks.model_checkpoint.every_n_epochs=10 \
callbacks.model_checkpoint.save_on_train_epoch_end=True \
callbacks.model_checkpoint.verbose=True \
data.batch_size=2048
#+trainer.limit_train_batches=10.0 \
