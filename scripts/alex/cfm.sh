#!/bin/bash
#SBATCH -J train                 # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=24G                     # server memory requested (per node)
#SBATCH -t 2-00:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=main,long
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 2
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption
#SBATCH --signal=SIGUSR1@90

#python src/train.py -m launcher=mila_48gb \
#srun python -u src/train.py \
python -u src/train.py \
experiment=aldp logger=wandb \
data=aldp \
trainer=ddp trainer.max_epochs=1000 \
tags=[al,cnf,v6] \
trainer.strategy=ddp_find_unused_parameters_true \
trainer.check_val_every_n_epoch=200 \
data.batch_size=512 \
model.sampling_config.batch_size=256 \
model.sampling_config.num_proposal_samples=1000 \
model.sampling_config.num_test_proposal_samples=100000 \
model.net.hidden_nf=256 \
seed=44 \
data.num_workers=2 \
train=False \
ckpt_path="/network/scratch/a/alexander.tong/fast-tbg/logs/train/multiruns/2025-01-22_19-52-01/0/checkpoints/last.ckpt"
#hydra.launcher.gres=gpu:a100l:1 \
#data=al3,al4 \
#ckpt_path="/network/scratch/a/alexander.tong/fast-tbg/logs/train/multiruns/2025-01-21_21-15-55/0/checkpoints/last.ckpt" \
#hydra.run.dir="/network/scratch/a/alexander.tong/fast-tbg/logs/train/multiruns/2025-01-21_21-15-55/0" \
#logger.wandb.id=ojm8bj6g \
# python src/train.py experiment=aldp trainer=ddp data.batch_size=512 data=aldp model.sampling_config.batch_size=50 model.sampling_config.num_proposal_samples=1000 +trainer.num_sanity_val_steps=0 trainer.strategy=ddp_find_unused_parameters_true callbacks.model_checkpoint.monitor=null callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.every_n_epochs=20 callbacks.model_checkpoint.save_on_train_epoch_end=True callbacks.model_checkpoint.verbose=True seed=46 trainer.check_val_every_n_epoch=200 hydra.run.dir=/network/scratch/a/alexander.tong/fast-tbg/logs/train/multiruns/2025-01-22_06-44-35/1/ ckpt_path=/network/scratch/a/alexander.tong/fast-tbg/logs/train/multiruns/2025-01-22_06-44-35/1/checkpoints/last.ckpt logger.wandb.id=iro43lip model.net.hidden_nf=256 tags=[al,cnf,v6]
#
# python src/train.py experiment=aldp logger=wandb data=al4 trainer=gpu trainer.max_epochs=1000 tags=[al,cnf,v6] trainer.check_val_every_n_epoch=200 data.batch_size=512 model.sampling_config.batch_size=50 model.sampling_config.num_proposal_samples=1000 model.net.hidden_nf=256 seed=42 ckpt_path="/network/scratch/a/alexander.tong/fast-tbg/logs/train/multiruns/2025-01-22_06-44-35/0/checkpoints/last.ckpt" logger.wandb.id=qmsz0pyn hydra.run.dir="/network/scratch/a/alexander.tong/fast-tbg/logs/train/multiruns/2025-01-22_06-44-35/0/"
#
