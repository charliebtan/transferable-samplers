#!/bin/bash
#SBATCH -J train                 # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64G                     # server memory requested (per node)
#SBATCH -t 1-00:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=long
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH -c 2
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption
#SBATCH --signal=SIGUSR1@90

#python src/train.py -m launcher=mila_48gb \

srun python src/train.py \
  experiment=aldp model/net=dit trainer=ddp  \
  data.batch_size=1024 +trainer.precision=bf16 \
  ++data.com_augmentation=True data=aldp model.sampling_config.batch_size=200 \
  model.sampling_config.num_proposal_samples=40 model.mean_free_prior=false \
  +trainer.num_sanity_val_steps=0 data.pin_memory=False \
  model.atol=1e-4 \
  model.rtol=1e-4 \
  model.div_estimator="exact_no_functional" \
  model.logp_tol_scale=1 \
  trainer.strategy=ddp_find_unused_parameters_true \
  callbacks.model_checkpoint.monitor=null \
  callbacks.model_checkpoint.save_top_k=-1 \
  callbacks.model_checkpoint.every_n_epochs=20 \
  callbacks.model_checkpoint.save_on_train_epoch_end=True \
  callbacks.model_checkpoint.verbose=True \
  seed=46 \
  trainer.check_val_every_n_epoch=200 \
  train=False \
  ckpt_path="/network/scratch/a/alexander.tong/fast-tbg/logs/train/runs/2025-01-24_16-54-30/checkpoints/epoch_999.ckpt" \
  #+model.clip_logits=0.01
  #ckpt_path="/network/scratch/a/alexander.tong/fast-tbg/logs/train/multiruns/2025-01-22_19-47-05/0/checkpoints/epoch_999.ckpt"
  #model.div_estimator="exact_no_functional" \
  #model.div_estimator="hutch_rademacher" \
  #model.div_estimator="exact_no_functional" \
  #hydra.run.dir=/network/scratch/a/alexander.tong/fast-tbg/logs/train/runs/2025-01-22_06-35-53/ \
  #logger.wandb.id=4gbrlzbs \
  #ckpt_path="/network/scratch/a/alexander.tong/fast-tbg/logs/train/runs/2025-01-22_06-35-53/checkpoints/last.ckpt"
  #ckpt_path=/network/scratch/a/alexander.tong/fast-tbg/logs/train/runs/2025-01-17_18-51-29/checkpoints/last.ckpt \
