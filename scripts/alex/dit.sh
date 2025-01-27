#!/bin/bash
python src/train.py \
  experiment=aldp model/net=dit trainer=gpu  \
  data.batch_size=1024 +trainer.precision=bf16 \
  ++data.com_augmentation=True data=aldp model.sampling_config.batch_size=200 \
  model.sampling_config.num_proposal_samples=40 model.mean_free_prior=false \
  +trainer.num_sanity_val_steps=0 data.pin_memory=False \
  model.atol=1e-5 \
  model.rtol=1e-5 \
  model.div_estimator="exact_no_functional" \
  trainer.check_val_every_n_epoch=200 \
  train=False \
  ckpt_path="/network/scratch/a/alexander.tong/fast-tbg/logs/train/multiruns/2025-01-22_19-47-05/0/checkpoints/epoch_999.ckpt"
