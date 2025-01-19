#!/bin/bash
python src/train.py -m launcher=mila_48gb \
model=invertible_reflow logger=wandb \
data=aldp \
trainer=gpu trainer.max_epochs=1000 \
model.optimizer._target_=torch.optim.AdamW \
tags=[tarflow,mle,aldp,reflow_wall_grid] \
model.net.num_blocks=4 \
model.net.channels=256 \
trainer.check_val_every_n_epoch=50 \
model.mean_free_prior=1 \
+data.com_augmentation=1 \
+model.aligned_loss_fn=1 \
model.base_flow_ckpt_path=/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/runs/2025-01-16_08-21-37/checkpoints/last.ckpt \
model.optimizer.lr=0.0001,0.0005 \
model.optimizer.weight_decay=0.0001,0.001,0.01 \
model.net.in_channels=3 \
data.batch_size=512 \
data.num_workers=0 \
model.sampling_config.num_proposal_samples = 1000 \
model.sampling_config.num_jarzynski_samples = 100 \
model.sampling_config.num_eval_samples = 5000 \
model.sampling_config.batch_size = 1000