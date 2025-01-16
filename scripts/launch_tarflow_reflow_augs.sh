#!/bin/bash
python src/train.py -m launcher=mila_48gb \
model=invertible_reflow logger=wandb \
data=aldp \
trainer=gpu trainer.max_epochs=1000 \
model.optimizer._target_=torch.optim.AdamW \
model.optimizer.weight_decay=0.0001 \
tags=[tarflow,mle,aldp,reflow_augs_grid] \
model.net.num_blocks=4 \
model.net.channels=256 \
trainer.check_val_every_n_epoch=50 \
+model.mean_free_prior=1,0 \
+data.com_augmentation=1,0 \
+model.aligned_loss_fn=1,0 \
model.base_flow_ckpt_path=/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/runs/2025-01-16_08-21-37/checkpoints/last.ckpt \
data.batch_size=1024 \
data.num_workers=0
