#!/bin/bash
python src/train.py -m launcher=mila_48gb \
model=normalizing_flow logger=wandb \
data=aldp \
trainer=gpu trainer.max_epochs=1000 \
model.optimizer._target_=torch.optim.AdamW \
model.optimizer.weight_decay=0.0001 \
tags=[tarflow,mle,best_practice] \
model.net.num_blocks=4 \
model.net.layers_per_block=4 \
model.net.channels=256 \
trainer.check_val_every_n_epoch=50 \
+model.force_gaussian_loss=1 \
model.mean_free_prior=1 \
+data.com_augmentation=1 \
model.net.in_channels=3 \
data.batch_size=256
