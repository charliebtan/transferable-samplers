#!/bin/bash
python src/train.py -m launcher=mila_l40s \
model=normalizing_flow logger=wandb \
data=aldp \
trainer=gpu trainer.max_epochs=1000 \
model.optimizer._target_=torch.optim.AdamW \
model.optimizer.weight_decay=0.0001 \
tags=[tarflow,mle,aldp,epochs_grid] \
model.net.num_blocks=4 \
model.net.channels=256 \
trainer.check_val_every_n_epoch=20 \
data.com_augmentation=0 \
trainer.max_epochs=250,500,1000,2000