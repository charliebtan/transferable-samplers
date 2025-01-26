#!/bin/bash
python src/train.py -m launcher=mila_48gb \
model=normalizing_flow logger=wandb \
data=aldp \
trainer=gpu trainer.max_epochs=1000 \
tags=[tarflow,mle,best_practice] \
model.net.num_blocks=4 \
model.net.layers_per_block=4 \
model.net.channels=256 \
trainer.check_val_every_n_epoch=50 \
+data.com_augmentation=1 \
model.sampling_config.num_proposal_samples=100_000 \
data.batch_size=256
