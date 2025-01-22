#!/bin/bash
python src/train.py -m launcher=mila_48gb \
model=normalizing_flow logger=wandb \
data=aldp \
trainer=gpu trainer.max_epochs=1000 \
model.optimizer._target_=torch.optim.AdamW \
tags=[tarflow,mle,aldp_final] \
model.net.num_blocks=4 \
model.net.layers_per_block=4 \
model.optimizer.weight_decay=4e-4 \
model.net.channels=256,128 \
trainer.check_val_every_n_epoch=50 \
+data.com_augmentation=1 \
model.net.in_channels=3 \
model.sampling_config.num_proposal_samples=100_000 \
data.batch_size=256 \
model.optimizer.lr=1e-4 \
seed=0,1,2