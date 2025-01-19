#!/bin/bash
python src/train.py -m launcher=mila_48gb \
model=normalizing_flow logger=wandb \
data=al3,al4 \
trainer=gpu trainer.max_epochs=1000 \
model.optimizer._target_=torch.optim.AdamW \
model.optimizer.weight_decay=0.0001 \
tags=[tarflow,mle,aldp,al34_model_size_grid,v2] \
model.net.num_blocks=4,6,8 \
model.net.layers_per_block=2,4,6 \
model.net.channels=256,512,1024 \
trainer.check_val_every_n_epoch=50 \
+model.force_gaussian_loss=1 \
model.mean_free_prior=1 \
+data.com_augmentation=1 \
data.batch_size=256
