#!/bin/bash
python src/train.py -m launcher=mila_48gb \
model=normalizing_flow logger=wandb \
data=aldp \
trainer=gpu trainer.max_epochs=1000 \
model.optimizer._target_=torch.optim.AdamW \
model.optimizer.weight_decay=0.0001 \
tags=[tarflow,mle,aldp] \
model.net.num_blocks=4,8,12,16 \
model.net.channels=64,128,256,512
