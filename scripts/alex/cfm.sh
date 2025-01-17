#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=aldp logger=wandb \
data=aldp,al3,al4 \
trainer=gpu trainer.max_epochs=1000 \
tags=[tarflow,mle,al,cnf,v5] \
trainer.check_val_every_n_epoch=50 \
data.batch_size=1024 \
model.net.hidden_nf=256 \
data.num_workers=2
