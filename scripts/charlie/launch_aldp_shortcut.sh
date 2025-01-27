#!/bin/bash
python src/train.py -m launcher=ox_a10 \
model=flow_matching_aldp logger=wandb \
trainer=gpu trainer.max_epochs=1000 \
model.optimizer.weight_decay=0.001,0.0001 \
model.optimizer.lr=0.0001,0.0005,0.001 \
tags=[ALDP,CNF,LR_search,WD_search]
