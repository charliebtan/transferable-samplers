#!/bin/bash
python src/train.py -m launcher=mila_l40 \
experiment=training/tarflow_al3_tf \
tags=[al3,tarflow,transfer] \
logger=wandb \
trainer=gpu \
trainer.max_epochs=2000 \
data.batch_size=256 \
model.energy_kl_weight=0,1e-3 \
seed=0,1,2
