#!/bin/bash
python src/train.py -m launcher=mila_l40 \
experiment=training/ecnf++_al3 \
tags=[sgb,al3,ecnf++_sweep_v3] \
model.optimizer.weight_decay=1e-4 \
logger=wandb \
seed=0,1,2