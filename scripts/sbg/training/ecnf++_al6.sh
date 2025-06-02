#!/bin/bash
python src/train.py -m launcher=mila_l40 \
experiment=training/ecnf++_al6 \
tags=[sgb,al6,ecnf++_sweep_v2] \
model.optimizer.weight_decay=1e-2,1e-3,1e-4 \
logger=wandb \
seed=0,1,2