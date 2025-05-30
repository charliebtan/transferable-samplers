#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=training/ecnf++_al4 \
tags=[sgb,al4,ecnf++_sweep_v1] \
model.optimizer.weight_decay=1e-3,1e-4 \
logger=wandb \
seed=0,1,2