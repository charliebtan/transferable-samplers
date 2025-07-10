#!/bin/bash
python src/train.py -m launcher=mila_l40 \
experiment=training/tarflow_al3 \
model.net.dropout=0.0,0.1,0.2 \
tags=[sgb,al3,sbg_sweep_v1] \
logger=wandb \
seed=0,1,2