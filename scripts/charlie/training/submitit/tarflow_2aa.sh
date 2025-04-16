#!/bin/bash
python src/train.py -m launcher=mila_l40_unkillable \
experiment=training/tarflow_2aa \
tags=[2aa] \
logger=wandb \
trainer=gpu \
seed=0
