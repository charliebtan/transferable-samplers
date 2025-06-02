#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=training/ecnf_al2 \
tags=[sgb,al2,ecnf_sweep_v4] \
logger=wandb \
seed=0,1,2