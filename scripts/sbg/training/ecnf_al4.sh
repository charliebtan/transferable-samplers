#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=training/ecnf_al4 \
tags=[sgb,al4,ecnf_sweep_v1] \
logger=wandb \
seed=0,1,2