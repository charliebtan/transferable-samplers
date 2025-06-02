#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=training/ecnf_al3 \
tags=[sgb,al3,ecnf_sweep_v3] \
logger=wandb \
seed=0,1,2